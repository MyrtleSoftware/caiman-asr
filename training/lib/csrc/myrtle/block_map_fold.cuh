// Note: Any modifications to this CUDA code require recompilation for the changes to take effect.

#pragma once

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <ATen/cuda/DeviceUtils.cuh> // For WARP_SHFL_DOWN

#include "myrtle/math.cuh"

namespace myrtle {

namespace detail {

template <class T>
__host__ __device__ T deduce(T);

// Get the type of the static I::value member.
template <class I>
using deduce_value_t = std::invoke_result_t<I>;

// Reduce over a warp/subwarp implementation.
template <std::size_t N>
struct warp {
  template <class I, class T, class F>
  __device__ __forceinline__ static auto fold(I mask, T my_acc, F bop) -> T {
    return warp<N / 2>::fold(mask, bop(my_acc, WARP_SHFL_DOWN(my_acc, N / 2, warpSize, mask)), bop);
  }
};
template <>
struct warp<1> {
  template <class I, class T, class F>
  __device__ __forceinline__ static auto fold(I, T my_acc, F) -> T {
    return my_acc;
  }
};

/* [markdown]
 *
 * # Sub-warp reduction
 *
 * Fold `my_acc` across a warp or part of a warp.
 *
 * This is guaranteed to inlined into a set of `__shfl_down_sync` intrinsics.
 *
 * **TODO:** In a future version of CUDA `__reduce_add_sync` may work with floating-point
 * and this function will need to be reworked.
 */
template <std::size_t ThreadsInWarp, std::size_t WarpSize, class T, class F>
__device__ __forceinline__ auto sub_warp_fold(T my_acc, F bop) -> T {
  //
  static_assert(ThreadsInWarp <= WarpSize, "Can't have more threads per warp than the warp size!");

  static_assert(has_single_bit(ThreadsInWarp), "ThreadsInWarp must be a power of 2!");
  static_assert(has_single_bit(WarpSize), "We assume the the warp size is a power of 2!");

  // Need to build a mask for the threads in the warp that will participate (and need to be
  // synchronized).

  static_assert(WarpSize <= 32, "Would need a bigger int type for mask");

  // MD Each shift **removes** a thread from the mask.
  uint32_t constexpr mask = 0xffffffff >> (WarpSize - ThreadsInWarp);

  return detail::warp<ThreadsInWarp>::fold(mask, my_acc, bop);
}

} // namespace detail

// A device functor that returns its argument.
struct identity_fn {
  template <class T>
  __device__ __forceinline__ T operator()(T x) const {
    return x;
  }
};

/* [markdown]
 *
 * # Block map-fold
 *
 * Use a single block of threads to map-fold an array of length n. Only the thread with tid = 0
 * will return the correct result (the rest have undefined values). This is designed to be most
 * efficient for small `k > 0` such that: `n ≈ 2 * k * ThreadsInBlock`.
 *
 * A full specification in haskell'ish notation:
 *
 * Let `map :: T -> R`
 * Let `bop :: R -> R -> R`
 * Let `id` be `Identity::value` and of type `R`
 * Let `a`, `b` be any values of type `R`
 *
 * Requirement: `(bop, id)` forms a commutative monoid over `R`, that is bop must be commutative and
 * associative and `id` must be its right/left identity, concretely:
 * ```haskell
 *           bop a b == bop b a
 *   bop a (bop b c) == bop (bop a b) c
 *          bop a id == a
 *          bop id a == a
 * ```
 *
 * **NOTE:** this is slightly over-constrained but it's closer to the mathematical intention.
 *
 * Then for a sequence of values `[a, b, c, ...]` this function will return:
 *
 *   `id ⊗ (map a) ⊗ (map b) ⊗ (map c) ⊗ ...`
 *
 * Where `⊗` is an infix version of `bop` such that `a ⊗ b == bop a b`.
 */
template <
    uint32_t ThreadsInBlock,
    class Identity,
    class R = detail::deduce_value_t<Identity>,
    class T,
    class Bop,
    class Map = identity_fn //
    >
__device__ R block_map_fold(uint32_t tid, T const* in, uint32_t n, Bop bop = {}, Map map = {}) {
  // We need the warp size at compile time but warpSize is a runtime constant in CUDA.
  uint32_t constexpr warp_size = 32;
  assert(warp_size == warpSize);

  static_assert(has_single_bit(ThreadsInBlock), "(It's ok to launch more than n threads)");
  static_assert(has_single_bit(warp_size), "We assume the the warp size is a power of 2!");

  R my_acc = Identity{}(); // Per-thread register to accumulate into.

  /* [markdown]
   *
   * Perform first level of reduction, all threads in the block will read from global memory, and
   * accumulate into `my_acc` (register). We fold multiple elements per thread if required.
   */
  for (uint32_t i = tid; i < n; i += ThreadsInBlock) {
    my_acc = bop(my_acc, map(in[i]));
  }

  /* [markdown]
   *
   * To perform second level of reduction, we will fold each warp to a single value. Currently,
   * every thread in the block has a partial reduction of the input stored in its `my_acc` register
   * (or a mapped identity element if `tid >= n`).
   *
   * Let `W <- warp_size`
   *
   * As ThreadsInBlock and warp_size are powers of two we have three cases:
   *    1. `ThreadsInBlock ∈ {1, 2,... W/2}`
   *    2. `ThreadsInBlock ∈ {W}`
   *    3. `ThreadsInBlock ∈ {2W, 4W, 8W...}`
   *
   * In cases (1) and (2) a single warp can handle the entire map-fold. But, in case (1)
   * `sub_warp_fold` needs threads_in_warp < warp_size. In case (3) we need multiple
   * warps to handle the map-fold but all the warps have threads_in_warp == warp_size.
   */

  static_assert(ThreadsInBlock < warp_size || ThreadsInBlock % warp_size == 0, "");

  uint32_t constexpr num_warps = ceil_div(ThreadsInBlock, warp_size);
  uint32_t constexpr threads_in_warp = num_warps == 1 ? ThreadsInBlock : warp_size;

  my_acc = detail::sub_warp_fold<threads_in_warp, warp_size>(my_acc, bop);

  if // Save a leaky macro as we can.
#ifdef __cpp_if_constexpr
      constexpr
#endif
      (num_warps > 1) {

    /* [markdown]
     *
     * Iff we have more than one warp's worth of threads we need a third and final level of
     * reduction between warps. We communicate the warp's partial reductions via shared memory then
     * the lowest warp will `sub_warp_fold` again.
     *
     * On a gpu shared memory blocks are divided into 32 banks, reads/writes to the same banks are
     * serialized. Address of **byte** `k` is in bank `(k / 4) mod 32`.
     *
     * If `R` is 4 bytes (e.g. `float`) as `num_warps <= 32` we have no bank conflicts.
     *
     * If `R` is greater than 4 bytes (e.g. `double`) We will have bank conflicts if `num_warps *
     * sizeof(R)` is greater than 4 * 32. This is not solvable without a further reduction. To avoid
     * this launch with appropriate number of threads.
     *
     * If `R` is less than 4 bytes (e.g. `__half`) we will have bank conflicts for adjacent warps,
     * e.g. between warp 0 and 1. To avoid this we will pad the shared memory.
     */

    std::size_t constexpr bank_width = 4;

    // Alignment forces compiler to pad such that arrays are contiguous.
    struct alignas(std::max(bank_width, alignof(R))) bank_padded {
      R val;
    };

    __shared__ bank_padded partials[num_warps];

    if ((tid % warp_size) == 0) {
      // Lowest thread in warp thread puts its local sum into shared memory.
      partials[tid / warp_size].val = my_acc;
    }

    __syncthreads(); // First and only block level sync!

    if (tid < num_warps) {
      // MD **NOTE:** `num_warps` is a power of two, see `static_assert`s above.
      my_acc = detail::sub_warp_fold<num_warps, warp_size>(partials[tid].val, bop);
    }
  }

  return my_acc;
}

} // namespace myrtle
