// Note: Any modifications to this CUDA code require recompilation for the changes to take effect.

#include <algorithm>
#include <cstdint>
#include <limits>

#include <ATen/AccumulateType.h>

#include <torch/extension.h>
#include "ATen/cuda/CUDAContext.h"
#include "c10/core/ScalarType.h"
#include "c10/cuda/CUDAException.h"

#include "myrtle/block_map_fold.cuh"
#include "myrtle/math.cuh"
#include "myrtle/utility.hpp"

namespace myrtle {

namespace {

template <typename T>
struct device_plus {
  __device__ __forceinline__ T operator()(T lhs, T rhs) const {
    return lhs + rhs;
  }
};

template <typename T>
struct device_max {
  __device__ __forceinline__ T operator()(T x, T y) const {
    return at::_isnan(x) ? x : x > y ? x : y;
  }
};

template <typename T>
struct device_zero {
  __device__ __forceinline__ T operator()() const {
    return 0;
  }
};

template <typename T>
struct device_lowest {
  __device__ __forceinline__ T operator()() const {
    return -std::numeric_limits<T>::infinity();
  }
};

// MD A functor to compute `exp(x - shift)`
template <typename T>
struct device_shift_exp {
  //
  T shift; // The shift value is a finite number.

  // When we use this operator we know there are no NaNs
  // or +inf in the input as the max pass would have
  // picked them up.

  __device__ __forceinline__ T operator()(T x) const {
    return ::exp(x - shift);
  }
};

template <uint32_t ThreadsInBlock, class Acc, class T, class O>
__global__ void batched_logsumexp_kernel(
    T const* __restrict__ in,
    uint32_t n,
    uint32_t stride,
    O* __restrict__ out //
) {
  T const* my_in = in + blockIdx.x * stride;

  uint32_t tid = threadIdx.x;

  // First-step compute max of each batch.

  T my_max =
      myrtle::block_map_fold<ThreadsInBlock, device_lowest<T>>(tid, my_in, n, device_max<T>{});

  __shared__ T global_max;

  if (tid == 0) {
    global_max = my_max;
  }

  __syncthreads();

  my_max = global_max; // Broadcast (no bank conflict).

  if (!::isfinite(my_max)) {
    if (tid == 0) {
      out[blockIdx.x] = my_max;
    }
    return;
  }
  // Now we can compute the sum of exp(x - max) for each batch.

  Acc my_sum = block_map_fold<ThreadsInBlock, device_zero<Acc>, Acc>(
      tid, my_in, n, device_plus<Acc>{}, device_shift_exp<Acc>{my_max});

  if (tid == 0) {
    out[blockIdx.x] = my_max + ::log(my_sum);
  }
};

template <class Acc, class T, class U>
void batched_logsumexp_dispatch(
    uint32_t max_threads,
    T const* d_in,
    uint32_t n,
    uint32_t stride,
    uint32_t batches,
    U* d_out) {
  //
  uint32_t constexpr lo = 1;
  uint32_t constexpr hi = 1024;

  using namespace detail;

  max_threads = std::max(lo, std::min(max_threads, hi)); // std::clamp is C++17
  max_threads = bit_floor(max_threads);
  max_threads = [&] {
    if (n <= max_threads * 2) {
      return bit_ceil(ceil_div<uint32_t>(n, 2));
    }
    return max_threads;
  }();

  uint32_t constexpr warp_size = 32; // Threads
  uint32_t constexpr bank_size = 4;  // Bytes
  uint32_t constexpr num_banks = 32;

  // Avoid bank conflicts, see comment in `block_map_fold`.
  while (ceil_div(max_threads, warp_size) * sizeof(Acc) > num_banks * bank_size) {
    // MD **NOTE:** `max_threads` is a power of two by bit_[floor/ceil] above
    max_threads /= 2;
  }

  uint32_t blocks = batches;

  TORCH_CHECK(max_threads != 0, "max_threads is zero");

  switch (max_threads) {
    case 1024:
      batched_logsumexp_kernel<1024, Acc><<<blocks, max_threads>>>(d_in, n, stride, d_out);
      break;
    case 512:
      batched_logsumexp_kernel<512, Acc><<<blocks, max_threads>>>(d_in, n, stride, d_out);
      break;
    case 256:
      batched_logsumexp_kernel<256, Acc><<<blocks, max_threads>>>(d_in, n, stride, d_out);
      break;
    case 128:
      batched_logsumexp_kernel<128, Acc><<<blocks, max_threads>>>(d_in, n, stride, d_out);
      break;
    case 64:
      batched_logsumexp_kernel<64, Acc><<<blocks, max_threads>>>(d_in, n, stride, d_out);
      break;
    case 32:
      batched_logsumexp_kernel<32, Acc><<<blocks, max_threads>>>(d_in, n, stride, d_out);
      break;
    case 16:
      batched_logsumexp_kernel<16, Acc><<<blocks, max_threads>>>(d_in, n, stride, d_out);
      break;
    case 8:
      batched_logsumexp_kernel<8, Acc><<<blocks, max_threads>>>(d_in, n, stride, d_out);
      break;
    case 4:
      batched_logsumexp_kernel<4, Acc><<<blocks, max_threads>>>(d_in, n, stride, d_out);
      break;
    case 2:
      batched_logsumexp_kernel<2, Acc><<<blocks, max_threads>>>(d_in, n, stride, d_out);
      break;
    case 1:
      batched_logsumexp_kernel<1, Acc><<<blocks, max_threads>>>(d_in, n, stride, d_out);
      break;
    default:
      TORCH_CHECK(false, "unreachable");
  }
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
// Log Sum Exp (pytorch)
////////////////////////////////////////////////////////////////////////////////

torch::Tensor batched_logsumexp(torch::Tensor in, uint32_t max_threads, bool promote) {
  //
  TORCH_CHECK(in.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(in.dim() == 2, "input must be a 2D tensor");

  TORCH_CHECK(in.stride(1) == 1, "input must be contiguous in the last dimension");
  TORCH_CHECK(in.stride(0) >= in.size(1), "input tensor must not alias itself");

  uint32_t const batches = in.size(0);
  uint32_t const n = in.size(1);
  uint32_t const stride = in.stride(0);

  auto opt = [&]() {
    if (promote) {
      return in.options().dtype(at::toAccumulateType(in.scalar_type(), true));
    } else {
      return in.options();
    }
  }();

  torch::Tensor out = torch::empty({batches}, opt, at::MemoryFormat::Contiguous);

  uint32_t device_max_threads = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;

  // clang-format off

  MYRTLE_DISPATCH_FLOATING_TYPES(in.scalar_type(), "logsumexp_cuda", ([&] {

    using acc_t = at::acc_type<scalar_t, true>;

    if (promote) {
      batched_logsumexp_dispatch<acc_t>(
        std::min(max_threads, device_max_threads),
        in.data_ptr<scalar_t>(),
        n,
        stride,
        batches,
        out.data_ptr<acc_t>()
      );
    } else {
      batched_logsumexp_dispatch<acc_t>(
        std::min(max_threads, device_max_threads),
        in.data_ptr<scalar_t>(),
        n,
        stride,
        batches,
        out.data_ptr<scalar_t>()
      );
    }
  }));

  C10_CUDA_CHECK(cudaGetLastError());

  return out;
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  //
  module.doc() = "Implementation of logsumexp with no memory overhead";

  module.def(
      "logsumexp",
      &myrtle::batched_logsumexp,
      "Compte the logSumExp along the last dimension of a rank-2 tensor with no memory overhead",
      py::arg("input"),
      py::arg("max_threads") = uint32_t(128),
      py::arg("promote") = false
  );
}
