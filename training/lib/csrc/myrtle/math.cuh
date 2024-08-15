// Note: Any modifications to this CUDA code require recompilation for the changes to take effect.

#pragma once

#include <type_traits>

#include <cmath>

#include <ATen/NumericUtils.h>

#if defined __has_include

#if __has_include(<bit>)
#include <bit>
#endif

#if __has_include(<version>)
#include <version>
#endif

#endif

namespace myrtle {

namespace detail {

template <typename T, typename R = T>
using if_unsigned = typename std::enable_if<std::is_unsigned<T>::value, R>::type;

} // namespace detail

// Compute the result of `x / y` but rounding up instead of down.
template <typename T>
__device__ __host__ constexpr auto ceil_div(T x, T y) noexcept -> detail::if_unsigned<T> {
  return (x + y - 1) / y;
}

// Calculates the smallest integral power of two that is not smaller than `x`.
template <typename T>
__device__ __host__ constexpr auto bit_ceil(T x) noexcept -> detail::if_unsigned<T> {
  --x;
#pragma unroll
  for (T i = 1; i < sizeof(T) * CHAR_BIT; i *= 2) {
    x |= x >> i;
  }
  return ++x;
}

// If `x` is not zero, calculates the largest integral power of two that is not greater than x. If x
// is zero, returns zero.
template <typename T>
__device__ __host__ constexpr auto bit_floor(T x) noexcept -> detail::if_unsigned<T> {
  if (x == 0) {
    return 0;
  }
#pragma unroll
  for (T i = 1; i < sizeof(T) * CHAR_BIT; i *= 2) {
    x |= x >> i;
  }
  return x - (x >> 1);
}

// Checks if `x` is an integral power of two.
template <typename T>
__device__ __host__ constexpr auto has_single_bit(T n) noexcept -> detail::if_unsigned<T, bool> {
  return n && !(n & (n - 1));
}

} // namespace myrtle
