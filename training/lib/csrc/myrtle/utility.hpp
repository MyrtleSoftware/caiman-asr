// Note: Any modifications to this C++ code require recompilation for the changes to take effect.

#pragma once

#include <limits>
#include <type_traits>

#include <torch/extension.h>

#include <ATen/cuda/Exceptions.h>
#include "ATen/Dispatch.h"

#include <cublasLt.h>

namespace myrtle {

/**
 * Until C++23's static_assert(false, "message")
 */
template <typename>
struct dependent_false : std::false_type {};

/**
 * The AT_DISPATCH_* macros does not seem to include bfloat16.
 */
#define MYRTLE_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, MYRTLE_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#define MYRTLE_DISPATCH_CASE_FLOATING_TYPES(...)        \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)   \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

static constexpr auto kMax32 = std::numeric_limits<int32_t>::max();

/**
 * See:
 *
 * https://github.com/pytorch/pytorch/blob/979f826015cbd2b353f02e93865a9b9a8877b414/aten/src/ATen/native/IndexingUtils.cpp#L5
 *
 * (it's not part of the public API)
 */
inline bool can_use_32bit_math(at::TensorBase const& t, int64_t max_elem = kMax32) {
  //
  auto elements = t.sym_numel();

  if (elements >= max_elem) {
    return false;
  }

  if (elements == 0) {
    return max_elem > 0;
  }

  c10::SymInt offset = 0;
  auto linearId = elements - 1;

  // NOTE: Assumes all strides are positive, which is true for now
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  for (int i = t.dim() - 1; i >= 0; --i) {
    auto curDimIndex = linearId % t.sym_size(i);
    auto curDimOffset = curDimIndex * t.sym_stride(i);
    offset += curDimOffset;
    linearId = linearId / t.sym_size(i);
  }

  if (offset >= max_elem) {
    return false;
  }

  return true;
}

inline bool same_scalar_type(torch::Tensor const& head) {
  return true;
}

template <typename... Args>
bool same_scalar_type(torch::Tensor const& ref, torch::Tensor const& head, Args const&... tail) {
  return ref.scalar_type() == head.scalar_type() && same_scalar_type(ref, tail...);
}

/**
 * A noop.
 */
template <typename Scalar>
struct scoped_math {
  scoped_math(cublasHandle_t) {}
};

/**
 * Disallow CUBLAS' reduced precision reductions for the duration of the object's lifetime.
 */
template <>
struct scoped_math<at::Half> {
  //
  scoped_math(scoped_math const&) = delete;
  scoped_math& operator=(scoped_math const&) = delete;

  scoped_math(cublasHandle_t handle) : m_handle(handle) {
    //
    cublasMath_t cublas_flags = CUBLAS_DEFAULT_MATH;

    constexpr auto reduced = CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION;

    if (!at::globalContext().allowFP16ReductionCuBLAS()) {
      cublas_flags = static_cast<cublasMath_t>(cublas_flags | reduced);
    }

    // Disallow fp16 reductions that could lead to unexpected overflow issues.
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, cublas_flags));
  }

  ~scoped_math() noexcept {
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(m_handle, CUBLAS_DEFAULT_MATH));
  }

 private:
  cublasHandle_t m_handle;
};

/**
 * Disallow CUBLAS' reduced precision reductions for the duration of the object's lifetime.
 */
template <>
struct scoped_math<at::BFloat16> : scoped_math<at::Half> {
  using scoped_math<at::Half>::scoped_math;
};

} // namespace myrtle
