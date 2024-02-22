// Note: Any modifications to this C++ code require recompilation for the changes to take effect.

#pragma once

#include <ATen/OpMathType.h>
#include <ATen/cuda/Exceptions.h>

#include <cublasLt.h>

#include "myrtle/utility.hpp"

// Here we expose the underlying cublas gemm functions to avoid torch's dynamic dispatch overhead.

namespace myrtle {

#define MYRTLE_CUSTOM_GEMM_ARGS(Scalar)                                                        \
  cublasHandle_t handle, cublasOperation_t opA, cublasOperation_t opB, int64_t m, int64_t n,   \
      int64_t k, at::opmath_type<Scalar> alpha, Scalar const *a, int64_t lda, Scalar const *b, \
      int64_t ldb, at::opmath_type<Scalar> beta, Scalar *c, int64_t ldc

namespace impl {

template <typename Scalar>
struct dispatch {
  static_assert(dependent_false<Scalar>::value, "Unsupported type passed to gemm");
};

template <>
struct dispatch<float> {
  static cublasStatus_t gemm(MYRTLE_CUSTOM_GEMM_ARGS(float)) {
    return cublasSgemm(handle, opA, opB, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
  }
};

template <>
struct dispatch<double> {
  static cublasStatus_t gemm(MYRTLE_CUSTOM_GEMM_ARGS(double)) {
    return cublasDgemm(handle, opA, opB, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
  }
};

// clang-format off

template <>
struct dispatch<at::Half> {
  static cublasStatus_t gemm(MYRTLE_CUSTOM_GEMM_ARGS(at::Half)) {
      return cublasGemmEx(
        handle, opA, opB, m, n, k, &alpha, a, CUDA_R_16F, lda, b, CUDA_R_16F, ldb, &beta, c, CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
  }
};

template <>
struct dispatch<at::BFloat16> {
  static cublasStatus_t gemm(MYRTLE_CUSTOM_GEMM_ARGS(at::BFloat16)) {
    return cublasGemmEx(
        handle, opA, opB, m, n, k, &alpha, a, CUDA_R_16BF, lda, b, CUDA_R_16BF, ldb, &beta, c, CUDA_R_16BF, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
  }
};

// clang-format on

} // namespace impl

template <typename Scalar>
void gemm(MYRTLE_CUSTOM_GEMM_ARGS(Scalar)) {
  //

  constexpr int64_t one = 1;

  if (n <= one) {
    ldc = std::max(m, one);
  }

  if (opA != CUBLAS_OP_N) {
    if (m <= one) {
      lda = std::max(k, one);
    }
  } else {
    if (k <= one) {
      lda = std::max(m, one);
    }
  }

  if (opB != CUBLAS_OP_N) {
    if (k <= one) {
      ldb = std::max(n, one);
    }
  } else {
    if (n <= one) {
      ldb = std::max(k, one);
    }
  }

  // clang-format off

  TORCH_CUDABLAS_CHECK(
      impl::dispatch<Scalar>::gemm(handle, opA, opB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
  );

  // clang-format on
}

#undef MYRTLE_CUSTOM_GEMM_ARGS

} // namespace myrtle
