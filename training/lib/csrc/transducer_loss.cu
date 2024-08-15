// Copyright (c) 2024, Myrtle Software Limited. All rights reserved.
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in
//       the documentation and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
//       may be used to endorse or promote products derived from this software
//       without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
// THE POSSIBILITY OF SUCH DAMAGE.

// This file is modified from:
//     https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/transducer

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "myrtle/math.cuh"
#include "myrtle/utility.hpp"

namespace {

template <typename T>
struct log_sum_exp {
  __device__ __forceinline__ T operator()(T a, T b) const {
    // standard log-sum-exp trick is used here to provide better numerical stability
    return (a >= b) ? a + std::log1p(exp(b - a)) : b + std::log1p(exp(a - b));
  }
};

template <typename Acc>
__device__ __forceinline__ Acc delay_penalty(Acc lambda, Acc t, Acc T) {
  return lambda * ((T - 1) / 2 - t);
}

template <typename Acc>
__device__ __forceinline__ Acc sub_or_nan(Acc num, Acc den) {
  return ::isfinite(den) ? num - den : std::numeric_limits<Acc>::quiet_NaN();
}

// Vanilla transducer loss function (i.e. forward-backward algorithm)
// Detail of this loss function can be found in:
// [1] Sequence Transduction with Recurrent Neural Networks.
// [2] DELAY-PENALIZED TRANSDUCER FOR LOW-LATENCY STREAMING ASR

// Forward (alpha) and backward (beta) path are launched together. Input is assumed to be
// converted into log scale by the preceding log_softmax layer Diagonal wavefront advancing
// usually used in dynamic programming is leveraged here. alpha and beta are of Acc type, as
// they are essentially accumulators.

// This loss function supports packed input where a tensor of shape [B, T, U, H] is packed into
// [B_packed, H].
// Don't-care region (t > aud_len) or (u > txt_len) is removed.
// To support the packed input, the starting offsets for each batch need to be specified with
// batch_offset.
template <typename Scalar, typename Acc>
__global__ void transducer_loss_forward_kernal(
    const Scalar* x,
    const Acc* denom,
    const int* label,
    const int* aud_len,
    const int* txt_len,
    const int64_t* batch_offset,
    myrtle::type_identity_t<Acc> lambda, // Disable type deduction as host has sigma as double.
    int64_t dict_size,                   // 64-bit indexing for data tensor
    int64_t blank_idx,
    int64_t max_flen,
    int64_t max_glen,
    bool packed,
    Acc* alpha,
    Acc* beta,
    Acc* loss //
) {
  const int batch = blockIdx.y;
  const int tid = threadIdx.x;
  const auto my_flen = aud_len[batch];
  // Note that start of the sentence is added as 1 here
  const auto my_glen = txt_len[batch] + 1;
  const auto my_label = label + batch * (max_glen - 1);
  const int64_t my_batch_offset =
      packed ? (batch == 0 ? 0 : batch_offset[batch - 1]) : batch * max_flen * max_glen;
  const int64_t my_stride = packed ? my_glen : max_glen;
  const Scalar* my_x = x + my_batch_offset * dict_size;
  const Acc* my_denom = denom + my_batch_offset;
  int u = tid;

  auto log_softmax_x = [=](auto t, auto u, auto k) -> Acc {
    return sub_or_nan<Acc>(
        my_x[(t * my_stride + u) * dict_size + k], my_denom[(t * my_stride + u)]);
  };

  if (blockIdx.x == 0) {
    // alpha path
    Acc* my_alpha = alpha + batch * max_flen * max_glen;

    if (u == 0) {
      my_alpha[0] = 0;
    }

    __syncthreads();

    for (int64_t step = 1; step < my_flen + my_glen - 1; ++step) {
      //
      // Move along the diagonal wavefront to leverage available parallelism
      //
      // step = 1 ... (T + U - 1)

      for (u = tid; u < my_glen; u += blockDim.x) {
        //
        // clang-format off

        int64_t t = step - u;

        if (t >= 0 and t < my_flen and u >= 0 and u < my_glen) {
          // Eq(16) in [1]
          if (u == 0) {
            // T != 0 because for t==0, u==0, we would require step == 0

            // alpha(t, u) = alpha(t-1, u) * null(t-1, u)
            my_alpha[t * max_glen + u] = my_alpha[(t - 1) * max_glen] + log_softmax_x(t - 1, 0, blank_idx);

          } else {

            // In both the following branches we consider y(t, u-1).
            // Follow Eq. 19 in [2] we need to apply:
            //
            //   y'(t, u - 1) = y(t, u - 1) + L * d(t)
            //
            // with:
            //
            //   d(t) = (T - 1) / 2 - t

            Acc penalty = delay_penalty<Acc>(lambda, t, my_flen);

            if (t == 0) {
              // alpha(t, u-1) = alpha(t, u-1) * y'(t, u-1)
              my_alpha[u] = my_alpha[u - 1] + log_softmax_x(t, u - 1, my_label[u - 1])  + penalty;
            } else {
              // alpha(t, u) = alpha(t-1, u) * null(t-1, u) + alpha(t, u-1) * y'(t, u-1)
              my_alpha[t * max_glen + u] = log_sum_exp<Acc>{}(
                my_alpha[(t - 1) * max_glen + u] + log_softmax_x(t - 1, u, blank_idx),
                my_alpha[(t) * max_glen + u - 1] + log_softmax_x(t, u - 1, my_label[u - 1]) + penalty
              );
            }
          }
        }

        // clang-format on
      }
      __syncthreads();
    }
  } else if (blockIdx.x == 1) {
    // beta path
    Acc* my_beta = beta + batch * max_flen * max_glen;

    // clang-format off

    if (u == 0) {
      // No delay penalty as emitting a blank
      my_beta[(my_flen - 1) * max_glen + my_glen - 1] = log_softmax_x(my_flen - 1, my_glen - 1, blank_idx);
    }

    __syncthreads();

    for (int64_t step = my_flen + my_glen - 3; step >= 0; --step) {
      for (u = tid; u < my_glen; u += blockDim.x) {

        int64_t t = step - u;

        if (t >= 0 and t < my_flen and u >= 0 and u < my_glen) {
          // Eq(18) in [1]
          if (u == my_glen - 1) {
            // beta(t, u) = beta(t+1, u) * null(t, u)
            my_beta[t * max_glen + u] = my_beta[(t + 1) * max_glen + u] + log_softmax_x(t, u, blank_idx);
          } else {

            // Penalty just like in the alpha path.

            Acc penalty = delay_penalty<Acc>(lambda, t, my_flen);

            if (t == my_flen - 1) {
              // beta(t, u) = beta(t, u+1) * y'(t, u)
              my_beta[t * max_glen + u] = my_beta[t * max_glen + u + 1] + log_softmax_x(t, u, my_label[u]) + penalty;
            } else {
              // beta(t, u) = beta(t+1, u) * null(t, u) + beta(t, u+1) * y'(t, u)
              my_beta[t * max_glen + u] = log_sum_exp<Acc>{}(
                my_beta[(t + 1) * max_glen + u + 0] + log_softmax_x(t, u, blank_idx),
                my_beta[(t + 0) * max_glen + u + 1] + log_softmax_x(t, u, my_label[u]) + penalty
              );
            }
          }
        }
      }
      __syncthreads();
    }

    // clang-format on

    if (tid == 0) {
      loss[batch] = -my_beta[0];
    }
  }
}

// Fused transudcer loss backward operation.
// Detail of this loss function can be found in:
// [1] Sequence Transduction with Recurrent Neural Networks.
// The bwd op of the preceding softmax layer is fused in this kernel.
// Each thread block works on [batch, t, u, :] of data. Each thread works on a specific h at a time

// To support the packed input, the starting offsets for each batch need to be specified with
// batch_offset.
template <typename Scalar, typename Acc>
__global__ void transducer_loss_fused_backward_kernal(
    const Scalar* x,
    const Acc* denom,
    const Acc* loss_grad,
    const int* aud_len,
    const int* txt_len,
    const int* label,
    const Acc* alpha,
    const Acc* beta,
    const int64_t* batch_offset,
    myrtle::type_identity_t<Acc> lambda, // Disable type deduction as host has sigma as double.
    int64_t dict_size,
    int64_t blank_idx,
    int64_t max_flen,
    int64_t max_glen,
    bool packed,
    Scalar* x_grad //
) {
  const int tid = threadIdx.x;
  const int u = blockIdx.x;
  const int t = blockIdx.y;
  const int batch = blockIdx.z;
  const int64_t my_flen = aud_len[batch];
  const int64_t my_glen = txt_len[batch] + 1;
  const int64_t my_batch_offset =
      packed ? (batch == 0 ? 0 : batch_offset[batch - 1]) : batch * max_flen * max_glen;
  const int64_t my_stride = packed ? my_glen : max_glen;

  __shared__ Acc common, beta_TU, beta_TUp1, beta_Tp1U, my_label_shared;

  auto my_x_grad = x_grad + (my_batch_offset + t * my_stride + u) * dict_size;

  if (t < my_flen and u < my_glen) {
    auto my_x = x + (my_batch_offset + t * my_stride + u) * dict_size;
    auto my_denom = denom + my_batch_offset + t * my_stride + u;
    auto my_alpha = alpha + batch * max_flen * max_glen;
    auto my_beta = beta + batch * max_flen * max_glen;
    auto my_label = label + batch * (max_glen - 1);

    // load and store shared variables in SMEM
    if (tid == 0) {
      common = std::log(loss_grad[batch]) + my_alpha[t * max_glen + u] - my_beta[0];
      beta_TU = my_beta[t * max_glen + u];

      if (t != my_flen - 1) {
        beta_Tp1U = my_beta[(t + 1) * max_glen + u];
      }

      if (u != my_glen - 1) {
        beta_TUp1 = my_beta[t * max_glen + u + 1];
        my_label_shared = my_label[u];
      }
    }

    __syncthreads();

    for (int64_t h = tid; h < dict_size; h += blockDim.x) {
      Acc grad = common + sub_or_nan<Acc>(my_x[h], *my_denom);
      Acc my_grad = std::exp(grad + beta_TU); //

      if (u != my_glen - 1 and h == my_label_shared) {
        // Must apply the emmission penalty  just like in the forward pass.
        my_grad -= std::exp(grad + beta_TUp1 + delay_penalty<Acc>(lambda, t, my_flen));
      } else if (h == blank_idx) {
        if (t == my_flen - 1 and u == my_glen - 1) {
          my_grad -= std::exp(grad);
        } else if (t != my_flen - 1) {
          my_grad -= std::exp(grad + beta_Tp1U);
        }
      }

      my_x_grad[h] = my_grad;
    }
  } else if (!packed) {
    // In non-pack mode, need to make sure the gradients for don't-care regions are zero.
    for (int64_t h = tid; h < dict_size; h += blockDim.x) {
      my_x_grad[h] = 0;
    }
  }
}

std::vector<torch::Tensor> transducer_loss_cuda_forward(
    torch::Tensor x,
    torch::Tensor denom,
    torch::Tensor label,
    torch::Tensor aud_len,
    torch::Tensor txt_len,
    torch::Tensor batch_offset,
    double lambda,
    int max_flen,
    int blank_idx,
    bool packed //
) {
  MYRTLE_CHECK_INPUT(x);
  MYRTLE_CHECK_INPUT(denom);
  MYRTLE_CHECK_INPUT(label);
  MYRTLE_CHECK_INPUT(aud_len);
  MYRTLE_CHECK_INPUT(txt_len);

  if (packed) {
    MYRTLE_CHECK_INPUT(batch_offset);
  }

  auto scalar_type = x.scalar_type();
  auto tensor_opt = x.options();

  const int batch_size = label.size(0);
  const int max_glen = label.size(1) + 1;
  const int dict_size = x.size(-1);

  TORCH_CHECK(
      blank_idx >= 0 and blank_idx < dict_size,
      "Expected blank index to be in the range of 0 to ",
      dict_size - 1,
      ", but got ",
      blank_idx);

  // The data type of alpha and beta will be resolved at dispatch time,
  // hence defined here and assigned later
  torch::Tensor alpha;
  torch::Tensor beta;
  torch::Tensor loss;

  const auto device_prop = at::cuda::getCurrentDeviceProperties();
  const auto max_threads_per_block = device_prop->maxThreadsPerBlock;
  const auto batch_offset_ptr = packed ? batch_offset.data_ptr<int64_t>() : nullptr;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int threads = std::min(max_threads_per_block, max_glen);
  const dim3 blocks(2, batch_size, 1);

  MYRTLE_DISPATCH_FLOATING_TYPES(
      scalar_type, "transducer_loss_cuda_forward", ([&] {
        // resolve accumulation type
        using acc_t = at::acc_type<scalar_t, true>;
        auto acc_type = c10::CppTypeToScalarType<acc_t>::value;
        auto acc_tensor_opt = tensor_opt.dtype(acc_type);

        alpha = torch::empty({batch_size, max_flen, max_glen}, acc_tensor_opt);
        beta = torch::empty({batch_size, max_flen, max_glen}, acc_tensor_opt);
        loss = torch::empty({batch_size}, acc_tensor_opt);

        transducer_loss_forward_kernal<<<blocks, threads, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            denom.data_ptr<acc_t>(),
            label.data_ptr<int>(),
            aud_len.data_ptr<int>(),
            txt_len.data_ptr<int>(),
            batch_offset_ptr,
            lambda,
            dict_size,
            blank_idx,
            max_flen,
            max_glen,
            packed,
            alpha.data_ptr<acc_t>(),
            beta.data_ptr<acc_t>(),
            loss.data_ptr<acc_t>() //
        );
      }));

  C10_CUDA_CHECK(cudaGetLastError());

  return {std::move(alpha), std::move(beta), std::move(loss)};
}

torch::Tensor transducer_loss_cuda_backward(
    torch::Tensor x,
    torch::Tensor denom,
    torch::Tensor loss_grad,
    torch::Tensor alpha,
    torch::Tensor beta,
    torch::Tensor aud_len,
    torch::Tensor txt_len,
    torch::Tensor label,
    torch::Tensor batch_offset,
    double lambda,
    int max_flen,
    int blank_idx,
    bool packed //
) {
  MYRTLE_CHECK_INPUT(x);
  MYRTLE_CHECK_INPUT(denom);
  MYRTLE_CHECK_INPUT(label);
  MYRTLE_CHECK_INPUT(loss_grad);
  MYRTLE_CHECK_INPUT(alpha);
  MYRTLE_CHECK_INPUT(beta);
  MYRTLE_CHECK_INPUT(aud_len);
  MYRTLE_CHECK_INPUT(txt_len);

  if (packed) {
    MYRTLE_CHECK_INPUT(batch_offset);
  }

  auto dtype = x.scalar_type();
  torch::Tensor x_grad;
  const int batch_size = label.size(0);
  const int max_glen = label.size(1) + 1;
  const int dict_size = x.size(-1);
  const auto device_prop = at::cuda::getCurrentDeviceProperties();
  const int max_threads_per_block = device_prop->maxThreadsPerBlock;
  const int warp_size = device_prop->warpSize;
  const auto batch_offset_ptr = packed ? batch_offset.data_ptr<int64_t>() : nullptr;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // alloc empty tensors for performance, hence need to ensure zeros are writtern to
  // don't-care region in the kernel.
  x_grad = torch::empty_like(x);

  // Would like each thread to work on 4 hidden units
  const int work_per_thread = 4;
  // Don't want to have more than 128 threads per thread block
  const int max_thread_per_elem = std::min(128, max_threads_per_block);
  const int threads = std::min(
      max_thread_per_elem,
      std::max(warp_size, (dict_size + work_per_thread - 1) / work_per_thread));
  const dim3 blocks(max_glen, max_flen, batch_size);

  MYRTLE_DISPATCH_FLOATING_TYPES(
      dtype, "transducer_loss_cuda_backward", ([&] {
        //
        using acc_t = at::acc_type<scalar_t, true>;

        transducer_loss_fused_backward_kernal<<<blocks, threads, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            denom.data_ptr<acc_t>(),
            loss_grad.data_ptr<acc_t>(),
            aud_len.data_ptr<int>(),
            txt_len.data_ptr<int>(),
            label.data_ptr<int>(),
            alpha.data_ptr<acc_t>(),
            beta.data_ptr<acc_t>(),
            batch_offset_ptr,
            lambda,
            dict_size,
            blank_idx,
            max_flen,
            max_glen,
            packed,
            x_grad.data_ptr<scalar_t>());
      }));

  C10_CUDA_CHECK(cudaGetLastError());

  return x_grad;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  //
  m.doc() = "Apex's transducer loss fwd/bwd with emission penalty kernel";

  m.def("forward", &transducer_loss_cuda_forward, "transducer loss forward (CUDA)");
  m.def("backward", &transducer_loss_cuda_backward, "transducer loss backward (CUDA)");
}
