// Note: Any modifications to this CUDA code require recompilation for the changes to take effect.

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include "ATen/cuda/CUDAContext.h"

#include "myrtle/gemm.hpp"
#include "myrtle/utility.hpp"

namespace {

enum class Kind {
  hard, // Use hard-tanh and hard-sigmoid.
  soft, // Use tanh and sigmoid.
};

template <typename Scalar, Kind>
struct Math {
  static_assert(myrtle::dependent_false<Scalar>::value, "Math requested for unsupported kind");
};

template <typename Scalar>
struct Math<Scalar, Kind::soft> {
  __device__ __forceinline__ static Scalar sigm(Scalar z) noexcept {
    return 1.0 / (1.0 + ::exp(-z));
  }

  __device__ __forceinline__ static Scalar tanh(Scalar z) noexcept {
    return ::tanh(z);
  }

  __device__ __forceinline__ static Scalar sigm_prime(Scalar z) noexcept {
    return (1 - z) * z;
  }

  __device__ __forceinline__ static Scalar tanh_prime(Scalar z) noexcept {
    return 1 - (z * z);
  }
};

template <typename Scalar>
struct Math<Scalar, Kind::hard> {
  __device__ __forceinline__ static Scalar clamp(Scalar z, Scalar lo, Scalar hi) noexcept {
    return ::max(lo, ::min(z, hi));
  }

  __device__ __forceinline__ static Scalar sigm(Scalar z) noexcept {
    Scalar const half = .5; // Constants stop implicit conversion to double.
    Scalar const frac = 8.; // Cheaper in hardware than non-powers-of-2
    return clamp(half + z / frac, 0, 1);
  }

  __device__ __forceinline__ static Scalar tanh(Scalar z) noexcept {
    return clamp(z, -1, 1);
  }

  __device__ __forceinline__ static Scalar sigm_prime(Scalar z) noexcept {
    // This is compiled to branchless code.
    // Floating point comparison is exact as they are generated from clamp.
    if (z == 0 || z == 1) {
      return 0;
    } else {
      return 1. / 8.;
    }
  }

  __device__ __forceinline__ static Scalar tanh_prime(Scalar z) noexcept {
    // This is compiled to branchless code.
    // Floating point comparison is exact as they are generated from clamp.
    if (z == -1 || z == 1) {
      return 0;
    } else {
      return 1;
    }
  }
};

/**
 * Number of threads per block to use for CUDA kernels.
 *
 * Must be less than or equal to 1024, 256 seems to give best performance.
 */
static constexpr std::size_t kThreadsPerBlock = 256;

template <typename Index, typename Scalar, Kind K>
__global__ void lstm_cuda_fwd_kernel(
    Scalar const* __restrict__ ct_0,
    Scalar* __restrict__ gt,
    Scalar* __restrict__ ct_1,
    Scalar* __restrict__ yt_1,
    Index state_size //
) {
  // Batch index
  Index const b = blockIdx.y;
  // Column index
  Index const n = blockIdx.x * blockDim.x + threadIdx.x;

  if (n < state_size) {
    Index const oI = Index{0} * state_size + Index{4} * b * state_size + n;
    Index const oF = Index{1} * state_size + Index{4} * b * state_size + n;
    Index const oG = Index{2} * state_size + Index{4} * b * state_size + n;
    Index const oO = Index{3} * state_size + Index{4} * b * state_size + n;

    Index const off = b * state_size + n;

    Scalar* const I = gt + oI;
    Scalar* const F = gt + oF;
    Scalar* const G = gt + oG;
    Scalar* const O = gt + oO;

    auto* const Ct_0 = ct_0 + off;
    auto* const Ct_1 = ct_1 + off;
    auto* const Yt_1 = yt_1 + off;

    // Load, add, transform.

    Scalar i = Math<Scalar, K>::sigm(*I);
    Scalar f = Math<Scalar, K>::sigm(*F);
    Scalar g = Math<Scalar, K>::tanh(*G);
    Scalar o = Math<Scalar, K>::sigm(*O);

    Scalar c = i * g + (f * *Ct_0);
    Scalar y = o * Math<Scalar, K>::tanh(c);

    // Write

    *I = i;
    *F = f;
    *G = g;
    *O = o;

    *Ct_1 = c;
    *Yt_1 = y;
  }
}

template <typename Index, typename Scalar, Kind K>
__global__ void lstm_cuda_bwd_kernel(
    Scalar const* __restrict__ gates,
    Scalar const* __restrict__ dY,
    Scalar const* __restrict__ c_prev,
    Scalar const* __restrict__ c,
    Scalar* __restrict__ dGates, // out
    Scalar* __restrict__ dC,     // in/out
    Index state_size             //
) {
  // Batch index
  Index const b = blockIdx.y;
  // Column index
  Index const n = blockIdx.x * blockDim.x + threadIdx.x;

  if (n < state_size) {
    // Index for load

    Index const oI = Index{0} * state_size + Index{4} * b * state_size + n;
    Index const oF = Index{1} * state_size + Index{4} * b * state_size + n;
    Index const oG = Index{2} * state_size + Index{4} * b * state_size + n;
    Index const oO = Index{3} * state_size + Index{4} * b * state_size + n;

    Index const off = b * state_size + n;

    Scalar const* const I = gates + oI;
    Scalar const* const F = gates + oF;
    Scalar const* const G = gates + oG;
    Scalar const* const O = gates + oO;

    Scalar const* const _dY = dY + off;
    Scalar const* const _c_prev = c_prev + off;
    Scalar const* const _c = c + off;

    Scalar const* const c_fut = dC + off;

    // Load

    Scalar const dc_fut = *c_fut;

    Scalar const i = *I;
    Scalar const f = *F;
    Scalar const g = *G;
    Scalar const o = *O;

    Scalar const dy = *_dY;
    Scalar const prev = *_c_prev;
    Scalar const c_tanh = Math<Scalar, K>::tanh(*_c);

    // Compute

    Scalar const dO = dy * c_tanh * Math<Scalar, K>::sigm_prime(o);
    Scalar const dc = dy * o * Math<Scalar, K>::tanh_prime(c_tanh) + dc_fut;
    Scalar const dF = dc * prev * Math<Scalar, K>::sigm_prime(f);
    Scalar const dI = dc * g * Math<Scalar, K>::sigm_prime(i);
    Scalar const dG = dc * i * Math<Scalar, K>::tanh_prime(g);

    // Index for store

    Scalar* const I_ = dGates + oI;
    Scalar* const F_ = dGates + oF;
    Scalar* const G_ = dGates + oG;
    Scalar* const O_ = dGates + oO;

    Scalar* const dC_ = dC + off;

    // Store

    *I_ = dI;
    *F_ = dF;
    *G_ = dG;
    *O_ = dO;

    *dC_ = dc * f;
  }
}

template <typename Index, typename Scalar, Kind K>
void lstm_fused_fwd_impl(
    torch::Tensor const R,
    torch::Tensor gates,
    torch::Tensor c,
    torch::Tensor y //
) {
  //
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  myrtle::scoped_math<Scalar> scope(handle);

  int64_t const seq_length = gates.size(0);
  int64_t const batch_size = c.dim() == 3 ? c.size(1) : 1;
  int64_t const state_size = c.dim() == 3 ? c.size(2) : c.size(1);

  // If state_size is less than kThreadsPerBlock we should not waste threads.
  // Use the next multiple of 32 (e.g.) the warp size (hardware constraint).
  int64_t const threads = std::min(int64_t{kThreadsPerBlock}, ((state_size + 31) / 32) * 32);

  // Grid will be (_, batch_size)
  dim3 const blocks((state_size + threads - 1) / threads, batch_size);

  // All this needs to be outside the loop because pytorch does
  // dynamic-dispatch/indirection for each access to the tensor's
  // extents/strides/data_ptr.

  int64_t const y_off = y[0].numel();
  int64_t const c_off = c[0].numel();
  int64_t const g_off = gates[0].numel();

  Scalar* const py = y.template data_ptr<Scalar>();
  Scalar* const pc = c.template data_ptr<Scalar>();
  Scalar* const pg = gates.template data_ptr<Scalar>();

  Scalar* const A = R.template data_ptr<Scalar>();

  int const m = gates.size(2);
  int const n = gates.size(1);
  int const k = R.size(1);

  int const lda = k;
  int const ldb = k;
  int const ldc = m;

  for (int64_t t = 0; t < seq_length; ++t) {
    //
    Scalar* const B = py + y_off * t;
    Scalar* const C = pg + g_off * t;

    // Cublas is column major (opposite of pytorch) so this is really: gates[t] += y[t] @ R.t()
    myrtle::gemm<Scalar>(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, 1, A, lda, B, ldb, 1, C, ldc);

    lstm_cuda_fwd_kernel<Index, Scalar, K><<<blocks, threads, 0, stream>>>(
        pc + c_off * (t), pg + g_off * (t), pc + c_off * (t + 1), py + y_off * (t + 1), state_size);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <typename Index, typename Scalar, Kind K>
void lstm_fused_bwd_impl(
    torch::Tensor const R,
    torch::Tensor const gates,
    torch::Tensor const c,
    torch::Tensor partials,
    torch::Tensor dG //
) {
  cudaStream_t const stream = c10::cuda::getCurrentCUDAStream();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  myrtle::scoped_math<Scalar> scoped(handle);

  int64_t const seq_length = partials.size(0);
  int64_t const batch_size = c.dim() == 3 ? c.size(1) : 1;
  int64_t const state_size = c.dim() == 3 ? c.size(2) : c.size(1);

  // If state_size is less than kThreadsPerBlock we should not waste threads.
  // Use the next multiple of 32 (e.g.) the warp size (hardware constraint).
  int64_t const threads = std::min(int64_t{kThreadsPerBlock}, ((state_size + 31) / 32) * 32);

  // Grid will be (_, batch_size)
  dim3 const blocks((state_size + threads - 1) / threads, batch_size);

  // Init with zeros such that on first call to kernel addition is a noop.
  torch::Tensor dC = torch::zeros_like(c[0], {}, at::MemoryFormat::Contiguous);

  // All this needs to be outside the loop because pytorch does dynamic-dispatch/indirection for
  // each access to the tensor's extents/strides/data_ptr.

  int64_t const N = seq_length - 1;

  int64_t const c_off = c[0].numel();
  int64_t const g_off = gates[0].numel();
  int64_t const p_off = partials[0].numel();

  Scalar* const pc = c.template data_ptr<Scalar>();
  Scalar* const pg = gates.template data_ptr<Scalar>();
  Scalar* const pp = partials.template data_ptr<Scalar>();
  Scalar* const pG = dG.template data_ptr<Scalar>();
  Scalar* const pC = dC.template data_ptr<Scalar>();

  Scalar* const A = R.template data_ptr<Scalar>();

  int const m = partials.size(2);
  int const n = partials.size(1);
  int const k = R.size(0);

  int const lda = m;
  int const ldb = k;
  int const ldc = m;

  for (int64_t t = N; t >= 0; --t) {
    //
    if (t < N) {
      Scalar* const B = pG + g_off * (t + 1);
      Scalar* const C = pp + p_off * (t);

      // Cublas is column major (opposite of pytorch) so this is: partials[t] += dG[t + 1] @ R
      myrtle::gemm<Scalar>(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 1, A, lda, B, ldb, 1, C, ldc);
    }

    lstm_cuda_bwd_kernel<Index, Scalar, K><<<blocks, threads, 0, stream>>>(
        pg + g_off * (t),
        pp + p_off * (t),
        pc + c_off * (t),
        pc + c_off * (t + 1),
        pG + g_off * (t),
        pC,
        state_size);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

} // namespace

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x)                   \
  TORCH_CHECK(                                \
      x.is_contiguous(),                      \
      #x " must be contiguous but got shape", \
      x.sizes(),                              \
      " and strides ",                        \
      x.strides(),                            \
      " such that numel is ",                 \
      x.numel())

#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

/**
 * Check inputs and dispatch to typed-implementation
 */
template <Kind K>
void lstm_fused_fwd(
    torch::Tensor const R,
    torch::Tensor gates,
    torch::Tensor c,
    torch::Tensor y //
) {
  CHECK_INPUT(R);
  CHECK_INPUT(gates);
  CHECK_INPUT(c);
  CHECK_INPUT(y);

  MYRTLE_DISPATCH_FLOATING_TYPES(gates.scalar_type(), "lstm_fused_fwd", [&] {
    if (myrtle::can_use_32bit_math(gates)) {
      lstm_fused_fwd_impl<int32_t, scalar_t, K>(R, gates, c, y);
    } else {
      lstm_fused_fwd_impl<int64_t, scalar_t, K>(R, gates, c, y);
    }
  });
}

/**
 * Check inputs and dispatch to typed-implementation
 */
template <Kind K>
void lstm_fused_bwd(
    torch::Tensor const R,
    torch::Tensor const gates,
    torch::Tensor const c,
    torch::Tensor const delta,
    torch::Tensor dG //
) {
  // Contiguous not required as we will make a copy.
  CHECK_CUDA(delta);

  CHECK_INPUT(R);
  CHECK_INPUT(gates);
  CHECK_INPUT(c);
  CHECK_INPUT(dG);

  // Copy needed as partials will be modified in-place.
  torch::Tensor partials = torch::empty_like(delta, {}, at::MemoryFormat::Contiguous);

  partials.copy_(delta);

  MYRTLE_DISPATCH_FLOATING_TYPES(gates.scalar_type(), "lstm_fused_bwd", [&] {
    if (myrtle::can_use_32bit_math(gates)) {
      lstm_fused_bwd_impl<int32_t, scalar_t, K>(R, gates, c, partials, dG);
    } else {
      lstm_fused_bwd_impl<int64_t, scalar_t, K>(R, gates, c, partials, dG);
    }
  });
}

/*
 * Boiler plate code to expose the C++ functions to  Python.
 */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  //
  m.doc() = "Implementation of LSTM passes supporting both hard/soft activation functions";

  m.def(
      "lstm_fused_fwd_soft",
      &lstm_fused_fwd<Kind::soft>,
      "Compute the LSTM forward pass with soft activation functions",
      py::arg("R"),
      py::arg("gates"),
      py::arg("c"),
      py::arg("y") //
  );

  m.def(
      "lstm_fused_fwd_hard",
      &lstm_fused_fwd<Kind::hard>,
      "Compute the LSTM forward pass with hard activation functions",
      py::arg("R"),
      py::arg("gates"),
      py::arg("c"),
      py::arg("y") //
  );

  m.def(
      "lstm_fused_bwd_soft",
      &lstm_fused_bwd<Kind::soft>,
      "Compute the LSTM backward pass with soft activation functions",
      py::arg("R"),
      py::arg("gates"),
      py::arg("c"),
      py::arg("delta"),
      py::arg("dG") //
  );

  m.def(
      "lstm_fused_bwd_hard",
      &lstm_fused_bwd<Kind::hard>,
      "Compute the LSTM backward pass with hard activation functions",
      py::arg("R"),
      py::arg("gates"),
      py::arg("c"),
      py::arg("delta"),
      py::arg("dG") //
  );
}
