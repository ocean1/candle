// GEMV kernel adapted from MLX:
// https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/gemv.metal
// Copyright © 2023-2024 Apple Inc.

#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const
#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

// elem_to_loc for nc=1 batch handling
//
// Templated on the *pointer types* rather than fixing them to `constant`,
// because the packed entry points below receive `batch_shape` and the three
// stride arrays in `device` buffers instead. Deducing the address space from
// the argument keeps this one body: rewriting the arithmetic per address space
// was the alternative, and it is the destructive one. Same remedy issue #38
// used for `strided_indexer` (`DESIGN.md` §11.3d).
//
// `stride_t` stays an explicit parameter -- it is the *value* type and callers
// name it -- while `ShapeP`/`StrideP` are deduced.
template <typename stride_t, typename ShapeP, typename StrideP>
METAL_FUNC stride_t gemv_elem_to_loc(
    uint elem,
    ShapeP shape,
    StrideP strides,
    int ndim) {
  stride_t loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * strides[i];
    elem /= shape[i];
  }
  return loc;
}

template <typename U>
struct GemvDefaultAccT {
  using type = float;
};

///////////////////////////////////////////////////////////////////////////////
/// Matrix-vector: mat [M_out, K] × vec [K] → out [M_out]
/// (mat rows = output dimension, mat cols = K)
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int BM,       // Threadgroup rows (in simdgroups)
    const int BN,       // Threadgroup cols (in simdgroups)
    const int SM,       // Simdgroup rows (in threads)
    const int SN,       // Simdgroup cols (in threads)
    const int TM,       // Thread rows (in elements)
    const int TN,       // Thread cols (in elements)
    const bool kDoAxpby,
    typename AccT = typename GemvDefaultAccT<T>::type>
struct GEMVKernel {
  using acc_type = AccT;

  MLX_MTL_CONST int threadsM = BM * SM;
  MLX_MTL_CONST int threadsN = BN * SN;

  MLX_MTL_CONST int blockM = threadsM * TM;
  MLX_MTL_CONST int blockN = threadsN * TN;

  static_assert(SM * SN == 32, "simdgroup must have 32 threads");
  static_assert(SN == 4 || SN == 8 || SN == 16 || SN == 32,
                "gemv block must have width 4, 8, 16, or 32");

  MLX_MTL_CONST short tgp_mem_size = BN > 1 ? BN * (blockM + TM) : 0;
  MLX_MTL_CONST bool needs_tgp_reduction = BN > 1;

  template <typename U = T>
  static METAL_FUNC void load_unsafe(
      const device T* src, thread U dst[TN], const int src_offset = 0) {
    MLX_MTL_PRAGMA_UNROLL
    for (int tn = 0; tn < TN; tn++) {
      dst[tn] = static_cast<U>(src[src_offset + tn]);
    }
  }

  template <typename U = T>
  static METAL_FUNC void load_safe(
      const device T* src,
      thread U dst[TN],
      const int src_offset = 0,
      const int src_size = TN) {
    if (src_offset + TN <= src_size) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        dst[tn] = static_cast<U>(src[src_offset + tn]);
      }
    } else {
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        dst[tn] = src_offset + tn < src_size
            ? static_cast<U>(src[src_offset + tn])
            : U(0);
      }
    }
  }

  static METAL_FUNC void run(
      const device T* mat,
      const device T* in_vec,
      const device T* bias,
      device T* out_vec,
      const int in_vec_size,
      const int out_vec_size,
      const int matrix_ld,
      const float alpha,
      const float beta,
      const int bias_stride,
      threadgroup AccT* tgp_memory,
      uint3 tid,
      uint3 lid,
      uint simd_gid,
      uint simd_lid) {
    (void)lid;

    thread AccT result[TM] = {0};
    thread T inter[TN];
    thread AccT v_coeff[TN];

    const int thrM = SN != 32 ? (int)(simd_lid / SN) : 0;
    const int thrN = SN != 32 ? (int)(simd_lid % SN) : (int)simd_lid;

    const int sgN = BN != 1 ? (int)(simd_gid % BN) : 0;
    const int simdM = BN != 1 ? SM * (int)(simd_gid / BN) : (int)(SM * simd_gid);
    const int simdN = BN != 1 ? SN * (int)(simd_gid % BN) : 0;

    int bm = (simdM + thrM) * TM;
    int bn = (simdN + thrN) * TN;

    // Block position (output row)
    int out_row = tid.x * blockM + bm;

    if (out_row >= out_vec_size) return;

    out_row = out_row + TM <= out_vec_size ? out_row : out_vec_size - TM;

    mat += out_row * matrix_ld;

    const int loop_stride = blockN;
    const int in_size = in_vec_size;
    const int n_iter = in_size / loop_stride;
    const int last_iter = loop_stride * n_iter;
    const int leftover = in_size - last_iter;

    for (int i = 0; i < n_iter; ++i) {
      load_unsafe<AccT>(in_vec, v_coeff, bn);

      int mat_offset = 0;
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        load_unsafe(mat, inter, mat_offset + bn);
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          result[tm] += inter[tn] * v_coeff[tn];
        }
        mat_offset += matrix_ld;
      }
      bn += blockN;
    }

    if (leftover > 0) {
      load_safe<AccT>(in_vec, v_coeff, bn, in_size);
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        load_safe(&mat[tm * matrix_ld], inter, bn, in_size);
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          result[tm] += inter[tn] * v_coeff[tn];
        }
      }
    }

    MLX_MTL_PRAGMA_UNROLL
    for (int tm = 0; tm < TM; tm++) {
      MLX_MTL_PRAGMA_UNROLL
      for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
        result[tm] += simd_shuffle_down(result[tm], sn);
      }
    }

    if (needs_tgp_reduction) {
      threadgroup AccT* tgp_results = tgp_memory + sgN * (blockM + TM) + bm;
      if (thrN == 0) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
          tgp_results[tm] = result[tm];
        }
        threadgroup_barrier(mem_flags::mem_none);
        if (sgN == 0) {
          MLX_MTL_PRAGMA_UNROLL
          for (int sgn = 1; sgn < BN; sgn++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tm = 0; tm < TM; tm++) {
              result[tm] += tgp_results[sgn * (blockM + TM) + tm];
            }
          }
        }
      }
    }

    if (simdN == 0 && thrN == 0) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        if (kDoAxpby) {
          out_vec[out_row + tm] =
              static_cast<T>(alpha) * static_cast<T>(result[tm]) +
              static_cast<T>(beta) * bias[(out_row + tm) * bias_stride];
        } else {
          out_vec[out_row + tm] = static_cast<T>(result[tm]);
        }
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
/// Vector-matrix: vec [K] × mat [K, N_out] → out [N_out]
/// (mat rows = K, mat cols = output dimension)
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int BM,
    const int BN,
    const int SM,
    const int SN,
    const int TM,
    const int TN,
    const bool kDoAxpby,
    typename AccT = typename GemvDefaultAccT<T>::type>
struct GEMVTKernel {
  using acc_type = AccT;

  MLX_MTL_CONST int threadsM = BM * SM;
  MLX_MTL_CONST int threadsN = BN * SN;

  MLX_MTL_CONST int blockM = threadsM * TM;
  MLX_MTL_CONST int blockN = threadsN * TN;

  static_assert(SM * SN == 32, "simdgroup must have 32 threads");

  MLX_MTL_CONST short tgp_mem_size = BM > 1 ? BM * (blockN + TN) : 0;
  MLX_MTL_CONST bool needs_tgp_reduction = BM > 1;

  static METAL_FUNC void run(
      const device T* mat,
      const device T* in_vec,
      const device T* bias,
      device T* out_vec,
      const int in_vec_size,
      const int out_vec_size,
      const int marix_ld,
      const float alpha,
      const float beta,
      const int bias_stride,
      threadgroup AccT* tgp_memory,
      uint3 tid,
      uint3 lid,
      uint simd_gid,
      uint simd_lid) {
    (void)lid;

    AccT result[TN] = {0};
    T inter[TN];
    AccT v_coeff[TM];

    const int thrM = SN != 32 ? (int)(simd_lid / SN) : 0;
    const int thrN = SN != 32 ? (int)(simd_lid % SN) : (int)simd_lid;

    const int sgM = BN != 1 ? (int)(simd_gid / BN) : (int)simd_gid;
    const int sgN = BN != 1 ? (int)(simd_gid % BN) : 0;

    const int simdM = SM * sgM;
    const int simdN = SN * sgN;

    int cm = (simdM + thrM);
    int cn = (simdN + thrN);

    int bm = cm * TM;
    int bn = cn * TN;

    int out_col = tid.x * blockN + bn;

    const int loop_stride = blockM;
    const int in_size = in_vec_size;
    const int n_iter = in_size / loop_stride;
    const int last_iter = loop_stride * n_iter;
    const int leftover = in_size - last_iter;

    if (out_col < out_vec_size) {
      out_col = out_col + TN <= out_vec_size ? out_col : out_vec_size - TN;

      for (int i = 0; i < n_iter; ++i) {
        threadgroup_barrier(mem_flags::mem_none);

        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
          v_coeff[tm] = static_cast<AccT>(in_vec[bm + tm]);
        }

        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
          auto vc = static_cast<AccT>(v_coeff[tm]);
          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            inter[tn] = mat[(bm + tm) * marix_ld + out_col + tn];
          }
          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            result[tn] += vc * inter[tn];
          }
        }

        bm += blockM;
      }

      if (leftover > 0) {
        for (int tm = 0; tm < TM && bm + tm < in_vec_size; tm++) {
          v_coeff[tm] = static_cast<AccT>(in_vec[bm + tm]);
          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            inter[tn] = mat[(bm + tm) * marix_ld + out_col + tn];
          }
          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            result[tn] += v_coeff[tm] * inter[tn];
          }
        }
      }
    }

    MLX_MTL_PRAGMA_UNROLL
    for (int tn = 0; tn < TN; tn++) {
      MLX_MTL_PRAGMA_UNROLL
      for (ushort sm = (SM / 2); sm >= 1; sm >>= 1) {
        result[tn] += simd_shuffle_down(result[tn], SN * sm);
      }
    }

    if (needs_tgp_reduction) {
      threadgroup AccT* tgp_results = tgp_memory + sgM * (blockN + TN) + bn;
      if (thrM == 0) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          tgp_results[tn] = result[tn];
        }
        threadgroup_barrier(mem_flags::mem_none);
        if (sgM == 0) {
          MLX_MTL_PRAGMA_UNROLL
          for (int sgm = 1; sgm < BM; sgm++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tn = 0; tn < TN; tn++) {
              result[tn] += tgp_results[sgm * (blockN + TN) + tn];
            }
          }
        }
      }
    }

    if (cm == 0 && out_col < out_vec_size) {
      MLX_MTL_PRAGMA_UNROLL
      for (int j = 0; j < TN; j++) {
        if (kDoAxpby) {
          out_vec[out_col + j] =
              static_cast<T>(alpha) * static_cast<T>(result[j]) +
              static_cast<T>(beta) * bias[(out_col + j) * bias_stride];
        } else {
          out_vec[out_col + j] = static_cast<T>(result[j]);
        }
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
/// Packed parameter block
///////////////////////////////////////////////////////////////////////////////

// The scalars `gemv` and `gemv_t` take as `constant int&` today, as one struct
// an `MTLIndirectComputeCommand` can bind.
//
// `MTLIndirectComputeCommand` has no `setBytes` in any form (`DESIGN.md`
// §3.7c); `setKernelBuffer` is its only binding primitive, so a kernel whose
// scalars arrive inline cannot be encoded into an ICB at all. Both entry points
// below are instantiated from the same body, so in §7.1's terms the binding
// style is a compile-tier variant axis alongside dtype — not a migration.
//
// Field order mirrors the classical argument order, so `set_params!` fills the
// block in the order the arguments are already written. `batch_shape` and the
// three stride arrays are deliberately *not* fields: their length is a property
// of the call, so they stay separate bindings — which an ICB can express, since
// `setKernelBuffer` binds a buffer of any length (`DESIGN.md` §11.3d).
//
// Every field is a 4-byte scalar, so this is 28 bytes at 4-byte alignment with
// no padding anywhere. Checked rather than assumed: the `static_assert` below
// pins size and alignment, and `gemv_params_layout` reports every offset for
// the Rust side to compare against `offset_of!`.
struct GemvParams {
  int in_vec_size;
  int out_vec_size;
  int matrix_ld;
  float alpha;
  float beta;
  int batch_ndim;
  int bias_stride;
};

static_assert(sizeof(GemvParams) == 28, "GemvParams layout");
static_assert(alignof(GemvParams) == 4, "GemvParams alignment");

///////////////////////////////////////////////////////////////////////////////
/// gemv kernel: mat [M, K] × vec [K] → out [M]
///////////////////////////////////////////////////////////////////////////////

// The batch offsets and the call into `GEMVKernel::run`, shared by both binding
// styles.
//
// Note what this takes and what it does *not*: the scalars arrive **by value**,
// already unpacked by whichever wrapper called it. That is the whole answer to
// issue #41's inner-loop question. `GEMVKernel::run` already takes
// `const int in_vec_size` / `const int matrix_ld` by value, so the loop bound
// and the row stride are in registers before the first iteration under either
// style; neither `constant int&` nor `device GemvParams*` is ever dereferenced
// inside the loop. Promoting them to a buffer therefore moves *where the load
// happens once*, not how often. `regpressure.m` checks that against the
// compiler rather than leaving it as a reading of the source.
//
// `ShapeP`/`StrideP` are deduced, so `constant`-space arrays (classical) and
// `device`-space ones (packed) both reach the same arithmetic.
template <
    typename T,
    const int BM, const int BN,
    const int SM, const int SN,
    const int TM, const int TN,
    const bool kDoNCBatch,
    const bool kDoAxpby,
    typename ShapeP, typename StrideP>
METAL_FUNC void gemv_body(
    const device T* mat,
    const device T* in_vec,
    const device T* bias,
    device T* out_vec,
    const int in_vec_size,
    const int out_vec_size,
    const int matrix_ld,
    const float alpha,
    const float beta,
    const int batch_ndim,
    ShapeP batch_shape,
    StrideP vector_batch_stride,
    StrideP matrix_batch_stride,
    StrideP bias_batch_stride,
    const int bias_stride,
    threadgroup typename GEMVKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>::acc_type* tgp_memory,
    uint3 tid,
    uint3 lid,
    uint simd_gid,
    uint simd_lid) {
  using kernel_t = GEMVKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;

  if (kDoNCBatch) {
    in_vec += gemv_elem_to_loc<int64_t>(tid.z, batch_shape, vector_batch_stride, batch_ndim);
    mat    += gemv_elem_to_loc<int64_t>(tid.z, batch_shape, matrix_batch_stride, batch_ndim);
    if (kDoAxpby)
      bias += gemv_elem_to_loc<int64_t>(tid.z, batch_shape, bias_batch_stride, batch_ndim);
  } else {
    in_vec += tid.z * vector_batch_stride[0];
    mat    += tid.z * matrix_batch_stride[0];
    if (kDoAxpby)
      bias += tid.z * bias_batch_stride[0];
  }
  out_vec += tid.z * out_vec_size;

  kernel_t::run(mat, in_vec, bias, out_vec,
                in_vec_size, out_vec_size, matrix_ld,
                alpha, beta, bias_stride,
                tgp_memory,
                tid, lid, simd_gid, simd_lid);
}

// The threadgroup array cannot move into `gemv_body`: MSL permits a
// threadgroup-address-space variable only inside a `[[kernel]]`-qualified
// function, which is issue #38's finding (`DESIGN.md` §11.3d) reproducing here
// unchanged. So the allocation stays in each wrapper and is passed down, and
// what the two wrappers differ in is the lines that unpack the scalars.
#define gemv_tgp_memory(KT) \
  threadgroup typename KT::acc_type tgp_memory[KT::tgp_mem_size == 0 ? 1 : KT::tgp_mem_size]

template <
    typename T,
    const int BM, const int BN,
    const int SM, const int SN,
    const int TM, const int TN,
    const bool kDoNCBatch,
    const bool kDoAxpby>
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]]
void gemv(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& matrix_ld [[buffer(6)]],
    const constant float& alpha [[buffer(7)]],
    const constant float& beta [[buffer(8)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant int64_t* vector_batch_stride [[buffer(11)]],
    const constant int64_t* matrix_batch_stride [[buffer(12)]],
    const constant int64_t* bias_batch_stride [[buffer(13)]],
    const constant int& bias_stride [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using kernel_t = GEMVKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;
  gemv_tgp_memory(kernel_t);
  gemv_body<T, BM, BN, SM, SN, TM, TN, kDoNCBatch, kDoAxpby>(
      mat, in_vec, bias, out_vec,
      in_vec_size, out_vec_size, matrix_ld, alpha, beta,
      batch_ndim, batch_shape, vector_batch_stride, matrix_batch_stride,
      bias_batch_stride, bias_stride,
      kernel_t::tgp_mem_size == 0 ? nullptr : tgp_memory,
      tid, lid, simd_gid, simd_lid);
}

// The ICB-expressible sibling: one `device const GemvParams*` in place of the
// seven `constant &` scalars, everything else unchanged.
//
// The buffer indices are renumbered rather than left with holes — params at 0,
// then the bound tensors, then the arrays — because diverting scalars out of
// the argument list shifts every binding that remains. `ParamCapture` on the
// Rust side performs exactly that renumbering, so the two agree by construction
// rather than by matching two hand-written lists (`DESIGN.md` §11.3d, third
// finding).
//
// **`bias` sits last, at 8, and that is not cosmetic.** `call_mlx_gemv` passes
// `()` for bias — no kernel LFM2 dispatches uses `axpby`, so nothing is bound
// there — and `EncoderParam for ()` binds nothing and consumes no slot. Under
// capture the renumbering therefore assigns consecutive slots to the buffers
// that *are* bound: mat 1, in_vec 2, out_vec 3, then the four arrays at 4..7.
// Declaring `bias` at 3 in source order would leave every later binding one
// slot low and the kernel would read `out_vec`'s contents as its input vector.
//
// This is a *different* failure from the hole #38 describes: there a diverted
// scalar shifted the bindings and the Rust side compensated. Here an argument
// that was never bound at all shifts them, and the compensation has to happen
// in the *kernel signature*, because the Rust side cannot renumber a slot no
// one asked for. Under `HazardTrackingModeUntracked` the symptom is not a
// crash — it hung the GPU, which is how it was found.
//
// `bias` stays a parameter rather than being deleted because `kDoAxpby` is a
// template axis and half the instantiations do read it; it is unbound only for
// the `axpby0` half, exactly as on the classical path.
template <
    typename T,
    const int BM, const int BN,
    const int SM, const int SN,
    const int TM, const int TN,
    const bool kDoNCBatch,
    const bool kDoAxpby>
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]]
void gemv_packed(
    const device GemvParams* pp [[buffer(0)]],
    const device T* mat [[buffer(1)]],
    const device T* in_vec [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const device int* batch_shape [[buffer(4)]],
    const device int64_t* vector_batch_stride [[buffer(5)]],
    const device int64_t* matrix_batch_stride [[buffer(6)]],
    const device int64_t* bias_batch_stride [[buffer(7)]],
    const device T* bias [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using kernel_t = GEMVKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;
  gemv_tgp_memory(kernel_t);
  GemvParams p = *pp;
  gemv_body<T, BM, BN, SM, SN, TM, TN, kDoNCBatch, kDoAxpby>(
      mat, in_vec, bias, out_vec,
      p.in_vec_size, p.out_vec_size, p.matrix_ld, p.alpha, p.beta,
      p.batch_ndim, batch_shape, vector_batch_stride, matrix_batch_stride,
      bias_batch_stride, p.bias_stride,
      kernel_t::tgp_mem_size == 0 ? nullptr : tgp_memory,
      tid, lid, simd_gid, simd_lid);
}

///////////////////////////////////////////////////////////////////////////////
/// gemv_t kernel: vec [K] × mat [K, N] → out [N]
///////////////////////////////////////////////////////////////////////////////

// As `gemv_body`, for the transposed kernel. Separate rather than templated on
// the kernel type because the two select different `GEMVKernel`/`GEMVTKernel`
// accumulator types for their threadgroup array, and folding them would put a
// second template parameter in the way of that for no saving.
template <
    typename T,
    const int BM, const int BN,
    const int SM, const int SN,
    const int TM, const int TN,
    const bool kDoNCBatch,
    const bool kDoAxpby,
    typename ShapeP, typename StrideP>
METAL_FUNC void gemv_t_body(
    const device T* mat,
    const device T* in_vec,
    const device T* bias,
    device T* out_vec,
    const int in_vec_size,
    const int out_vec_size,
    const int matrix_ld,
    const float alpha,
    const float beta,
    const int batch_ndim,
    ShapeP batch_shape,
    StrideP vector_batch_stride,
    StrideP matrix_batch_stride,
    StrideP bias_batch_stride,
    const int bias_stride,
    threadgroup typename GEMVTKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>::acc_type* tgp_memory,
    uint3 tid,
    uint3 lid,
    uint simd_gid,
    uint simd_lid) {
  using kernel_t = GEMVTKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;

  if (kDoNCBatch) {
    in_vec += gemv_elem_to_loc<int64_t>(tid.z, batch_shape, vector_batch_stride, batch_ndim);
    mat    += gemv_elem_to_loc<int64_t>(tid.z, batch_shape, matrix_batch_stride, batch_ndim);
    if (kDoAxpby)
      bias += gemv_elem_to_loc<int64_t>(tid.z, batch_shape, bias_batch_stride, batch_ndim);
  } else {
    in_vec += tid.z * vector_batch_stride[0];
    mat    += tid.z * matrix_batch_stride[0];
    if (kDoAxpby)
      bias += tid.z * bias_batch_stride[0];
  }
  out_vec += tid.z * out_vec_size;

  kernel_t::run(mat, in_vec, bias, out_vec,
                in_vec_size, out_vec_size, matrix_ld,
                alpha, beta, bias_stride,
                tgp_memory,
                tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int BM, const int BN,
    const int SM, const int SN,
    const int TM, const int TN,
    const bool kDoNCBatch,
    const bool kDoAxpby>
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]]
void gemv_t(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& matrix_ld [[buffer(6)]],
    const constant float& alpha [[buffer(7)]],
    const constant float& beta [[buffer(8)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant int64_t* vector_batch_stride [[buffer(11)]],
    const constant int64_t* matrix_batch_stride [[buffer(12)]],
    const constant int64_t* bias_batch_stride [[buffer(13)]],
    const constant int& bias_stride [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using kernel_t = GEMVTKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;
  gemv_tgp_memory(kernel_t);
  gemv_t_body<T, BM, BN, SM, SN, TM, TN, kDoNCBatch, kDoAxpby>(
      mat, in_vec, bias, out_vec,
      in_vec_size, out_vec_size, matrix_ld, alpha, beta,
      batch_ndim, batch_shape, vector_batch_stride, matrix_batch_stride,
      bias_batch_stride, bias_stride,
      kernel_t::tgp_mem_size == 0 ? nullptr : tgp_memory,
      tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int BM, const int BN,
    const int SM, const int SN,
    const int TM, const int TN,
    const bool kDoNCBatch,
    const bool kDoAxpby>
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]]
void gemv_t_packed(
    const device GemvParams* pp [[buffer(0)]],
    const device T* mat [[buffer(1)]],
    const device T* in_vec [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const device int* batch_shape [[buffer(4)]],
    const device int64_t* vector_batch_stride [[buffer(5)]],
    const device int64_t* matrix_batch_stride [[buffer(6)]],
    const device int64_t* bias_batch_stride [[buffer(7)]],
    const device T* bias [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using kernel_t = GEMVTKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;
  gemv_tgp_memory(kernel_t);
  GemvParams p = *pp;
  gemv_t_body<T, BM, BN, SM, SN, TM, TN, kDoNCBatch, kDoAxpby>(
      mat, in_vec, bias, out_vec,
      p.in_vec_size, p.out_vec_size, p.matrix_ld, p.alpha, p.beta,
      p.batch_ndim, batch_shape, vector_batch_stride, matrix_batch_stride,
      bias_batch_stride, p.bias_stride,
      kernel_t::tgp_mem_size == 0 ? nullptr : tgp_memory,
      tid, lid, simd_gid, simd_lid);
}

// Reports `GemvParams`'s size and every field's offset *as the compiled kernel
// sees them*, for the host-side layout test to compare against Rust's own
// `size_of`/`offset_of!`.
//
// A `static_assert` proves only that the device side agrees with itself; only
// shipping the numbers across the boundary proves the two sides agree, which is
// the failure this exists to catch — a field at the wrong offset does not
// crash, it reads a well-formed number from the wrong place and computes a
// plausible wrong answer (`DESIGN.md` §3.5, §15.1). Offsets cannot be
// `static_assert`ed in MSL: there is no `<cstddef>` and the null-pointer-member
// form is not a constant expression, so they are measured at runtime from a
// real `thread` instance (`DESIGN.md` §11.3b).
//
// Writes 8 slots; `GEMV_LAYOUT_SLOTS` in `kernels/params.rs` says what each is.
#define gemv_offsetof_rt(S, F) \
    ((uint)((thread const char *)&(probe_##S.F) - (thread const char *)&probe_##S))

[[kernel]] void gemv_params_layout(
    device uint *out,
    uint tid [[thread_position_in_grid]]
) {
  if (tid != 0) { return; }
  GemvParams probe_GemvParams;
  out[0] = sizeof(GemvParams);
  out[1] = gemv_offsetof_rt(GemvParams, in_vec_size);
  out[2] = gemv_offsetof_rt(GemvParams, out_vec_size);
  out[3] = gemv_offsetof_rt(GemvParams, matrix_ld);
  out[4] = gemv_offsetof_rt(GemvParams, alpha);
  out[5] = gemv_offsetof_rt(GemvParams, beta);
  out[6] = gemv_offsetof_rt(GemvParams, batch_ndim);
  out[7] = gemv_offsetof_rt(GemvParams, bias_stride);
}

///////////////////////////////////////////////////////////////////////////////
/// Instantiations
///////////////////////////////////////////////////////////////////////////////

// Use decltype-based instantiation (MLX defines.h pattern):
//   template [[host_name(...)]] [[kernel]] decltype(func<...>) func<...>;
// This avoids redeclaring parameter attributes, which Metal rejects.
#define instantiate_gemv_helper(func, nm, itype, bm, bn, sm, sn, tm, tn, nc, axpby) \
  template [[host_name(                                                               \
      #func "_" #nm "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn                         \
            "_tm" #tm "_tn" #tn "_nc" #nc "_axpby" #axpby)]]                         \
  [[kernel]] decltype(func<itype, bm, bn, sm, sn, tm, tn, (bool)nc, (bool)axpby>)   \
             func<itype, bm, bn, sm, sn, tm, tn, (bool)nc, (bool)axpby>;

// The packed sibling, named by appending `_packed` to the classical name.
//
// It has to be a *distinct function template* rather than a second
// `[[host_name]]` on the same instantiation: MSL rejects that outright with
// "duplicate explicit instantiation of `gemv_t<bfloat, 1, 16, 4, 8, 4, 4, true,
// true>`", since an explicit instantiation is unique per template-argument list
// however many names are attached to it. Measured while pricing the doubling —
// see `measurements/probes/issue-41/`.
//
// Generated from the same axis lists as the classical names below, so the two
// sets cannot drift: adding a tile configuration adds both, and a name that
// exists on one side only fails `gemv_names_resolve` against the compiled
// library (`DESIGN.md` §8.1b).
#define instantiate_gemv_packed_helper(func, nm, itype, bm, bn, sm, sn, tm, tn, nc, axpby) \
  template [[host_name(                                                                      \
      #func "_" #nm "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn                                \
            "_tm" #tm "_tn" #tn "_nc" #nc "_axpby" #axpby "_packed")]]                      \
  [[kernel]] decltype(func##_packed<itype, bm, bn, sm, sn, tm, tn, (bool)nc, (bool)axpby>)  \
             func##_packed<itype, bm, bn, sm, sn, tm, tn, (bool)nc, (bool)axpby>;

#define instantiate_gemv_nc_axpby(func, nm, itype, bm, bn, sm, sn, tm, tn)    \
  instantiate_gemv_helper(func, nm, itype, bm, bn, sm, sn, tm, tn, 0, 0)      \
  instantiate_gemv_helper(func, nm, itype, bm, bn, sm, sn, tm, tn, 0, 1)      \
  instantiate_gemv_helper(func, nm, itype, bm, bn, sm, sn, tm, tn, 1, 0)      \
  instantiate_gemv_helper(func, nm, itype, bm, bn, sm, sn, tm, tn, 1, 1)      \
  instantiate_gemv_packed_helper(func, nm, itype, bm, bn, sm, sn, tm, tn, 0, 0) \
  instantiate_gemv_packed_helper(func, nm, itype, bm, bn, sm, sn, tm, tn, 0, 1) \
  instantiate_gemv_packed_helper(func, nm, itype, bm, bn, sm, sn, tm, tn, 1, 0) \
  instantiate_gemv_packed_helper(func, nm, itype, bm, bn, sm, sn, tm, tn, 1, 1)

// gemv blocks: mat×vec (output size = M)
// bm=4/8 for large output; bm=1,bn=8 for K-heavy; bm=1,bn=1,sm=8,sn=4 for small K
#define instantiate_gemv_blocks(nm, itype)                              \
  instantiate_gemv_nc_axpby(gemv, nm, itype, 1,  8, 1, 32, 4, 4)      \
  instantiate_gemv_nc_axpby(gemv, nm, itype, 1,  8, 1, 32, 1, 4)      \
  instantiate_gemv_nc_axpby(gemv, nm, itype, 1,  1, 8,  4, 4, 4)      \
  instantiate_gemv_nc_axpby(gemv, nm, itype, 1,  1, 8,  4, 1, 4)      \
  instantiate_gemv_nc_axpby(gemv, nm, itype, 4,  1, 1, 32, 1, 4)      \
  instantiate_gemv_nc_axpby(gemv, nm, itype, 4,  1, 1, 32, 4, 4)      \
  instantiate_gemv_nc_axpby(gemv, nm, itype, 8,  1, 1, 32, 4, 4)

// gemv_t blocks: vec×mat (output size = N)
// bn=2/4/16 for various output sizes; sm/sn tuned for K size
#define instantiate_gemv_t_blocks(nm, itype)                              \
  instantiate_gemv_nc_axpby(gemv_t, nm, itype, 1,  2,  8, 4, 4, 1)      \
  instantiate_gemv_nc_axpby(gemv_t, nm, itype, 1,  2,  8, 4, 4, 4)      \
  instantiate_gemv_nc_axpby(gemv_t, nm, itype, 1,  4,  8, 4, 4, 4)      \
  instantiate_gemv_nc_axpby(gemv_t, nm, itype, 1, 16,  8, 4, 4, 4)      \
  instantiate_gemv_nc_axpby(gemv_t, nm, itype, 1, 16,  4, 8, 4, 4)

instantiate_gemv_blocks(float32, float)
instantiate_gemv_blocks(float16, half)
instantiate_gemv_blocks(bfloat16, bfloat)

instantiate_gemv_t_blocks(float32, float)
instantiate_gemv_t_blocks(float16, half)
instantiate_gemv_t_blocks(bfloat16, bfloat)
