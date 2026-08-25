#include <metal_stdlib>
using namespace metal;

template <typename T>
inline T max_value();

template <>
inline int64_t max_value<int64_t>() {
    return 0x7FFFFFFFFFFFFFFF;
}

template <>
inline uint32_t max_value<uint32_t>() {
    return 0xFFFFFFFFu;
}

template <>
inline uint8_t max_value<uint8_t>() {
    return 0xFF;
}

METAL_FUNC uint get_strided_index(
    uint idx,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides
) {
    uint strided_i = 0;
    for (uint d = 0; d < num_dims; d++) {
        uint dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

template<typename TYPENAME, typename INDEX_TYPENAME>
[[kernel]] void index(
    constant size_t &dst_size,
    constant size_t &left_size,
    constant size_t &src_dim_size,
    constant size_t &right_size,
    constant size_t &ids_size,
    constant bool &contiguous,
    constant size_t *src_dims,
    constant size_t *src_strides,
    const device TYPENAME *input,
    const device INDEX_TYPENAME *input_ids,
    device TYPENAME *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dst_size) {
        return;
    }
    const size_t id_i = (tid / right_size) % ids_size;
    if (input_ids[id_i] == max_value<INDEX_TYPENAME>()) {
      output[tid] = static_cast<TYPENAME>(0);
    } else {
      const INDEX_TYPENAME input_i = min(input_ids[id_i], (INDEX_TYPENAME)(src_dim_size - 1));
      const size_t right_rank_i = tid % right_size;
      const size_t left_rank_i = tid / right_size / ids_size;
      /*
      // Force prevent out of bounds indexing
      // since there doesn't seem to be a good way to force crash
      // No need to check for zero we're only allowing unsized.
      */
      const size_t src_i = left_rank_i * src_dim_size * right_size + input_i * right_size + right_rank_i;
      const size_t strided_src_i = contiguous ? src_i : get_strided_index(src_i, src_dim_size, src_dims, src_strides);
      output[tid] = input[strided_src_i];
    }
}

template<typename TYPENAME, typename INDEX_TYPENAME>
[[kernel]] void gather(
    constant size_t &dst_size,
    constant size_t &left_size,
    constant size_t &src_dim_size,
    constant size_t &right_size,
    constant size_t &ids_size,
    const device TYPENAME *input,
    const device INDEX_TYPENAME *input_ids,
    device TYPENAME *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dst_size) {
        return;
    }
    const INDEX_TYPENAME input_i = input_ids[tid];
    if (input_i == max_value<INDEX_TYPENAME>()) {
      output[tid] = static_cast<TYPENAME>(0);
    } else {
      const size_t right_rank_i = tid % right_size;
      const size_t left_rank_i = tid / right_size / ids_size;
      const size_t src_i = (left_rank_i * src_dim_size + input_i) * right_size + right_rank_i;
      output[tid] = input[src_i];
    }
}

template<typename TYPENAME, typename INDEX_TYPENAME>
[[kernel]] void scatter(
    constant size_t &dst_size,
    constant size_t &left_size,
    constant size_t &src_dim_size,
    constant size_t &right_size,
    constant size_t &dst_dim_size,
    const device TYPENAME *input,
    const device INDEX_TYPENAME *input_ids,
    device TYPENAME *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dst_size) {
        return;
    }
    const size_t right_rank_i = tid % right_size;
    const size_t left_rank_i = tid / right_size;
    for (unsigned int j = 0; j < src_dim_size; ++j) {
        const size_t src_i = (left_rank_i * src_dim_size + j) * right_size + right_rank_i;
        const INDEX_TYPENAME idx = input_ids[src_i];
        if (idx < max_value<INDEX_TYPENAME>()) {
          const size_t dst_i = (left_rank_i * dst_dim_size + idx) * right_size + right_rank_i;
          output[dst_i] = input[src_i];
        }
    }
}

template<typename TYPENAME, typename INDEX_TYPENAME>
[[kernel]] void scatter_add(
    constant size_t &dst_size,
    constant size_t &left_size,
    constant size_t &src_dim_size,
    constant size_t &right_size,
    constant size_t &dst_dim_size,
    const device TYPENAME *input,
    const device INDEX_TYPENAME *input_ids,
    device TYPENAME *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dst_size) {
        return;
    }
    const size_t right_rank_i = tid % right_size;
    const size_t left_rank_i = tid / right_size;
    for (unsigned int j = 0; j < src_dim_size; ++j) {
        const size_t src_i = (left_rank_i * src_dim_size + j) * right_size + right_rank_i;
        const INDEX_TYPENAME idx = input_ids[src_i];
        if (idx < max_value<INDEX_TYPENAME>()) {
          const size_t dst_i = (left_rank_i * dst_dim_size + idx) * right_size + right_rank_i;
          output[dst_i] += input[src_i];
        }
    }
}

template<typename TYPENAME, typename INDEX_TYPENAME>
[[kernel]] void index_add(
    constant size_t &dst_size,
    constant size_t &left_size,
    constant size_t &src_dim_size,
    constant size_t &right_size,
    constant size_t &dst_dim_size,
    constant size_t &ids_dim_size,
    const device TYPENAME *input,
    const device INDEX_TYPENAME *input_ids,
    device TYPENAME *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dst_size) {
        return;
    }
    const size_t right_rank_i = tid % right_size;
    const size_t left_rank_i = tid / right_size;
    for (unsigned int j = 0; j < ids_dim_size; ++j) {
        const INDEX_TYPENAME idx = input_ids[j];
        if (idx < max_value<INDEX_TYPENAME>()) {
          const size_t src_i = (left_rank_i * src_dim_size + j) * right_size + right_rank_i;
          const size_t dst_i = (left_rank_i * dst_dim_size + idx) * right_size + right_rank_i;
          output[dst_i] += input[src_i];
        }
    }
}

// Explicit instantiation. `decltype(func<...>)` restates the template's own
// signature, so a variant is declared by naming the type arguments and the
// `[[host_name]]` string only -- the parameter list is written once, in the
// template above.
//
// The macro form this replaces spelled every signature twice: a `kernel void`
// wrapper per instantiation whose only job was to restate the template's
// parameters and forward to it. `DESIGN.md` §8.1c records why that is worse
// than verbose -- a parameter added to one list and not the other is a compile
// error only when the argument *count* changes, and **reordering two
// same-typed parameters in one list is silent**. `index` takes five
// consecutive `constant size_t &` parameters and `gather` five more, so that
// shape was reachable in this file.
//
// Same spelling as unary.metal, binary.metal, affine.metal, conv.metal and
// reduce.metal, which candle already migrated.
#define init_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

// The four op families, each keyed on a `(index dtype, value dtype)` **pair**.
//
// That pair is what makes this file different from every other family candle
// has templated: `conv` and `reduce` vary over one dtype, so their names are
// `<stem>_<dtype>`. Here both the index type and the value type vary
// independently, and the name carries both -- `is_u32_f16` is `index` over an
// f16 tensor with u32 indices. The registry on the Rust side
// (`IndexingKernel`) is keyed the same way for the same reason.
//
// Note the argument order: the *name* reads `<stem>_<index>_<value>` while the
// template takes `<value, index>`. That inversion is inherited from the macro
// form and is preserved deliberately -- changing it would rename all 62
// kernels for no gain, and the registry checks the spelling rather than
// assuming it.
#define init_index(iname, tname, itype, ttype) \
    init_kernel("is_" #iname "_" #tname, index, ttype, itype)

#define init_gather(iname, tname, itype, ttype) \
    init_kernel("gather_" #iname "_" #tname, gather, ttype, itype)

#define init_scatter(iname, tname, itype, ttype) \
    init_kernel("s_" #iname "_" #tname, scatter, ttype, itype)

#define init_scatter_add(iname, tname, itype, ttype) \
    init_kernel("sa_" #iname "_" #tname, scatter_add, ttype, itype)

#define init_index_add(iname, tname, itype, ttype) \
    init_kernel("ia_" #iname "_" #tname, index_add, ttype, itype)

// ---------------------------------------------------------------------------
// index_select -- 18 variants, three index types over six value types.
//
// `is_i64_u8` and `is_i64_u32` were **absent before this conversion** while
// `candle-core` named both, so `index_select` on a U8 or U32 tensor with I64
// indices was a runtime `LoadFunctionError` on `lloom/integration`. That is the
// fourth firing of the absent-variant class `DESIGN.md` §8.1b tracks, after
// #26's 48 reduce variants and conv's. They are declared here, and
// `indexing_names_resolve` is what keeps the two sides from drifting again.
//
// `is_u32_f16` is the embedding lookup -- the one kernel in this file on the
// LFM2 decode path (§11.3h), one dispatch per token.
// ---------------------------------------------------------------------------
init_index(i64, i64, int64_t, int64_t)
init_index(i64, f32, int64_t, float)
init_index(i64, f16, int64_t, half)
init_index(i64, u8, int64_t, uint8_t)
init_index(i64, u32, int64_t, uint32_t)
#if defined(__HAVE_BFLOAT__)
init_index(i64, bf16, int64_t, bfloat)
#endif

init_index(u32, u8, uint32_t, uint8_t)
init_index(u32, u32, uint32_t, uint32_t)
init_index(u32, i64, uint32_t, int64_t)
init_index(u32, f32, uint32_t, float)
init_index(u32, f16, uint32_t, half)
#if defined(__HAVE_BFLOAT__)
init_index(u32, bf16, uint32_t, bfloat)
#endif

init_index(u8, u8, uint8_t, uint8_t)
init_index(u8, u32, uint8_t, uint32_t)
init_index(u8, i64, uint8_t, int64_t)
init_index(u8, f32, uint8_t, float)
init_index(u8, f16, uint8_t, half)
#if defined(__HAVE_BFLOAT__)
init_index(u8, bf16, uint8_t, bfloat)
#endif

// ---------------------------------------------------------------------------
// gather -- 16 variants.
//
// The pre-conversion list spelled `uint` for the u32 index type here and
// `uint32_t` in the index_select block. They are the same type in MSL, so this
// is a spelling difference rather than a behaviour one; `uint32_t` is used
// throughout below so the instantiation rows read uniformly.
// ---------------------------------------------------------------------------
init_gather(u8, f32, uint8_t, float)
init_gather(u8, f16, uint8_t, half)
init_gather(i64, f32, int64_t, float)
init_gather(i64, f16, int64_t, half)
init_gather(u32, f32, uint32_t, float)
init_gather(u32, f16, uint32_t, half)
#if defined(__HAVE_BFLOAT__)
init_gather(u8, bf16, uint8_t, bfloat)
init_gather(i64, bf16, int64_t, bfloat)
init_gather(u32, bf16, uint32_t, bfloat)
#endif
init_gather(u8, u8, uint8_t, uint8_t)
init_gather(u8, i64, uint8_t, int64_t)
init_gather(u8, u32, uint8_t, uint32_t)
init_gather(u32, u32, uint32_t, uint32_t)
init_gather(u32, i64, uint32_t, int64_t)
init_gather(i64, u32, int64_t, uint32_t)
init_gather(i64, i64, int64_t, int64_t)

// ---------------------------------------------------------------------------
// scatter_add -- 10 variants.
// ---------------------------------------------------------------------------
init_scatter_add(u32, f32, uint32_t, float)
init_scatter_add(u8, f32, uint8_t, float)
init_scatter_add(i64, f32, int64_t, float)
init_scatter_add(u32, u32, uint32_t, uint32_t)
init_scatter_add(u32, f16, uint32_t, half)
init_scatter_add(u8, f16, uint8_t, half)
init_scatter_add(i64, f16, int64_t, half)
#if defined(__HAVE_BFLOAT__)
init_scatter_add(u32, bf16, uint32_t, bfloat)
init_scatter_add(u8, bf16, uint8_t, bfloat)
init_scatter_add(i64, bf16, int64_t, bfloat)
#endif

// ---------------------------------------------------------------------------
// scatter -- 10 variants. Same key set as scatter_add; the two differ in
// whether the destination is assigned or accumulated into, which is the kernel
// body rather than the name.
// ---------------------------------------------------------------------------
init_scatter(u32, f32, uint32_t, float)
init_scatter(u8, f32, uint8_t, float)
init_scatter(i64, f32, int64_t, float)
init_scatter(u32, u32, uint32_t, uint32_t)
init_scatter(u32, f16, uint32_t, half)
init_scatter(u8, f16, uint8_t, half)
init_scatter(i64, f16, int64_t, half)
#if defined(__HAVE_BFLOAT__)
init_scatter(u32, bf16, uint32_t, bfloat)
init_scatter(u8, bf16, uint8_t, bfloat)
init_scatter(i64, bf16, int64_t, bfloat)
#endif

// ---------------------------------------------------------------------------
// index_add -- 18 variants, the same three-by-six grid as index_select.
// ---------------------------------------------------------------------------
init_index_add(i64, f16, int64_t, half)
init_index_add(i64, f32, int64_t, float)
init_index_add(i64, i64, int64_t, int64_t)
init_index_add(i64, u32, int64_t, uint32_t)
init_index_add(i64, u8, int64_t, uint8_t)
#if defined(__HAVE_BFLOAT__)
init_index_add(i64, bf16, int64_t, bfloat)
#endif

init_index_add(u32, f16, uint32_t, half)
init_index_add(u32, f32, uint32_t, float)
init_index_add(u32, i64, uint32_t, int64_t)
init_index_add(u32, u32, uint32_t, uint32_t)
init_index_add(u32, u8, uint32_t, uint8_t)
#if defined(__HAVE_BFLOAT__)
init_index_add(u32, bf16, uint32_t, bfloat)
#endif

init_index_add(u8, f16, uint8_t, half)
init_index_add(u8, f32, uint8_t, float)
init_index_add(u8, i64, uint8_t, int64_t)
init_index_add(u8, u32, uint8_t, uint32_t)
init_index_add(u8, u8, uint8_t, uint8_t)
#if defined(__HAVE_BFLOAT__)
init_index_add(u8, bf16, uint8_t, bfloat)
#endif
