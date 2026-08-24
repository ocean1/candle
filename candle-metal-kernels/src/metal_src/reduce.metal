#include <metal_stdlib>
#include <metal_limits>
using namespace metal;

template<uint Y>
constexpr uint div_ceil(uint x) {
    return x / Y + (x % Y > 0);
}

template<uint X, uint Y>
constexpr uint div_ceil() {
    return X / Y + (X % Y > 0);
}

template<typename T>
constexpr uint work_per_thread() {
    return div_ceil<8, sizeof(T)>();
}

METAL_FUNC uint nonzero(uint n) {
    return n == 0 ? 1 : n;
}

template<uint N>
constexpr uint nonzero() {
    return N == 0 ? 1 : N;
}

template<typename T>
constexpr ushort granularity() {
    return nonzero<vec_elements<T>::value>();
}

METAL_FUNC uint next_p2(uint x) {
    return 1 << (32 - clz(x - 1));
}

METAL_FUNC uint prev_p2(uint x) {
    return 1 << (31 - clz(x));
}

constant uint MAX_SHARED_MEM = 32767;

template<typename T>
METAL_FUNC uint max_shared_mem(uint n) {
    return min(n, div_ceil<MAX_SHARED_MEM, sizeof(T)>());
}


template<ushort D, typename IndexT>
struct strided_indexer {
    constant const IndexT *dims;
    constant const IndexT *strides;
    strided_indexer<D - 1, IndexT> next {dims, strides};

    METAL_FUNC IndexT operator()(IndexT idx) const {
        IndexT dim = dims[D - 1];
        IndexT i = (idx % dim) * strides[D - 1];
        idx /= dim;
        return i + next(idx);
    }
};

template<typename IndexT>
struct strided_indexer<1, IndexT> {
    constant const IndexT *dims;
    constant const IndexT *strides;

    METAL_FUNC IndexT operator()(IndexT idx) const {
        return idx * strides[0];
    }
};

template<ushort D, typename IndexT>
METAL_FUNC IndexT get_strided_idx_fallback(
    IndexT idx,
    constant const IndexT &num_dims,
    constant const IndexT *dims,
    constant const IndexT *strides
) {
    strided_indexer<D, IndexT> next {dims, strides};

    IndexT strided_i = 0;
    for (IndexT d = D; d < num_dims; d++) {
        IndexT dim_idx = num_dims - 1 - (d - D);
        IndexT dim = dims[dim_idx];
        strided_i += (idx % dim) * strides[dim_idx];
        idx /= dim;
    }
    return strided_i + next(idx);
}

template<typename IndexT>
METAL_FUNC IndexT get_strided_index_t(
    IndexT idx,
    constant const IndexT &num_dims,
    constant const IndexT *dims,
    constant const IndexT *strides
) {
    switch (num_dims) {
        case 1: return strided_indexer<1, IndexT>{dims, strides}(idx);
        case 2: return strided_indexer<2, IndexT>{dims, strides}(idx);
        case 3: return strided_indexer<3, IndexT>{dims, strides}(idx);
        case 4: return strided_indexer<4, IndexT>{dims, strides}(idx);
        //case 5: return strided_indexer<5, IndexT>{dims, strides}(idx);
        //case 6: return strided_indexer<6, IndexT>{dims, strides}(idx);
        default: return get_strided_idx_fallback<4, IndexT>(idx, num_dims, dims, strides);
    }
}

template<typename IndexT, bool STRIDED>
struct indexer_t {
    typedef IndexT I;
};

template<typename IndexT>
struct indexer_t<IndexT, false> {
    typedef IndexT I;

    const IndexT last_dim = 0;

    METAL_FUNC IndexT operator()(IndexT i) const {
        return i;
    }
};

template<typename IndexT>
struct indexer_t<IndexT, true> {
    typedef IndexT I;

    constant const IndexT &num_dims;
    constant const IndexT *dims;
    constant const IndexT *strides;
    const IndexT last_dim;

    METAL_FUNC IndexT operator()(IndexT i) const {
        return get_strided_index_t(i, num_dims, dims, strides);
    }
};

struct Divide {
    template<typename T>
    METAL_FUNC T operator()(T a, T b) { return a / b; }
    METAL_FUNC float  operator()(float  a, float  b) { return fast::divide(a, b); }
    METAL_FUNC half   operator()(half   a, half   b) { return divide(a, b); }
    #if defined(__HAVE_BFLOAT__)
    METAL_FUNC bfloat  operator()(bfloat  a, bfloat  b) { return static_cast<bfloat>(fast::divide(a, b)); }
    #endif
};

struct Exp {
    template<typename T>
    METAL_FUNC T operator()(T a) { return fast::exp(a); }
    METAL_FUNC float  operator()(float  a) { return fast::exp(a); }
    METAL_FUNC half   operator()(half   a) { return exp(a); }
    #if defined(__HAVE_BFLOAT__)
    METAL_FUNC bfloat  operator()(bfloat  a) { return static_cast<bfloat>(fast::exp(a)); }
    #endif
};


// Keeps track of the index of the value in the reduction operation (argmin, argmax, etc.)
// and the value itself. The index is also used to break ties in the reduction operation.
template <typename T>
struct indexed {
    uint i;
    T val;

    constexpr indexed<T>() threadgroup = default;
};

template <typename T>
struct is_indexed_type {
    static constant constexpr bool value = false;
};

template <typename T>
constexpr constant bool is_indexed_t = is_indexed_type<T>::value;

template <typename T>
struct is_indexed_type<indexed<T>> {
    static constant constexpr bool value = true;
};

template <typename T>
constexpr constant bool not_indexed_t = !is_indexed_t<T>;

template<typename T>
constexpr METAL_FUNC bool operator<(indexed<T> lhs, indexed<T> rhs) {
    return lhs.val < rhs.val || (lhs.val == rhs.val && lhs.i < rhs.i);
}

template<typename T>
constexpr METAL_FUNC bool operator>(indexed<T> lhs, indexed<T> rhs) {
    return lhs.val > rhs.val || (lhs.val == rhs.val && lhs.i < rhs.i);
}

template<typename T>
struct _numeric_limits_impl<indexed<T>> {
    static constexpr METAL_FUNC indexed<T> lowest() {
        return indexed<T>{ 0, numeric_limits<T>::lowest() };
    }

    static constexpr METAL_FUNC indexed<T> max() {
        return indexed<T>{ 0, numeric_limits<T>::max() };
    }
};

#if __METAL_VERSION__ >= 220
METAL_FUNC int64_t simd_shuffle_down(int64_t data, uint16_t delta) {
  return as_type<int64_t>(simd_shuffle_down(as_type<uint2>(data), delta));
}
#endif


#if defined(__HAVE_BFLOAT__)
// Metal does not have simd_shuffle_down for bfloat16
METAL_FUNC bfloat simd_shuffle_down(bfloat value, ushort delta) {
    return as_type<bfloat>(simd_shuffle_down(as_type<ushort>(value), delta));
}
#endif

template <typename T>
METAL_FUNC indexed<T> simd_shuffle_down(indexed<T> iv, ushort delta) {
    return indexed<T> {
        simd_shuffle_down(iv.i, delta),
        simd_shuffle_down(iv.val, delta)
    };
}

template<typename T>
struct Sum {
    static constexpr METAL_FUNC T init() {
        return 0;
    }
    static METAL_FUNC T simd_op(T a) {
        return simd_sum(a);
    }

    template<typename V>
    METAL_FUNC V operator()(V a, V b) {
        return a + b;
    }
};

template<typename T>
struct Mul {
    static constexpr METAL_FUNC T init() {
        return 1;
    }
    static METAL_FUNC T simd_op(T a) {
        return simd_product(a);
    }

    template<typename V>
    METAL_FUNC V operator()(V a, V b) {
        return a * b;
    }
};

template<typename T>
struct Min {
    static constexpr METAL_FUNC T init() {
        return numeric_limits<T>::max();
    }
    static METAL_FUNC T simd_op(T a) {
        return simd_min(a);
    }

    template<typename V>
    METAL_FUNC V operator()(V a, V b) { return a < b ? a : b; }

    METAL_FUNC float operator()(float a, float b) { return fast::min(a, b); }
    METAL_FUNC half   operator()(half   a, half   b) { return min(a, b); }
    METAL_FUNC uint operator()(uint a, uint b) { return min(a, b); }
    METAL_FUNC uchar operator()(uchar a, uchar b) { return min(a, b); }

    #if __METAL_VERSION__ >= 220
    METAL_FUNC long operator()(long a, long b) { return min(a, b); }
    #endif

    #if defined(__HAVE_BFLOAT__)
    METAL_FUNC bfloat operator()(bfloat a, bfloat b) { return static_cast<bfloat>(fast::min(static_cast<float>(a), static_cast<float>(b))); }
    #endif
};

template<typename T>
struct Max {
    static constexpr METAL_FUNC T init() {
        return numeric_limits<T>::lowest();
    }
    static METAL_FUNC T simd_op(T a) {
        return simd_max(a);
    }

    template<typename V>
    METAL_FUNC V operator()(V a, V b) { return a > b ? a : b; }

    METAL_FUNC float operator()(float a, float b) { return fast::max(a, b); }
    METAL_FUNC half operator()(half a, half b) { return max(a, b); }
    METAL_FUNC uint operator()(uint a, uint b) { return max(a, b); }
    METAL_FUNC uchar operator()(uchar a, uchar b) { return max(a, b); }

    #if __METAL_VERSION__ >= 220
    METAL_FUNC long operator()(long a, long b) { return max(a, b); }
    #endif

    #if defined(__HAVE_BFLOAT__)
    METAL_FUNC bfloat operator()(bfloat a, bfloat b) { return static_cast<bfloat>(fast::max(static_cast<float>(a), static_cast<float>(b))); }
    #endif
};

template <typename T>
constexpr constant bool is_simd_t = __is_valid_simdgroup_type<T>::value;

template <typename T, typename _E = void>
struct is_valid_simd_type {
    static constant constexpr bool value = false;
};

template <typename T>
constexpr constant bool is_valid_simd_t = is_valid_simd_type<T>::value;

template <typename T>
struct is_valid_simd_type<T, typename metal::enable_if_t<is_simd_t<T>>> {
    static constant constexpr bool value = true;
};

template <typename T>
struct is_valid_simd_type<indexed<T>, typename metal::enable_if_t<is_valid_simd_t<T>>> {
    static constant constexpr bool value = true;
};

#if __METAL_VERSION__ >= 220
template <>
struct is_valid_simd_type<int64_t> {
    static constant constexpr bool value = true;
};
#endif

#if defined(__HAVE_BFLOAT__)
template <>
struct is_valid_simd_type<bfloat> {
    static constant constexpr bool value = true;
};
#endif

template <typename T, typename _E = void>
struct is_simd_op {
    static constant constexpr bool value = false;
};
template <typename T>
struct is_simd_op<Sum<T>, typename metal::enable_if_t<is_simd_t<T>>> {
    static constant constexpr bool value = true;
};
template <typename T>
struct is_simd_op<Mul<T>, typename metal::enable_if_t<is_simd_t<T>>> {
    static constant constexpr bool value = true;
};
template <typename T>
struct is_simd_op<Min<T>, typename metal::enable_if_t<is_simd_t<T>>> {
    static constant constexpr bool value = true;
};
template <typename T>
struct is_simd_op<Max<T>, typename metal::enable_if_t<is_simd_t<T>>> {
    static constant constexpr bool value = true;
};

// Helper struct for applying operators.
// The overloaded operator() function is used to apply an operator to two values.
template<typename OP, typename T>
struct operation;

// Specialization for scalar values.
template<typename OP, typename T>
struct operation {
    OP op;

    METAL_FUNC T operator()(T a, T b) {
        return op(a, b);
    }
};

// Specialization for indexed values.
template<typename OP, typename T>
struct operation<OP, indexed<T>> {
    OP op;

    METAL_FUNC indexed<T> operator()(indexed<T> a, indexed<T> b) {
        return op(a, b);
    }
    METAL_FUNC indexed<T> operator()(indexed<T> a, T b, uint idx) {
        return this->operator()(a, indexed<T>{ idx, b });
    }
};

// Load elements from global memory into shared memory.
// Handles both indexed and non-indexed types by using operate.
template<
    typename T,
    typename R,
    typename OP,
    ushort BLOCKSIZE,
    typename Indexer,
    typename IndexT,
    typename _E = void
>
struct loader;

template<
    typename T,
    typename R,
    typename OP,
    ushort BLOCKSIZE,
    typename Indexer,
    typename IndexT
>
struct loader<T, R, OP, BLOCKSIZE, Indexer, IndexT, typename metal::enable_if_t<not_indexed_t<R>>> {
    operation<OP, R> operate;

    METAL_FUNC R operator()(
        R value,
        Indexer indexer,
        constant IndexT &src_numel,
        constant IndexT &el_per_block,
        device const T *src,
        const IndexT offset,
        const uint tid
    ) {
        const IndexT idx = tid + offset;
        const IndexT stop_idx = min(el_per_block + offset, src_numel);

        #pragma clang loop unroll(full)
        for (IndexT i = idx; i < stop_idx; i += BLOCKSIZE) {
            value = operate(value, src[indexer(i)]);
        }
        return value;
    }
};

// Indexed
template<
    typename T,
    typename R,
    typename OP,
    ushort BLOCKSIZE,
    typename Indexer,
    typename IndexT
>
struct loader<T, R, OP, BLOCKSIZE, Indexer, IndexT, typename metal::enable_if_t<is_indexed_t<R>>> {
    operation<OP, R> operate;

    METAL_FUNC R operator()(
        R value,
        Indexer indexer,
        constant IndexT &src_numel,
        constant IndexT &el_per_block,
        device const T *src,
        const IndexT offset,
        const uint tid
    ) {
        const IndexT idx = tid + offset;
        const IndexT stop_idx = min(el_per_block + offset, src_numel);

        #pragma clang loop unroll(full)
        for (IndexT i = idx; i < stop_idx; i += BLOCKSIZE) {
            value = operate(value, src[indexer(i)], i % indexer.last_dim);
        }
        return value;
    }
};

template<
    typename OP,
    ushort BLOCKSIZE,
    typename T,
    typename _E = void
>
struct simdgroup_reducer;

// Specialization for built-in simd operations.
template<typename OP, ushort BLOCKSIZE, typename T>
struct simdgroup_reducer<OP, BLOCKSIZE, T, typename metal::enable_if_t<is_simd_op<OP>::value && is_valid_simd_t<T>>> {
    METAL_FUNC T operator()(T value) {
        return OP::simd_op(value);
    }
};

// Specialization for custom (non-built-in) simd operations.
template<typename OP, ushort BLOCKSIZE, typename T>
struct simdgroup_reducer<OP, BLOCKSIZE, T, typename metal::enable_if_t<!is_simd_op<OP>::value && is_valid_simd_t<T>>> {
    operation<OP, T> op;

    METAL_FUNC T operator()(T value) {
        if (BLOCKSIZE >= 32) value = op(value, simd_shuffle_down(value, 16));
        if (BLOCKSIZE >= 16) value = op(value, simd_shuffle_down(value,  8));
        if (BLOCKSIZE >=  8) value = op(value, simd_shuffle_down(value,  4));
        if (BLOCKSIZE >=  4) value = op(value, simd_shuffle_down(value,  2));
        if (BLOCKSIZE >=  2) value = op(value, simd_shuffle_down(value,  1));
        return value;
    }
};

template<typename T, typename OP, ushort BLOCKSIZE>
struct block_reducer {
    simdgroup_reducer<OP, BLOCKSIZE, T> simd_reduce;
    operation<OP, T> operate;
    threadgroup T *shared;

    block_reducer(threadgroup T shared[BLOCKSIZE]) {
        this->shared = shared;
    }

    METAL_FUNC T operator()(T value, const uint tid) {
        if (BLOCKSIZE >= 64) {
            // Only store in threadgroup shared memory if needed.
            shared[tid] = value;
            // Threadgroup barrier is needed to ensure that all threads have written to shared memory
            threadgroup_barrier(mem_flags::mem_none);
        }

        #pragma clang loop unroll(full)
        for (ushort s = BLOCKSIZE / 2; s >= 64; s >>= 1) {
            if (tid < s) shared[tid] = operate(shared[tid], shared[tid + s]);
            threadgroup_barrier(mem_flags::mem_none);
        }
        if (tid < 32) {
            // Last shared memory reduce can be done without tid < s check.
            if (BLOCKSIZE >= 64) {
                value = operate(shared[tid], shared[tid + 32]);
                simdgroup_barrier(mem_flags::mem_none);
            }
            // Remaining 32 threads can be reduced with simdgroup_reduce.
            value = simd_reduce(value);
        }
        return value;
    }
};

template<typename T, typename _E = void>
struct storer;

template<typename T>
struct storer<T, typename metal::enable_if_t<not_indexed_t<T>>> {
    device T *dst;
    const uint tid;
    const uint dst_id;

    METAL_FUNC void operator()(T value) {
        if (tid == 0) {
            dst[dst_id] = value;
        }
    }
};

template<typename T>
struct storer<T, typename metal::enable_if_t<is_indexed_t<T>>> {
    device uint *dst;
    const uint tid;
    const uint dst_id;

    METAL_FUNC void operator()(T value) {
        if (tid == 0) {
            dst[dst_id] = value.i;
        }
    }
};

// Inspired by "Optimizing Parallel Reduction in CUDA" by Mark Harris
template<
    typename T,
    typename R,
    typename OP,
    ushort BLOCKSIZE,
    typename Indexer,
    typename IndexT = typename Indexer::IndexT
>
METAL_FUNC void reduce(
    Indexer indexer,
    constant IndexT &src_numel,
    constant IndexT &el_per_block,
    device const T *src,
    device R *dst,
    threadgroup R shared[BLOCKSIZE],
    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]]
) {
    loader<T, R, OP, BLOCKSIZE, Indexer, IndexT> load;
    block_reducer<R, OP, BLOCKSIZE> reduce(shared);
    storer<R> store { dst, tid, dst_id };

    // Calculate offset for the threadgroup of current thread
    const IndexT offset = dst_id * el_per_block;

    // Load with reduction from global memory into shared memory
    auto value = load(OP::init(), indexer, src_numel, el_per_block, src, offset, tid);

    // Complete reduction
    R result = reduce(value, tid);

    store(result);
}

// The threadgroup-size switch below is not the dtype axis that `conv.metal`'s
// conversion dealt with, and it does not fold into the kernel name.
//
// `threadgroup R shared[N]` needs a compile-time `N`, but the threadgroup size
// is chosen per dispatch on the CPU from the tensor shape
// (`kernels/reduce.rs`: `min(max_total_threads_per_threadgroup,
// (work_per_threadgroup / 2).next_power_of_two())`) and reaches the kernel as
// `block_dim`. So one entry point covers every block size by switching on it
// and calling the matching `BLOCKSIZE` instantiation — eleven instantiations
// behind one `[[host_name]]`, selected at runtime.
//
// Lifting `N` into the name would mean either recomputing `max_shared_mem`'s
// clamp on the Rust side to pick the variant — duplicating a rule that lives in
// exactly one place today, which is the coupling issue #8 removed — or fixing
// `N`, which changes occupancy and is a performance change. Both are out of
// scope for a conversion whose acceptance criterion is no behaviour change, so
// the switch stays exactly as it was and only the wrapper duplication goes.
// Two spellings because the clamp is on the type actually held in threadgroup
// memory, which is not always `T`: the reduce, arg-reduce and softmax families
// clamp on `T`, while `rms_norm` and `layer_norm` accumulate in `float` and
// clamp on that. The macro form spelled this difference out per family
// (`max_shared_mem<float>` in `impl_rms_norm` / `impl_layer_norm`); it is
// preserved rather than unified, since collapsing the two would change which
// block size a half-precision norm selects.
#define reduce_switch_on(A, CASE_MACRO)                 \
    switch (max_shared_mem<A>(block_dim)) {             \
        CASE_MACRO(1024)                                \
        CASE_MACRO( 512)                                \
        CASE_MACRO( 256)                                \
        CASE_MACRO( 128)                                \
        CASE_MACRO(  64)                                \
        CASE_MACRO(  32)                                \
        CASE_MACRO(  16)                                \
        CASE_MACRO(   8)                                \
        CASE_MACRO(   4)                                \
        CASE_MACRO(   2)                                \
        CASE_MACRO(   1)                                \
    }

#define reduce_switch(CASE_MACRO) reduce_switch_on(T, CASE_MACRO)

#define reduce_case(N)                                                  \
case N: {                                                               \
    threadgroup R shared[N];                                            \
    reduce<T, R, OP, N>(                                                \
        indexer, src_numel, el_per_block, src, dst, shared, tid, dst_id \
    );                                                                  \
    break;                                                              \
}

// `OP` is the reduction operator already applied to the accumulator type
// (`Sum<float>`), not a template-template parameter as the macro form took it.
// The macro spelled `OP<R>` at each use site; naming the applied type once in
// the instantiation keeps the accumulator visible in exactly one place per
// variant, which is what `avg_pool2d`'s per-dtype accumulators needed in #9.
template<typename T, typename R, typename OP>
[[kernel]] void reduce_kernel(
    constant uint &src_numel,
    constant uint &num_dims,
    constant uint *dims,
    constant uint &el_per_block,
    device const T *src,
    device R *dst,
    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]],
    uint block_dim [[ threads_per_threadgroup ]]
) {
    indexer_t<uint, false> indexer;
    reduce_switch(reduce_case)
}

template<typename T, typename R, typename OP>
[[kernel]] void reduce_kernel_strided(
    constant uint &src_numel,
    constant uint &num_dims,
    constant uint *dims,
    constant uint *strides,
    constant uint &el_per_block,
    device const T *src,
    device R *dst,
    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]],
    uint block_dim [[ threads_per_threadgroup ]]
) {
    indexer_t<uint, true> indexer {
        num_dims, dims, strides, dims[num_dims - 1]
    };
    reduce_switch(reduce_case)
}

#define init_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

// Both the contiguous and strided entry points for one (op, dtype), matching
// the pair `impl_reduce` used to generate.
//
// `opname` is spelled separately from the operator type rather than stringized
// from it: the type is `Sum` while the kernel is `fast_sum_*`, so `#op` would
// produce `fast_Sum_f32` — a name that compiles, resolves nowhere, and fails
// only when a dispatch asks for it. The registry test in `reduce_names.rs` is
// what catches that; it caught exactly this while this file was being written.
#define init_reduce(op, opname, tname, t)                               \
    init_kernel("fast_" #opname "_" #tname, reduce_kernel, t, t, op<t>) \
    init_kernel("fast_" #opname "_" #tname "_strided",                  \
                reduce_kernel_strided, t, t, op<t>)

template<
    typename T,
    typename ReductionOp,
    ushort BLOCKSIZE,
    typename Indexer,
    typename IndexT = typename Indexer::IndexT
>
METAL_FUNC void reduce(
    Indexer indexer,
    constant IndexT &src_numel,
    constant IndexT &el_per_block,
    device const T *src,
    device uint *dst,
    threadgroup indexed<T> shared[BLOCKSIZE],
    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]]
) {
    using I = indexed<T>;
    loader<T, I, ReductionOp, BLOCKSIZE, Indexer, IndexT> load;
    block_reducer<I, ReductionOp, BLOCKSIZE> reduce(shared);
    storer<I> store { dst, tid, dst_id };

    // Calculate offset for the threadgroup of current thread
    const uint offset = dst_id * el_per_block;

    // Load with reduction from global memory into shared memory
    auto value = load(
        ReductionOp::init(),
        indexer,
        src_numel,
        el_per_block,
        src,
        offset,
        tid
    );

    // Complete reduction
    I result = reduce(value, tid);

    // Return index of reduce result
    store(result);
}

#define arg_reduce_case(N)                              \
case N: {                                               \
    threadgroup I shared[N];                            \
    reduce<T, OP, N>(                                   \
        indexer,                                        \
        src_numel,                                      \
        el_per_block,                                   \
        src,                                            \
        dst,                                            \
        shared,                                         \
        tid,                                            \
        dst_id);                                        \
    break;                                              \
}

// `OP` is again the applied operator, here over the `indexed<T>` accumulator
// (`Min<indexed<float>>`), which is what carries the tie-breaking index
// alongside the value.
template<typename T, typename OP>
[[kernel]] void arg_reduce_kernel(
    constant uint &src_numel,
    constant uint &num_dims,
    constant uint *dims,
    constant uint &el_per_block,
    device const T *src,
    device uint *dst,
    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]],
    uint block_dim [[ threads_per_threadgroup ]]
) {
    using I = indexed<T>;
    indexer_t<uint, false> indexer {
        dims[num_dims - 1]
    };
    reduce_switch(arg_reduce_case)
}

template<typename T, typename OP>
[[kernel]] void arg_reduce_kernel_strided(
    constant uint &src_numel,
    constant uint &num_dims,
    constant uint *dims,
    constant uint *strides,
    constant uint &el_per_block,
    device const T *src,
    device uint *dst,
    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]],
    uint block_dim [[ threads_per_threadgroup ]]
) {
    using I = indexed<T>;
    indexer_t<uint, true> indexer {
        num_dims, dims, strides, dims[num_dims - 1]
    };
    reduce_switch(arg_reduce_case)
}

#define init_arg_reduce(op, opname, tname, t)                                 \
    init_kernel("fast_" #opname "_" #tname,                                   \
                arg_reduce_kernel, t, op<indexed<t>>)                         \
    init_kernel("fast_" #opname "_" #tname "_strided",                        \
                arg_reduce_kernel_strided, t, op<indexed<t>>)

// Contains the intermediate results for the online softmax calculation.
// m: max
// d: sum of the exponentials
template <typename T>
struct MD {
    T m;
    float d;

    constexpr MD<T>() = default;
    constexpr MD<T>() threadgroup = default;
};

// Enable operations for softmax MD
template<typename OP, typename T>
struct operation<OP, MD<T>> {
    OP op;

    METAL_FUNC MD<T> operator()(MD<T> a, MD<T> b) {
        return op(a, b);
    }

    METAL_FUNC MD<T> operator()(MD<T> a, T b) {
        return this->operator()(a, MD<T>{ b, static_cast<T>(1.0) });
    }
};

template <typename T>
METAL_FUNC MD<T> simd_shuffle_down(MD<T> md, ushort delta) {
    return MD<T> {
        simd_shuffle_down(md.m, delta),
        simd_shuffle_down(md.d, delta)
    };
}

// Enable simd_shuffle_down for softmax MD
template <typename T>
struct is_valid_simd_type<MD<T>, typename metal::enable_if_t<is_valid_simd_t<T>>> {
    static constant constexpr bool value = true;
};

template<typename T>
struct MDReduceOp {
    Exp fast_exp;

    static constexpr METAL_FUNC MD<T> init() {
        return MD<T>{ numeric_limits<T>::lowest(), 0 };
    }

    METAL_FUNC MD<T> operator()(MD<T> a, MD<T> b) {
        bool a_bigger = a.m > b.m;
        MD<T> bigger_m = a_bigger ? a : b;
        MD<T> smaller_m = a_bigger ? b : a;
        MD<T> res;
        res.d = bigger_m.d + smaller_m.d * fast_exp(smaller_m.m - bigger_m.m);
        res.m = bigger_m.m;
        return res;
    }
};

template<typename T, ushort BLOCKSIZE>
struct finalize_softmax {
    Divide fast_divide;
    Exp fast_exp;

    METAL_FUNC void operator()(
        device const T *src,
        device T *dst,
        threadgroup MD<T> &md_total,
        const uint thread_id,
        const uint stop_idx
    ) {
        const float d_total_inverse = fast_divide(1.0, md_total.d);
        for (uint idx = thread_id; idx < stop_idx; idx += BLOCKSIZE) {
            dst[idx] = static_cast<T>(fast_exp(src[idx] - md_total.m) * d_total_inverse);
        }
    }
};


// Welford's algorithm approach for an online softmax implementation.
// Same as the Online normalizer calculation for softmax: https://arxiv.org/pdf/1805.02867.pdf
template<typename T, ushort BLOCKSIZE>
METAL_FUNC void softmax(
    constant uint &src_numel,
    constant uint &el_per_block,
    device const T *src,
    device T *dst,
    threadgroup MD<T> shared[BLOCKSIZE],
    threadgroup MD<T> &md_total,

    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]]
) {
    using MDReduceOp = MDReduceOp<T>;
    using Indexer = indexer_t<uint, false>;
    Indexer indexer;
    loader<T, MD<T>, MDReduceOp, BLOCKSIZE, Indexer, uint> load;
    block_reducer<MD<T>, MDReduceOp, BLOCKSIZE> reduce(shared);
    finalize_softmax<T, BLOCKSIZE> softmax_finalize;

    // Calculate offset for the threadgroup of current thread;
    const uint offset = dst_id * el_per_block;

    // Calculate partial result for current thread
    MD<T> md_partial = MD<T> { numeric_limits<T>::lowest(), 0 };
    md_partial = load(
        md_partial,
        indexer,
        src_numel,
        el_per_block,
        src,
        offset,
        tid
    );

    // Reduce in shared memory
    MD<T> md = reduce(md_partial, tid);

    if (tid == 0) md_total = md;
    threadgroup_barrier(mem_flags::mem_none);

    // Finalize softmax
    const uint thread_id = tid + offset;
    const uint stop_idx = min(el_per_block + offset, src_numel);
    softmax_finalize(src, dst, md_total, thread_id, stop_idx);
}

#define softmax_case(N)                                 \
case N: {                                               \
    threadgroup MD<T> shared[N];                        \
    threadgroup MD<T> md_total;                         \
    softmax<T, N>(                                      \
        src_numel,                                      \
        el_per_block,                                   \
        src,                                            \
        dst,                                            \
        shared,                                         \
        md_total,                                       \
        tid,                                            \
        dst_id);                                        \
    break;                                              \
}

template<typename T>
[[kernel]] void softmax_kernel(
    constant uint &src_numel,
    constant uint &el_per_block,
    device const T *src,
    device T *dst,
    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]],
    uint block_dim [[ threads_per_threadgroup ]]
) {
    reduce_switch(softmax_case)
}

#define init_softmax(tname, t) \
    init_kernel("softmax_" #tname, softmax_kernel, t)


template<typename T>
METAL_FUNC void rmsnorm(
    constant size_t &src_numel,
    constant size_t &el_to_sum_per_block,
    device const T *src,
    device T *dst,
    device const T *alpha,
    constant float &eps,
    uint id,
    uint tid,
    uint dst_id,
    uint block_dim,
    threadgroup float * shared_memory
) {
    size_t start_idx = dst_id * el_to_sum_per_block;
    size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
    size_t idx = start_idx + tid;

    float tmp = 0;
    while (idx < stop_idx) {
        tmp = tmp + float(src[idx]) * float(src[idx]);
        idx += block_dim;
    }
    shared_memory[tid] = tmp;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_memory[tid] = shared_memory[tid] + shared_memory[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* wait for shared_memory[0] to be filled */
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float norm = sqrt(shared_memory[0] / float(el_to_sum_per_block) + eps);
    float inv_norm = 1.0f / norm;
    idx = start_idx + tid;
    while (idx < stop_idx) {
        float val = float(src[idx]) * inv_norm;
        if (alpha != nullptr) {
            val *= float(alpha[idx - start_idx]);
        }
        dst[idx] = T(val);
        idx += block_dim;
    }
}

template<typename T>
struct RMS {
    uint count;
    T sum_sq;

    constexpr RMS<T>() = default;
    constexpr RMS<T>() threadgroup = default;
};

template<typename T>
struct RMSLoadOp {
    static constexpr METAL_FUNC RMS<T> init() {
        return { 0, 0 };
    }

    METAL_FUNC RMS<T> operator()(RMS<T> a, RMS<T> b) {
        a.sum_sq += (b.sum_sq * b.sum_sq);
        a.count += 1;
        return a;
    }
};

template<typename T>
struct RMSReduceOp {
    static constexpr METAL_FUNC RMS<T> init() {
        return { 0, 0 };
    }

    METAL_FUNC RMS<T> operator()(RMS<T> a, RMS<T> b) {
        a.sum_sq += b.sum_sq;
        a.count += b.count;
        return a;
    }
};

template<typename OP, typename T>
struct operation<OP, RMS<T>> {
    OP op;

    METAL_FUNC RMS<T> operator()(RMS<T> a, RMS<T> b) {
        return op(a, b);
    }

    template<typename U>
    METAL_FUNC RMS<T> operator()(RMS<T> a, U b) {
        return this->operator()(a, RMS<T>{ 0, static_cast<T>(b) });
    }
};

template <typename T>
METAL_FUNC RMS<T> simd_shuffle_down(RMS<T> rms, ushort delta) {
    return RMS<T> {
        simd_shuffle_down(rms.count, delta),
        simd_shuffle_down(rms.sum_sq, delta)
    };
}

template <typename T>
struct is_valid_simd_type<RMS<T>, typename metal::enable_if_t<is_valid_simd_t<T>>> {
    static constant constexpr bool value = true;
};

// Kernels
template<
    typename T,
    ushort BLOCKSIZE
>
METAL_FUNC void rms_norm(
    constant uint &src_numel,
    constant uint &el_per_block,
    device const T *src,
    device T *dst,
    device const T *alpha,
    constant float &eps,
    threadgroup RMS<float> shared[BLOCKSIZE],
    threadgroup float &total,

    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]]
) {
    using Indexer = indexer_t<uint, false>;
    Indexer indexer;
    Divide fast_divide;
    loader<T, RMS<float>, RMSLoadOp<float>, BLOCKSIZE,  Indexer, uint> load;
    block_reducer<RMS<float>, RMSReduceOp<float>, BLOCKSIZE> reduce(shared);

    // Calculate offset for the threadgroup of current thread
    const uint offset = dst_id * el_per_block;
    const uint stop_idx = min(el_per_block + offset, src_numel);
    const uint idx = tid + offset;

    // Load with reduction from global memory into shared memory
    RMS<float> value = load(
        RMSLoadOp<float>::init(),
        indexer,
        src_numel,
        el_per_block,
        src,
        offset,
        tid
    );
    RMS<float> result = RMS<float> { value.count, static_cast<float>(value.sum_sq) };

    // Complete reduction
    result = reduce(result, tid);
    if (tid == 0) {
        total = rsqrt(fast_divide(result.sum_sq, float(el_per_block)) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (alpha == nullptr) {
        #pragma clang loop unroll(full)
        for (uint i = idx; i < stop_idx; i += BLOCKSIZE) {
            dst[i] = src[i] * static_cast<T>(total);
        }
    } else {
        #pragma clang loop unroll(full)
        for (uint i = idx; i < stop_idx; i += BLOCKSIZE) {
            T val = src[i] * static_cast<T>(total);
            val *= alpha[i - offset];
            dst[i] = val;
        }
    }
}


#define rms_norm_case(N)                                \
case N: {                                               \
    threadgroup RMS<float> shared[N];                   \
    threadgroup float total;                            \
    rms_norm<T, N>(                                     \
        src_numel,                                      \
        el_per_block,                                   \
        src,                                            \
        dst,                                            \
        alpha,                                          \
        eps,                                            \
        shared,                                         \
        total,                                          \
        tid,                                            \
        dst_id);                                        \
    break;                                              \
}

template<typename T>
[[kernel]] void rms_norm_kernel(
    constant uint &src_numel,
    constant uint &el_per_block,
    device const T *src,
    device T *dst,
    device const T *alpha,
    constant float &eps,
    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]],
    uint block_dim [[ threads_per_threadgroup ]]
) {
    reduce_switch_on(float, rms_norm_case)
}

#define init_rms_norm(tname, t) \
    init_kernel("rmsnorm_" #tname, rms_norm_kernel, t)

template<typename T>
struct LayerNormValue {
    uint count;
    T mean;
    T m2;

    constexpr LayerNormValue<T>() = default;
    constexpr LayerNormValue<T>() threadgroup = default;
};

template<typename T>
struct LNLoadOp {
    static constexpr METAL_FUNC LayerNormValue<T> init() {
        return { 0, 0, 0 };
    }

    METAL_FUNC LayerNormValue<T> operator()(LayerNormValue<T> a, LayerNormValue<T> b) {
        a.count += 1;
        T delta1 = b.mean - a.mean;
        a.mean += delta1 / a.count;
        T delta2 = b.mean - a.mean;
        a.m2 += delta1 * delta2;
        return a;
    }
};

template<typename T>
struct LNReduceOp {
    static constexpr METAL_FUNC LayerNormValue<T> init() {
        return { 0, 0, 0 };
    }

    METAL_FUNC LayerNormValue<T> operator()(LayerNormValue<T> a, LayerNormValue<T> b) {
        if (b.count == 0) {
            return a;
        }
        uint new_count = a.count + b.count;
        T nb_over_n = b.count / T(new_count);
        T delta = b.mean - a.mean;
        a.mean += delta * nb_over_n;
        a.m2 += b.m2 + delta * delta * a.count * nb_over_n;
        a.count = new_count;
        return a;
    }
};

template<typename OP, typename T>
struct operation<OP, LayerNormValue<T>> {
    OP op;

    METAL_FUNC LayerNormValue<T> operator()(LayerNormValue<T> a, LayerNormValue<T> b) {
        return op(a, b);
    }

    template<typename U>
    METAL_FUNC LayerNormValue<T> operator()(LayerNormValue<T> a, U b) {
        return this->operator()(a, LayerNormValue<T>{ 0, static_cast<T>(b), static_cast<T>(b) });
    }
};

template <typename T>
METAL_FUNC LayerNormValue<T> simd_shuffle_down(LayerNormValue<T> lnv, ushort delta) {
    return LayerNormValue<T> {
        simd_shuffle_down(lnv.count, delta),
        simd_shuffle_down(lnv.mean, delta),
        simd_shuffle_down(lnv.m2, delta)
    };
}

template <typename T>
struct is_valid_simd_type<LayerNormValue<T>, typename metal::enable_if_t<is_valid_simd_t<T>>> {
    static constant constexpr bool value = true;
};

// Kernels
template<
    typename T,
    ushort BLOCKSIZE
>
METAL_FUNC void layer_norm(
    constant uint &src_numel,
    constant uint &el_per_block,
    device const T *src,
    device T *dst,
    device const T *alpha,
    device const T *beta,
    constant float &eps,
    threadgroup LayerNormValue<float> shared[BLOCKSIZE],
    threadgroup float &mu,
    threadgroup float &sigma,

    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]],
    uint lane_id [[thread_index_in_simdgroup]]
) {
    using Indexer = indexer_t<uint, false>;
    Indexer indexer;
    Divide fast_divide;
    loader<T, LayerNormValue<float>, LNLoadOp<float>, BLOCKSIZE,  Indexer, uint> load;
    block_reducer<LayerNormValue<float>, LNReduceOp<float>, BLOCKSIZE> reduce(shared);

    // Calculate offset for the threadgroup of current thread
    const uint offset = dst_id * el_per_block;
    const uint stop_idx = min(el_per_block + offset, src_numel);
    const uint idx = tid + offset;

    // Load with reduction from global memory into shared memory
    LayerNormValue<float> value = load(
        LNReduceOp<float>::init(),
        indexer,
        src_numel,
        el_per_block,
        src,
        offset,
        tid
    );
    LayerNormValue<float> result = LayerNormValue<float> { value.count, static_cast<float>(value.mean), static_cast<float>(value.m2) };

    // Complete reduction
    result = reduce(result, tid);
    if (tid == 0) {
        mu = result.mean;
        sigma = rsqrt(fast_divide(result.m2, float(result.count)) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (alpha == nullptr || beta == nullptr) {
        if (alpha == nullptr) {
            #pragma clang loop unroll(full)
            for (uint i = idx; i < stop_idx; i += BLOCKSIZE) {
                T val = src[i];
                T normalized = (val - static_cast<T>(mu)) * static_cast<T>(sigma);
                dst[i] = normalized + beta[i - offset];
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint i = idx; i < stop_idx; i += BLOCKSIZE) {
                T val = src[i];
                T normalized = (val - static_cast<T>(mu)) * static_cast<T>(sigma);
                dst[i] = normalized * alpha[i - offset];
            }
        }
    } else {
        #pragma clang loop unroll(full)
        for (uint i = idx; i < stop_idx; i += BLOCKSIZE) {
            T val = src[i];
            T normalized = (val - static_cast<T>(mu)) * static_cast<T>(sigma);
            dst[i] = static_cast<T>(fma(normalized, alpha[i - offset], beta[i - offset]));
        }
    }
}

#define layer_norm_case(N)                              \
case N: {                                               \
    threadgroup LayerNormValue<float> shared[N];        \
    threadgroup float mu;                               \
    threadgroup float sigma;                            \
    layer_norm<T, N>(                                   \
        src_numel,                                      \
        el_per_block,                                   \
        src,                                            \
        dst,                                            \
        alpha,                                          \
        beta,                                           \
        eps,                                            \
        shared,                                         \
        mu,                                             \
        sigma,                                          \
        tid,                                            \
        dst_id,                                         \
        lane_id);                                       \
    break;                                              \
}

template<typename T>
[[kernel]] void layer_norm_kernel(
    constant uint &src_numel,
    constant uint &el_per_block,
    device const T *src,
    device T *dst,
    device const T *alpha,
    device const T *beta,
    constant float &eps,
    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint block_dim [[ threads_per_threadgroup ]]
) {
    reduce_switch_on(float, layer_norm_case)
}

#define init_layer_norm(tname, t) \
    init_kernel("layernorm_" #tname, layer_norm_kernel, t)

template<typename T>
METAL_FUNC void ropei(
    constant size_t &bh,
    constant size_t &td,
    constant size_t &stride_b,
    device const T *src,
    device const T *cos,
    device const T *sin,
    device T *dst,
    uint tid
) {
    if (2 * tid >= bh * td) {
        return;
    }
    size_t rope_idx = tid % (td / 2);
    if (stride_b > 0) {
      size_t b_idx = (2 * tid) / stride_b;
      rope_idx += b_idx * (td / 2);
    }
    T c = cos[rope_idx];
    T s = sin[rope_idx];
    dst[2 * tid] = src[2 * tid] * c - src[2 * tid + 1] * s;
    dst[2 * tid + 1] = src[2 * tid] * s + src[2 * tid + 1] * c;
}

template<typename T>
METAL_FUNC void rope(
    constant size_t &bh,
    constant size_t &td,
    constant size_t &d,
    constant size_t &stride_b,
    device const T *src,
    device const T *cos,
    device const T *sin,
    device T *dst,
    uint idx
) {
    if (2 * idx >= bh * td) {
        return;
    }
    size_t i_bh = idx / (td / 2);
    size_t i_td = idx - (td / 2) * i_bh;
    size_t i_t = i_td / (d / 2);
    size_t i_d = i_td - (d / 2) * i_t;
    size_t i1 = i_bh * td + i_t * d + i_d;
    size_t i2 = i1 + d / 2;
    size_t i_cs = i_t * (d / 2) + i_d;
    if (stride_b > 0) {
      size_t b_idx = (2 * idx) / stride_b;
      i_cs += b_idx * (td / 2);
    }
    T c = cos[i_cs];
    T s = sin[i_cs];
    dst[i1] = src[i1] * c - src[i2] * s;
    dst[i2] = src[i1] * s + src[i2] * c;
}

template<typename T>
METAL_FUNC void rope_thd(
    constant size_t &b,
    constant size_t &t,
    constant size_t &h,
    constant size_t &d,
    constant size_t &stride_b,
    device const T *src,
    device const T *cos,
    device const T *sin,
    device T *dst,
    uint idx
) {
    if (2 * idx >= b * t * h * d) {
        return;
    }
    const size_t i_bth = idx / (d / 2);
    const size_t i_d = idx - (d / 2) * i_bth;
    const size_t i_t = (i_bth / h) % t;
    const size_t i1 = i_bth * d + i_d;
    const size_t i2 = i1 + d / 2;
    size_t i_cs = i_t * (d / 2) + i_d;
    if (stride_b > 0) {
      const size_t b_idx = (2 * idx) / stride_b;
      i_cs += b_idx * ((t * d) / 2);
    }
    T c = cos[i_cs];
    T s = sin[i_cs];
    dst[i1] = src[i1] * c - src[i2] * s;
    dst[i2] = src[i1] * s + src[i2] * c;
}

template<typename T>
[[kernel]] void rope_i_kernel(
    constant size_t &bh,
    constant size_t &td,
    constant size_t &stride_b,
    device const T *src,
    device const T *cos,
    device const T *sin,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
    ropei<T>(bh, td, stride_b, src, cos, sin, dst, tid);
}

template<typename T>
[[kernel]] void rope_kernel(
    constant size_t &bh,
    constant size_t &td,
    constant size_t &d,
    constant size_t &stride_b,
    device const T *src,
    device const T *cos,
    device const T *sin,
    device T *dst,
    uint idx [[ thread_position_in_grid ]]
) {
    rope<T>(bh, td, d, stride_b, src, cos, sin, dst, idx);
}

template<typename T>
[[kernel]] void rope_thd_kernel(
    constant size_t &b,
    constant size_t &t,
    constant size_t &h,
    constant size_t &d,
    constant size_t &stride_b,
    device const T *src,
    device const T *cos,
    device const T *sin,
    device T *dst,
    uint idx [[ thread_position_in_grid ]]
) {
    rope_thd<T>(b, t, h, d, stride_b, src, cos, sin, dst, idx);
}

// The three variants a single `ROPE(...)` used to generate. Note the name
// shapes differ between them — `rope_f32`, `rope_i_f32`, `rope_thd_f32` — so
// the dtype suffix is not a common tail and each is spelled out.
#define init_rope(tname, t)                                     \
    init_kernel("rope_" #tname, rope_kernel, t)                 \
    init_kernel("rope_i_" #tname, rope_i_kernel, t)             \
    init_kernel("rope_thd_" #tname, rope_thd_kernel, t)

init_rms_norm(f32, float)
init_rms_norm(f16, half)
init_layer_norm(f32, float)
init_layer_norm(f16, half)
init_rope(f32, float)
init_rope(f16, half)

init_reduce(Sum, sum, f32, float)
init_reduce(Sum, sum, u32, uint)
init_reduce(Sum, sum, f16, half)
init_reduce(Sum, sum, u8, uint8_t)

init_reduce(Mul, mul, f32, float)
init_reduce(Mul, mul, u32, uint)
init_reduce(Mul, mul, f16, half)
init_reduce(Mul, mul, u8, uint8_t)

init_reduce(Max, max, f32, float)
init_reduce(Max, max, u32, uint)
init_reduce(Max, max, f16, half)
init_reduce(Max, max, u8, uint8_t)

init_reduce(Min, min, f32, float)
init_reduce(Min, min, u32, uint)
init_reduce(Min, min, f16, half)
init_reduce(Min, min, u8, uint8_t)

init_arg_reduce(Min, argmin, f32, float)
init_arg_reduce(Min, argmin, f16, half)
init_arg_reduce(Min, argmin, u32, uint)
init_arg_reduce(Min, argmin, u8, uint8_t)

init_arg_reduce(Max, argmax, f32, float)
init_arg_reduce(Max, argmax, f16, half)
init_arg_reduce(Max, argmax, u32, uint)
init_arg_reduce(Max, argmax, u8, uint8_t)

init_softmax(f32, float)
init_softmax(f16, half)

// `int64_t` gains `simd_shuffle_down` and a valid-simd-type marking only at
// Metal 2.2, so its variants are gated exactly as before.
#if __METAL_VERSION__ >= 220
init_reduce(Sum, sum, i64, int64_t)
init_reduce(Mul, mul, i64, int64_t)
init_reduce(Min, min, i64, int64_t)
init_reduce(Max, max, i64, int64_t)

init_arg_reduce(Min, argmin, i64, int64_t)
init_arg_reduce(Max, argmax, i64, int64_t)
#endif

// `__HAVE_BFLOAT__`, not `__METAL_VERSION__ >= 310`. Both guards appear in this
// tree and they are not interchangeable (#9 §1.2): everything bfloat in this
// file is reached through the operator overloads and `simd_shuffle_down`
// shim above, all of which are themselves under `__HAVE_BFLOAT__`.
#if defined(__HAVE_BFLOAT__)
init_reduce(Sum, sum, bf16, bfloat)
init_reduce(Mul, mul, bf16, bfloat)
init_reduce(Max, max, bf16, bfloat)
init_reduce(Min, min, bf16, bfloat)

init_arg_reduce(Min, argmin, bf16, bfloat)
init_arg_reduce(Max, argmax, bf16, bfloat)

init_softmax(bf16, bfloat)

init_rms_norm(bf16, bfloat)
init_layer_norm(bf16, bfloat)
init_rope(bf16, bfloat)
#endif
