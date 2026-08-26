pub const AFFINE: &str = include_str!("metal_src/affine.metal");
pub const ARENA_ALLOC: &str = include_str!("metal_src/arena_alloc.metal");
pub const BINARY: &str = include_str!("metal_src/binary.metal");
pub const CAST: &str = include_str!("metal_src/cast.metal");
pub const CONV: &str = include_str!("metal_src/conv.metal");
pub const FILL: &str = include_str!("metal_src/fill.metal");
pub const INDEXING: &str = include_str!("metal_src/indexing.metal");
pub const GEMV: &str = include_str!("metal_src/gemv.metal");
pub const MLX_GEMM: &str = include_str!("metal_src/mlx_gemm.metal");
pub const MLX_SORT: &str = include_str!("metal_src/mlx_sort.metal");
pub const QUANTIZED: &str = include_str!("metal_src/quantized.metal");
pub const RANDOM: &str = include_str!("metal_src/random.metal");
pub const REDUCE: &str = include_str!("metal_src/reduce.metal");
pub const SCRATCH: &str = include_str!("metal_src/scratch.metal");
pub const SORT: &str = include_str!("metal_src/sort.metal");
pub const TERNARY: &str = include_str!("metal_src/ternary.metal");
pub const UNARY: &str = include_str!("metal_src/unary.metal");
pub const SDPA: &str = include_str!("metal_src/scaled_dot_product_attention.metal");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Affine,
    /// GPU-side bump allocation over the activation arena (`DESIGN.md` §9.2d,
    /// issue #70). Not dispatched by any tensor op -- the arena drives it.
    ArenaAlloc,
    Binary,
    Cast,
    Conv,
    Fill,
    Gemm,
    Gemv,
    Indexing,
    MlxSort,
    Quantized,
    Random,
    Reduce,
    /// FlashDecoding partials: the scratch class and its sizing policy
    /// (`DESIGN.md` §9.1, issue #71). Not dispatched by any tensor op -- the
    /// scratch arena drives it, and the kernel is a stub until Phase 4/5
    /// implements FlashDecoding itself.
    Scratch,
    Sort,
    Ternary,
    Unary,
    Sdpa,
}
