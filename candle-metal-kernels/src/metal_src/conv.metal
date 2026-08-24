#include <metal_stdlib>

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))

template <typename T>
[[kernel]] void im2col(
    constant size_t &dst_numel,
    constant size_t &h_out,
    constant size_t &w_out,
    constant size_t &h_k,
    constant size_t &w_k,
    constant size_t &stride,
    constant size_t &padding,
    constant size_t &dilation,
    constant size_t *src_dims,
    constant size_t *src_strides,
    device const T *src,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  // dst: (b_size, h_out, w_out, c_in, h_k, w_k)
  // src: (b_size, c_in, h_in, w_in)
  if (tid >= dst_numel) {
    return;
  }
  const size_t b_in = src_dims[0];
  const size_t c_in = src_dims[1];
  const size_t h_in = src_dims[2];
  const size_t w_in = src_dims[3];

  const size_t dst_s4 = w_k;
  const size_t dst_s3 = h_k * dst_s4;
  const size_t dst_s2 = c_in * dst_s3;
  const size_t dst_s1 = w_out * dst_s2;
  const size_t dst_s0 = h_out * dst_s1;

  size_t tmp_tid = tid;
  const size_t b_idx = tmp_tid / dst_s0;
  tmp_tid -= b_idx * dst_s0;
  const size_t h_idx = tmp_tid / dst_s1;
  tmp_tid -= h_idx * dst_s1;
  const size_t w_idx = tmp_tid / dst_s2;
  tmp_tid -= w_idx * dst_s2;
  const size_t c_idx = tmp_tid / dst_s3;
  tmp_tid -= c_idx * dst_s3;
  const size_t h_k_idx = tmp_tid / dst_s4;
  tmp_tid -= h_k_idx * dst_s4;
  const size_t w_k_idx = tmp_tid;
  size_t src_h_idx = h_idx * stride + h_k_idx * dilation;
  size_t src_w_idx = w_idx * stride + w_k_idx * dilation;
  if (src_h_idx < padding || src_h_idx >= h_in + padding) {
    dst[tid] = static_cast<T>(0);
  }
  else if (src_w_idx < padding || src_w_idx >= w_in + padding) {
    dst[tid] = static_cast<T>(0);
  }
  else {
    src_h_idx -= padding;
    src_w_idx -= padding;
    const size_t src_i =
      b_idx * src_strides[0]
      + c_idx * src_strides[1]
      + src_h_idx * src_strides[2]
      + src_w_idx * src_strides[3];
    dst[tid] = src[src_i];
  }
}

template <typename T>
[[kernel]] void col2im1d(
    constant size_t &dst_el,
    constant size_t &l_out,
    constant size_t &l_in,
    constant size_t &c_out,
    constant size_t &k_size,
    constant size_t &stride,
    device const T *src,
    device T *dst,
    uint dst_i [[ thread_position_in_grid ]]
) {
  // src: (b_size, l_in, c_out, l_k)
  // dst: (b_size, c_out, l_out)
  if (dst_i >= dst_el) {
    return;
  }

  const size_t dst_s0 = c_out * l_out;
  const size_t dst_s1 = l_out;
  const size_t src_s0 = c_out * k_size * l_in;
  const size_t src_s1 = c_out * k_size;
  const size_t src_s2 = k_size;

  size_t tmp_dst_i = dst_i;
  const size_t b_idx = tmp_dst_i / dst_s0;
  tmp_dst_i -= b_idx * dst_s0;
  const size_t c_idx = tmp_dst_i / dst_s1;
  tmp_dst_i -= c_idx * dst_s1;
  const int l_out_idx = tmp_dst_i;

  dst[dst_i] = static_cast<T>(0);

  int l_in_idx = l_out_idx / stride;
  int k0 = l_out_idx - l_in_idx * stride;
  // l_out_idx = l_in_idx * stride + k0
  for (; k0 < k_size && l_in_idx >= 0; k0 += stride, --l_in_idx) {
    if (l_in_idx < l_in) {
      const size_t src_i = b_idx * src_s0 + l_in_idx * src_s1 + c_idx * src_s2 + k0;
      dst[dst_i] += src[src_i];
    }
  }
}

template <typename T>
[[kernel]] void im2col1d(
    constant size_t &dst_numel,
    constant size_t &l_out,
    constant size_t &l_k,
    constant size_t &stride,
    constant size_t &padding,
    constant size_t &dilation,
    constant size_t *src_dims,
    constant size_t *src_strides,
    device const T *src,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  // dst: (b_size, l_out, c_in, l_k)
  // src: (b_size, c_in, l_in)
  if (tid >= dst_numel) {
    return;
  }
  const size_t b_in = src_dims[0];
  const size_t c_in = src_dims[1];
  const size_t l_in = src_dims[2];

  const size_t dst_s2 = l_k;
  const size_t dst_s1 = c_in * dst_s2;
  const size_t dst_s0 = l_out * dst_s1;

  size_t tmp_dst_i = tid;
  const size_t b_idx = tmp_dst_i / dst_s0;
  tmp_dst_i -= b_idx * dst_s0;
  const size_t l_idx = tmp_dst_i / dst_s1;
  tmp_dst_i -= l_idx * dst_s1;
  const size_t c_idx = tmp_dst_i / dst_s2;
  tmp_dst_i -= c_idx * dst_s2;
  const size_t l_k_idx = tmp_dst_i;
  size_t src_l_idx = l_idx * stride + l_k_idx * dilation;
  if (src_l_idx < padding || src_l_idx >= l_in + padding) {
    dst[tid] = static_cast<T>(0);
  }
  else {
    src_l_idx -= padding;
    const size_t src_i = b_idx * src_strides[0] + c_idx * src_strides[1] + src_l_idx * src_strides[2];
    dst[tid] = src[src_i];
  }
}

template <typename T>
[[kernel]] void upsample_nearest2d(
    constant size_t &w_out,
    constant size_t &h_out,
    constant float &w_scale,
    constant float &h_scale,
    constant size_t *src_dims,
    constant size_t *src_s,
    device const T *src,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  // src: (b_size, c_in, w_in, h_in)

  const size_t c = src_dims[1];
  const size_t w_in = src_dims[2];
  const size_t h_in = src_dims[3];

  if (tid >= src_dims[0] * c * w_out * h_out) {
    return;
  }

  // TODO: Improve this.
  const size_t b_idx = tid / (w_out * h_out * c);
  const size_t c_idx = (tid / (w_out * h_out)) % c;
  const size_t dst_w = (tid / h_out) % w_out;
  const size_t dst_h = tid % h_out;

  size_t src_w = static_cast<size_t>(dst_w * w_scale);
  size_t src_h = static_cast<size_t>(dst_h * h_scale);
  if (src_w >= w_in) {
    src_w = w_in - 1;
  }
  if (src_h >= h_in) {
    src_h = h_in - 1;
  }

  const size_t src_i = b_idx * src_s[0] + c_idx * src_s[1] + src_w * src_s[2] + src_h * src_s[3];
  dst[tid] = src[src_i];
}

// Buffer indices are explicit here because the macro that used to generate this
// kernel's signature spelled them out; keeping them pinned means the binding
// layout stays fixed even if a parameter is added above.
template <typename T>
[[kernel]] void upsample_bilinear2d(
    constant size_t &w_out [[buffer(0)]],
    constant size_t &h_out [[buffer(1)]],
    constant bool &align_corners [[buffer(2)]],
    constant bool &has_scale_h [[buffer(3)]],
    constant float &scale_h_factor [[buffer(4)]],
    constant bool &has_scale_w [[buffer(5)]],
    constant float &scale_w_factor [[buffer(6)]],
    constant size_t *src_dims [[buffer(7)]],
    constant size_t *src_s [[buffer(8)]],
    device const T *src [[buffer(9)]],
    device T *dst [[buffer(10)]],
    uint tid [[thread_position_in_grid]]
) {
    // src: (b_size, c_in, h_in, w_in)  // Standard NCHW layout
    const size_t c = src_dims[1];
    const size_t h_in = src_dims[2];  // dims[2] = height
    const size_t w_in = src_dims[3];  // dims[3] = width
    
    if (tid >= src_dims[0] * c * h_out * w_out) {
        return;
    }
    
    // Compute output position (NCHW layout)
    const size_t b_idx = tid / (h_out * w_out * c);
    const size_t c_idx = (tid / (h_out * w_out)) % c;
    const size_t dst_h = (tid / w_out) % h_out;
    const size_t dst_w = tid % w_out;
    
    // Calculate scale factors following PyTorch's area_pixel_compute_scale logic
    float h_scale, w_scale;
    if (align_corners) {
        h_scale = (h_out > 1) ? static_cast<float>(h_in - 1) / (h_out - 1) : 0.0f;
        w_scale = (w_out > 1) ? static_cast<float>(w_in - 1) / (w_out - 1) : 0.0f;
    } else {
        // PyTorch's compute_scales_value logic
        h_scale = has_scale_h ? (1.0f / scale_h_factor) : (static_cast<float>(h_in) / h_out);
        w_scale = has_scale_w ? (1.0f / scale_w_factor) : (static_cast<float>(w_in) / w_out);
    }
    
    // Compute source position
    float src_h_fp, src_w_fp;
    if (align_corners) {
        src_h_fp = h_scale * dst_h;
        src_w_fp = w_scale * dst_w;
    } else {
        src_h_fp = h_scale * (dst_h + 0.5f) - 0.5f;
        src_w_fp = w_scale * (dst_w + 0.5f) - 0.5f;
    }
    
    // Clamp to valid range
    src_h_fp = max(0.0f, src_h_fp);
    src_w_fp = max(0.0f, src_w_fp);
    
    // Get integer indices
    size_t h0 = static_cast<size_t>(floor(src_h_fp));
    size_t w0 = static_cast<size_t>(floor(src_w_fp));
    size_t h1 = min(h0 + 1, h_in - 1);
    size_t w1 = min(w0 + 1, w_in - 1);
    
    // Compute interpolation weights
    float weight_h = src_h_fp - h0;
    float weight_w = src_w_fp - w0;
    weight_h = clamp(weight_h, 0.0f, 1.0f);
    weight_w = clamp(weight_w, 0.0f, 1.0f);
    
    // Get base index
    const size_t base = b_idx * src_s[0] + c_idx * src_s[1];
    
    // Read four neighboring pixels
    const T v00 = src[base + h0 * src_s[2] + w0 * src_s[3]];
    const T v10 = src[base + h0 * src_s[2] + w1 * src_s[3]];
    const T v01 = src[base + h1 * src_s[2] + w0 * src_s[3]];
    const T v11 = src[base + h1 * src_s[2] + w1 * src_s[3]];
    
    // Bilinear interpolation
    const float v_top = float(v00) * (1.0f - weight_w) + float(v10) * weight_w;
    const float v_bottom = float(v01) * (1.0f - weight_w) + float(v11) * weight_w;
    const float value = v_top * (1.0f - weight_h) + v_bottom * weight_h;
    
    dst[tid] = T(value);
}

template <typename T, typename A>
[[kernel]] void avg_pool2d(
    constant size_t &w_k,
    constant size_t &h_k,
    constant size_t &w_stride,
    constant size_t &h_stride,
    constant size_t *src_dims,
    constant size_t *src_strides,
    device const T *src,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  const size_t c = src_dims[1];
  const size_t w_in = src_dims[2];
  const size_t h_in = src_dims[3];

  const size_t w_out = (w_in - w_k) / w_stride + 1;
  const size_t h_out = (h_in - h_k) / h_stride + 1;
  if (tid >= src_dims[0] * c * w_out * h_out) {
    return;
  }

  const size_t b_idx = tid / (w_out * h_out * c);
  const size_t c_idx = (tid / (w_out * h_out)) % c;
  const size_t dst_w = (tid / h_out) % w_out;
  const size_t dst_h = tid % h_out;

  const size_t src_idx0 = b_idx * src_strides[0];
  A d = 0;
  for (size_t w_offset = 0; w_offset < w_k; ++w_offset) {
    size_t src_w = w_stride * dst_w + w_offset;
    if (src_w >= w_in){
      continue;
    }
    for (size_t h_offset = 0; h_offset < h_k; ++h_offset) {
      size_t src_h = h_stride * dst_h + h_offset;
      if (src_h >= h_in) {
        continue;
      }
      const size_t src_idx = src_idx0 + c_idx * src_strides[1] + src_w * src_strides[2] + src_h * src_strides[3];
      d += static_cast<A>(src[src_idx]);
    }
  }
  dst[tid] = static_cast<T>(d / (w_k * h_k));
}

template <typename T>
[[kernel]] void max_pool2d(
    constant size_t &w_k,
    constant size_t &h_k,
    constant size_t &w_stride,
    constant size_t &h_stride,
    constant size_t *src_dims,
    constant size_t *src_strides,
    device const T *src,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  const size_t c = src_dims[1];
  const size_t w_in = src_dims[2];
  const size_t h_in = src_dims[3];

  const size_t w_out = (w_in - w_k) / w_stride + 1;
  const size_t h_out = (h_in - h_k) / h_stride + 1;
  if (tid >= src_dims[0] * c * w_out * h_out) {
    return;
  }

  const size_t b_idx = tid / (w_out * h_out * c);
  const size_t c_idx = (tid / (w_out * h_out)) % c;
  const size_t dst_w = (tid / h_out) % w_out;
  const size_t dst_h = tid % h_out;

  const size_t src_idx0 = b_idx * src_strides[0];
  T d = 0;
  bool set = false;
  for (size_t w_offset = 0; w_offset < w_k; ++w_offset) {
    size_t src_w = w_stride * dst_w + w_offset;
    if (src_w >= w_in){
      continue;
    }
    for (size_t h_offset = 0; h_offset < h_k; ++h_offset) {
      size_t src_h = h_stride * dst_h + h_offset;
      if (src_h >= h_in) {
        continue;
      }
      const size_t src_idx = src_idx0 + c_idx * src_strides[1] + src_w * src_strides[2] + src_h * src_strides[3];
      if (set) {
        d = MAX(d, src[src_idx]);
      }
      else {
        d = src[src_idx];
        set = true;
      }
    }
  }
  dst[tid] = d;
}


// Naive implementation of conv_transpose1d.
template <typename T, typename A>
[[kernel]] void conv_transpose1d(
    constant size_t &l_out,
    constant size_t &stride,
    constant size_t &padding,
    constant size_t &out_padding,
    constant size_t &dilation,
    constant size_t *src_dims,
    constant size_t *src_strides,
    constant size_t *k_dims,
    constant size_t *k_strides,
    device const T *src,
    device const T *k,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  // src: (b_size, c_in, l_in)
  // kernel: (c_in, c_out, l_k)
  const size_t l_k = k_dims[2];
  const size_t c_out = k_dims[1];
  const size_t c_in = src_dims[1];
  const size_t l_in = src_dims[2];
  if (tid >= src_dims[0] * c_out * l_out) {
    return;
  }

  const size_t b_idx = tid / (l_out * c_out);
  const size_t dst_c_idx = (tid / l_out) % c_out;
  const size_t out_x = tid % l_out;

  const size_t src_idx0 = b_idx * src_strides[0];
  A d = 0;
  for (int k_x = 0; k_x < (int)l_k; ++k_x) {
      // let out_x = inp_x * p.stride + k_x * p.dilation - p.padding;
      int inp_x_stride = (int)(out_x + padding) - k_x * dilation;
      if (inp_x_stride < 0 || inp_x_stride % stride) {
          continue;
      }
      int inp_x = inp_x_stride / stride;
      if (inp_x >= l_in) continue;
      for (size_t src_c_idx = 0; src_c_idx < c_in; ++src_c_idx) {
          const size_t src_idx = src_idx0 + src_c_idx * src_strides[1] + inp_x * src_strides[2];
          const size_t k_idx = src_c_idx * k_strides[0] + dst_c_idx * k_strides[1] + k_x * k_strides[2];
          d += static_cast<A>(src[src_idx]) * static_cast<A>(k[k_idx]);
      }
  }
  dst[tid] = static_cast<T>(d);
}

template <typename T, typename A>
[[kernel]] void conv_transpose2d(
  constant size_t &w_out,
  constant size_t &h_out,
  constant size_t &stride,
  constant size_t &padding,
  constant size_t &out_padding,
  constant size_t &dilation,
  constant size_t *input_dims,
  constant size_t *input_stride,
  constant size_t *k_dims,
  constant size_t *k_stride,
  device const T *src,
  device const T *k,
  device T *dst,
  uint tid [[ thread_position_in_grid ]]
) {
  const size_t h_k = k_dims[2];
  const size_t w_k = k_dims[3];
  const size_t c_out = k_dims[1];
  const size_t c_in = input_dims[1];
  const size_t h_in = input_dims[2];
  const size_t w_in = input_dims[3];

  if (tid >= input_dims[0] * c_out * w_out * h_out) {
    return;
  }

  const size_t b_idx = tid / (w_out * h_out * c_out);
  const size_t dst_c_idx = (tid / (w_out * h_out)) % c_out;
  const size_t out_y = (tid / w_out) % h_out;
  const size_t out_x = tid % w_out;

  const size_t src_idx0 = b_idx * input_stride[0];

  A d = 0;
  for (int k_x = 0; k_x < (int)w_k; ++k_x) {
      const int inp_x_stride = (int)(out_x + padding) - k_x * dilation;
      if (inp_x_stride < 0 || inp_x_stride % stride) {
          continue;
      }
      const int inp_x = inp_x_stride / stride;
      if (inp_x >= w_in) continue;
      for (int k_y = 0; k_y < (int)h_k; ++k_y) {
          const int inp_y_stride = (int)(out_y + padding) - k_y * dilation;
          if (inp_y_stride < 0 || inp_y_stride % stride) {
              continue;
          }
          const int inp_y = inp_y_stride / stride;
          if (inp_y >= h_in) continue;
          for (size_t src_c_idx = 0; src_c_idx < c_in; ++src_c_idx) {
              const size_t src_idx = src_idx0 + src_c_idx * input_stride[1] + inp_y * input_stride[2] + inp_x * input_stride[3];
              const size_t k_idx = src_c_idx * k_stride[0] + dst_c_idx * k_stride[1] + k_y * k_stride[2] + k_x * k_stride[3];
              d += static_cast<A>(src[src_idx]) * static_cast<A>(k[k_idx]);
          }
      }
  }
  dst[tid] = static_cast<T>(d);
}

// Depthwise 1D convolution, fused.
//
// The generic path builds an im2col matrix, runs a matmul, then transposes the
// result into place — three dispatches and two intermediate buffers. For a
// depthwise conv (groups == c_in, so each output channel reads only its own
// input channel) the matmul is degenerate: every output is a short dot product
// of k_size taps. candle also splits grouped convs into one conv per group and
// concatenates, so a 2048-channel depthwise layer becomes thousands of
// dispatches. This does the whole layer in one, writing straight to the final
// (b, c, l_out) layout so no transpose copy is needed.
template <typename T, typename A>
[[kernel]] void conv1d_depthwise(
    constant size_t &dst_numel,
    constant size_t &l_out,
    constant size_t &k_size,
    constant size_t &stride,
    constant size_t &padding,
    constant size_t &dilation,
    constant size_t *src_dims,
    constant size_t *src_strides,
    device const T *src,
    device const T *weight,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  if (tid >= dst_numel) {
    return;
  }

  const size_t c = src_dims[1];
  const size_t l_in = src_dims[2];

  // dst is contiguous (b, c, l_out); recover the index directly.
  const size_t l_idx = tid % l_out;
  const size_t tmp = tid / l_out;
  const size_t c_idx = tmp % c;
  const size_t b_idx = tmp / c;

  // Accumulate in a wider type: f16 sums lose precision quickly, and this
  // costs nothing since the taps are few.
  A acc = static_cast<A>(0);
  const size_t src_base = b_idx * src_strides[0] + c_idx * src_strides[1];
  const size_t w_base = c_idx * k_size;

  for (size_t k = 0; k < k_size; ++k) {
    const size_t pos = l_idx * stride + k * dilation;
    // Positions inside the padding contribute nothing.
    if (pos < padding) {
      continue;
    }
    const size_t src_l = pos - padding;
    if (src_l >= l_in) {
      continue;
    }
    acc += static_cast<A>(src[src_base + src_l * src_strides[2]])
         * static_cast<A>(weight[w_base + k]);
  }

  dst[tid] = static_cast<T>(acc);
}

// Depthwise 1D convolution, specialized for the common contiguous case.
//
// Same computation as `conv1d_depthwise` above, with three things moved from
// runtime to compile time:
//
//   - K_SIZE is a template parameter, so the tap loop has a constant trip count
//     and unrolls. The generic kernel reads k_size from a buffer, so it cannot.
//   - stride and dilation are fixed at 1, so `pos` is `l_idx + k` — one add,
//     no multiplies.
//   - the source is contiguous, so addressing is `(b*c + c_idx)*l_in + src_l`
//     rather than three loads from src_strides[] and three multiplies.
//
// The padding branches stay, but they are now comparisons against compile-time
// K_SIZE-derived values inside an unrolled body, so the compiler hoists what it
// can. They are *not* eliminated: `l_idx` is a runtime value, so which taps fall
// in the padding genuinely varies per thread. See the measurement note in the
// issue-10 write-up — the branches diverge only in the simdgroups that straddle
// an l_out boundary, which for a long prefill is a small minority.
//
// Preconditions the caller must check (see `call_conv1d_depthwise_k`):
// stride == 1, dilation == 1, contiguous source, and K_SIZE matching an
// instantiated variant. Anything else uses the generic kernel above.
template <typename T, typename A, ushort K_SIZE>
[[kernel]] void conv1d_depthwise_k(
    constant size_t &dst_numel,
    constant size_t &l_out,
    constant size_t &padding,
    constant size_t *src_dims,
    device const T *src,
    device const T *weight,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
  if (tid >= dst_numel) {
    return;
  }

  const size_t c = src_dims[1];
  const size_t l_in = src_dims[2];

  const size_t l_idx = tid % l_out;
  const size_t tmp = tid / l_out;
  const size_t c_idx = tmp % c;
  const size_t b_idx = tmp / c;

  // Accumulate in a wider type for the same reason the generic kernel does.
  A acc = static_cast<A>(0);
  // Contiguous (b, c, l_in): the row base is a single multiply-add.
  const size_t src_base = (b_idx * c + c_idx) * l_in;
  const size_t w_base = c_idx * K_SIZE;

#pragma clang loop unroll(full)
  for (ushort k = 0; k < K_SIZE; ++k) {
    // stride == 1 and dilation == 1 are preconditions, so this is just l_idx + k.
    const size_t pos = l_idx + k;
    if (pos < padding) {
      continue;
    }
    const size_t src_l = pos - padding;
    if (src_l >= l_in) {
      continue;
    }
    acc += static_cast<A>(src[src_base + src_l])
         * static_cast<A>(weight[w_base + k]);
  }

  dst[tid] = static_cast<T>(acc);
}

// Explicit instantiation. `decltype(func<...>)` restates the template's own
// signature, so a variant is declared by naming the type arguments and the
// `[[host_name]]` string only — the parameter list is written once, in the
// template. The macro-per-family form this replaces spelled every signature
// twice, and the two could drift silently.
//
// Same spelling as unary.metal, binary.metal and affine.metal, which candle
// already migrated.
#define init_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

// bfloat is gated per family exactly as before this conversion: the depthwise
// kernel on __METAL_VERSION__ >= 310, every other family on __HAVE_BFLOAT__.
// The two guards are not interchangeable, so they are kept as they were rather
// than unified.
#define init_conv1d_depthwise(tname, t, acc) \
    init_kernel("conv1d_depthwise_" #tname, conv1d_depthwise, t, acc)

// The specialized variant carries k_size in its name, per DESIGN.md §7.4's
// `<op>_<dtype>_k<K>` shape. K is a compile-tier axis: it fixes the tap loop's
// trip count, so it changes the generated code and the register allocation.
#define init_conv1d_depthwise_k(tname, t, acc, k) \
    init_kernel("conv1d_depthwise_" #tname "_k" #k, conv1d_depthwise_k, t, acc, k)

#define init_im2col1d(tname, t) \
    init_kernel("im2col1d_" #tname, im2col1d, t)

#define init_im2col(tname, t) \
    init_kernel("im2col_" #tname, im2col, t)

#define init_col2im1d(tname, t) \
    init_kernel("col2im1d_" #tname, col2im1d, t)

#define init_upsample_nearest2d(tname, t) \
    init_kernel("upsample_nearest2d_" #tname, upsample_nearest2d, t)

#define init_upsample_bilinear2d(tname, t) \
    init_kernel("upsample_bilinear2d_" #tname, upsample_bilinear2d, t)

#define init_max_pool2d(tname, t) \
    init_kernel("max_pool2d_" #tname, max_pool2d, t)

#define init_avg_pool2d(tname, t, acc) \
    init_kernel("avg_pool2d_" #tname, avg_pool2d, t, acc)

#define init_conv_transpose1d(tname, t, acc) \
    init_kernel("conv_transpose1d_" #tname, conv_transpose1d, t, acc)

#define init_conv_transpose2d(tname, t, acc) \
    init_kernel("conv_transpose2d_" #tname, conv_transpose2d, t, acc)

init_im2col(f32, float);
init_im2col(f16, half);
init_im2col(u8, uint8_t);
init_im2col(u32, uint32_t);
#if defined(__HAVE_BFLOAT__)
init_im2col(bf16, bfloat);
#endif

init_col2im1d(f32, float);
init_col2im1d(f16, half);
init_col2im1d(u8, uint8_t);
init_col2im1d(u32, uint32_t);
#if defined(__HAVE_BFLOAT__)
init_col2im1d(bf16, bfloat);
#endif

init_conv1d_depthwise(f32, float, float);
init_conv1d_depthwise(f16, half, float);
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 310
init_conv1d_depthwise(bf16, bfloat, float);
#endif

// k = 2, 3, 4 covers the depthwise widths that occur in practice; LFM2 uses 3
// (conv_L_cache). Anything outside the set falls back to the generic kernel
// above, so the list is a performance choice and never a correctness one.
init_conv1d_depthwise_k(f32, float, float, 2);
init_conv1d_depthwise_k(f32, float, float, 3);
init_conv1d_depthwise_k(f32, float, float, 4);
init_conv1d_depthwise_k(f16, half, float, 2);
init_conv1d_depthwise_k(f16, half, float, 3);
init_conv1d_depthwise_k(f16, half, float, 4);
#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 310
init_conv1d_depthwise_k(bf16, bfloat, float, 2);
init_conv1d_depthwise_k(bf16, bfloat, float, 3);
init_conv1d_depthwise_k(bf16, bfloat, float, 4);
#endif

init_im2col1d(f32, float);
init_im2col1d(f16, half);
init_im2col1d(u8, uint8_t);
init_im2col1d(u32, uint32_t);
#if defined(__HAVE_BFLOAT__)
init_im2col1d(bf16, bfloat);
#endif

init_upsample_nearest2d(f32, float);
init_upsample_nearest2d(f16, half);
init_upsample_nearest2d(u8, uint8_t);
init_upsample_nearest2d(u32, uint32_t);
#if defined(__HAVE_BFLOAT__)
init_upsample_nearest2d(bf16, bfloat);
#endif

init_upsample_bilinear2d(f32, float);
init_upsample_bilinear2d(f16, half);
init_upsample_bilinear2d(u8, uint8_t);
init_upsample_bilinear2d(u32, uint32_t);
#if defined(__HAVE_BFLOAT__)
init_upsample_bilinear2d(bf16, bfloat);
#endif

init_max_pool2d(f32, float);
init_max_pool2d(f16, half);
init_max_pool2d(u32, uint32_t);
init_max_pool2d(u8, uint8_t);
#if defined(__HAVE_BFLOAT__)
init_max_pool2d(bf16, bfloat);
#endif

// Accumulator types differ per dtype and are load-bearing: the integer
// instantiations accumulate in their own type, so their averaging truncates
// exactly as it did before. Widening them would be a behaviour change.
init_avg_pool2d(f32, float, float);
init_avg_pool2d(f16, half, float);
init_avg_pool2d(u32, uint32_t, uint32_t);
init_avg_pool2d(u8, uint8_t, uint8_t);
#if defined(__HAVE_BFLOAT__)
init_avg_pool2d(bf16, bfloat, float);
#endif

init_conv_transpose1d(f32, float, float);
init_conv_transpose1d(f16, half, float);
init_conv_transpose1d(u8, uint8_t, uint8_t);
init_conv_transpose1d(u32, uint32_t, uint32_t);
#if defined(__HAVE_BFLOAT__)
init_conv_transpose1d(bf16, bfloat, float);
#endif

init_conv_transpose2d(f32, float, float);
init_conv_transpose2d(f16, half, float);
#if defined(__HAVE_BFLOAT__)
init_conv_transpose2d(bf16, bfloat, float);
#endif
