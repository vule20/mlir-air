//===- mm_engine.cc - GEMM engine kernels: bfp16 GEMM + fused epilogues -*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
//
// mm_bfp16.cc plus the epilogues gemm_engine.py folds into its GEMM jobs, so
// one object serves every phase of the engine herd (AIR links one object per
// core):
//   add_residual_blocked   acc += a residual tile that arrived on the A channel
//   zero_rows / sumsq_rows_blocked   per-row sum of squares of the A chunks
//                                    (RMSNorm statistics of the GEMM input)
//   rows_rstd              ss -> rsqrt(ss/RMS_K + eps)
//   f32_to_bf16_rms_swiglu drain: row scale by rstd, then SwiGLU, into one half
//                          of the output tile
//
// Layouts: acc (DIM_N/8, DIM_M/8, 8, 8) N-outer; A tile (DIM_M/8, DIM_K/8, 8, 8);
// every 8x8 block is row-major (m row, n or k column).
//
//===----------------------------------------------------------------------===//

#include "mm_bfp16.cc"

#ifndef RMS_K
#define RMS_K 960
#endif
#ifndef RMS_EPS
#define RMS_EPS 1e-5f
#endif

extern "C" {

// Columns [half * DIM_N, half * DIM_N + DIM_N) of a DIM_M x DIM_K A tile, added
// in f32 to the accumulator.
void SYM(add_residual_blocked)(float *acc, bfloat16 *a, int32_t half) {
  constexpr unsigned T = 8, NB = DIM_N / T, MB = DIM_M / T, KB = DIM_K / T;
  constexpr unsigned BE = T * T, VW = 16;
  static_assert(DIM_K == 2 * DIM_N, "one A chunk holds two output tiles");
  const aie::vector<bfloat16, VW> one_v =
      aie::broadcast<bfloat16, VW>((bfloat16)1.0f);
  for (unsigned nb = 0; nb < NB; nb++) {
    for (unsigned mb = 0; mb < MB; mb++) {
      float *pc = acc + (nb * MB + mb) * BE;
      const bfloat16 *pa = a + (mb * KB + half * NB + nb) * BE;
      for (unsigned e = 0; e < BE; e += VW) {
        aie::accum<accfloat, VW> c(aie::load_v<VW>(pc + e));
        c = aie::mac(c, aie::load_v<VW>(pa + e), one_v);
        aie::store_v(pc + e, c.template to_vector<float>());
      }
    }
  }
}

void SYM(zero_rows)(float *ss) {
  for (unsigned i = 0; i < DIM_M; i++)
    ss[i] = 0.0f;
}

void SYM(sumsq_rows_blocked)(bfloat16 *a, float *ss) {
  constexpr unsigned T = 8, MB = DIM_M / T, KB = DIM_K / T, BE = T * T;
  constexpr unsigned VW = 16;
  for (unsigned mb = 0; mb < MB; mb++) {
    for (unsigned rp = 0; rp < T / 2; rp++) {
      aie::accum<accfloat, VW> s = aie::zeros<accfloat, VW>();
      for (unsigned kb = 0; kb < KB; kb++) {
        aie::vector<bfloat16, VW> v =
            aie::load_v<VW>(a + (mb * KB + kb) * BE + rp * VW);
        s = aie::mac(s, v, v);
      }
      aie::vector<float, VW> f = s.template to_vector<float>();
      float lo = 0.0f, hi = 0.0f;
      for (unsigned i = 0; i < T; i++) {
        lo += f[i];
        hi += f[i + T];
      }
      ss[mb * T + 2 * rp] += lo;
      ss[mb * T + 2 * rp + 1] += hi;
    }
  }
}

// Sum of squares -> rstd = rsqrt(ss / RMS_K + eps), in place.
void SYM(rows_rstd)(float *ss) {
  for (unsigned r = 0; r < DIM_M; r++)
    ss[r] = aie::invsqrt(ss[r] * (1.0f / RMS_K) + RMS_EPS);
}

// g and u are narrowed to bf16, scaled by the per-row rstd, then silu_and_mul's
// math, as f32_to_bf16_swiglu_mn.
void SYM(f32_to_bf16_rms_swiglu)(float *src, float *ss, bfloat16 *dst,
                                 int32_t half) {
  constexpr unsigned VW = 16, T = 8;
  constexpr unsigned NB = DIM_N / T, H = NB / 2, BE = DIM_M * T;
  static_assert(NB % 2 == 0, "tile_n must hold whole gate/up block pairs");
  const aie::vector<bfloat16, VW> half_v =
      aie::broadcast<bfloat16, VW>((bfloat16)0.5f);
  const aie::vector<bfloat16, VW> one_v =
      aie::broadcast<bfloat16, VW>((bfloat16)1.0f);
  const aie::mask<VW> hi_rows = aie::mask<VW>::from_uint32(0xFF00u);
  for (unsigned jb = 0; jb < H; jb++) {
    const float *pg = src + jb * BE;
    const float *pu = src + (jb + H) * BE;
    bfloat16 *pd = dst + (half * H + jb) * BE;
    for (unsigned e = 0; e < BE; e += VW) {
      // e = mb * 64 + r * 8 + t: lanes 0-7 are row mb*8+r, 8-15 the next row.
      const unsigned row = e / T;
      const aie::vector<bfloat16, VW> sv =
          aie::select(aie::broadcast<bfloat16, VW>((bfloat16)ss[row]),
                      aie::broadcast<bfloat16, VW>((bfloat16)ss[row + 1]),
                      hi_rows);
      ::aie::set_rounding(aie::rounding_mode::conv_even);
      aie::vector<bfloat16, VW> g =
          narrow_f32_to_bf16<VW>(aie::load_v<VW>(pg + e));
      aie::vector<bfloat16, VW> u =
          narrow_f32_to_bf16<VW>(aie::load_v<VW>(pu + e));
      g = aie::mul(g, sv).template to_vector<bfloat16>();
      u = aie::mul(u, sv).template to_vector<bfloat16>();
      ::aie::set_rounding(aie::rounding_mode::floor);
      aie::vector<bfloat16, VW> g_half = aie::mul(g, half_v);
      aie::accum<accfloat, VW> tanh_in;
      tanh_in.from_vector(g_half);
      aie::vector<bfloat16, VW> tanh_val =
          aie::tanh<bfloat16>(tanh_in.template to_vector<float>());
      aie::vector<bfloat16, VW> sigmoid =
          aie::mul(half_v, aie::add(one_v, tanh_val));
      aie::vector<bfloat16, VW> silu = aie::mul(g, sigmoid);
      aie::vector<bfloat16, VW> out = aie::mul(silu, u);
      aie::store_v(pd + e, out);
    }
  }
}

} // extern "C"
