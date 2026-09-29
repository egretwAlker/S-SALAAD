#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <immintrin.h>
#include <cstdint>
#include <stdexcept>

namespace {

void check_inputs(
    const at::Tensor& out,
    const at::Tensor& x,
    const at::Tensor& crow,
    const at::Tensor& col,
    const at::Tensor& values,
    const at::Tensor& active_rows) {
  TORCH_CHECK(out.device().is_cpu(), "out must be CPU");
  TORCH_CHECK(x.device().is_cpu(), "x must be CPU");
  TORCH_CHECK(crow.device().is_cpu(), "crow must be CPU");
  TORCH_CHECK(col.device().is_cpu(), "col must be CPU");
  TORCH_CHECK(values.device().is_cpu(), "values must be CPU");
  TORCH_CHECK(active_rows.device().is_cpu(), "active_rows must be CPU");

  TORCH_CHECK(out.dtype() == at::kFloat, "out must be float32");
  TORCH_CHECK(x.dtype() == at::kFloat, "x must be float32");
  TORCH_CHECK(values.dtype() == at::kFloat, "values must be float32");
  TORCH_CHECK(crow.dtype() == at::kInt, "crow must be int32");
  TORCH_CHECK(col.dtype() == at::kInt, "col must be int32");
  TORCH_CHECK(active_rows.dtype() == at::kInt, "active_rows must be int32");

  TORCH_CHECK(out.dim() == 2, "out must be 2D");
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(crow.dim() == 1, "crow must be 1D");
  TORCH_CHECK(col.dim() == 1, "col must be 1D");
  TORCH_CHECK(values.dim() == 3, "values must be 3D: (nnz, block_q, block_p)");
  TORCH_CHECK(active_rows.dim() == 1, "active_rows must be 1D");

  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(crow.is_contiguous(), "crow must be contiguous");
  TORCH_CHECK(col.is_contiguous(), "col must be contiguous");
  TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
  TORCH_CHECK(active_rows.is_contiguous(), "active_rows must be contiguous");

  const auto block_p = values.size(2);
  const auto block_q = values.size(1);
  TORCH_CHECK(block_p == 8 || block_p == 16 || block_p == 32 || block_p == 64,
              "AVX2 BSR CPU kernel supports block_p 8, 16, 32 or 64, got ", block_p);
  TORCH_CHECK(block_q == block_p, "AVX2 BSR CPU kernel requires square blocks, got ",
              block_p, "x", block_q);
  TORCH_CHECK(out.size(1) == (crow.size(0) - 1) * block_p, "out N does not match crow/block_p");
  TORCH_CHECK(x.size(1) % block_q == 0, "x K must be divisible by block_q");
  TORCH_CHECK(values.size(0) == col.size(0), "values nnz must match col nnz");
}

// One ymm register holds 8 float32 lanes.
constexpr int64_t kLanes = 8;
constexpr int64_t kGrain = 8;

// Single kernel for every supported block size.
//
// ROW_TILE is not a free parameter. VEC = BLOCK/8 ymm registers hold one block
// row, so ROW_TILE * VEC accumulators are live across a block-row; AVX2 has 16
// architectural ymm registers and this pins that product at 8, giving
// 8x8 -> 8, 16x16 -> 4, 32x32 -> 2, 64x64 -> 1. The static_assert enforces it.
template <int64_t BLOCK, int64_t ROW_TILE>
inline void run_block(
    float* __restrict__ out,
    const float* __restrict__ x,
    const int32_t* __restrict__ crow,
    const int32_t* __restrict__ col,
    const float* __restrict__ values,
    const int32_t* __restrict__ active_rows,
    int64_t M,
    int64_t K,
    int64_t NB,
    int64_t ACTIVE_NB) {
  constexpr int64_t VEC = BLOCK / kLanes;
  static_assert(BLOCK % kLanes == 0, "BLOCK must be a multiple of 8 lanes");
  static_assert(ROW_TILE * VEC == 8, "keep 8 live ymm accumulators per block-row");

  at::parallel_for(0, M, kGrain, [&](int64_t m_begin, int64_t m_end) {
    for (int64_t m0 = m_begin; m0 < m_end; m0 += ROW_TILE) {
      const int64_t rows = (m0 + ROW_TILE <= m_end) ? ROW_TILE : (m_end - m0);
      for (int64_t ai = 0; ai < ACTIVE_NB; ++ai) {
        const int64_t by = static_cast<int64_t>(active_rows[ai]);
        const int32_t row_start = crow[by];
        const int32_t row_end   = crow[by + 1];

        __m256 acc[ROW_TILE][VEC];
        for (int64_t r = 0; r < rows; ++r) {
          float* out_ptr = out + (m0 + r) * NB * BLOCK + by * BLOCK;
          for (int64_t c = 0; c < VEC; ++c) {
            acc[r][c] = _mm256_loadu_ps(out_ptr + c * kLanes);
          }
        }

        for (int32_t ji = row_start; ji < row_end; ++ji) {
          const int64_t x_base = static_cast<int64_t>(col[ji]) * BLOCK;
          const float* v = values + static_cast<int64_t>(ji) * BLOCK * BLOCK;
          for (int64_t k = 0; k < BLOCK; ++k) {
            const float* v_row = v + k * BLOCK;
            if constexpr (ROW_TILE > 1) {
              // Hoist the weight row: its VEC loads are amortized over
              // ROW_TILE rows of x. Live set = 8 accumulators + VEC weights
              // + 1 broadcast, i.e. at most 13 ymm.
              __m256 w[VEC];
              for (int64_t c = 0; c < VEC; ++c) {
                w[c] = _mm256_loadu_ps(v_row + c * kLanes);
              }
              for (int64_t r = 0; r < rows; ++r) {
                const __m256 xv = _mm256_broadcast_ss(x + (m0 + r) * K + x_base + k);
                for (int64_t c = 0; c < VEC; ++c) {
                  acc[r][c] = _mm256_fmadd_ps(xv, w[c], acc[r][c]);
                }
              }
            } else {
              // ROW_TILE == 1 (BLOCK == 64): hoisting would need 17 ymm, one
              // past the register file, so the compiler would spill. With a
              // single row there is nothing to amortize the load over.
              const __m256 xv = _mm256_broadcast_ss(x + m0 * K + x_base + k);
              for (int64_t c = 0; c < VEC; ++c) {
                acc[0][c] = _mm256_fmadd_ps(
                    xv, _mm256_loadu_ps(v_row + c * kLanes), acc[0][c]);
              }
            }
          }
        }

        for (int64_t r = 0; r < rows; ++r) {
          float* out_ptr = out + (m0 + r) * NB * BLOCK + by * BLOCK;
          for (int64_t c = 0; c < VEC; ++c) {
            _mm256_storeu_ps(out_ptr + c * kLanes, acc[r][c]);
          }
        }
      }
    }
  });
}

}  // namespace

void bsr_addmm_avx2_(
    at::Tensor out,
    at::Tensor x,
    at::Tensor crow,
    at::Tensor col,
    at::Tensor values,
    at::Tensor active_rows) {
  check_inputs(out, x, crow, col, values, active_rows);

  const int64_t M        = out.size(0);
  const int64_t K        = x.size(1);
  const int64_t NB       = crow.size(0) - 1;
  const int64_t ACTIVE_NB = active_rows.size(0);
  const int64_t block_p  = values.size(2);

  float*         out_ptr        = out.data_ptr<float>();
  const float*   x_ptr          = x.data_ptr<float>();
  const int32_t* crow_ptr       = crow.data_ptr<int32_t>();
  const int32_t* col_ptr        = col.data_ptr<int32_t>();
  const float*   values_ptr     = values.data_ptr<float>();
  const int32_t* active_rows_ptr = active_rows.data_ptr<int32_t>();

  // ROW_TILE is fixed by the register budget, not chosen — see run_block.
  switch (block_p) {
    case 8:  run_block<8,  8>(out_ptr, x_ptr, crow_ptr, col_ptr, values_ptr, active_rows_ptr, M, K, NB, ACTIVE_NB); break;
    case 16: run_block<16, 4>(out_ptr, x_ptr, crow_ptr, col_ptr, values_ptr, active_rows_ptr, M, K, NB, ACTIVE_NB); break;
    case 32: run_block<32, 2>(out_ptr, x_ptr, crow_ptr, col_ptr, values_ptr, active_rows_ptr, M, K, NB, ACTIVE_NB); break;
    case 64: run_block<64, 1>(out_ptr, x_ptr, crow_ptr, col_ptr, values_ptr, active_rows_ptr, M, K, NB, ACTIVE_NB); break;
    default: TORCH_CHECK(false, "unsupported block_p ", block_p);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bsr_addmm_avx2_", &bsr_addmm_avx2_, "In-place float32 BSR addmm AVX2/FMA");
}
