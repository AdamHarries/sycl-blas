/***************************************************************************
 *  @license
 *  Copyright (C) Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCL-BLAS: BLAS implementation using SYCL
 *
 *  @filename blas3_trees_gemm.hpp
 *
 **************************************************************************/

#ifndef BLAS3_TSGEMM_HPP
#define BLAS3_TSGEMM_HPP

#include <CL/sycl.hpp>

#include <sstream>

#define dbg(expr)                            \
  [&]() -> decltype(expr) {                  \
    auto val = expr;                         \
    std::cerr << #expr " = " << val << ", "; \
    return val;                              \
  }()

#define dbg_str(expr)           \
  [&]() -> std::string {        \
    std::stringstream sstr;     \
    auto val = expr;            \
    sstr << #expr " = " << val; \
    return sstr.str();          \
  }()

template <typename IndexType, IndexType NumTiles = 8,
          IndexType TileSizeDimM = 8, IndexType TileSizeDimK = 16,
          IndexType TileSizeDimN = 16, IndexType WorkPerThreadM = 16,
          IndexType WorkPerThreadN = 1, IndexType LocalThreadSizeN = 4,
          IndexType LocalThreadSizeM = 4>
struct TSGEMMTile {
  /* The number of tiles to be processed */
  static constexpr IndexType num_tiles = NumTiles;
  /* tile size, dimension M */
  static constexpr IndexType tile_size_dim_m = TileSizeDimM;
  /* tile size, dimension N */
  static constexpr IndexType tile_size_dim_n = TileSizeDimN;
  /* tile size, dimension K */
  static constexpr IndexType tile_size_dim_k = TileSizeDimK;

  /* workload per thread M */
  static constexpr IndexType work_per_thread_m = WorkPerThreadM;
  /* workload per thread N */
  static constexpr IndexType work_per_thread_n = WorkPerThreadN;

  /* local thread size n */
  static constexpr IndexType local_thread_size_n = LocalThreadSizeN;
  /* local thread size m */
  static constexpr IndexType local_thread_size_m = LocalThreadSizeM;

  /*!
   * @brief Get tile type as human readable string.
   */
  // static inline std::string get_type_string() noexcept {
  //   return std::string("Tile<") + std::to_string(item_rows) + ", " +
  //          std::to_string(item_cols) + ", " + std::to_string(wg_rows) + ", "
  //          + std::to_string(wg_cols) + ", " + std::to_string(tl_rows) + ", "
  //          + std::to_string(tl_cols) + ">";
  // }
};

// So far, more or less just copied from eigen.
// template <typename OutScalar, typename LhsScalar, typename RhsScalar,
// typename OutAccessor, typename TempAcc, typename LhsMapper,
// typename RhsMapper, typename Scratch, typename Index,
// typename PanelParameters, bool Vectorizable, bool NoEdge, boolIsFinal>
template <typename RHS0, typename RHS1, typename ScratchT, typename LocalT,
          typename T, int WgSize, bool TransA, bool TransB, typename tile_type>
class TallSkinnyGemmFactory {
 public:
  using ValueType = T;
  // using IndexType = typename std::make_signed<typename
  // RHS0::IndexType>::type;
  using IndexType = int;

  RHS0 _A;
  RHS0 _B;
  RHS1 _C;

  ValueType alpha;
  ValueType beta;

  ScratchT scratch;
  LocalT temp_res;

  // OutAccessor out_res;
  // OutAccessor temp_res;
  // const LhsMapper lhs;
  // const RhsMapper rhs;
  // Scratch scratch;

  /* Size of dimension M */
  const IndexType M;
  /* Size of dimension N */
  const IndexType N;
  /* Size of dimension K */
  const IndexType K;

  static constexpr IndexType local_thread_size_n =
      tile_type::local_thread_size_n;
  static constexpr IndexType local_thread_size_m =
      tile_type::local_thread_size_m;

  /* The number of tiles to be processed */
  static constexpr IndexType num_tiles = tile_type::num_tiles;
  /* tile size, dimension M */
  static constexpr IndexType tile_size_dim_m = tile_type::tile_size_dim_m;
  /* tile size, dimension N */
  static constexpr IndexType tile_size_dim_n = tile_type::tile_size_dim_n;
  /* tile size, dimension K */
  static constexpr IndexType tile_size_dim_k = tile_type::tile_size_dim_k;

  /* workload per thread M */
  static constexpr IndexType work_per_thread_m = tile_type::work_per_thread_m;
  /* workload per thread N */
  static constexpr IndexType work_per_thread_n = tile_type::work_per_thread_n;

  /* the number of groups, dimension M */
  const IndexType group_count_m;
  /* the number of groups, dimension n */
  const IndexType group_count_n;
  /* the number of groups, dimension k */
  const IndexType group__k;

  TallSkinnyGemmFactory(RHS0 A, RHS0 B, RHS1 C, IndexType M, IndexType N,
                        IndexType K, T alpha, T beta, ScratchT scratch,
                        const IndexType group_count_m,
                        const IndexType group_count_n, const IndexType group__k)
      : _A(A),
        _B(B),
        _C(C),
        M(M),
        N(N),
        K(K),
        alpha(alpha),
        beta(beta),
        scratch(scratch),
        group_count_m(group_count_m),
        group_count_n(group_count_n),
        group__k(group__k) {}

#ifdef TESTING_GUARD
  inline void eval(IndexType lcl_tid, IndexType wg_id) noexcept {
#else
  inline void eval(cl::sycl::nd_item<1> id) noexcept {
#endif

/* references to the matrices */
#ifdef TESTING_GUARD
      // Use this to see if the matrices are std::vectors
      auto A = _A.data();
  auto B = _B.data();
  auto C = _C.data();
  auto tmp_ptr = temp_res.data();
  auto scratch_ptr = scratch.data();
#else
    auto A = _A.getData().get_pointer().get() + _A.getDisp();
    auto B = _B.getData().get_pointer().get() + _B.getDisp();
    auto C = _C.getData().get_pointer().get() + _C.getDisp();
    /* references to the temporary memory, scratch memory, and rhs scratch
     * memory*/
    auto tmp_ptr = temp_res.get_pointer().get();
    auto scratch_ptr = scratch.get_pointer().get();
#endif

  auto rhs_scratch_ptr = scratch_ptr + (2 * tile_size_dim_m * tile_size_dim_k);

  T private_res[work_per_thread_m * work_per_thread_n];

/* workgroup id */
#ifndef TESTING_GUARD
  const IndexType wg_id = id.get_group(0);
#endif
/* Local thread id */
#ifndef TESTING_GUARD
  const IndexType lcl_tid = id.get_local_id();
#endif

  std::cerr << "-----------------------------------------" << std::endl;
  std::cerr << "Workgroup id: " << wg_id << ", Local id: " << lcl_tid
            << std::endl;
  std::cerr << "-----------------------------------------" << std::endl;

  /* Local ID Column */
  const IndexType n_lcl_tid = lcl_tid / local_thread_size_m;

  /* Local ID row */
  const IndexType m_lcl_tid = lcl_tid - (n_lcl_tid * local_thread_size_m);

  /* workgroup id, local column */
  const IndexType tmp = wg_id / group_count_m;

  /* Workgroup id row */
  const IndexType mgroup_id = wg_id % group_count_m;
  // wg_id - (tmp * group_count_m);  // isn't this always 0???

  const IndexType kgroup_id = (wg_id / group_count_m) / group_count_n;

  const IndexType ngroup_id = wg_id % group_count_n;

  // tmp - (kgroup_id * group_count_n);

  /* register offsets */
  const IndexType global_mix_offset = mgroup_id * tile_size_dim_m;
  const IndexType global_nix_offset = ngroup_id * tile_size_dim_n;
  const IndexType global_kix_offset = kgroup_id * tile_size_dim_k;

  /* initialise the private res summation registers */
  for (auto i = 0; i < work_per_thread_m * work_per_thread_n; i++) {
    private_res[i] = static_cast<T>(0);
  }

  /* Load tiles, LHS and RHS */

  // Tile LHS
  // for now, assume that the LHS isn't transposed.
  load_tile(A, scratch_ptr, lcl_tid, global_mix_offset, global_kix_offset, 0,
            work_per_thread_m, M, K);
  // Tile RHS
  load_and_transpose_tile(B, rhs_scratch_ptr, lcl_tid, global_nix_offset,
                          global_kix_offset, 0, work_per_thread_n, K, N);

#ifndef TESTING_GUARD
  id.barrier(cl::sycl::access::fence_space::local_space);
#endif

  const IndexType start_lhs_index = m_lcl_tid;
  const IndexType start_rhs_index = n_lcl_tid;
  IndexType firstHalf = 0;
  /* Loop over all tiles allocated to this particular workgroup size */
  do {
// Synchronise
#ifndef TESTING_GUARD
    id.barrier(cl::sycl::access::fence_space::local_space);
#endif
    IndexType next_half = firstHalf + 1;
    // If we need ot swap, or not?
    if (next_half < num_tiles) {
      /* Load tiles, LHS and RHS into local memory */
      // Tile LHS
      load_tile(A, scratch_ptr, lcl_tid, global_mix_offset, global_kix_offset,
                next_half, work_per_thread_m, M, K);
      // Tile RHS
      load_and_transpose_tile(B, rhs_scratch_ptr, lcl_tid, global_nix_offset,
                              global_kix_offset, next_half, work_per_thread_n,
                              K, N);
    }
    // Calculate offsets into the temporary memory.

    IndexType lhs_offset =
        ((firstHalf & 1) * (tile_size_dim_m * tile_size_dim_k)) +
        start_lhs_index;
    IndexType rhs_offset =
        ((firstHalf & 1) * (tile_size_dim_k * tile_size_dim_n)) +
        start_rhs_index;

    /* Loop over the values of a single tile */
    for (IndexType k = 0; k < tile_size_dim_k; k++) {
      auto idx = 0;
      auto rhs_index = 0;
      for (IndexType wLPTN = 0; wLPTN < work_per_thread_n; wLPTN++) {
        // load a RHS element from the scratch buffer
        ValueType privateRhs = rhs_scratch_ptr[rhs_index + rhs_offset];

        IndexType lhs_index = 0;
        for (IndexType wLPTM = 0; wLPTM < work_per_thread_m; wLPTM++) {
          // load a LHS element from the scratch buffer
          ValueType privateLhs = scratch_ptr[lhs_index + lhs_offset];

          // Perform a manual MAD.
          private_res[wLPTM + idx] = lcl_tid;
          // private_res[wLPTM + idx] + (privateLhs * privateRhs);

          lhs_index += work_per_thread_m;
        }
        idx += work_per_thread_m;
        rhs_index += work_per_thread_n;
      }
      lhs_offset += tile_size_dim_m;
      rhs_offset += tile_size_dim_n;
    }
    // Next tile
    firstHalf++;
  } while (firstHalf < num_tiles);

#ifndef TESTING_GUARD
  id.barrier(cl::sycl::access::fence_space::local_space);
#endif

  // Store the final results in C
  IndexType global_col_offset = (ngroup_id * tile_size_dim_n) + (n_lcl_tid);
  IndexType global_row_offset = (mgroup_id * tile_size_dim_m) + (m_lcl_tid);
  IndexType global_k_offset = kgroup_id * M * N;
  IndexType c_index = global_col_offset * M;
  IndexType private_index_offset = 0;

  for (IndexType wLPTN = 0; wLPTN < work_per_thread_n; wLPTN++) {
    IndexType private_index = private_index_offset;

    // Disregard anything involving `i` - it simply specifies a stride
    // for (IndexType i = 0; i < PacketSize; i++) {
    IndexType global_col = global_col_offset;  // + i;
    IndexType global_row = global_row_offset;
    for (IndexType wLPTM = 0; wLPTM < work_per_thread_m; wLPTM++) {
      if (/*(NoEdge) ||*/ (global_row < M && global_col < N)) {
        // Store the final results in C
        C[c_index + global_row + global_k_offset] =
            private_res[wLPTM + private_index];
      }
    }
    c_index += M;
    private_index += (work_per_thread_m);
    //}
    global_col_offset += local_thread_size_n;
    c_index = global_col_offset * M;
    private_index_offset += work_per_thread_m;
  }
}
// We need two load functions: one that loads normally, one that
// loads + transposes on load.

// Load a "left hand" tile, or "right hand transposed" tile
// What is NoEdge for??
template <typename GlobalPointerType, typename LocalPointerType>
static inline void load_tile(GlobalPointerType glb_ptr,
                             LocalPointerType lcl_ptr,
                             IndexType linear_local_thread_id,
                             IndexType global_m_offset,
                             IndexType global_k_offset, IndexType next_half,
                             IndexType load_per_thread_lhs, IndexType M,
                             IndexType K) {
  std::cerr << std::endl << "Load Tile" << std::endl;
  // Our rhs linear id is the same as our thread id to start with
  IndexType local_lhs_linear_id = dbg(linear_local_thread_id);
  // Local id offset depends on whether we're on the first or second "half" of
  // the scratch memory. If we're on the first half (i.e. the lowest bit is
  // set to 0), then the offset is 0. If we're on the second half (i.e. the
  // lowest bit is set to 1), then the offset is the linear size of a RHS
  // tile: tile_size_dim_m * tile_size_dim_k
  IndexType linear_local_id_offset =
      (next_half & 1) * (tile_size_dim_m * tile_size_dim_k);

  for (IndexType lPTL = 0; lPTL < load_per_thread_lhs; lPTL++) {
    IndexType local_thread_k = local_lhs_linear_id / tile_size_dim_m;
    IndexType local_thread_m =
        // dbg(local_lhs_linear_id % local_thread_k);
        local_lhs_linear_id - (local_thread_k * tile_size_dim_m);

    IndexType global_m_index = global_m_offset + local_thread_m;
    IndexType global_k_index = global_k_offset + local_thread_k;
    IndexType linear_local_id_index =
        local_thread_m + (local_thread_k * tile_size_dim_m);

    // We can ignore this check, as we're not using packet types right now
    // if (/*(NoEdge) ||*/ ((global_m_index < M) && (global_k_index < K))) {
    // load from matrix according to global_m_index and
    // global_k_index

    std::cerr << "glb_ptr[" << dbg_str(global_m_index + (global_k_index * M))
              << "]\n";

    T val = glb_ptr[global_m_index + (global_k_index * M)];
    std::cerr << "Val: " << val << std::endl;

    dbg(linear_local_id_index);
    dbg(linear_local_id_offset);
    std::cerr << std::endl;

    std::cerr << "lcl_ptr["
              << dbg_str(linear_local_id_index + linear_local_id_offset)
              << "]\n";

    // check we're not writing beyond the range of the tile
    lcl_ptr[linear_local_id_index + linear_local_id_offset] = val;
    // }

    local_lhs_linear_id += (local_thread_size_n * local_thread_size_m);
  }
}

template <typename GlobalPointerType, typename LocalPointerType>
static inline void load_and_transpose_tile(
    GlobalPointerType glb_ptr, LocalPointerType lcl_ptr,
    IndexType linear_local_thread_id, IndexType global_n_offset,
    IndexType global_k_offset, IndexType next_half,
    IndexType load_per_thread_rhs, IndexType K, IndexType N) {
  // std::cerr << "Load Tile and Transpose" << std::endl;
  // Our rhs linear id is the same as our thread id to start with
  IndexType local_rhs_linear_id = linear_local_thread_id;
  // Local id offset depends on whether we're on the first or second "half" of
  // the scratch memory. If we're on the first half (i.e. the lowest bit is
  // set to 0), then the offset is 0. If we're on the second half (i.e. the
  // lowest bit is set to 1), then the offset is the linear size of a RHS
  // tile: tile_size_dim_k * tile_size_dim_n
  IndexType linear_local_id_offset =
      (next_half & 1) * (tile_size_dim_k * tile_size_dim_n);
  for (IndexType lPTR = 0; lPTR < load_per_thread_rhs; lPTR++) {
    // Calculate the index in the 'n' dimension that this thread should access
    IndexType local_thread_n = local_rhs_linear_id / tile_size_dim_k;
    // Calculate the index in the 'k' dimension that this thread should access
    IndexType local_thread_k =
        local_rhs_linear_id - (tile_size_dim_k * local_thread_n);

    IndexType global_k_index = global_k_offset + local_thread_k;
    IndexType global_n_index = global_n_offset + local_thread_n;

    // Transpose RHS on the fly
    // IndexType linear_local_id_index =
    //     local_thread_n + (local_thread_k * tile_size_dim_n);
    IndexType linear_local_id_index =
        local_thread_k + (local_thread_n * tile_size_dim_k);

    std::cout << std::endl;

    auto val = T(0);
    if (/*(NoEdge) ||*/ ((global_k_index < K) && (global_n_index < N))) {
      val = glb_ptr[global_n_index + (global_k_index * N)];
    }

    lcl_ptr[linear_local_id_index + linear_local_id_offset] = val;
    local_rhs_linear_id += local_thread_size_n * local_thread_size_m;
  }
}
}
;

#endif  // BLAS3_TSGEMM_HPP