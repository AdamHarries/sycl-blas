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

/*!
 *
 *
 *
 *
 *
 */

// So far, more or less just copied from eigen.
// template <typename OutScalar, typename LhsScalar, typename RhsScalar,
// typename OutAccessor, typename TempAcc, typename LhsMapper,
// typename RhsMapper, typename Scratch, typename Index,
// typename PanelParameters, bool Vectorizable, bool NoEdge, boolIsFinal>
template <typename RHS0, typename RHS1, int WgSize, bool TransA, bool TransB,
          typename tile_type, typename T>
class TallSkinnyGemmFactory {
 public:
  using value_type = T;
  using IndexType = typename std::make_signed<typename RHS0::IndexType>::type;

  RHS0 _A;
  RHS0 _B;
  RHS1 _C;

  value_type alpha;
  value_type beta;

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

  /* the size of the group, dimension M */
  const IndexType group_size_m;
  /* the size of the group, dimension n */
  const IndexType group_size_n;
  /* the size of the group, dimension k */
  const IndexType group_size_k;

  TallSkinnyGemmFactory(RHS0 A, RHS0 B, RHS1 C, T alpha, T beta,
                        const Index group_size_m)
      : _A(A), _B(B), _C(C), alpha(alpha), beta(beta) {}

  static inline void eval(cl::sycl::nd_item<1> id) noexcept {
    /* references to the matrices */
    auto A = _A.getData().get_pointer().get() + _A.getDisp();
    auto B = _B.getData().get_pointer().get() + _B.getDisp();
    auto C = _C.getData().get_pointer().get() + _C.getDisp();

    /* references to the temporary memory, scratch memory, and rhs scratch
     * memory*/
    auto tmp_ptr = temp_res.get_pointer().get();
    auto scratch_ptr = scratch.get_pointer().get();
    auto rhs_scratch_ptr =
        scratch_ptr + (2 * tile_size_tim_m * tile_size_dim_k);

    T private_res[work_per_thread_m * work_per_thread_n];

    /* Local thread id */
    const IndexType lcl_tid = id.get_local_id();

    /* Local ID Column */
    const IndexType n_lcl_tid = lcl_tid / local_thread_size_n;

    /* Local ID row */
    const IndexType m_lcl_tid = lcl_tid - (n_lcl_tid * local_thread_size_m);

    /* workgroup id */
    const IndexType wg_id = id.get_group(0);
    /* workgroup id, local column */
    const IndexType = wg_id / group_size_m;

    /* Workgroup id row */
    const IndexType mgroup_id =
        wg_id - (tmp * group_size_m);  // isn't this always 0???

    const IndexType kgroup_id = tmp / group_size_n;

    const IndexType ngroup_id = tmp - (kgroup_id * group_size_n);

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
    // Lhs<is_lhs_transposed>::template LoadLocalTile<
    //     PacketReturnType, LhsScalar, Index,
    //     PannelParameters::LoadPerThreadLhs, PacketSize,
    //     PannelParameters::LocalThreadSizeM,
    //     PannelParameters::LocalThreadSizeN, PannelParameters::TileSizeDimM,
    //     PannelParameters::TileSizeDimK, Vectorizable, NoEdge>(
    //     lhs, scratch_ptr, linearLocalThreadId, GlobalMIndex_offset,
    //     GlobalKIndex_offset, 0, M, K);
    // // Tile RHS
    // Rhs<is_rhs_transposed>::template LoadLocalTile<
    //     PacketReturnType, RhsScalar, Index,
    //     PannelParameters::LoadPerThreadRhs, PacketSize,
    //     PannelParameters::LocalThreadSizeM,
    //     PannelParameters::LocalThreadSizeN, tile_size_dim_n,
    //     PannelParameters::TileSizeDimK, Vectorizable, NoEdge>(
    //     rhs, rhs_scratch_ptr, linearLocalThreadId, GlobalNIndex_offset,
    //     GlobalKIndex_offset, 0, K, N);

    id.barrier(cl::sycl::access::fence_space::local_space);

    const IndexType start_lhs_index = m_lcl_tid;
    const IndexType start_rhs_index = n_lcl_tid;
    IndexType firstHalf = 0;
    /* Loop over all tiles allocated to this particular workgroup size */
    do {
      // Synchronise
      id.barrier(cl::sycl::access::fence_space::local_space);
      IndexType nextHalf = firstHalf + 1;
      // If we need ot swap, or not?
      if (nextHalf < numTiles) {
        /* Load tiles, LHS and RHS into local memory */
        // Tile LHS
        // Lhs<is_lhs_transposed>::template LoadLocalTile<
        //     PacketReturnType, LhsScalar, Index, LoadPerThreadLhs, PacketSize,
        //     LocalThreadSizeM, LocalThreadSizeN, TileSizeDimM, TileSizeDimK,
        //     Vectorizable, NoEdge>(lhs, scratch_ptr, linearLocalThreadId,
        //                           GlobalMIndex_offset, TileSizeDimK *
        //                           nextHalf, nextHalf, M, K);
        // Tile RHS
        // Rhs<is_rhs_transposed>::template LoadLocalTile<
        //     PacketReturnType, RhsScalar, Index, LoadPerThreadRhs, PacketSize,
        //     LocalThreadSizeM, LocalThreadSizeN, TileSizeDimN, TileSizeDimK,
        //     Vectorizable, NoEdge>(rhs, rhs_scratch_ptr, linearLocalThreadId,
        //                           GlobalNIndex_offset, TileSizeDimK *
        //                           nextHalf, nextHalf, K, N);
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
        for (Index wLPTN = 0; wLPTN < workload_per_thread_n; wLPTN++) {
          // load a RHS element from the scratch buffer
          ValueType privateRhs = rhs_scratch_ptr[rhs_index + rhs_offset];

          IndexType lhs_index = 0;
          for (IndexType wLPTM = 0; wLPTM < work_per_thread_m; wLPTM++) {
            // load a LHS element from the scratch buffer
            ValueType privateLhs = scratch_ptr[lhs_index + lhs_offset];

            // Perform a manual MAD.
            private_res[wLPTM + idx] =
                private_res[wLPTM + idx] + (privateLhs * privateRhs);

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
    } while (firstHalf < numTiles);

    id.barrier(cl::sycl::access::fence_space::local_space);

    // Store the final results in C
    IndexType global_col_offset = (ngroup_id * tile_size_dim_n) + (n_lcl_tid);
    IndexType global_row_offset = (mgroup_id * tile_size_dim_m) + (m_lcl_tid);
    IndexType global_k_offset = kgroup_id * M * N;
    IndexType cIndex = global_col_offset * M;
    IndexType private_index_offset = 0;

    for (Index wLPTN = 0; wLPTN < work_per_thread_n; wLPTN++) {
      Index private_index = private_index_offset;
      // for (Index i = 0; i < PacketSize; i++) {
      Index globalCol = global_col_offset;  // + i;
      Index globalRow = global_row_offset;
      for (Index wLPTM = 0; wLPTM < work_per_thread_m; wLPTM++) {
        if ((NoEdge) || ((globalRow + PacketSize - 1) < M && globalCol < N)) {
          // Store the final results in C
          if (IsFinal) {
            // vec_t::store(out_ptr + cIndex + globalRow + global_k_offset,
            //  *(privateRes + wLPTM + private_index));
          } else {
            // vec_t::store(tmp_ptr + cIndex + globalRow + global_k_offset,
            //  *(privateRes + wLPTM + private_index));
          }
        } else {
          for (Index j = 0; j < PacketSize; j++) {
            Index offset = globalRow + j + cIndex;
            PacketReturnType res = *(privateRes + wLPTM + private_index);
            if ((NoEdge) || ((globalRow + j) < M && globalCol < N)) {
              if (IsFinal) {
                out_ptr[offset] = Eigen::TensorSycl::internal::PacketWrapper<
                    PacketReturnType, PacketSize>::scalarize(j, res);
              } else {
                tmp_ptr[offset + global_k_offset] =
                    Eigen::TensorSycl::internal::PacketWrapper<
                        PacketReturnType, PacketSize>::scalarize(j, res);
              }
            }
          }
        }
        globalRow += (local_thread_size_n);
      }
      cIndex += M;
      private_index += (work_per_thread_m);
      // }
      //   global_col_offset += (PacketSize *
      //   PannelParameters::LocalThreadSizeN); cIndex = global_col_offset *
      //   M; private_index_offset += PannelParameters::WorkLoadPerThreadM;
    }
  }
};

#endif  // BLAS3_TSGEMM_HPP