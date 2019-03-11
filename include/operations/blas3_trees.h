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
 *  @filename blas3_trees.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS3_TREES_GEMM_H
#define SYCL_BLAS_BLAS3_TREES_GEMM_H

#include <CL/sycl.hpp>

#include <string>
#include <type_traits>

namespace blas {
/*
 * @brief Determines the type of the GEMM kernel.
 * It can be either a naive kernel; a kernel uses local memory or a kernel that
 * does not use local memory
 */
enum class Gemm_t : int { naive = 0, local_memory = 1, no_local_memory = 2 };

/*!
 * @brief The Tile structure determines the tiling configuration of a gemm
 *        implementation.
 *
 * The structure defines a hierarchical mapping of work items to matrix blocks,
 * and the Tile parameter have the largest impact on performance.
 * The blocking is done in 3 layers.
 *
 * The largest, top-level layer groups multiple consecutive work groups into a
 * top-level tile. The size of the tile (expressed in the number of work groups
 * in a tile) is determined by TlRows and TlColumns template parameters.
 * Different settings of the top-level layer do not have any impact on the
 * amount of required resources, but can impact data locality between
 * neighboring work groups, which can improve cache hit rates of high-level
 * caches if they are shared between multiple work groups.
 *
 * The second, block-level layer groups multiple work items into a block-level
 * tile. One block-level tile is assigned to one work-group, and hence
 * determines the number of work items within a work group.
 * It impacts local memory requirement (larger tile size requires more
 * local memory). A larger tile will also increase global data reuse
 * (average number of arithmetic operations performed per each data element
 * fetched from global memory). Generally, the larger block-level tile the
 * better, but its size is restricted by the maximal work-group size, and by
 * the available amount of shared memory.
 *
 * The last, item-level layer determines the size of the item-level tile,
 * which represents the size of the matrix block processed by a single work
 * item. A larger tile results in higher global data reuse, as well as local
 * data reuse (average number of arithmetic operations performed per each data
 * element fetched from local). However, larger tiles require more
 * register space, as well as more local memory.
 *
 * @note Square, or close-to-square tiles achieve the highest data reuse rate
 *       among all tiles that use the same amount of local / register
 *       space.
 *
 * @tparam ItemRows  the number of rows processed by each work item
 * @tparam ItemCols  the number of columns processed by each work item
 * @tparam WgRows  the number of item-level tiles within each column of
 *                 block-level tile
 * @tparam WgCols  the number of item-level tiles within each row of
 *                 block-level tile
 * @tparam TlRows  the number of block-level tiles within each column of
 *                 top-level tile
 * @tparam TlCols  the number of block-level tiles within each row of
 *                 top-level tile
 *
 * @see GemmFactory
 */
template <int ItemRows = 8, int ItemCols = 8, int WgRows = 16, int WgCols = 16,
          int TlRows = 1, int TlCols = 1>
struct Tile {
  static constexpr int item_rows = ItemRows;
  static constexpr int item_cols = ItemCols;
  static constexpr int wg_rows = WgRows;
  static constexpr int wg_cols = WgCols;
  static constexpr int tl_rows = TlRows;
  static constexpr int tl_cols = TlCols;
  /*!
   * @brief Get tile type as human readable string.
   */
  static std::string get_type_string() noexcept;
};
/*!
 * @brief GemmFactory is a template class whose instantiations provide
 *        different implementations of the GEMM device function. It also support
 * batch GEMM
 *
 * To use the function, each item of a kernel launched with nd_range given by
 * GemmFactory::get_nd_range() should call GemmFactory::run(). The size of
 * local memory required per work group can be queried with
 * GemmFactory::local_memory_size.
 *
 * @tparam DoubleBuffer  iff true,  enables the use of double buffering
 *                       (doubles the amount of consumed local memory,
 *                        but halves the number of required local barriers)
 * @tparam NbcA  iff true, avoids bank conflicts when accessing blocks of
 *               matrix A in local memory (slightly increases local
 *               memory consumption) - may be useful in combination with TranA
 * @tparam NbcA  iff true, avoids bank conflicts when accessing blocks of
 *               matrix B in local memory (slightly increases local
 *               memory consumption) - may be useful in combination with TranB
 * @tparam ClSize  the size of the cache line of the architecture
 *                 (If the value passed in is smaller than the actual cache
 *                 line size, some values fetched will be wasted, which can
 *                 significantly reduce performance. It can be set to a
 *                 multiple of the physical cache line size. In this case, it
 *                 will significantly increase local memory usage, but
 *                 will result in fewer local barriers.)
 * @tparam TileType  determines the size of the local, work group, and top
 *                   level tiles to use, see Tile
 * @tparam TransA  iff true, matrix A will be transposed on the fly
 * @tparam TransB  iff true, matrix B will be transposed on the fly
 * @tparam T  type of matrix elements
 * @param _A the LHS matrix
 * @param _B the RHS matrix
 * @param _C the output matrix
 * @param Alpha specifies the scalar alpha
 * @param beta specifies the scalar beta
 * @param m  the number  of rows  of the  matrix _C
 * @param n  the number  of column  of the  matrix _C
 * @param k the contracting dimension between _A and _B
 * @param lda the leading dimension of the matrix _A
 * @param ldb the leading dimension of the matrix _B
 * @param ldc the leading dimension of the matrix _C
 * @param m_batch_size the number batches of matrices of _A _B _C
 */
template <typename RHS0, typename RHS1, bool DoubleBuffer, bool NbcA, bool NbcB,
          int ClSize, typename tile_type, bool TransA, bool TransB, typename T,
          bool is_beta_zero, int Gemm_type>
class Gemm {
 public:
  using value_type = T;
  using IndexType = typename std::make_signed<typename RHS0::IndexType>::type;
  static constexpr int type = Gemm_type;
  static constexpr int wg_size = tile_type::wg_rows * tile_type::wg_cols;
  static constexpr bool trans_a = TransA;
  static constexpr bool trans_b = TransB;
  static constexpr int local_memory_size = 0;
  RHS0 _A;
  RHS0 _B;
  RHS1 _C;
  T alpha;
  T beta;
  IndexType m;
  IndexType n;
  IndexType k;
  IndexType lda;
  IndexType ldb;
  IndexType ldc;
  IndexType m_batch_size;
  Gemm(RHS0 A, RHS0 B, RHS1 C, T alpha, T beta, IndexType batch_size);
  static std::string get_type_string() noexcept;
  static IndexType get_workgroup_cluster(IndexType m, IndexType n) noexcept;
  static IndexType get_num_workgroup_cluster(IndexType m, IndexType n,
                                             IndexType compute_units) noexcept;
  static cl::sycl::nd_range<1> get_nd_range(IndexType m, IndexType n,
                                            IndexType compute_units) noexcept;
  IndexType getSize() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  void eval(cl::sycl::nd_item<1> id) noexcept;
  void bind(cl::sycl::handler &h);
};

/*
 * @brief a helper function used for constructing the GEMM
 *  see GEMM for the parammeters passed here.
 */
template <bool DoubleBuffer, bool ConflictA, bool ConflictB, int ClSize,
          typename TileType, bool TransA, bool TransB, int Gemm_type,
          bool is_beta_zero, typename RHS1, typename RHS2, typename T,
          typename IndexType>
inline Gemm<RHS1, RHS2, DoubleBuffer, ConflictA, ConflictB, ClSize, TileType,
            TransA, TransB, T, is_beta_zero, Gemm_type>
make_gemm(RHS1 buffer_a, RHS1 buffer_b, RHS2 buffer_c, T alpha, T beta,
          IndexType batch_size) {
  return Gemm<RHS1, RHS2, DoubleBuffer, ConflictA, ConflictB, ClSize, TileType,
              TransA, TransB, T, is_beta_zero, Gemm_type>(
      buffer_a, buffer_b, buffer_c, alpha, beta, batch_size);
}

// template <typename IndexType, IndexType NumTiles = 8,
//           IndexType TileSizeDimM = 8, IndexType TileSizeDimK = 16,
//           IndexType TileSizeDimN = 16, IndexType WorkPerThreadM = 16,
//           IndexType WorkPerThreadN = 1, IndexType LocalThreadSizeN = 4,
//           IndexType LocalThreadSizeM = 4>
// struct TSGEMMTile {
//   /* The number of tiles to be processed */
//   static constexpr IndexType num_tiles = NumTiles;
//   /* tile size, dimension M */
//   static constexpr IndexType tile_size_dim_m = TileSizeDimM;
//   /* tile size, dimension N */
//   static constexpr IndexType tile_size_dim_n = TileSizeDimN;
//   /* tile size, dimension K */
//   static constexpr IndexType tile_size_dim_k = TileSizeDimK;

//   /* workload per thread M */
//   static constexpr IndexType work_per_thread_m = WorkPerThreadM;
//   /* workload per thread N */
//   static constexpr IndexType work_per_thread_n = WorkPerThreadN;

//   /* local thread size n */
//   static constexpr IndexType local_thread_size_n = LocalThreadSizeN;
//   /* local thread size m */
//   static constexpr IndexType local_thread_size_m = LocalThreadSizeM;

//   /*!
//    * @brief Get tile type as human readable string.
//    */
//   // static inline std::string get_type_string() noexcept {
//   //   return std::string("Tile<") + std::to_string(item_rows) + ", " +
//   //          std::to_string(item_cols) + ", " + std::to_string(wg_rows) + ",
//   "
//   //          + std::to_string(wg_cols) + ", " + std::to_string(tl_rows) + ",
//   "
//   //          + std::to_string(tl_cols) + ">";
//   // }
// };

// // So far, more or less just copied from eigen.
// // template <typename OutScalar, typename LhsScalar, typename RhsScalar,
// // typename OutAccessor, typename TempAcc, typename LhsMapper,
// // typename RhsMapper, typename Scratch, typename Index,
// // typename PanelParameters, bool Vectorizable, bool NoEdge, boolIsFinal>
// template <typename RHS0, typename RHS1, typename ScratchT, typename T,
//           int WgSize, bool TransA, bool TransB, typename tile_type>
// class TallSkinnyGemmFactory {
//  public:
//   using ValueType = T;
//   // using IndexType = typename std::make_signed<typename
//   // RHS0::IndexType>::type;
//   using IndexType = int;

//   RHS0 _A;
//   RHS0 _B;
//   RHS1 _C;

//   ValueType alpha;
//   ValueType beta;

//   ScratchT scratch;

//   // OutAccessor out_res;

//   // const LhsMapper lhs;
//   // const RhsMapper rhs;
//   // Scratch scratch;

//   /* Size of dimension M */
//   const IndexType M;
//   /* Size of dimension N */
//   const IndexType N;
//   /* Size of dimension K */
//   const IndexType K;

//   static constexpr IndexType local_thread_size_n =
//       tile_type::local_thread_size_n;
//   static constexpr IndexType local_thread_size_m =
//       tile_type::local_thread_size_m;

//   /* The number of tiles to be processed */
//   static constexpr IndexType num_tiles = tile_type::num_tiles;
//   /* tile size, dimension M */
//   static constexpr IndexType tile_size_dim_m = tile_type::tile_size_dim_m;
//   /* tile size, dimension N */
//   static constexpr IndexType tile_size_dim_n = tile_type::tile_size_dim_n;
//   /* tile size, dimension K */
//   static constexpr IndexType tile_size_dim_k = tile_type::tile_size_dim_k;

//   /* workload per thread M */
//   static constexpr IndexType work_per_thread_m =
//   tile_type::work_per_thread_m;
//   /* workload per thread N */
//   static constexpr IndexType work_per_thread_n =
//   tile_type::work_per_thread_n;

//   /* the number of groups, dimension M */
//   const IndexType group_count_m;
//   /* the number of groups, dimension n */
//   const IndexType group_count_n;
//   /* the number of groups, dimension k */
//   const IndexType group__k;

//   inline TallSkinnyGemmFactory(RHS0 A, RHS0 B, RHS1 C, IndexType M, IndexType
//   N,
//                                IndexType K, T alpha, T beta, ScratchT
//                                scratch, const IndexType group_count_m, const
//                                IndexType group_count_n, const IndexType
//                                group__k);

//   inline void eval(cl::sycl::nd_item<1> id) const noexcept;
//   // We need two load functions: one that loads normally, one that
//   // loads + transposes on load.

//   // Load a "left hand" tile, or "right hand transposed" tile
//   // What is NoEdge for??
//   template <typename GlobalPointerType, typename LocalPointerType>
//   static inline void load_tile(GlobalPointerType glb_ptr,
//                                LocalPointerType lcl_ptr,
//                                IndexType linear_local_thread_id,
//                                IndexType global_m_offset,
//                                IndexType global_k_offset, IndexType
//                                next_half, IndexType load_per_thread_lhs,
//                                IndexType M, IndexType K);

//   template <typename GlobalPointerType, typename LocalPointerType>
//   static inline void load_and_transpose_tile(
//       GlobalPointerType glb_ptr, LocalPointerType lcl_ptr,
//       IndexType linear_local_thread_id, IndexType global_n_offset,
//       IndexType global_k_offset, IndexType next_half,
//       IndexType load_per_thread_rhs, IndexType K, IndexType N);
// };

}  // namespace blas

#endif  // BLAS3_TREES_GEMM_H
