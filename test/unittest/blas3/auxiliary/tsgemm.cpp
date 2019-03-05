/***************************************************************************
 *
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
 *  @filename blas2_tsgemm.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"
#include "unittest/util/matrix_debug_tools.hpp"

// #define TESTING_GUARD
#include <operations/blas3_tsgemm.hpp>

typedef ::testing::Types<blas_test_float<>> BlasTypes;

TYPED_TEST_SUITE(BLAS_Test, BlasTypes);

REGISTER_PREC(float, 1e-4, tsgemm_lhs_tile)
REGISTER_PREC(double, 1e-8, tsgemm_lhs_tile)
REGISTER_PREC(long double, 1e-8, tsgemm_lhs_tile)

using IndexType = int;

TYPED_TEST(BLAS_Test, tsgemm_matmul) {
  using ScalarT = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class tsgemm_matmul;
  ScalarT prec = 0.01;  // TestClass::template test_prec<test>();

  size_t m = 8;
  size_t k = 4;
  size_t n = 4;

  constexpr size_t group_count_m = 2;
  constexpr size_t group_count_n = 1;
  constexpr size_t group_count_k = 1;

  constexpr size_t groups = group_count_m * group_count_n * group_count_k;

  constexpr size_t local_size_m = 4;
  constexpr size_t local_size_n = 4;

  constexpr size_t tile_size_dim_m = 4;
  constexpr size_t tile_size_dim_k = 4;
  constexpr size_t tile_size_dim_n = 4;

  std::vector<ScalarT> C_expt(m * n);
  std::vector<ScalarT> A(m * k);
  std::vector<ScalarT> B(k * n);
  std::vector<ScalarT> C(m * n);

  // Fill the input matrices with 0..'s
  auto i = 0;
  for (auto &a : A) {
    a = i++;
  }
  for (auto &b : B) {
    b = i++;
  }
  // Fill the output with -42, so we know when it's modified.
  for (auto &c : C) {
    c = -42;
  }
  for (auto &c : C_expt) {
    c = 0;
  }

  // system gemm implementation
  gemm("n", "n", m, n, k, 1.0, A.data(), m, B.data(), k, 1.0, C_expt.data(), m);

  auto lhs_tile_size = tile_size_dim_m * tile_size_dim_k;
  auto rhs_tile_size = tile_size_dim_n * tile_size_dim_k;

  std::vector<ScalarT> scratch((2 * lhs_tile_size) + (2 * rhs_tile_size));

  ScalarT alpha(0.0);
  ScalarT beta(0.0);

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  cl::sycl::buffer<ScalarT, 1> a_gpu(A.data(), cl::sycl::range<1>(A.size()));
  cl::sycl::buffer<ScalarT, 1> b_gpu(B.data(), cl::sycl::range<1>(B.size()));
  cl::sycl::buffer<ScalarT, 1> c_gpu(C.data(), cl::sycl::range<1>(C.size()));
  cl::sycl::buffer<ScalarT, 1> scratch_gpu(scratch.data(),
                                           cl::sycl::range<1>(scratch.size()));

  auto event = q.submit([&](cl::sycl::handler &cgh) {
    auto a_ptr = a_gpu.template get_access<cl::sycl::access::mode::read>(cgh);
    auto b_ptr = b_gpu.template get_access<cl::sycl::access::mode::read>(cgh);
    auto c_ptr =
        c_gpu.template get_access<cl::sycl::access::mode::read_write>(cgh);
    cl::sycl::accessor<ScalarT, 1, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::local>
        scratch_ptr(cl::sycl::range<1>(scratch.size()), cgh);

    TallSkinnyGemmFactory<
        cl::sycl::accessor<ScalarT, 1, cl::sycl::access::mode::read,
                           cl::sycl::access::target::global_buffer>,  // RHS0
        cl::sycl::accessor<ScalarT, 1, cl::sycl::access::mode::read_write,
                           cl::sycl::access::target::global_buffer>,  // RHS1
        cl::sycl::accessor<ScalarT, 1, cl::sycl::access::mode::read_write,
                           cl::sycl::access::target::local>,  // ScratchT
        ScalarT,                                              // T
        groups,                                               // WgSize
        false,                                                // TransA
        false,                                                // TransB
        TSGEMMTile<                                           // TileType
            int,                                              // IndexType
            2,                                                // Numtiles
            tile_size_dim_m,                                  // TileSizeDimM
            tile_size_dim_k,                                  // TileSizeDimK
            tile_size_dim_n,                                  // TileSizeDimN
            1,                                                // WorkPerThreadM
            1,                                                // WorkPerThreadN
            local_size_n,  // LocalThreadSizeN
            local_size_m   // LocalThreadSizeM
            >>
        tsgf(a_ptr,          // RHS0 A
             b_ptr,          // RHS0 B
             c_ptr,          // RHS1 C
             m,              // IndexType M
             n,              // IndexType N
             k,              // IndexType K
             alpha,          // T alpha
             beta,           // T beta
             scratch_ptr,    // ScratchT scratch
             group_count_m,  // const IndexType group_count_m
             group_count_n,  // const IndexType group_count_n
             group_count_k   // const IndexType group_count_k
        );

    cgh.parallel_for<cl::sycl::kernel>(
        cl::sycl::nd_range<1>(
            cl::sycl::range<1>(groups * local_size_n * local_size_m),
            cl::sycl::range<1>(local_size_n * local_size_m)),
        [=](cl::sycl::nd_item<1> item) { tsgf.eval(item); });

    cgh.copy(c_ptr, C.data());
  });

  ex.wait(event);

  for (auto i = 0; i < C.size(); ++i) {
    ASSERT_NEAR(C[i], C_expt[i], prec);
  }

  DEBUG_PRINT(std::cerr << "A before: " << std::endl);
  MatrixPrinter<true>::eval(k, m, A);

  DEBUG_PRINT(std::cerr << "B before: " << std::endl);
  MatrixPrinter<true>::eval(n, k, B);

  // the matrix is now in tsgf._C
  DEBUG_PRINT(std::cerr << "C expected: " << std::endl);
  MatrixPrinter<true>::eval(n, m, C_expt);

  DEBUG_PRINT(std::cerr << "C afterwards: " << std::endl);
  MatrixPrinter<true>::eval(n, m, C);
}
