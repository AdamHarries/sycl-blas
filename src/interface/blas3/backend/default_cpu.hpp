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
 *  @filename default_cpu.hpp
 *
 **************************************************************************/
#ifndef SYCL_BLAS_GEMM_DEFAULT_CPU_BACKEND_HPP
#define SYCL_BLAS_GEMM_DEFAULT_CPU_BACKEND_HPP
#include "interface/gemm_launcher.h"

namespace blas {
namespace gemm {
namespace backend {

template <bool _t_a, bool _t_b, bool is_beta_zero, typename Executor,
          typename ContainerT0, typename ContainerT1, typename ContainerT2,
          typename T, typename IndexType>
typename Executor::Policy::event_type _gemm(
    Executor& ex, IndexType _M, IndexType _N, IndexType _K, T _alpha,
    ContainerT0 _A, IndexType _lda, ContainerT1 _B, IndexType _ldb, T _beta,
    ContainerT2 _C, IndexType _ldc, IndexType batch_size) {
  return blas::Gemm_Launcher<
      64, false, false, false, 64, Tile<8, 8, 8, 8>, _t_a, _t_b,
#if defined(NAIVE_GEMM)

      static_cast<int>(Gemm_t::naive)
#else

      static_cast<int>(Gemm_t::no_local_memory)
#endif
          ,
      is_beta_zero>::template _select_gemm(ex, _M, _N, _K, _alpha, _A, _lda, _B,
                                           _ldb, _beta, _C, _ldc, batch_size);
}
}  // namespace backend
}  // namespace gemm
}  // namespace blas
#endif