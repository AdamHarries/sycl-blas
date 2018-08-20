#include "../blas3_matrix_formats.hpp"
#include "blas_test.hpp"

typedef ::testing::Types<blas_test_float<MatrixFormats<Conjugate, Conjugate>>,
                         blas_test_double<MatrixFormats<Conjugate, Conjugate>>>
    BlasTypes;

#define BlasTypes BlasTypes
#define TestName gemm_normal_normal

#include "blas3_gemm_def.hpp"