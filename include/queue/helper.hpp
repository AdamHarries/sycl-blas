#ifndef BLAS_HELPER_HPP
#define BLAS_HELPER_HPP
#include <queue/sycl_iterator.hpp>
#include <types/sycl_types.hpp>

namespace blas {
namespace helper {

// construct buffer from host pointer/data
template <typename scalar_t, typename index_t>
inline blas::buffer_iterator<scalar_t> make_sycl_iterator_buffer(
    scalar_t* data, index_t size,
    const cl::sycl::property_list& propList = {}) {
  using buff_t = blas::buffer_t<scalar_t, 1>;
  return blas::buffer_iterator<scalar_t>{
      buff_t{data, cl::sycl::range<1>(size), propList}};
}

// construct buffer from std::vector
template <typename scalar_t, typename index_t>
inline buffer_iterator<scalar_t> make_sycl_iterator_buffer(
    std::vector<scalar_t>& data, index_t size,
    const cl::sycl::property_list& propList = {}) {
  using buff_t = blas::buffer_t<scalar_t, 1>;
  return blas::buffer_iterator<scalar_t>{
      buff_t{data.data(), cl::sycl::range<1>(size), propList}};
}

// construct buffer with size, no data
template <typename scalar_t, typename index_t>
inline blas::buffer_iterator<scalar_t> make_sycl_iterator_buffer(
    index_t size, const cl::sycl::property_list& propList = {}) {
  using buff_t = blas::buffer_t<scalar_t, 1>;
  return blas::buffer_iterator<scalar_t>{
      buff_t{cl::sycl::range<1>(size), propList}};
}

// construct buffer from buffer
template <typename scalar_t, typename index_t>
inline blas::buffer_iterator<scalar_t> make_sycl_iterator_buffer(
    blas::buffer_t<scalar_t, 1> buff_) {
  return blas::buffer_iterator<scalar_t>{buff_};
}

}  // namespace helper
}  // namespace blas
#endif  // BLAS_HELPER_HPP
