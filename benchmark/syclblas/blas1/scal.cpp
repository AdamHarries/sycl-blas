/**************************************************************************
 *
 *  @license
 *  Copyright (C) 2016 Codeplay Software Limited
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
 *  @filename scal.cpp
 *
 **************************************************************************/

#include "utils.hpp"

template <typename scalar_t>
void BM_Scal(benchmark::State& state) {
  // Standard test setup.
  const index_t size = static_cast<index_t>(state.range(0));

  // Google-benchmark counters are double.
  double size_d = static_cast<double>(size);
  state.counters["size"] = size_d;
  state.counters["n_fl_ops"] = size_d;
  state.counters["bytes_processed"] = 2 * size_d * sizeof(scalar_t);

  SyclExecutorType ex = *Global::executorInstancePtr;

  // Create data
  std::vector<scalar_t> v1 = benchmark::utils::random_data<scalar_t>(size);
  scalar_t alpha = benchmark::utils::random_scalar<scalar_t>();

  auto in = blas::make_sycl_iterator_buffer<scalar_t>(v1, size);

  // Warmup
  for (int i = 0; i < 10; i++) {
    _scal(ex, size, alpha, in, 1);
  }
  ex.get_policy_handler().wait();

  state.counters["best_event_time"] = ULONG_MAX;
  state.counters["best_overall_time"] = ULONG_MAX;

  // Measure
  for (auto _ : state) {
    // Run
    std::tuple<double, double> times =
        benchmark::utils::timef([&]() -> std::vector<cl::sycl::event> {
          auto event = _scal(ex, size, alpha, in, 1);
          ex.get_policy_handler().wait(event);
          return event;
        });

    // Report
    state.PauseTiming();

    double overall_time, event_time;
    std::tie(overall_time, event_time) = times;

    state.counters["total_event_time"] += event_time;
    state.counters["best_event_time"] =
        std::min<double>(state.counters["best_event_time"], event_time);

    state.counters["total_overall_time"] += overall_time;
    state.counters["best_overall_time"] =
        std::min<double>(state.counters["best_overall_time"], overall_time);

    state.ResumeTiming();
  };

  state.counters["avg_event_time"] =
      state.counters["total_event_time"] / state.iterations();
  state.counters["avg_overall_time"] =
      state.counters["total_overall_time"] / state.iterations();
}

BENCHMARK_TEMPLATE(BM_Scal, float)->RangeMultiplier(2)->Range(2 << 5, 2 << 18);
#ifdef DOUBLE_SUPPORT
BENCHMARK_TEMPLATE(BM_Scal, double)->RangeMultiplier(2)->Range(2 << 5, 2 << 18);
#endif
