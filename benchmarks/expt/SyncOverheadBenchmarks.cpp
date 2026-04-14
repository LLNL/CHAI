//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "benchmark/benchmark.h"

#include "chai/config.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

#if defined(CHAI_ENABLE_CUDA)
#include <cuda_runtime.h>
#elif defined(CHAI_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace {
#if defined(CHAI_ENABLE_CUDA)
  using ErrorType = cudaError_t;
  using StreamType = cudaStream_t;
  using EventType = cudaEvent_t;

  constexpr ErrorType kSuccess = cudaSuccess;

  inline const char* getErrorString(ErrorType error)
  {
    return cudaGetErrorString(error);
  }

  inline ErrorType createStream(StreamType* stream)
  {
    return cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking);
  }

  inline ErrorType destroyStream(StreamType stream)
  {
    return cudaStreamDestroy(stream);
  }

  inline ErrorType createEvent(EventType* event)
  {
    return cudaEventCreateWithFlags(event, cudaEventDisableTiming);
  }

  inline ErrorType destroyEvent(EventType event)
  {
    return cudaEventDestroy(event);
  }

  inline ErrorType recordEvent(EventType event, StreamType stream)
  {
    return cudaEventRecord(event, stream);
  }

  inline ErrorType deviceSynchronize()
  {
    return cudaDeviceSynchronize();
  }

  inline ErrorType streamSynchronize(StreamType stream)
  {
    return cudaStreamSynchronize(stream);
  }

  inline ErrorType eventSynchronize(EventType event)
  {
    return cudaEventSynchronize(event);
  }

  inline ErrorType eventQuery(EventType event)
  {
    return cudaEventQuery(event);
  }
#elif defined(CHAI_ENABLE_HIP)
  using ErrorType = hipError_t;
  using StreamType = hipStream_t;
  using EventType = hipEvent_t;

  constexpr ErrorType kSuccess = hipSuccess;

  inline const char* getErrorString(ErrorType error)
  {
    return hipGetErrorString(error);
  }

  inline ErrorType createStream(StreamType* stream)
  {
    return hipStreamCreateWithFlags(stream, hipStreamNonBlocking);
  }

  inline ErrorType destroyStream(StreamType stream)
  {
    return hipStreamDestroy(stream);
  }

  inline ErrorType createEvent(EventType* event)
  {
    return hipEventCreateWithFlags(event, hipEventDisableTiming);
  }

  inline ErrorType destroyEvent(EventType event)
  {
    return hipEventDestroy(event);
  }

  inline ErrorType recordEvent(EventType event, StreamType stream)
  {
    return hipEventRecord(event, stream);
  }

  inline ErrorType deviceSynchronize()
  {
    return hipDeviceSynchronize();
  }

  inline ErrorType streamSynchronize(StreamType stream)
  {
    return hipStreamSynchronize(stream);
  }

  inline ErrorType eventSynchronize(EventType event)
  {
    return hipEventSynchronize(event);
  }

  inline ErrorType eventQuery(EventType event)
  {
    return hipEventQuery(event);
  }
#endif

  [[noreturn]] void failRuntimeCall(const char* call, ErrorType error)
  {
    std::fprintf(stderr, "%s failed: %s\n", call, getErrorString(error));
    std::abort();
  }

  void checkRuntimeCall(ErrorType error, const char* call)
  {
    if (error != kSuccess)
    {
      failRuntimeCall(call, error);
    }
  }

  class SyncResources
  {
    public:
      SyncResources()
      {
        checkRuntimeCall(createStream(&m_stream), "createStream");
        checkRuntimeCall(createEvent(&m_event), "createEvent");

        // Keep the event in a completed state so benchmarks measure synchronization
        // call overhead rather than the cost of waiting for outstanding work.
        checkRuntimeCall(recordEvent(m_event, m_stream), "recordEvent");
        checkRuntimeCall(streamSynchronize(m_stream), "streamSynchronize");
      }

      ~SyncResources()
      {
        checkRuntimeCall(destroyEvent(m_event), "destroyEvent");
        checkRuntimeCall(destroyStream(m_stream), "destroyStream");
      }

      StreamType stream() const
      {
        return m_stream;
      }

      EventType event() const
      {
        return m_event;
      }

    private:
      StreamType m_stream{};
      EventType m_event{};
  };

  class EventPool
  {
    public:
      explicit EventPool(std::size_t initial_size = 0)
        : m_initial_size{initial_size}
      {
        prewarm();
      }

      ~EventPool()
      {
        for (EventType event : m_events)
        {
          checkRuntimeCall(destroyEvent(event), "destroyEvent");
        }
      }

      EventType acquire()
      {
        if (m_events.empty())
        {
          EventType event{};
          checkRuntimeCall(createEvent(&event), "createEvent");
          return event;
        }

        EventType event = m_events.back();
        m_events.pop_back();
        return event;
      }

      void release(EventType event)
      {
        m_events.push_back(event);
      }

    private:
      void prewarm()
      {
        for (std::size_t i = 0; i < m_initial_size; ++i)
        {
          EventType event{};
          checkRuntimeCall(createEvent(&event), "createEvent");
          m_events.push_back(event);
        }
      }

      std::size_t m_initial_size{0};
      std::vector<EventType> m_events{};
  };

  class MutexEventPool
  {
    public:
      explicit MutexEventPool(std::size_t initial_size = 0)
        : m_initial_size{initial_size}
      {
        prewarm();
      }

      ~MutexEventPool()
      {
        for (EventType event : m_events)
        {
          checkRuntimeCall(destroyEvent(event), "destroyEvent");
        }
      }

      EventType acquire()
      {
        std::lock_guard<std::mutex> lock{m_mutex};

        if (m_events.empty())
        {
          EventType event{};
          checkRuntimeCall(createEvent(&event), "createEvent");
          return event;
        }

        EventType event = m_events.back();
        m_events.pop_back();
        return event;
      }

      void release(EventType event)
      {
        std::lock_guard<std::mutex> lock{m_mutex};
        m_events.push_back(event);
      }

    private:
      void prewarm()
      {
        std::lock_guard<std::mutex> lock{m_mutex};

        for (std::size_t i = 0; i < m_initial_size; ++i)
        {
          EventType event{};
          checkRuntimeCall(createEvent(&event), "createEvent");
          m_events.push_back(event);
        }
      }

      std::mutex m_mutex{};
      std::size_t m_initial_size{0};
      std::vector<EventType> m_events{};
  };

  void setPoolCounters(benchmark::State& state,
                       std::int64_t calls_per_iteration,
                       std::int64_t pool_size,
                       double pool_ops,
                       double record_calls,
                       double sync_calls)
  {
    state.SetItemsProcessed(state.iterations() * calls_per_iteration);
    state.counters["call_sites"] =
        benchmark::Counter(static_cast<double>(calls_per_iteration),
                           benchmark::Counter::kAvgIterations);
    state.counters["pool_size"] =
        benchmark::Counter(static_cast<double>(pool_size),
                           benchmark::Counter::kAvgIterations);
    state.counters["pool_ops"] =
        benchmark::Counter(pool_ops, benchmark::Counter::kAvgIterations);
    state.counters["record_calls"] =
        benchmark::Counter(record_calls, benchmark::Counter::kAvgIterations);
    state.counters["sync_calls"] =
        benchmark::Counter(sync_calls, benchmark::Counter::kAvgIterations);
    state.counters["api_calls"] =
        benchmark::Counter(record_calls + sync_calls,
                           benchmark::Counter::kAvgIterations);
  }

  void addCallSiteAndPoolSizeArgs(benchmark::internal::Benchmark* benchmark)
  {
    for (std::int64_t calls = 1; calls <= 256; calls *= 2)
    {
      for (std::int64_t pool_size = 1; pool_size <= calls; pool_size *= 2)
      {
        benchmark->Args({calls, pool_size});
      }
    }
  }

  template <typename Call>
  void runRepeatedCalls(benchmark::State& state, Call&& call)
  {
    const auto calls_per_iteration = static_cast<std::int64_t>(state.range(0));

    for (auto _ : state)
    {
      for (std::int64_t i = 0; i < calls_per_iteration; ++i)
      {
        call();
      }
    }

    state.SetItemsProcessed(state.iterations() * calls_per_iteration);
    state.counters["call_sites"] =
        benchmark::Counter(static_cast<double>(calls_per_iteration),
                           benchmark::Counter::kAvgIterations);
    state.counters["api_calls"] =
        benchmark::Counter(static_cast<double>(calls_per_iteration),
                           benchmark::Counter::kAvgIterations);
  }

  template <typename Call>
  void runGuardedCallsNonThreadSafe(benchmark::State& state, Call&& call)
  {
    const auto calls_per_iteration = static_cast<std::int64_t>(state.range(0));

    for (auto _ : state)
    {
      bool already_synchronized = false;

      for (std::int64_t i = 0; i < calls_per_iteration; ++i)
      {
        if (!already_synchronized)
        {
          call();
          already_synchronized = true;
        }
      }
    }

    state.SetItemsProcessed(state.iterations() * calls_per_iteration);
    state.counters["call_sites"] =
        benchmark::Counter(static_cast<double>(calls_per_iteration),
                           benchmark::Counter::kAvgIterations);
    state.counters["api_calls"] =
        benchmark::Counter(calls_per_iteration > 0 ? 1.0 : 0.0,
                           benchmark::Counter::kAvgIterations);
  }

  template <typename Call>
  void runGuardedCallsAtomic(benchmark::State& state, Call&& call)
  {
    const auto calls_per_iteration = static_cast<std::int64_t>(state.range(0));

    for (auto _ : state)
    {
      std::atomic<bool> already_synchronized{false};

      for (std::int64_t i = 0; i < calls_per_iteration; ++i)
      {
        bool expected = false;
        if (already_synchronized.compare_exchange_strong(
                expected,
                true,
                std::memory_order_acq_rel,
                std::memory_order_acquire))
        {
          call();
        }
      }
    }

    state.SetItemsProcessed(state.iterations() * calls_per_iteration);
    state.counters["call_sites"] =
        benchmark::Counter(static_cast<double>(calls_per_iteration),
                           benchmark::Counter::kAvgIterations);
    state.counters["api_calls"] =
        benchmark::Counter(calls_per_iteration > 0 ? 1.0 : 0.0,
                           benchmark::Counter::kAvgIterations);
  }

  template <typename Pool>
  void runPooledEventCalls(benchmark::State& state, Pool& pool, StreamType stream)
  {
    const auto calls_per_iteration = static_cast<std::int64_t>(state.range(0));

    for (auto _ : state)
    {
      for (std::int64_t i = 0; i < calls_per_iteration; ++i)
      {
        EventType event = pool.acquire();
        checkRuntimeCall(recordEvent(event, stream), "recordEvent");
        checkRuntimeCall(eventSynchronize(event), "eventSynchronize");
        pool.release(event);
      }
    }

    state.SetItemsProcessed(state.iterations() * calls_per_iteration);
    state.counters["call_sites"] =
        benchmark::Counter(static_cast<double>(calls_per_iteration),
                           benchmark::Counter::kAvgIterations);
    state.counters["api_calls"] =
        benchmark::Counter(static_cast<double>(calls_per_iteration),
                           benchmark::Counter::kAvgIterations);
  }

  template <typename Pool>
  void runPooledEventCallsGuardedBool(benchmark::State& state, Pool& pool, StreamType stream)
  {
    const auto calls_per_iteration = static_cast<std::int64_t>(state.range(0));

    for (auto _ : state)
    {
      bool already_synchronized = false;

      for (std::int64_t i = 0; i < calls_per_iteration; ++i)
      {
        if (!already_synchronized)
        {
          EventType event = pool.acquire();
          checkRuntimeCall(recordEvent(event, stream), "recordEvent");
          checkRuntimeCall(eventSynchronize(event), "eventSynchronize");
          pool.release(event);
          already_synchronized = true;
        }
      }
    }

    state.SetItemsProcessed(state.iterations() * calls_per_iteration);
    state.counters["call_sites"] =
        benchmark::Counter(static_cast<double>(calls_per_iteration),
                           benchmark::Counter::kAvgIterations);
    state.counters["api_calls"] =
        benchmark::Counter(calls_per_iteration > 0 ? 1.0 : 0.0,
                           benchmark::Counter::kAvgIterations);
  }

  template <typename Pool>
  void runPooledEventCallsGuardedAtomic(benchmark::State& state, Pool& pool, StreamType stream)
  {
    const auto calls_per_iteration = static_cast<std::int64_t>(state.range(0));

    for (auto _ : state)
    {
      std::atomic<bool> already_synchronized{false};

      for (std::int64_t i = 0; i < calls_per_iteration; ++i)
      {
        bool expected = false;
        if (already_synchronized.compare_exchange_strong(
                expected,
                true,
                std::memory_order_acq_rel,
                std::memory_order_acquire))
        {
          EventType event = pool.acquire();
          checkRuntimeCall(recordEvent(event, stream), "recordEvent");
          checkRuntimeCall(eventSynchronize(event), "eventSynchronize");
          pool.release(event);
        }
      }
    }

    state.SetItemsProcessed(state.iterations() * calls_per_iteration);
    state.counters["call_sites"] =
        benchmark::Counter(static_cast<double>(calls_per_iteration),
                           benchmark::Counter::kAvgIterations);
    state.counters["api_calls"] =
        benchmark::Counter(calls_per_iteration > 0 ? 1.0 : 0.0,
                           benchmark::Counter::kAvgIterations);
  }

  template <typename Pool>
  void runPooledAcquireReleaseOnly(benchmark::State& state)
  {
    const auto calls_per_iteration = static_cast<std::int64_t>(state.range(0));
    const auto pool_size = static_cast<std::int64_t>(state.range(1));

    for (auto _ : state)
    {
      state.PauseTiming();
      {
        Pool pool{static_cast<std::size_t>(pool_size)};
        state.ResumeTiming();

        for (std::int64_t i = 0; i < calls_per_iteration; ++i)
        {
          EventType event = pool.acquire();
          pool.release(event);
        }

        state.PauseTiming();
      }
      state.ResumeTiming();
    }

    setPoolCounters(state,
                    calls_per_iteration,
                    pool_size,
                    static_cast<double>(2 * calls_per_iteration),
                    0.0,
                    0.0);
  }

  template <typename Pool>
  void runPooledRecordOnlySameStream(benchmark::State& state, StreamType stream)
  {
    const auto calls_per_iteration = static_cast<std::int64_t>(state.range(0));
    const auto pool_size = static_cast<std::int64_t>(state.range(1));

    for (auto _ : state)
    {
      state.PauseTiming();
      {
        Pool pool{static_cast<std::size_t>(pool_size)};
        state.ResumeTiming();

        for (std::int64_t i = 0; i < calls_per_iteration; ++i)
        {
          EventType event = pool.acquire();
          checkRuntimeCall(recordEvent(event, stream), "recordEvent");
          pool.release(event);
        }

        state.PauseTiming();
        checkRuntimeCall(streamSynchronize(stream), "streamSynchronize");
      }
      state.ResumeTiming();
    }

    setPoolCounters(state,
                    calls_per_iteration,
                    pool_size,
                    static_cast<double>(2 * calls_per_iteration),
                    static_cast<double>(calls_per_iteration),
                    0.0);
  }

  template <typename Pool>
  void runRecordOneSynchronizeAllSharedToken(benchmark::State& state, StreamType stream)
  {
    const auto calls_per_iteration = static_cast<std::int64_t>(state.range(0));
    const auto pool_size = static_cast<std::int64_t>(state.range(1));

    for (auto _ : state)
    {
      state.PauseTiming();
      {
        Pool pool{static_cast<std::size_t>(pool_size)};
        state.ResumeTiming();

        EventType event = pool.acquire();
        checkRuntimeCall(recordEvent(event, stream), "recordEvent");

        for (std::int64_t i = 0; i < calls_per_iteration; ++i)
        {
          checkRuntimeCall(eventSynchronize(event), "eventSynchronize");
        }

        pool.release(event);
        state.PauseTiming();
      }
      state.ResumeTiming();
    }

    setPoolCounters(state,
                    calls_per_iteration,
                    pool_size,
                    2.0,
                    1.0,
                    static_cast<double>(calls_per_iteration));
  }

  template <typename Pool>
  void runRecordAllSynchronizeOneDistinctTokens(benchmark::State& state, StreamType stream)
  {
    const auto calls_per_iteration = static_cast<std::int64_t>(state.range(0));
    const auto pool_size = static_cast<std::int64_t>(state.range(1));

    for (auto _ : state)
    {
      state.PauseTiming();
      {
        Pool pool{static_cast<std::size_t>(pool_size)};
        std::vector<EventType> events;
        events.reserve(static_cast<std::size_t>(calls_per_iteration));
        state.ResumeTiming();

        for (std::int64_t i = 0; i < calls_per_iteration; ++i)
        {
          EventType event = pool.acquire();
          checkRuntimeCall(recordEvent(event, stream), "recordEvent");
          events.push_back(event);
        }

        checkRuntimeCall(eventSynchronize(events.front()), "eventSynchronize");

        state.PauseTiming();
        checkRuntimeCall(streamSynchronize(stream), "streamSynchronize");
        for (EventType event : events)
        {
          pool.release(event);
        }
      }
      state.ResumeTiming();
    }

    setPoolCounters(state,
                    calls_per_iteration,
                    pool_size,
                    static_cast<double>(2 * calls_per_iteration),
                    static_cast<double>(calls_per_iteration),
                    1.0);
  }

  template <typename Pool>
  void runRecordAllSynchronizeAllDistinctTokens(benchmark::State& state, StreamType stream)
  {
    const auto calls_per_iteration = static_cast<std::int64_t>(state.range(0));
    const auto pool_size = static_cast<std::int64_t>(state.range(1));

    for (auto _ : state)
    {
      state.PauseTiming();
      {
        Pool pool{static_cast<std::size_t>(pool_size)};
        std::vector<EventType> events;
        events.reserve(static_cast<std::size_t>(calls_per_iteration));
        state.ResumeTiming();

        for (std::int64_t i = 0; i < calls_per_iteration; ++i)
        {
          EventType event = pool.acquire();
          checkRuntimeCall(recordEvent(event, stream), "recordEvent");
          events.push_back(event);
        }

        for (EventType event : events)
        {
          checkRuntimeCall(eventSynchronize(event), "eventSynchronize");
        }

        state.PauseTiming();
        for (EventType event : events)
        {
          pool.release(event);
        }
      }
      state.ResumeTiming();
    }

    setPoolCounters(state,
                    calls_per_iteration,
                    pool_size,
                    static_cast<double>(2 * calls_per_iteration),
                    static_cast<double>(calls_per_iteration),
                    static_cast<double>(calls_per_iteration));
  }

  template <typename Pool>
  void runSameStreamRefreshReplaceToken(benchmark::State& state, StreamType stream)
  {
    const auto calls_per_iteration = static_cast<std::int64_t>(state.range(0));
    const auto pool_size = static_cast<std::int64_t>(state.range(1));

    for (auto _ : state)
    {
      state.PauseTiming();
      {
        Pool pool{static_cast<std::size_t>(pool_size)};
        EventType current{};
        bool has_current = false;
        state.ResumeTiming();

        for (std::int64_t i = 0; i < calls_per_iteration; ++i)
        {
          EventType next = pool.acquire();
          checkRuntimeCall(recordEvent(next, stream), "recordEvent");

          if (has_current)
          {
            pool.release(current);
          }

          current = next;
          has_current = true;
        }

        state.PauseTiming();
        checkRuntimeCall(streamSynchronize(stream), "streamSynchronize");
        if (has_current)
        {
          pool.release(current);
        }
      }
      state.ResumeTiming();
    }

    setPoolCounters(state,
                    calls_per_iteration,
                    pool_size,
                    static_cast<double>(2 * calls_per_iteration),
                    static_cast<double>(calls_per_iteration),
                    0.0);
  }

  static void DeviceSynchronize_Repeated(benchmark::State& state)
  {
    runRepeatedCalls(state, []() {
      checkRuntimeCall(deviceSynchronize(), "deviceSynchronize");
    });
  }

  BENCHMARK(DeviceSynchronize_Repeated)
    ->RangeMultiplier(2)
    ->Range(1, 256)
    ->Unit(benchmark::kNanosecond);

  static void DeviceSynchronize_GuardedBool(benchmark::State& state)
  {
    runGuardedCallsNonThreadSafe(state, []() {
      checkRuntimeCall(deviceSynchronize(), "deviceSynchronize");
    });
  }

  BENCHMARK(DeviceSynchronize_GuardedBool)
    ->RangeMultiplier(2)
    ->Range(1, 256)
    ->Unit(benchmark::kNanosecond);

  static void DeviceSynchronize_GuardedAtomic(benchmark::State& state)
  {
    runGuardedCallsAtomic(state, []() {
      checkRuntimeCall(deviceSynchronize(), "deviceSynchronize");
    });
  }

  BENCHMARK(DeviceSynchronize_GuardedAtomic)
    ->RangeMultiplier(2)
    ->Range(1, 256)
    ->Unit(benchmark::kNanosecond);

  static void StreamSynchronize_Repeated(benchmark::State& state)
  {
    SyncResources resources{};

    runRepeatedCalls(state, [&resources]() {
      checkRuntimeCall(streamSynchronize(resources.stream()), "streamSynchronize");
    });
  }

  BENCHMARK(StreamSynchronize_Repeated)
    ->RangeMultiplier(2)
    ->Range(1, 256)
    ->Unit(benchmark::kNanosecond);

  static void StreamSynchronize_GuardedBool(benchmark::State& state)
  {
    SyncResources resources{};

    runGuardedCallsNonThreadSafe(state, [&resources]() {
      checkRuntimeCall(streamSynchronize(resources.stream()), "streamSynchronize");
    });
  }

  BENCHMARK(StreamSynchronize_GuardedBool)
    ->RangeMultiplier(2)
    ->Range(1, 256)
    ->Unit(benchmark::kNanosecond);

  static void StreamSynchronize_GuardedAtomic(benchmark::State& state)
  {
    SyncResources resources{};

    runGuardedCallsAtomic(state, [&resources]() {
      checkRuntimeCall(streamSynchronize(resources.stream()), "streamSynchronize");
    });
  }

  BENCHMARK(StreamSynchronize_GuardedAtomic)
    ->RangeMultiplier(2)
    ->Range(1, 256)
    ->Unit(benchmark::kNanosecond);

  static void EventSynchronize_Repeated(benchmark::State& state)
  {
    SyncResources resources{};

    runRepeatedCalls(state, [&resources]() {
      checkRuntimeCall(eventSynchronize(resources.event()), "eventSynchronize");
    });
  }

  BENCHMARK(EventSynchronize_Repeated)
    ->RangeMultiplier(2)
    ->Range(1, 256)
    ->Unit(benchmark::kNanosecond);

  static void EventSynchronize_GuardedBool(benchmark::State& state)
  {
    SyncResources resources{};

    runGuardedCallsNonThreadSafe(state, [&resources]() {
      checkRuntimeCall(eventSynchronize(resources.event()), "eventSynchronize");
    });
  }

  BENCHMARK(EventSynchronize_GuardedBool)
    ->RangeMultiplier(2)
    ->Range(1, 256)
    ->Unit(benchmark::kNanosecond);

  static void EventSynchronize_GuardedAtomic(benchmark::State& state)
  {
    SyncResources resources{};

    runGuardedCallsAtomic(state, [&resources]() {
      checkRuntimeCall(eventSynchronize(resources.event()), "eventSynchronize");
    });
  }

  BENCHMARK(EventSynchronize_GuardedAtomic)
    ->RangeMultiplier(2)
    ->Range(1, 256)
    ->Unit(benchmark::kNanosecond);

  static void EventQuery_Repeated(benchmark::State& state)
  {
    SyncResources resources{};

    runRepeatedCalls(state, [&resources]() {
      checkRuntimeCall(eventQuery(resources.event()), "eventQuery");
    });
  }

  BENCHMARK(EventQuery_Repeated)
    ->RangeMultiplier(2)
    ->Range(1, 256)
    ->Unit(benchmark::kNanosecond);

  static void EventQuery_GuardedBool(benchmark::State& state)
  {
    SyncResources resources{};

    runGuardedCallsNonThreadSafe(state, [&resources]() {
      checkRuntimeCall(eventQuery(resources.event()), "eventQuery");
    });
  }

  BENCHMARK(EventQuery_GuardedBool)
    ->RangeMultiplier(2)
    ->Range(1, 256)
    ->Unit(benchmark::kNanosecond);

  static void EventQuery_GuardedAtomic(benchmark::State& state)
  {
    SyncResources resources{};

    runGuardedCallsAtomic(state, [&resources]() {
      checkRuntimeCall(eventQuery(resources.event()), "eventQuery");
    });
  }

  BENCHMARK(EventQuery_GuardedAtomic)
    ->RangeMultiplier(2)
    ->Range(1, 256)
    ->Unit(benchmark::kNanosecond);

  static void EventPoolSynchronize_Repeated(benchmark::State& state)
  {
    SyncResources resources{};
    EventPool pool{};

    runPooledEventCalls(state, pool, resources.stream());
  }

  BENCHMARK(EventPoolSynchronize_Repeated)
    ->RangeMultiplier(2)
    ->Range(1, 256)
    ->Unit(benchmark::kNanosecond);

  static void EventPoolSynchronize_GuardedBool(benchmark::State& state)
  {
    SyncResources resources{};
    EventPool pool{};

    runPooledEventCallsGuardedBool(state, pool, resources.stream());
  }

  BENCHMARK(EventPoolSynchronize_GuardedBool)
    ->RangeMultiplier(2)
    ->Range(1, 256)
    ->Unit(benchmark::kNanosecond);

  static void EventPoolSynchronize_GuardedAtomic(benchmark::State& state)
  {
    SyncResources resources{};
    EventPool pool{};

    runPooledEventCallsGuardedAtomic(state, pool, resources.stream());
  }

  BENCHMARK(EventPoolSynchronize_GuardedAtomic)
    ->RangeMultiplier(2)
    ->Range(1, 256)
    ->Unit(benchmark::kNanosecond);

  static void MutexEventPoolSynchronize_Repeated(benchmark::State& state)
  {
    SyncResources resources{};
    MutexEventPool pool{};

    runPooledEventCalls(state, pool, resources.stream());
  }

  BENCHMARK(MutexEventPoolSynchronize_Repeated)
    ->RangeMultiplier(2)
    ->Range(1, 256)
    ->Unit(benchmark::kNanosecond);

  static void MutexEventPoolSynchronize_GuardedBool(benchmark::State& state)
  {
    SyncResources resources{};
    MutexEventPool pool{};

    runPooledEventCallsGuardedBool(state, pool, resources.stream());
  }

  BENCHMARK(MutexEventPoolSynchronize_GuardedBool)
    ->RangeMultiplier(2)
    ->Range(1, 256)
    ->Unit(benchmark::kNanosecond);

  static void MutexEventPoolSynchronize_GuardedAtomic(benchmark::State& state)
  {
    SyncResources resources{};
    MutexEventPool pool{};

    runPooledEventCallsGuardedAtomic(state, pool, resources.stream());
  }

  BENCHMARK(MutexEventPoolSynchronize_GuardedAtomic)
    ->RangeMultiplier(2)
    ->Range(1, 256)
    ->Unit(benchmark::kNanosecond);

  static void EventPool_AcquireReleaseOnly(benchmark::State& state)
  {
    runPooledAcquireReleaseOnly<EventPool>(state);
  }

  {
    auto* benchmark = BENCHMARK(EventPool_AcquireReleaseOnly)->Unit(benchmark::kNanosecond);
    addCallSiteAndPoolSizeArgs(benchmark);
  }

  static void MutexEventPool_AcquireReleaseOnly(benchmark::State& state)
  {
    runPooledAcquireReleaseOnly<MutexEventPool>(state);
  }

  {
    auto* benchmark = BENCHMARK(MutexEventPool_AcquireReleaseOnly)->Unit(benchmark::kNanosecond);
    addCallSiteAndPoolSizeArgs(benchmark);
  }

  static void EventPool_RecordOnlySameStream(benchmark::State& state)
  {
    SyncResources resources{};
    runPooledRecordOnlySameStream<EventPool>(state, resources.stream());
  }

  {
    auto* benchmark = BENCHMARK(EventPool_RecordOnlySameStream)->Unit(benchmark::kNanosecond);
    addCallSiteAndPoolSizeArgs(benchmark);
  }

  static void MutexEventPool_RecordOnlySameStream(benchmark::State& state)
  {
    SyncResources resources{};
    runPooledRecordOnlySameStream<MutexEventPool>(state, resources.stream());
  }

  {
    auto* benchmark = BENCHMARK(MutexEventPool_RecordOnlySameStream)->Unit(benchmark::kNanosecond);
    addCallSiteAndPoolSizeArgs(benchmark);
  }

  static void EventPool_RecordOneSynchronizeAll_SharedToken(benchmark::State& state)
  {
    SyncResources resources{};
    runRecordOneSynchronizeAllSharedToken<EventPool>(state, resources.stream());
  }

  {
    auto* benchmark =
        BENCHMARK(EventPool_RecordOneSynchronizeAll_SharedToken)->Unit(benchmark::kNanosecond);
    addCallSiteAndPoolSizeArgs(benchmark);
  }

  static void EventPool_RecordAllSynchronizeOne_DistinctTokens(benchmark::State& state)
  {
    SyncResources resources{};
    runRecordAllSynchronizeOneDistinctTokens<EventPool>(state, resources.stream());
  }

  {
    auto* benchmark =
        BENCHMARK(EventPool_RecordAllSynchronizeOne_DistinctTokens)->Unit(benchmark::kNanosecond);
    addCallSiteAndPoolSizeArgs(benchmark);
  }

  static void EventPool_RecordAllSynchronizeAll_DistinctTokens(benchmark::State& state)
  {
    SyncResources resources{};
    runRecordAllSynchronizeAllDistinctTokens<EventPool>(state, resources.stream());
  }

  {
    auto* benchmark =
        BENCHMARK(EventPool_RecordAllSynchronizeAll_DistinctTokens)->Unit(benchmark::kNanosecond);
    addCallSiteAndPoolSizeArgs(benchmark);
  }

  static void EventPool_SameStreamRefreshReplaceToken(benchmark::State& state)
  {
    SyncResources resources{};
    runSameStreamRefreshReplaceToken<EventPool>(state, resources.stream());
  }

  {
    auto* benchmark =
        BENCHMARK(EventPool_SameStreamRefreshReplaceToken)->Unit(benchmark::kNanosecond);
    addCallSiteAndPoolSizeArgs(benchmark);
  }
}  // namespace

BENCHMARK_MAIN();
