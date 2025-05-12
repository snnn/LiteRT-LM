// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/framework/threadpool.h"

#include <set>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/framework/thread_options.h"

namespace litert::lm {
namespace {

TEST(ThreadPoolTest, DestroyWithoutStart) {
  auto thread_pool_ptr =
      ThreadPool::CreateThreadPool(ThreadOptions(), "testpool", 10);
  EXPECT_OK(thread_pool_ptr);
}

TEST(ThreadPoolTest, EmptyThread) {
  auto thread_pool_ptr =
      ThreadPool::CreateThreadPool(ThreadOptions(), "testpool", 0);
  EXPECT_OK(thread_pool_ptr);
  ThreadPool& thread_pool = **thread_pool_ptr;
  EXPECT_EQ(1, thread_pool.num_threads());
  thread_pool.StartWorkers();
}

TEST(ThreadPoolTest, SingleThread) {
  absl::Mutex mu;
  int n = 100;
  {
    auto thread_pool_ptr =
        ThreadPool::CreateThreadPool(ThreadOptions(), "testpool", 1);
    EXPECT_OK(thread_pool_ptr);
    ThreadPool& thread_pool = **thread_pool_ptr;
    EXPECT_EQ(1, thread_pool.num_threads());
    thread_pool.StartWorkers();

    for (int i = 0; i < 100; ++i) {
      thread_pool.Schedule([&n, &mu]() mutable {
        absl::MutexLock l(&mu);
        --n;
      });
    }
  }

  EXPECT_EQ(0, n);
}

TEST(ThreadPoolTest, MultiThreads) {
  absl::Mutex mu;
  int n = 100;
  {
    auto thread_pool_ptr =
        ThreadPool::CreateThreadPool(ThreadOptions(), "testpool", 10);
    EXPECT_OK(thread_pool_ptr);
    ThreadPool& thread_pool = **thread_pool_ptr;
    ASSERT_EQ(10, thread_pool.num_threads());
    thread_pool.StartWorkers();

    for (int i = 0; i < 100; ++i) {
      thread_pool.Schedule([&n, &mu]() mutable {
        absl::MutexLock l(&mu);
        --n;
      });
    }
  }

  EXPECT_EQ(0, n);
}

TEST(ThreadPoolTest, CreateWithThreadOptions) {
  auto thread_pool_ptr =
      ThreadPool::CreateThreadPool(ThreadOptions(), "testpool", 10);
  EXPECT_OK(thread_pool_ptr);
  ThreadPool& thread_pool = **thread_pool_ptr;
  ASSERT_EQ(10, thread_pool.num_threads());
  thread_pool.StartWorkers();
}

TEST(ThreadPoolTest, CreateWithThreadPriority) {
  ThreadOptions thread_options = ThreadOptions().set_nice_priority_level(-10);

  auto thread_pool_ptr =
      ThreadPool::CreateThreadPool(thread_options, "testpool", 10);
  EXPECT_OK(thread_pool_ptr);
  ThreadPool& thread_pool = **thread_pool_ptr;

  EXPECT_EQ(10, thread_pool.num_threads());
  EXPECT_EQ(-10, thread_pool.thread_options().nice_priority_level());
  thread_pool.StartWorkers();
}

TEST(ThreadPoolTest, CreateWithCPUAffinity) {
  ThreadOptions thread_options = ThreadOptions().set_cpu_set({0});

  auto thread_pool_ptr =
      ThreadPool::CreateThreadPool(thread_options, "testpool", 10);
  EXPECT_OK(thread_pool_ptr);
  ThreadPool& thread_pool = **thread_pool_ptr;
  ASSERT_EQ(10, thread_pool.num_threads());
  ASSERT_EQ(1, thread_pool.thread_options().cpu_set().size());
  thread_pool.StartWorkers();
}

TEST(ThreadPoolTest, WaitUntilIdle) {
  absl::Mutex mu;
  std::vector<int> v;

  auto thread_pool_ptr =
      ThreadPool::CreateThreadPool(ThreadOptions(), "testpool", 1);
  EXPECT_OK(thread_pool_ptr);
  ThreadPool& thread_pool = **thread_pool_ptr;
  EXPECT_EQ(1, thread_pool.num_threads());
  thread_pool.StartWorkers();

  for (int i = 0; i < 10; ++i) {
    thread_pool.Schedule([&v, &mu, i]() mutable {
      absl::MutexLock l(&mu);
      // Simulate a task that takes some time to execute.
      absl::SleepFor(absl::Milliseconds(500));
      v.push_back(i);
    });
  }
  EXPECT_OK(thread_pool.WaitUntilIdle(absl::Seconds(50)));
  // WaitUntilIdle() should wait until the task queue is empty. Note that when
  // the function returns, the last task is still being executed so the vector
  // will only have 9 elements instead of 10.
  EXPECT_THAT(v, testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8));
}

TEST(ThreadPoolTest, WaitUntilDone) {
  absl::Mutex mu;
  std::vector<int> v;

  auto thread_pool_ptr =
      ThreadPool::CreateThreadPool(ThreadOptions(), "testpool", 1);
  EXPECT_OK(thread_pool_ptr);
  ThreadPool& thread_pool = **thread_pool_ptr;
  EXPECT_EQ(1, thread_pool.num_threads());
  thread_pool.StartWorkers();

  for (int i = 0; i < 10; ++i) {
    thread_pool.Schedule([&v, &mu, i]() mutable {
      absl::MutexLock l(&mu);
      // Simulate a task that takes some time to execute.
      absl::SleepFor(absl::Milliseconds(500));
      v.push_back(i);
    });
  }
  EXPECT_OK(thread_pool.WaitUntilDone(absl::Seconds(50)));
  // Even without destroying the thread pool and force all threads to join,
  // calling WaitUntilDone() should be enough to ensure that all of the
  // scheduled tasks are executed.
  EXPECT_THAT(v, testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
}

}  // namespace
}  // namespace litert::lm
