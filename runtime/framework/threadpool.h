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

#ifndef THIRD_PARTY_LITERT_LM_RUNTIME_FRAMEWORK_THREADPOOL_H_
#define THIRD_PARTY_LITERT_LM_RUNTIME_FRAMEWORK_THREADPOOL_H_

#include <functional>
#include <memory>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/framework/thread_options.h"

namespace litert::lm {

// A thread pool consists of a set of threads that sit around waiting
// for callbacks to appear on a queue.  When that happens, one of the
// threads pulls a callback off the queue and runs it.
//
// The thread pool is shut down when the pool is destroyed.
//
// Sample usage:
//
// {
//   ThreadPool pool("testpool", num_workers);
//   pool.StartWorkers();
//   for (int i = 0; i < N; ++i) {
//     pool.Schedule([i]() { DoWork(i); });
//   }
// }
//
class ThreadPool {
 public:
  // Create a thread pool that creates and can use up to "num_threads"
  // threads.  Any standard thread options, such as stack size, should
  // be passed via "thread_options".  "name_prefix" specifies the
  // thread name prefix.
  static absl::StatusOr<std::unique_ptr<ThreadPool>> CreateThreadPool(
      const ThreadOptions& thread_options, const std::string& name_prefix,
      int num_threads);

  // Waits for closures (if any) to complete. May be called without
  // having called StartWorkers().
  virtual ~ThreadPool() = default;

  // REQUIRES: StartWorkers has not been called
  // Actually start the worker threads.
  virtual void StartWorkers() = 0;

  // REQUIRES: StartWorkers has been called
  // Add specified callback to queue of pending callbacks.  Eventually a
  // thread will pull this callback off the queue and execute it. Note that
  // this does not guarantee that the callback is executed in the order it was
  // scheduled.
  virtual void Schedule(std::function<void()> callback) = 0;

  // Waits until the task queue is empty. The function will return an error if
  // the timeout is reached before the task queue is empty.
  // Note that this only indicates that there are no pending callbacks in the
  // queue, and does not guarantee that all scheduled callbacks have finished
  // executing. This is helpful for the caller to get a sense about the status
  // of the pool, but should not be used for synchronization.
  virtual absl::Status WaitUntilIdle(absl::Duration timeout) = 0;

  // Waits until all the scheduled callbacks are executed and finished. The
  // function will return an error if the timeout is reached before all the
  // callbacks are finished.
  virtual absl::Status WaitUntilDone(absl::Duration timeout) = 0;

  // Provided for debugging and testing only.
  // The number of threads in the pool.
  virtual int num_threads() const = 0;

  // Standard thread options.  Use this accessor to get them.
  virtual const ThreadOptions& thread_options() const = 0;

 private:
  // The number of threads in the pool.
  int num_threads_;

  // Thread options.
  ThreadOptions thread_options_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_LITERT_LM_RUNTIME_FRAMEWORK_THREADPOOL_H_
