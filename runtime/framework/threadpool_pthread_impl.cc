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

#include <errno.h>
#include <pthread.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <deque>
#include <functional>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/framework/thread_options.h"
#include "runtime/framework/threadpool.h"

namespace litert::lm {
namespace {

// Create a thread name from the given prefix and thread id.
// - thread_id is not portable
// - the 16-byte limit is Linux-specific
// - the std::thread implementation has a copy of this but doesn't use it
// - why do we even need the thread id in the name? any thread list should show
//   the id too.
std::string CreateThreadName(const std::string& prefix, int thread_id) {
  std::string name = absl::StrCat(prefix, "/", thread_id);
  // 16 is the limit allowed by `pthread_setname_np`, including
  // the terminating null byte ('\0')
  constexpr size_t kMaxThreadNameLength = 15;
  name.resize(std::min(name.length(), kMaxThreadNameLength));
  return name;
}

}  // namespace

class ThreadPoolPthreadImpl : public ThreadPool {
 public:
  explicit ThreadPoolPthreadImpl(int num_threads) {
    num_threads_ = (num_threads == 0) ? 1 : num_threads;
  }

  ThreadPoolPthreadImpl(const std::string& name_prefix, int num_threads)
      : name_prefix_(name_prefix) {
    num_threads_ = (num_threads == 0) ? 1 : num_threads;
  }

  ThreadPoolPthreadImpl(const ThreadOptions& thread_options,
                        const std::string& name_prefix, int num_threads)
      : name_prefix_(name_prefix), thread_options_(thread_options) {
    num_threads_ = (num_threads == 0) ? 1 : num_threads;
    ABSL_LOG(INFO) << "ThreadPoolPthreadImpl: Created with " << num_threads_
                   << " threads.";
  }

  ~ThreadPoolPthreadImpl() override;

  void StartWorkers() override;

  void Schedule(std::function<void()> callback) override;

  absl::Status WaitUntilIdle(absl::Duration timeout) override;

  absl::Status WaitUntilDone(absl::Duration timeout) override;

  int num_threads() const override { return num_threads_; }

  const ThreadOptions& thread_options() const override {
    return thread_options_;
  }

 private:
  void RunWorker();

  class WorkerThread;
  friend ThreadPoolPthreadImpl::WorkerThread;

  std::string name_prefix_;
  std::vector<std::unique_ptr<WorkerThread>> threads_;
  int num_threads_;

  absl::Mutex mutex_;
  absl::CondVar condition_;
  // Whether the pool is stopped.
  bool stopped_ ABSL_GUARDED_BY(mutex_) = false;
  // The tasks are stored in a queue using the Schedule() method and will be
  // executed by the threads.
  std::deque<std::function<void()>> tasks_ ABSL_GUARDED_BY(mutex_);
  // Count the number of active tasks that are being executed by the threads.
  int active_tasks_ ABSL_GUARDED_BY(mutex_) = 0;

  ThreadOptions thread_options_;
};

class ThreadPoolPthreadImpl::WorkerThread {
 public:
  // Creates and starts a thread that runs pool->RunWorker().
  WorkerThread(ThreadPoolPthreadImpl* pool, const std::string& name_prefix);

  // REQUIRES: Join() must have been called.
  ~WorkerThread();

  // Joins with the running thread.
  void Join();

 private:
  static void* ThreadBody(void* arg);

  ThreadPoolPthreadImpl* pool_;
  const std::string name_prefix_;
  pthread_t thread_;
  // Track if this thread is joined.
  bool joined_;
};

ThreadPoolPthreadImpl::WorkerThread::WorkerThread(
    ThreadPoolPthreadImpl* pool, const std::string& name_prefix)
    : pool_(pool), name_prefix_(name_prefix), joined_(false) {
  int res = pthread_create(&thread_, nullptr, ThreadBody, this);
  ABSL_CHECK_EQ(res, 0) << "pthread_create failed";
}

ThreadPoolPthreadImpl::WorkerThread::~WorkerThread() {
  if (!joined_) {
    ABSL_LOG(WARNING)
        << "WorkerThread for pool " << name_prefix_
        << " destroyed without Join(). Potential resource leak or race.";
  }
}

void ThreadPoolPthreadImpl::WorkerThread::Join() {
  if (!joined_) {
    int res = pthread_join(thread_, nullptr);
    if (res != 0) {
      ABSL_LOG(ERROR) << "pthread_join failed for pool " << name_prefix_ << ": "
                      << strerror(res);
    }
    joined_ = true;
  }
}

void* ThreadPoolPthreadImpl::WorkerThread::ThreadBody(void* arg) {
  auto thread = reinterpret_cast<WorkerThread*>(arg);
  int nice_priority_level =
      thread->pool_->thread_options().nice_priority_level();
  const std::set<int> selected_cpus = thread->pool_->thread_options().cpu_set();
#if defined(__linux__)
  const std::string name =
      CreateThreadName(thread->name_prefix_, syscall(SYS_gettid));
  if (nice_priority_level != 0) {
    if (nice(nice_priority_level) != -1 || errno == 0) {
      ABSL_VLOG(1) << "Changed the nice priority level by "
                   << nice_priority_level;
    } else {
      ABSL_LOG(ERROR) << "Error : " << strerror(errno) << std::endl
                      << "Could not change the nice priority level by "
                      << nice_priority_level;
    }
  }
  if (!selected_cpus.empty()) {
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    for (const int cpu : selected_cpus) {
      CPU_SET(cpu, &cpu_set);
    }
    if (sched_setaffinity(syscall(SYS_gettid), sizeof(cpu_set_t), &cpu_set) !=
            -1 ||
        errno == 0) {
      ABSL_VLOG(1) << "Pinned the thread pool executor to processor "
                   << absl::StrJoin(selected_cpus, ", processor ") << ".";
    } else {
      ABSL_LOG(ERROR) << "Error : " << strerror(errno) << std::endl
                      << "Failed to set processor affinity. Ignore processor "
                         "affinity setting for now.";
    }
  }
  int error = pthread_setname_np(pthread_self(), name.c_str());
  if (error != 0) {
    ABSL_LOG(ERROR) << "Error : " << strerror(error) << std::endl
                    << "Failed to set name for thread: " << name;
  }
#else
  const std::string name = CreateThreadName(thread->name_prefix_, 0);
  if (nice_priority_level != 0 || !selected_cpus.empty()) {
    ABSL_LOG(ERROR) << "Thread priority and processor affinity feature aren't "
                       "supported on the current platform.";
  }
#if __APPLE__
  int error = pthread_setname_np(name.c_str());
  if (error != 0) {
    ABSL_LOG(ERROR) << "Error : " << strerror(error) << std::endl
                    << "Failed to set name for thread: " << name;
  }
#endif  // __APPLE__
#endif  // __linux__
  thread->pool_->RunWorker();
  return nullptr;
}

ThreadPoolPthreadImpl::~ThreadPoolPthreadImpl() {
  ABSL_VLOG(2) << "ThreadPoolPthreadImpl '" << name_prefix_
               << "': Shutting down...";
  {
    absl::MutexLock lock(&mutex_);
    stopped_ = true;
    // Wake up all workers
    condition_.SignalAll();
  }

  for (auto& thread_ptr : threads_) {
    // Check if pointer is valid before calling Join.
    if (thread_ptr) {
      // Wait for each worker thread to finish.
      thread_ptr->Join();
    }
  }
  threads_.clear();

  // Log active_tasks_ state at shutdown. Should be 0 if all workers exited
  // cleanly.
  int final_active_tasks = 0;
  {
    absl::MutexLock lock(&mutex_);
    final_active_tasks = active_tasks_;
  }
  ABSL_VLOG(2) << "ThreadPoolPthreadImpl '" << name_prefix_
               << "': Shutdown complete. " << final_active_tasks
               << " active tasks recorded at the end (should ideally be 0).";
}

void ThreadPoolPthreadImpl::StartWorkers() {
  absl::MutexLock lock(&mutex_);
  if (!threads_.empty() || stopped_) {
    ABSL_LOG(WARNING)
        << "ThreadPoolPthreadImpl '" << name_prefix_
        << "': StartWorkers called on an already started or stopped pool.";
    return;
  }
  threads_.reserve(num_threads_);
  for (int i = 0; i < num_threads_; ++i) {
    threads_.push_back(std::make_unique<WorkerThread>(this, name_prefix_));
  }
  ABSL_VLOG(2) << "ThreadPoolPthreadImpl '" << name_prefix_ << "': Started "
               << num_threads_ << " workers.";
}

void ThreadPoolPthreadImpl::Schedule(std::function<void()> callback) {
  absl::MutexLock lock(&mutex_);
  if (stopped_) {
    ABSL_LOG(WARNING) << "ThreadPoolPthreadImpl '" << name_prefix_
                      << "': Schedule called on a stopped pool. Task for pool '"
                      << name_prefix_ << "' ignored.";
    return;
  }
  tasks_.push_back(std::move(callback));
  condition_.Signal();
}

absl::Status ThreadPoolPthreadImpl::WaitUntilIdle(absl::Duration timeout) {
  absl::MutexLock lock(&mutex_);
  absl::Time deadline = absl::Now() + timeout;
  while (!tasks_.empty()) {
    // Check if the current time has passed the deadline
    if (absl::Now() >= deadline) {
      // Re-check the condition one last time
      if (!tasks_.empty()) {
        return absl::DeadlineExceededError(absl::StrCat(
            "Timeout waiting for task queue to become idle in pool '",
            name_prefix_, "'. Tasks still in queue: ", tasks_.size()));
      } else {
        // Queue became empty just around or after the deadline check.
        return absl::OkStatus();
      }
    }
    // Wait until signaled OR the deadline is reached.
    condition_.WaitWithDeadline(&mutex_, deadline);
  }
  return absl::OkStatus();
}

absl::Status ThreadPoolPthreadImpl::WaitUntilDone(absl::Duration timeout) {
  absl::MutexLock lock(&mutex_);
  absl::Time deadline = absl::Now() + timeout;
  while (!tasks_.empty() || active_tasks_ > 0) {
    if (absl::Now() >= deadline) {
      // Check the condition one last time.
      if (!tasks_.empty() || active_tasks_ > 0) {
        return absl::DeadlineExceededError(
            absl::StrCat("Timeout waiting for all tasks to be done in pool '",
                         name_prefix_, "'. Tasks in queue: ", tasks_.size(),
                         ", Active tasks: ", active_tasks_));
      } else {
        // Condition was met just around the deadline.
        return absl::OkStatus();
      }
    }
    // WaitUntil will wait at most until the deadline.
    condition_.WaitWithDeadline(&mutex_, deadline);
  }
  return absl::OkStatus();
}

void ThreadPoolPthreadImpl::RunWorker() {
  mutex_.Lock();
  while (true) {
    while (tasks_.empty() && !stopped_) {
      condition_.Wait(&mutex_);
    }

    if (stopped_ && tasks_.empty()) {
      break;
    }

    // Should only be reachable if !stopped_ is false but
    // tasks got emptied by another thread
    if (tasks_.empty()) {
      // Re-evaluate conditions.
      continue;
    }

    std::function<void()> task_to_run = std::move(tasks_.front());
    tasks_.pop_front();
    active_tasks_++;

    // Signal if queue became empty. This is for WaitUntilIdle.
    if (tasks_.empty()) {
      condition_.SignalAll();
    }
    // Release mutex before executing task.
    mutex_.Unlock();

    // Execute the task.
    task_to_run();

    // Task finished. Re-acquire mutex and decrement active tasks.
    mutex_.Lock();
    active_tasks_--;

    // Signal if truly idle (no queued & no active tasks). This is for
    // WaitUntilDone.
    if ((tasks_.empty() && active_tasks_ == 0) ||
        (stopped_ && active_tasks_ == 0 && tasks_.empty())) {
      condition_.SignalAll();
    }
  }
  mutex_.Unlock();
}

absl::StatusOr<std::unique_ptr<ThreadPool>> ThreadPool::CreateThreadPool(
    const ThreadOptions& thread_options, const std::string& name_prefix,
    int num_threads) {
  if (num_threads < 0) {
    return absl::InvalidArgumentError(
        "Number of threads cannot be negative for ThreadPool creation.");
  }
  return std::make_unique<ThreadPoolPthreadImpl>(thread_options, name_prefix,
                                                 num_threads);
}

}  // namespace litert::lm
