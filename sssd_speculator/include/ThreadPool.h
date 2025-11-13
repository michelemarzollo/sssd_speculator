/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 */

/*
Based on progshj Threadpool https://github.com/progschj/ThreadPool, it adds the feature of
having tasks with different priorities:
- if there are available tasks for both priorities, the ones with higher priority are scheduled.
- if there aren't high priority tasks, at most half of the threads are used for low priority
tasks, in case high priority tasks would come.

The following is the original license (https://github.com/progschj/ThreadPool/blob/master/COPYING):

---
Copyright (c) 2012 Jakob Progsch, Václav Zeman

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

   1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.

   2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.

   3. This notice may not be removed or altered from any source
   distribution.
---
*/
#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <atomic>
#include <utils.hpp>

class ThreadPool {
public:
    explicit ThreadPool(size_t);

    // Modified enqueue to accept integer priority
    template <class F, class... Args>
    auto Enqueue(int priority, F &&f, Args &&...args) -> std::future<typename std::result_of<F(Args...)>::type>;

    ~ThreadPool();

private:
    // Keep track of threads so we can join them
    std::vector<std::thread> workers;

    // Task queues for different priorities
    std::queue<std::function<void()> > high_priority_tasks;
    std::queue<std::function<void()> > low_priority_tasks;

    // Synchronization
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;

    // Concurrency control for low priority tasks
    size_t maxLowPriorityConcurrency;
    std::atomic<size_t> currentLowPriorityTasks;
};

// Constructor launches worker threads
inline ThreadPool::ThreadPool(size_t threads) : stop(false), currentLowPriorityTasks(0)
{
    if (threads == 0) {
        throw std::invalid_argument("ThreadPool must have at least one thread.");
    }

    maxLowPriorityConcurrency = threads / 2;
    if (maxLowPriorityConcurrency == 0 && threads > 0) {
        maxLowPriorityConcurrency = 1;
    }

    for (size_t i = 0; i < threads; ++i)
        // capture i so (if desired later) you can make per-worker choices
        workers.emplace_back([this, i] {
            // --- One-time NUMA/affinity init for this worker thread ---
            auto logger = GetLogger();
            // Constrain scheduling of this thread to the NUMA node it started on (no-op if non-NUMA)
            PinThreadToLocalNUMANode(logger);
            // Prefer memory from that node (affects subsequent allocations in this thread)
            PreferLocalNodeMemory(logger);
            // --- End one-time init ---

            while (true) {
                std::function<void()> task;
                bool is_low_priority = false;

                {
                    std::unique_lock<std::mutex> lock(this->queueMutex);
                    this->condition.wait(lock, [this] {
                        return this->stop || !this->high_priority_tasks.empty() ||
                               (!this->low_priority_tasks.empty() &&
                                this->currentLowPriorityTasks.load() < this->maxLowPriorityConcurrency);
                    });

                    if (this->stop && this->high_priority_tasks.empty() && this->low_priority_tasks.empty())
                        return;

                    if (!this->high_priority_tasks.empty()) {
                        task = std::move(this->high_priority_tasks.front());
                        this->high_priority_tasks.pop();
                    } else if (!this->low_priority_tasks.empty() &&
                               this->currentLowPriorityTasks.load() < this->maxLowPriorityConcurrency) {
                        task = std::move(this->low_priority_tasks.front());
                        this->low_priority_tasks.pop();
                        is_low_priority = true;
                        this->currentLowPriorityTasks.fetch_add(1, std::memory_order_relaxed);
                    }
                }

                if (task) {
                    try {
                        task();
                    } catch (const std::exception &e) {
                        // Handle exceptions if necessary
                        // For now, we'll just ignore them
                    }

                    if (is_low_priority) {
                        this->currentLowPriorityTasks.fetch_sub(1, std::memory_order_relaxed);
                        this->condition.notify_all();
                    }
                }
            }
        });
}

// Modified enqueue method with integer priority
template <class F, class... Args>
auto ThreadPool::Enqueue(int priority, F &&f, Args &&...args) -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;

    // Create a packaged task
    auto task = std::make_shared<std::packaged_task<return_type()> >(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queueMutex);

        if (stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        // Enqueue based on priority
        if (priority == PRIORITY_HIGH) {
            high_priority_tasks.emplace([task]() { (*task)(); });
        } else if (priority == PRIORITY_LOW) {
            low_priority_tasks.emplace([task]() { (*task)(); });
        } else {
            throw std::invalid_argument("Invalid priority level. Use 0 for Low or 1 for High.");
        }
    }
    condition.notify_one();
    return res;
}

// Destructor joins all threads
inline ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread &worker : workers)
        worker.join();
}

#endif
