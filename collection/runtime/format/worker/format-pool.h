#pragma once

#include <atomic>
#include <thread>
#include <vector>
#include <mutex>
#include <queue>

using Job = std::function<void()>;

class FormatPool
{
public:
    FormatPool();

    void workerBody();
    void enqueue(const Job& job);
    void stop();

private:
    bool pop(Job& job);
    bool empty();

    std::mutex mutex;
    std::queue<Job> jobs;
    std::vector<std::thread> threads;
    std::atomic<bool> running{false};
};
