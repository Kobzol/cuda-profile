#include "format-pool.h"

FormatPool::FormatPool()
{
    int threadCount = std::thread::hardware_concurrency();
    this->running = true;
    for (int i = 0; i < threadCount; i++)
    {
        this->threads.emplace_back(&FormatPool::workerBody, this);
        this->threads.back().detach();
    }
}

void FormatPool::workerBody()
{
    while (this->running)
    {
        Job job;
        if (this->pop(job))
        {
            job();
        }
        else std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    while (!this->empty())
    {
        Job job;
        if (this->pop(job))
        {
            job();
        }
    }
}

void FormatPool::enqueue(const Job& job)
{
    std::lock_guard<decltype(this->mutex)> guard(this->mutex);
    this->jobs.push(job);
}

void FormatPool::stop()
{
    this->running = false;
}

bool FormatPool::pop(Job& job)
{
    std::lock_guard<decltype(this->mutex)> guard(this->mutex);
    if (this->jobs.empty()) return false;

    job = this->jobs.front();
    this->jobs.pop();
    return true;
}

bool FormatPool::empty()
{
    std::lock_guard<decltype(this->mutex)> guard(this->mutex);
    return this->jobs.empty();
}
