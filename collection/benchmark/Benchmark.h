#pragma once

#include <functional>
#include <string>
#include <vector>

struct TimeRecord
{
public:
    TimeRecord() = default;
    TimeRecord(double realTime, double cudaTime);

    TimeRecord operator+(const TimeRecord& other);
    TimeRecord operator/(double value);
    void operator+=(const TimeRecord& other);

    double realTime = 0.0;
    double cudaTime = 0.0;
};

class Benchmark
{
public:
    explicit Benchmark(std::string name);

    void measure(const std::function<void()>& body, size_t times);
    double average() const;

    void print();

private:
    std::string name;
    std::vector<double> times;
};
