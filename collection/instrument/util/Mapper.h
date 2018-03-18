#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>

template <typename T>
class Mapper
{
public:
    explicit Mapper(std::function<std::string(const T&)> mapping): mapping(mapping)
    {

    }

    uint64_t mapItem(const T& item)
    {
        std::string name = this->mapping(item);

        if (this->map.find(name) == this->map.end())
        {
            this->items.push_back(name);
            this->map.insert({name, this->map.size()});

            return this->map.size() - 1;
        }

        return this->map[name];
    }
    std::vector<std::string> getMappedItems()
    {
        return this->items;
    }

private:
    std::function<std::string(const T&)> mapping;

    std::vector<std::string> items;
    std::unordered_map<std::string, uint64_t> map;
};
