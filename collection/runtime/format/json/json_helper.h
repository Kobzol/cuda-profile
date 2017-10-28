#pragma once

#include "picojson.h"

#include <vector>

template <typename T>
inline picojson::value jsonify(const std::vector<T>& items)
{
    std::vector<picojson::value> jsonified;
    for (auto& item: items)
    {
        jsonified.push_back(picojson::value(item));
    }

    return picojson::value(jsonified);
}
