#pragma once

#include <tbb/tbb.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ZIRAN {

template <class T, int dim>
class UMGrid {
    using ULL = unsigned long long;
    std::unordered_map<ULL, T> data;

public:
    UMGrid() {}

    bool find(int x, int y) const
    {
        ULL hsh = ((ULL)x << 20) | (ULL)y;
        return data.find(hsh) != data.end();
    }

    bool find(int x, int y, int z) const
    {
        ULL hsh = ((ULL)x << 40) | ((ULL)y << 20) | (ULL)z;
        return data.find(hsh) != data.end();
    }

    bool find(const Vector<int, dim>& v)
    {
        if constexpr (dim == 2) {
            int x = v[0], y = v[1];
            ULL hsh = ((ULL)x << 20) | (ULL)y;
            return data.find(hsh) != data.end();
        }
        else {
            int x = v[0], y = v[1], z = v[2];
            ULL hsh = ((ULL)x << 40) | ((ULL)y << 20) | (ULL)z;
            return data.find(hsh) != data.end();
        }
    }

    T& operator()(int x, int y)
    {
        ULL hsh = ((ULL)x << 20) | (ULL)y;
        return data[hsh];
    }

    T& operator()(int x, int y, int z)
    {
        ULL hsh = ((ULL)x << 40) | ((ULL)y << 20) | (ULL)z;
        return data[hsh];
    }

    // not using ZIRAN_ASSERT to optimize Release mode
    const T& operator()(int x, int y) const
    {
        ULL hsh = ((ULL)x << 20) | (ULL)y;
        assert(data.find(hsh) != data.end());
        return data.find(hsh)->second;
    }

    // not using ZIRAN_ASSERT to optimize Release mode
    const T& operator()(int x, int y, int z) const
    {
        ULL hsh = ((ULL)x << 40) | ((ULL)y << 20) | (ULL)z;
        assert(data.find(hsh) != data.end());
        return data.find(hsh)->second;
    }

    T& operator[](const Vector<int, dim>& v)
    {
        if constexpr (dim == 2) {
            int x = v[0], y = v[1];
            ULL hsh = ((ULL)x << 20) | (ULL)y;
            return data[hsh];
        }
        else {
            int x = v[0], y = v[1], z = v[2];
            ULL hsh = ((ULL)x << 40) | ((ULL)y << 20) | (ULL)z;
            return data[hsh];
        }
    }

    template <typename OP>
    void iterateGrid(const OP& target) const
    {
        if constexpr (dim == 2) {
            for (const auto& [key, value] : data) {
                int x = (int)(key >> 20);
                int y = (int)(key & ((1 << 20) - 1));
                target(x, y, value);
            }
        }
        else {
            for (const auto& [key, value] : data) {
                int x = (int)(key >> 40);
                int y = (int)((key >> 20) & ((1 << 20) - 1));
                int z = (int)(key & ((1 << 20) - 1));
                target(x, y, z, value);
            }
        }
    }

    std::vector<std::pair<ULL, T>> caches;
    template <typename OP>
    void iterateGridParallel(const OP& target) const
    {
        if (caches.empty()) {
            for (const auto& [key, value] : data)
                const_cast<std::vector<std::pair<ULL, T>>&>(caches).emplace_back(key, value);
        }

        tbb::parallel_for(0, (int)caches.size(), [&](int i) {
            ULL key = caches[i].first;
            T value = caches[i].second;
            if constexpr (dim == 2) {
                int x = (int)(key >> 20);
                int y = (int)(key & ((1 << 20) - 1));
                target(x, y, value);
            }
            else {
                int x = (int)(key >> 40);
                int y = (int)((key >> 20) & ((1 << 20) - 1));
                int z = (int)(key & ((1 << 20) - 1));
                target(x, y, z, value);
            }
        });
    }

    void clear()
    {
        data.clear();
        caches.clear();
    }
};

} // namespace ZIRAN
