#pragma once

#include <Ziran/CS/Util/Debug.h>
#include <SPGrid/Core/SPGrid_Allocator.h>

namespace ZIRAN {

template <class T, int dim>
class SmallGrid {
    static constexpr int log2_page = 12;
    static constexpr int spgrid_size = 4096;
    using SparseGrid = SPGrid_Allocator<T, dim, log2_page>;
    using SparseMask = typename SparseGrid::template Array_type<>::MASK;
    std::unique_ptr<SparseGrid> grid;

public:
    SmallGrid()
    {
        ZIRAN_ASSERT(is_power_of_two(sizeof(T)), "Type size must be POT");
        if constexpr (dim == 2) {
            grid = std::make_unique<SparseGrid>(spgrid_size, spgrid_size);
        }
        else {
            grid = std::make_unique<SparseGrid>(spgrid_size, spgrid_size, spgrid_size);
        }
    }

    T& operator[](const Vector<int, dim>& v)
    {
        return grid->Get_Array()(to_std_array(v));
    }

    const T& operator[](const Vector<int, dim>& v) const
    {
        return grid->Get_Array()(to_std_array(v));
    }
};

template <typename OP>
void iterateRegion(const Vector<int, 2>& region, const OP& target)
{
    for (int i = 0; i < region[0]; ++i)
        for (int j = 0; j < region[1]; ++j)
            target(Vector<int, 2>(i, j));
}

template <typename OP>
void iterateRegion(const Vector<int, 3>& region, const OP& target)
{
    for (int i = 0; i < region[0]; ++i)
        for (int j = 0; j < region[1]; ++j)
            for (int k = 0; k < region[2]; ++k)
                target(Vector<int, 3>(i, j, k));
}

} // namespace ZIRAN