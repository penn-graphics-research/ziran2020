#pragma once

#include "SmallGrid.h"
#include <MPM/MpmSimulationBase.h>

namespace ZIRAN {

template <class T, int dim>
class ParticleLevelSet {
    typedef Vector<T, dim> TV;
    typedef Vector<int, dim> IV;
    typedef Matrix<T, dim, dim> TM;

    MpmGrid<T, dim, 1> grid;
    T dx, radius;

public:
    ParticleLevelSet(T dx, T radius)
        : dx(dx)
        , radius(radius)
    {
    }

    void rebuild(const StdVector<TV>& positions)
    {
        int layer = std::ceil(radius / dx) + 1;
        StdVector<uint64_t> particle_base_offset;
        grid.pollute(positions, particle_base_offset, dx, -1, 1);
        for (auto& Xp : positions) {
            BSplineWeights<T, dim, 1> spline(Xp, dx);
            IV region = IV::Ones() * (layer * 2 + 1);
            iterateRegion(region, [&](const IV& offset) {
                IV neighbor_node = spline.base_node + offset - IV::Ones() * layer;
                TV neighbor_pos = neighbor_node.template cast<T>() * dx;
                T signed_distance = (neighbor_pos - Xp).norm() - radius;
                if (signed_distance > 0) {
                    if (grid[neighbor_node] == 0) {
                        grid[neighbor_node] = signed_distance;
                    }
                    else {
                        grid[neighbor_node] = std::min(grid[neighbor_node], signed_distance);
                    }
                }
                else {
                    grid[neighbor_node] = -100;
                }
            });
        }
    }

    void
};

}; // namespace ZIRAN
