#pragma once

#include "SmallGrid.h"
#include <MPM/MpmSimulationBase.h>
#include "UMGrid.h"

namespace ZIRAN {

template <class T, int dim>
class LegoMarker {
    typedef Vector<T, dim> TV;
    typedef Vector<int, dim> IV;
    typedef Matrix<T, dim, dim> TM;

public:
    // 0 air
    // 1 fluid
    // 2 solid
    UMGrid<int, dim> cell_type;

    int total_id = 0;
    UMGrid<int, dim> face_ID;

    StdVector<IV> fluid_dof;
    StdVector<TV> volume_GQ;
    StdVector<T> volume_mass;
    StdVector<TV> volume_v;
    StdVector<TM> volume_C;
    StdVector<TV> face_GQ;
    StdVector<TV> face_N;

    void update(MpmSimulationBase<T, dim>& sim)
    {
        // reset cell_type
        cell_type.clear();
        for (int i = 0; i < sim.particles.count; ++i) {
            TV Xp = sim.particles.X.array[i];
            BSplineWeights<T, dim, 1> spline(Xp - TV::Ones() * 0.5 * sim.dx, sim.dx);
            IV region = IV::Ones() * 3;
            iterateRegion(region, [&](const IV& offset) {
                IV cell = spline.base_node + offset - IV::Ones();
                cell_type[cell] = 0;
            });
        }
        // mark fluid in cell_type
        for (int i = 0; i < sim.particles.count; ++i) {
            TV Xp = sim.particles.X.array[i];
            BSplineWeights<T, dim, 1> spline(Xp - TV::Ones() * 0.5 * sim.dx, sim.dx);
            cell_type[spline.base_node] = 3;
        }
        // mark solid in cell_type
        for (int i = 0; i < sim.particles.count; ++i) {
            TV Xp = sim.particles.X.array[i];
            BSplineWeights<T, dim, 1> spline(Xp - TV::Ones() * 0.5 * sim.dx, sim.dx);
            IV region = IV::Ones() * 3;
            iterateRegion(region, [&](const IV& offset) {
                IV cell = spline.base_node + offset - IV::Ones();
                TV center_x = cell.template cast<T>() * sim.dx + TV::Ones() * sim.dx;
                TV center_v = TV::Zero();
                TM normal_basis;
                bool collided = AnalyticCollisionObject<T, dim>::
                    multiObjectCollision(sim.collision_objects, center_x, center_v, normal_basis);
                if (collided) cell_type[cell] = 2;
            });
        }
        // collect fluid cells
        fluid_dof.clear();
        for (int i = 0; i < sim.particles.count; ++i) {
            TV Xp = sim.particles.X.array[i];
            BSplineWeights<T, dim, 1> spline(Xp - TV::Ones() * 0.5 * sim.dx, sim.dx);
            if (cell_type[spline.base_node] == 3) {
                fluid_dof.push_back(spline.base_node);
                cell_type[spline.base_node] = 1;
            }
        }
    }

    void generateFluidVolumeGQ(MpmSimulationBase<T, dim>& sim)
    {
        volume_GQ.clear();
        for (auto& cell : fluid_dof) {
            IV region = IV::Ones() * 2;
            iterateRegion(region, [&](const IV& offset) {
                TV base_Xp = cell.template cast<T>() * sim.dx + TV::Ones() * (1 - std::sqrt(3) / 6) * sim.dx;
                TV Xp = base_Xp + offset.template cast<T>() * std::sqrt(3) / 3 * sim.dx;
                volume_GQ.push_back(Xp);
            });
        }
        if (!volume_GQ.empty()) {
            volume_mass.assign(volume_GQ.size(), sim.particles.mass.array[0] * sim.particles.count / volume_GQ.size());
        }
    }

    void generateSolidFaceGQ(MpmSimulationBase<T, dim>& sim)
    {
        face_GQ.clear();
        face_N.clear();
        face_ID.clear();
        total_id = 0;
        if constexpr (dim == 2) {
            for (auto& cell : fluid_dof) {
                if (cell_type[IV(cell[0], cell[1])] == 1) {
                    if (cell_type[IV(cell[0] - 1, cell[1])] == 2) {
                        T x = (T)cell[0] * sim.dx + 0.5 * sim.dx;
                        T y0 = (T)cell[1] * sim.dx + sim.dx - std::sqrt(3) / 6 * sim.dx;
                        T y1 = (T)cell[1] * sim.dx + sim.dx + std::sqrt(3) / 6 * sim.dx;
                        face_GQ.emplace_back(x, y0);
                        face_GQ.emplace_back(x, y1);
                        face_N.emplace_back(-1, 0);
                        face_N.emplace_back(-1, 0);
                        if (!face_ID.find(cell[0], cell[1])) face_ID[IV(cell[0], cell[1])] = ++total_id;
                        if (!face_ID.find(cell[0], cell[1] + 1)) face_ID[IV(cell[0], cell[1] + 1)] = ++total_id;
                    }
                    if (cell_type[IV(cell[0] + 1, cell[1])] == 2) {
                        T x = (T)cell[0] * sim.dx + 1.5 * sim.dx;
                        T y0 = (T)cell[1] * sim.dx + sim.dx - std::sqrt(3) / 6 * sim.dx;
                        T y1 = (T)cell[1] * sim.dx + sim.dx + std::sqrt(3) / 6 * sim.dx;
                        face_GQ.emplace_back(x, y0);
                        face_GQ.emplace_back(x, y1);
                        face_N.emplace_back(1, 0);
                        face_N.emplace_back(1, 0);
                        if (!face_ID.find(cell[0] + 1, cell[1])) face_ID[IV(cell[0] + 1, cell[1])] = ++total_id;
                        if (!face_ID.find(cell[0] + 1, cell[1] + 1)) face_ID[IV(cell[0] + 1, cell[1] + 1)] = ++total_id;
                    }
                    if (cell_type[IV(cell[0], cell[1] - 1)] == 2) {
                        T y = (T)cell[1] * sim.dx + 0.5 * sim.dx;
                        T x0 = (T)cell[0] * sim.dx + sim.dx - std::sqrt(3) / 6 * sim.dx;
                        T x1 = (T)cell[0] * sim.dx + sim.dx + std::sqrt(3) / 6 * sim.dx;
                        face_GQ.emplace_back(x0, y);
                        face_GQ.emplace_back(x1, y);
                        face_N.emplace_back(0, -1);
                        face_N.emplace_back(0, -1);
                        if (!face_ID.find(cell[0], cell[1])) face_ID[IV(cell[0], cell[1])] = ++total_id;
                        if (!face_ID.find(cell[0] + 1, cell[1])) face_ID[IV(cell[0] + 1, cell[1])] = ++total_id;
                    }
                    if (cell_type[IV(cell[0], cell[1] + 1)] == 2) {
                        T y = (T)cell[1] * sim.dx + 1.5 * sim.dx;
                        T x0 = (T)cell[0] * sim.dx + sim.dx - std::sqrt(3) / 6 * sim.dx;
                        T x1 = (T)cell[0] * sim.dx + sim.dx + std::sqrt(3) / 6 * sim.dx;
                        face_GQ.emplace_back(x0, y);
                        face_GQ.emplace_back(x1, y);
                        face_N.emplace_back(0, 1);
                        face_N.emplace_back(0, 1);
                        if (!face_ID.find(cell[0], cell[1] + 1)) face_ID[IV(cell[0], cell[1] + 1)] = ++total_id;
                        if (!face_ID.find(cell[0] + 1, cell[1] + 1)) face_ID[IV(cell[0] + 1, cell[1] + 1)] = ++total_id;
                    }
                }
            }
        }
    }

    void work(MpmSimulationBase<T, dim>& sim)
    {
        // update(sim);
        // generateFluidVolumeGQ(sim);
        // generateSolidFaceGQ(sim);
    }
};

} // namespace ZIRAN