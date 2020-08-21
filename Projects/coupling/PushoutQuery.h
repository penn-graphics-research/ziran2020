#pragma once

#include "../splitting/SplittingSimulation.h"

namespace ZIRAN {

template <class T, int dim>
class SplittingSimulation;

namespace PUSHOUT_QUERIER {

template <class T, int dim>
void query(SplittingSimulation<T, dim>& solid_sim, SplittingSimulation<T, dim>& fluid_sim, StdVector<Vector<T, dim>>& dx, StdVector<Vector<T, dim>>& s_normal)
{
    using TV = Vector<T, dim>;
    using IV = Vector<int, dim>;

    auto& solid_particles = solid_sim.particles.X.array;
    auto& fluid_particles = fluid_sim.particles.X.array;
    SpatialHash<T, dim> hash_solid;
    T H_solid = 0.6 * solid_sim.dx;
    hash_solid.rebuild(H_solid, solid_particles);

    // calculate grid mass
    MpmGrid<T, dim, 3> subgrid;
    MpmGrid<T, dim, 1> lineargrid;
    StdVector<uint64_t> suboffsets;
    StdVector<uint64_t> linearsuboffsets;
    subgrid.pollute(solid_particles, suboffsets, solid_sim.dx);
    lineargrid.pollute(solid_particles, linearsuboffsets, solid_sim.dx);
    for (int i = 0; i < solid_particles.size(); ++i) {
        BSplineWeights<T, dim, 3> spline(solid_particles[i], solid_sim.dx);
        subgrid.iterateKernel(spline, suboffsets[i], [&](const IV& node, T w, const TV& dw, GridState<T, dim>& g) {
            g.m += w;
        });
    }
    StdVector<TV> solid_normals;
    solid_normals.resize(solid_particles.size());
    s_normal.resize(solid_particles.size());
    tbb::parallel_for(0, (int)solid_particles.size(), [&](int i) {
        TV normals = TV::Zero();
        BSplineWeights<T, dim, 3> splinetest(solid_particles[i], solid_sim.dx);
        subgrid.iterateKernel(splinetest, suboffsets[i], [&](IV node, T w, TV dw, GridState<T, dim>& g) {
            normals -= g.m * dw;
        });
        solid_normals[i] = normals.normalized();
        s_normal[i] = normals.normalized();
    });

    // query fluid particles
    dx.clear();
    dx.assign(fluid_particles.size(), TV::Zero());

    for (size_t i = 0; i < (int)fluid_particles.size(); ++i) {
        TV fluid_normal = TV::Zero();
        TV& pi = fluid_particles[i];
        StdVector<int> neighbors_solid;
        hash_solid.oneLayerNeighborsWithinRadius(pi + dx[i], solid_particles, H_solid, neighbors_solid);
        bool bad = false;
        for (auto id_solid : neighbors_solid) {
            TV n = solid_normals[id_solid];
            TV solid_pos = solid_particles[id_solid];
            TV fluid_pos = pi + dx[i];
            TV dir = fluid_pos - solid_pos;
            if (dir.dot(n) < 0) {
                bad = true;
                break;
            }
        }
        if (!bad) continue;

        // find fluid normal
        int closest_solid = -1;
        T closest_distance2 = 1000000;
        for (auto id_solid : neighbors_solid) {
            auto& xs = solid_particles[id_solid];
            T distance2 = (xs - pi).squaredNorm();
            if (distance2 < closest_distance2) {
                closest_distance2 = distance2;
                closest_solid = id_solid;
            }
        }
        fluid_normal = solid_normals[closest_solid];

        int iteration = 0;
        while (iteration++ < 20) {
            dx[i] += 0.1 * fluid_sim.dx * fluid_normal;
            neighbors_solid.clear();
            hash_solid.oneLayerNeighborsWithinRadius(pi + dx[i], solid_particles, H_solid, neighbors_solid);
            bool still_bad = false;
            for (auto id_solid : neighbors_solid) {
                TV n = solid_normals[id_solid];
                TV solid_pos = solid_particles[id_solid];
                TV fluid_pos = pi + dx[i];
                TV dir = fluid_pos - solid_pos;
                if (dir.dot(n) < 0) {
                    still_bad = true;
                    break;
                }
            }
            if (!still_bad) break;
        }
    }
}

} // namespace PUSHOUT_QUERIER

} // namespace ZIRAN