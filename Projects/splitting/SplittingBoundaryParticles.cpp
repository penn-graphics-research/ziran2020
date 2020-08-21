#include "SplittingSimulation.h"

#include <Ziran/Math/Geometry/AnalyticLevelSet.h>
#include <Ziran/Math/Geometry/PoissonDisk.h>

namespace ZIRAN {

template <class T, int dim>
void SplittingSimulation<T, dim>::sampleBoundaryParticlesInCollisionObjects(const AnalyticCollisionObject<T, dim>& object, TV min_corner, TV max_corner, T distance_threshold)
{
    static std::unordered_map<uint64_t, bool> boundary_markers;
    int boundary_per_dimension = 3;
    T grid_offset = 0.5;

    auto sampleInAnalyticLevelSetHelper = [&](AxisAlignedAnalyticBox<T, dim>& levelset, StdVector<Vector<T, dim>>& samples) {
        using IV = Vector<int, dim>;
        ZIRAN_ASSERT(boundary_per_dimension == 2 || boundary_per_dimension == 3);
        ZIRAN_ASSERT(grid_offset >= 0 && grid_offset < 1);
        int ppc = std::pow(boundary_per_dimension, dim);
        ZIRAN_ASSERT(ppc == 4 || ppc == 9 || ppc == 8 || ppc == 27);
        ZIRAN_INFO("dd");
        T per_particle_volume = std::pow(Base::dx, (T)dim) / (T)ppc;
        TV min_corner, max_corner;
        levelset.getBounds(min_corner, max_corner);
        ZIRAN_INFO("min_corner ", min_corner.transpose(), max_corner.transpose());
        min_corner -= 4 * Base::dx * TV::Ones();
        max_corner += 4 * Base::dx * TV::Ones();
        IV sample_grid_min_idx = IV::Zero();
        IV sample_grid_max_idx = IV::Zero();
        for (int d = 0; d < dim; d++) {
            sample_grid_min_idx(d) = (int)(min_corner(d) / Base::dx);
            sample_grid_max_idx(d) = (int)(max_corner(d) / Base::dx);
        }
        Box<int, dim> localBox(sample_grid_min_idx, sample_grid_max_idx);
        ZIRAN_INFO("sample grid min:", sample_grid_min_idx.transpose());
        ZIRAN_INFO("sample grid max:", sample_grid_max_idx.transpose());
        for (MaxExclusiveBoxIterator<dim> it(localBox); it.valid(); ++it) {
            TV cell_corner = TV::Zero();
            for (int d = 0; d < dim; d++) cell_corner(d) = it.index(d) * Base::dx + grid_offset * Base::dx;
            TV cell_center = cell_corner + TV::Ones() * Base::dx / 2;
            if (!levelset.inside(cell_center)) continue;

            if constexpr (dim == 2) {
                if (boundary_per_dimension == 2) {
                    for (int i = 0; i < 2; i++)
                        for (int j = 0; j < 2; j++)
                            samples.emplace_back(cell_corner + TV(0.25 * Base::dx + i * 0.5 * Base::dx, 0.25 * Base::dx + j * 0.5 * Base::dx));
                }
                else if (boundary_per_dimension == 3) {
                    for (int i = 0; i < 3; i++)
                        for (int j = 0; j < 3; j++)
                            samples.emplace_back(cell_corner + TV(Base::dx / 6 + i * Base::dx / 3, Base::dx / 6 + j * Base::dx / 3));
                }
            }
            else if constexpr (dim == 3) {
                if (boundary_per_dimension == 2) {
                    for (int i = 0; i < 2; i++)
                        for (int j = 0; j < 2; j++)
                            for (int k = 0; k < 2; k++)
                                samples.emplace_back(cell_corner + TV(0.25 * Base::dx + i * 0.5 * Base::dx, 0.25 * Base::dx + j * 0.5 * Base::dx, 0.25 * Base::dx + k * 0.5 * Base::dx));
                }
                else if (boundary_per_dimension == 3) {
                    for (int i = 0; i < 3; i++)
                        for (int j = 0; j < 3; j++)
                            for (int k = 0; k < 3; k++)
                                samples.emplace_back(cell_corner + TV(Base::dx / 6 + i * Base::dx / 3, Base::dx / 6 + j * Base::dx / 3, Base::dx / 6 + k * Base::dx / 3));
                }
            }
        }
        ZIRAN_INFO("sampled particle count: ", samples.size());
        boundary_particle_vol = per_particle_volume;
    };
    AxisAlignedAnalyticBox<T, dim> box(min_corner, max_corner);
    StdVector<TV> samples;
    sampleInAnalyticLevelSetHelper(box, samples);
    for (const auto& X : samples) {
        IV node = (X / (Base::dx / (T)4)).template cast<int>();
        uint64_t offset = Base::SparseMask::Linear_Offset(to_std_array(node));
        if (boundary_markers[offset]) continue;

        T distance = object.signedDistance(X);
        if (-distance_threshold < distance && distance < 0) {
            TV position = X;
            TV normal = object.ls->normal(X);
            boundary_positions.push_back(position);
            boundary_normals.push_back(normal);
        }
    }

    for (const auto& X : boundary_positions) {
        IV node = (X / (Base::dx / (T)4)).template cast<int>();
        uint64_t offset = Base::SparseMask::Linear_Offset(to_std_array(node));
        boundary_markers[offset] = true;
    }

    SpatialHash<T, dim> spatialHash;
    spatialHash.rebuild(Base::dx, boundary_positions);
    StdVector<TV> average_normal(boundary_normals.size());
    for (uint i = 0; i < boundary_positions.size(); ++i) {
        TV Xp = boundary_positions[i];
        TV normal = TV::Zero();
        StdVector<int> neighbors;
        spatialHash.oneLayerNeighbors(Xp, neighbors);
        for (auto& idx : neighbors) {
            if ((Xp - boundary_positions[idx]).squaredNorm() < Base::dx * Base::dx)
                normal += boundary_normals[idx];
        }
        average_normal[i] = normal.squaredNorm() < 1e-30 ? boundary_normals[i] : normal.normalized();
    }
    boundary_normals.assign(average_normal.begin(), average_normal.end());
}

template void SplittingSimulation<float, 2>::sampleBoundaryParticlesInCollisionObjects(const AnalyticCollisionObject<float, 2>& object, TV min_corner, TV max_corner, float distance);
template void SplittingSimulation<double, 2>::sampleBoundaryParticlesInCollisionObjects(const AnalyticCollisionObject<double, 2>& object, TV min_corner, TV max_corner, double distance);
template void SplittingSimulation<float, 3>::sampleBoundaryParticlesInCollisionObjects(const AnalyticCollisionObject<float, 3>& object, TV min_corner, TV max_corner, float distance);
template void SplittingSimulation<double, 3>::sampleBoundaryParticlesInCollisionObjects(const AnalyticCollisionObject<double, 3>& object, TV min_corner, TV max_corner, double distance);

}; // namespace ZIRAN
