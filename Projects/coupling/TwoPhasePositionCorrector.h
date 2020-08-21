#include <MPM/MpmGrid.h>
#include <MPM/MpmSimulationBase.h>
#include <Ziran/CS/DataStructure/SpatialHash.h>
#include <Ziran/Math/Geometry/CollisionObject.h>
#include <Ziran/Math/Geometry/Visualizer.h>
#include "../splitting/UMGrid.h"

namespace ZIRAN {

namespace TWO_PHASE_PARTICLE_CORRECTOR {

template <class T>
T smooth_kernel(T r2, T h)
{
    return std::max(1.0 - r2 / (h * h), 0.0);
}

template <class T>
T sharp_kernel(T r2, T h)
{
    return std::max(h * h / std::max(r2, (T)1.0e-5) - 1.0, 0.0);
}

template <class T>
T lerp(T v0, T v1, T c)
{
    return (1 - c) * v0 + c * v1;
}

template <class T>
T bilerp(const T& v00, const T& v10,
    const T& v01, const T& v11,
    T cx, T cy)
{
    return lerp(lerp(v00, v10, cx),
        lerp(v01, v11, cx),
        cy);
}

template <class T>
T trilerp(const T& v000, const T& v100,
    const T& v010, const T& v110,
    const T& v001, const T& v101,
    const T& v011, const T& v111,
    T fx, T fy, T fz)
{
    return lerp(bilerp(v000, v100, v010, v110, fx, fy),
        bilerp(v001, v101, v011, v111, fx, fy),
        fz);
}

template <class T, int dim>
void findInterface(UMGrid<T, dim>& cell_phi, UMGrid<int, dim>& cell_sign)
{
    cell_sign.clear();
    T LARGE_VALUE = 100;
    //TODO: MAYBE SEPERATE AND PARALLIZE
    cell_phi.iterateGrid([&](int x, int y, int kind) {
        if (cell_phi(x, y) > 0) {
            cell_sign(x, y) = 1;
            cell_phi(x, y) *= (T)cell_sign(x, y);
        }
        else {
            cell_sign(x, y) = -1;
            cell_phi(x, y) = LARGE_VALUE;
        }
    });
}

template <class T, int dim>
T findSmallValueX(UMGrid<T, dim>& cell_phi, int i, int j)
{
    if (!cell_phi.find(i - 1, j) && cell_phi.find(i + 1, j))
        return cell_phi(i + 1, j);
    else if (!cell_phi.find(i + 1, j) && cell_phi.find(i - 1, j))
        return cell_phi(i - 1, j);
    else if (!cell_phi.find(i + 1, j) && !cell_phi.find(i - 1, j))
        return cell_phi(i, j);
    else
        return std::min(cell_phi(i - 1, j), cell_phi(i + 1, j));
}

template <class T, int dim>
T findSmallValueY(UMGrid<T, dim>& cell_phi, int i, int j)
{
    if (!cell_phi.find(i, j - 1) && cell_phi.find(i, j + 1))
        return cell_phi(i, j + 1);
    else if (!cell_phi.find(i, j + 1) && cell_phi.find(i, j - 1))
        return cell_phi(i, j - 1);
    else if (!cell_phi.find(i, j + 1) && !cell_phi.find(i, j - 1))
        return cell_phi(i, j);
    else
        return std::min(cell_phi(i, j - 1), cell_phi(i, j + 1));
}

template <class T>
T calculateUbar(T a, T b, T h)
{
    if (std::abs(a - b) >= h)
        return std::min(a, b) + h;
    else
        return 0.5f * (a + b + std::sqrt(2 * h * h - (a - b) * (a - b)));
}

template <class T, int dim>
bool is_solid(Vector<int, dim> Xi, const MpmSimulationBase<T, dim>& sim)
{
    using TV = Vector<T, dim>;

    TV xi;
    TV vi = TV::Zero();
    T phi;
    TV n;
    Matrix<T, dim, dim> normal_basis;
    if constexpr (dim == 2) {
        xi(0) = ((T)Xi[0] + 0.5) * sim.dx;
        xi(1) = ((T)Xi[1] + 0.5) * sim.dx;
    }
    else {
        xi(0) = ((T)Xi[0] + 0.5) * sim.dx;
        xi(1) = ((T)Xi[1] + 0.5) * sim.dx;
        xi(2) = ((T)Xi[2] + 0.5) * sim.dx;
    }
    // return AnalyticCollisionObject<T, dim>::multiObjectCollision(sim.collision_objects, xi, vi, normal_basis);
    const StdVector<std::unique_ptr<AnalyticCollisionObject<T, dim>>>& my_collisionObjects = sim.collision_objects;
    for (size_t k = 0; k < my_collisionObjects.size(); k++) {
        if (my_collisionObjects[k]->queryInside(xi, phi, n, -sim.dx)) {
            return true;
        }
    }
    return false;
}

template <class T, int dim>
StdVector<Vector<T, dim>> precompute(const MpmSimulationBase<T, dim>& sim)
{
    ZIRAN_TIMER();
    ZIRAN_INFO("START PRECOMPUTE POSITION CORRECTION!");

    using TV = Vector<T, dim>;
    using IV = Vector<int, dim>;
    UMGrid<int, dim> cell_kind;
    StdVector<TV> wall_particles;

    cell_kind.clear();
    for (int i = 0; i < sim.particles.count; ++i) {
        TV& Xp = sim.particles.X[i];
        BSplineWeights<T, dim, 1> spline(Xp - TV::Ones() * 0.5 * sim.dx, sim.dx);

        if constexpr (dim == 2) {
            for (int x = spline.base_node[0] - 1; x < spline.base_node[0] + ZIRAN_MPM_DEGREE + 1; ++x)
                for (int y = spline.base_node[1] - 1; y < spline.base_node[1] + ZIRAN_MPM_DEGREE + 1; ++y) {
                    if (is_solid(IV(x, y), sim)) {
                        cell_kind(x, y) = 2;
                        continue;
                    }
                    if (!cell_kind.find(x, y) || cell_kind(x, y) != 1) {
                        cell_kind(x, y) = 1;
                        int offset_x[] = { 0, 0, 1, -1, -1, -1, 1, 1 };
                        int offset_y[] = { 1, -1, 0, 0, -1, 1, -1, 1 };
                        for (int k = 0; k < 8; ++k) {
                            int new_x = x + offset_x[k];
                            int new_y = y + offset_y[k];
                            if (is_solid(IV(new_x, new_y), sim)) {
                                cell_kind(new_x, new_y) = 2;
                            }
                            if (!cell_kind.find(new_x, new_y)) cell_kind(new_x, new_y) = 0;
                        }
                    }
                }
        }
        else {
            for (int x = spline.base_node[0]; x < spline.base_node[0] + ZIRAN_MPM_DEGREE; ++x)
                for (int y = spline.base_node[1]; y < spline.base_node[1] + ZIRAN_MPM_DEGREE; ++y)
                    for (int z = spline.base_node[2]; z < spline.base_node[2] + ZIRAN_MPM_DEGREE; ++z) {
                        if (is_solid(IV(x, y, z), sim))
                            continue;
                        if (!cell_kind.find(x, y, z) || cell_kind(x, y, z) != 1) {
                            cell_kind(x, y, z) = 1;
                            int offset_x[] = { 0, 0, 1, -1, -1, -1, 1, 1, 0, 0, 0, 1, -1, -1, -1, 1, 1, 0, 0, 0, 1, -1, -1, -1, 1, 1 };
                            int offset_y[] = { 1, -1, 0, 0, -1, 1, -1, 1, 0, 1, -1, 0, 0, -1, 1, -1, 1, 0, 1, -1, 0, 0, -1, 1, -1, 1 };
                            int offset_z[] = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
                            for (int k = 0; k < 26; ++k) {
                                int new_x = x + offset_x[k];
                                int new_y = y + offset_y[k];
                                int new_z = z + offset_z[k];
                                if (is_solid(IV(new_x, new_y, new_z), sim)) {
                                    cell_kind(new_x, new_y, new_z) = 2;
                                }
                                if (!cell_kind.find(new_x, new_y, new_z)) cell_kind(new_x, new_y, new_z) = 0;
                            }
                        }
                    }
        }
    }

    // actually sample solid particles
    T volume = 1;
    for (int i = 0; i < dim; i++) {
        volume *= sim.dx;
    }

    // sample solid boundary particles
    auto* vol_pointer = &sim.particles.DataManager::get(AttributeName<T>("element measure"));
    T particles_vol = (*vol_pointer)[0];
    int ppc = std::round(volume / particles_vol);
    ZIRAN_INFO("Estimated particle per cell in Density Projection: ", ppc);
    int ppc_dim = std::round(std::sqrt(ppc));
    T shift = (T)1 / ppc_dim;
    ZIRAN_INFO("PPC_DIM IS: ", ppc_dim, " Shift is: ", shift);
    if constexpr (dim == 2) {
        cell_kind.iterateGrid([&](int x, int y, int kind) {
            if (kind == 2) {
                for (int i = 0; i < ppc_dim; i++)
                    for (int j = 0; j < ppc_dim; j++) {
                        TV particle_pos = TV((T)x + 0.5 * shift + i * shift, (T)y + 0.5 * shift + j * shift) * sim.dx;
                        wall_particles.push_back(particle_pos);
                    }
            }
        });
    }
    else {
        // TODO: FIX 3D
        //        cell_kind.iterateGrid([&](int x, int y, int z, int kind) {
        //            if (kind == 2) {
        //                for (int i = 0; i < ppc_dim; i++)
        //                    for (int j = 0; j < ppc_dim; j++)
        //                        for (int k = 0; k < ppc_dim; k++) {
        //                            TV particle_pos = TV((T)x + i * shift, (T)y + j * shift, (T)z + k * shift) * sim.dx;
        //                            wall_particles.push_back(particle_pos);
        //                        }
        //            }
        //        });
    }

    return wall_particles;
}

template <class T, int dim>
void fastSweep(int xmin, int xmax, int ymin, int ymax, T dx, UMGrid<T, dim>& cell_phi, UMGrid<int, dim>& cell_sign, UMGrid<T, dim>& face_phi_x, UMGrid<T, dim>& face_phi_y)
{
    T ubar;
    findInterface(cell_phi, cell_sign);
    for (int i = xmin; i <= xmax; i++)
        for (int j = ymin; j <= ymax; j++) {
            if (cell_sign.find(i, j) && cell_sign(i, j) < 0) {
                T a = findSmallValueX(cell_phi, i, j);
                T b = findSmallValueY(cell_phi, i, j);
                ubar = calculateUbar(a, b, dx);
                cell_phi(i, j) = std::min(cell_phi(i, j), ubar);
            }
        }

    for (int i = xmax; i >= xmin; i--)
        for (int j = ymin; j <= ymax; j++) {
            if (cell_sign.find(i, j) && cell_sign(i, j) < 0) {
                T a = findSmallValueX(cell_phi, i, j);
                T b = findSmallValueY(cell_phi, i, j);
                ubar = calculateUbar(a, b, dx);
                cell_phi(i, j) = std::min(cell_phi(i, j), ubar);
            }
        }

    for (int i = xmax; i >= xmin; i--)
        for (int j = ymax; j >= ymin; j--) {
            if (cell_sign.find(i, j) && cell_sign(i, j) < 0) {
                T a = findSmallValueX(cell_phi, i, j);
                T b = findSmallValueY(cell_phi, i, j);
                ubar = calculateUbar(a, b, dx);
                cell_phi(i, j) = std::min(cell_phi(i, j), ubar);
            }
        }

    for (int i = xmin; i <= xmax; i++)
        for (int j = ymax; j >= ymin; j--) {
            if (cell_sign.find(i, j) && cell_sign(i, j) < 0) {
                T a = findSmallValueX(cell_phi, i, j);
                T b = findSmallValueY(cell_phi, i, j);
                ubar = calculateUbar(a, b, dx);
                cell_phi(i, j) = std::min(cell_phi(i, j), ubar);
            }
        }

    cell_phi.iterateGrid([&](int x, int y, int kind) {
        cell_phi(x, y) *= (T)cell_sign(x, y);
    });

    face_phi_x.clear();
    face_phi_y.clear();
    cell_phi.iterateGrid([&](int x, int y, int kind) {
        face_phi_x(x, y) = 0;
        face_phi_x(x + 1, y) = 0;
        face_phi_y(x, y) = 0;
        face_phi_y(x, y + 1) = 0;
    });
    cell_phi.iterateGrid([&](int x, int y, int kind) {
        face_phi_x(x, y) += cell_phi(x, y) / dx;
        face_phi_x(x + 1, y) -= cell_phi(x, y) / dx;
        face_phi_y(x, y) += cell_phi(x, y) / dx;
        face_phi_y(x, y + 1) -= cell_phi(x, y) / dx;
    });

    ZIRAN_INFO("Renormalize Signed Distance Field finished");
}

template <class T, int dim>
bool isInside(Vector<T, dim> particleX, T& phi, Vector<T, dim>& normal, T dx, UMGrid<T, dim>& cell_phi, UMGrid<int, dim>& cell_sign, UMGrid<T, dim>& face_phi_x, UMGrid<T, dim>& face_phi_y)
{
    using TV = Vector<T, dim>;
    BSplineWeights<T, dim, 1> spline(particleX - TV::Ones() * 0.5 * dx, dx);
    int x = spline.base_node[0];
    int y = spline.base_node[1];

    if (!cell_phi.find(x, y) || !cell_phi.find(x + 1, y) || !cell_phi.find(x + 1, y + 1) || !cell_phi.find(x, y + 1)) {
        return false;
    }
    else {
        ZIRAN_ASSERT(cell_phi.find(x, y));
        ZIRAN_ASSERT(cell_phi.find(x + 1, y));
        ZIRAN_ASSERT(cell_phi.find(x + 1, y + 1));
        ZIRAN_ASSERT(cell_phi.find(x, y + 1));
        T phi00 = cell_phi(x, y);
        T phi10 = cell_phi(x + 1, y);
        T phi11 = cell_phi(x + 1, y + 1);
        T phi01 = cell_phi(x, y + 1);

        T px = std::abs(particleX[0] / dx - ((T)x + 0.5));
        T py = std::abs(particleX[1] / dx - ((T)y + 0.5));
        phi = bilerp(phi00, phi10, phi01, phi11, px, py);
        if (phi < 0) {
            BSplineWeights<T, dim, 1> spline1(particleX - TV(0, 1) * 0.5 * dx, dx);
            BSplineWeights<T, dim, 1> spline2(particleX - TV(1, 0) * 0.5 * dx, dx);
            // interpolate x direction
            int x1 = spline1.base_node[0];
            int y1 = spline1.base_node[1];

            ZIRAN_ASSERT(face_phi_x.find(x1, y1));
            ZIRAN_ASSERT(face_phi_x.find(x1 + 1, y1));
            ZIRAN_ASSERT(face_phi_x.find(x1 + 1, y1 + 1));
            ZIRAN_ASSERT(face_phi_x.find(x1, y1 + 1));
            T vx00 = face_phi_x(x1, y1);
            T vx10 = face_phi_x(x1 + 1, y1);
            T vx11 = face_phi_x(x1 + 1, y1 + 1);
            T vx01 = face_phi_x(x1, y1 + 1);

            T px1 = std::abs(particleX[0] / dx - ((T)x1));
            T py1 = std::abs(particleX[1] / dx - ((T)y1 + 0.5));
            T dphidx = bilerp(vx00, vx10, vx01, vx11, px1, py1);

            // interpolate y direciton
            int x2 = spline2.base_node[0];
            int y2 = spline2.base_node[1];

            ZIRAN_ASSERT(face_phi_y.find(x2, y2));
            ZIRAN_ASSERT(face_phi_y.find(x2 + 1, y2));
            ZIRAN_ASSERT(face_phi_y.find(x2 + 1, y2 + 1));
            ZIRAN_ASSERT(face_phi_y.find(x2, y2 + 1));
            T vy00 = face_phi_y(x2, y2);
            T vy10 = face_phi_y(x2 + 1, y2);
            T vy11 = face_phi_y(x2 + 1, y2 + 1);
            T vy01 = face_phi_y(x2, y2 + 1);

            T px2 = std::abs(particleX[0] / dx - ((T)x2 + 0.5));
            T py2 = std::abs(particleX[1] / dx - ((T)y2));
            T dphidy = bilerp(vy00, vy10, vy01, vy11, px2, py2);

            normal = TV(dphidx, dphidy);
            normal = normal.normalized();
            return true;
        }
        else {
            return false;
        }
    }
}

// Ando, isotropic version
// http://research.nii.ac.jp/~rand/sheetflip/download/tvcg.pdf
template <class T, int dim>
StdVector<Vector<T, dim>> solve(SplittingSimulation<T, dim>& sim, SplittingSimulation<T, dim>& sim_solid, T ppc)
{
    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;

    ZIRAN_TIMER();

    // UMGrid<T, dim> cell_phi;
    // UMGrid<int, dim> cell_sign;
    // UMGrid<T, dim> face_phi_x;
    // UMGrid<T, dim> face_phi_y;
    // int xmin, xmax, ymin, ymax;
    // cell_phi.clear();
    // if (sim_solid.particles.count > 0) {
    //     for (int i = 0; i < sim_solid.particles.count; i++) {
    //         if constexpr (dim == 2) {
    //             TV& Xp = sim_solid.particles.X[i];
    //             BSplineWeights<T, dim, 1> spline(Xp - TV::Ones() * 0.5 * sim.dx, sim.dx);
    //             if (i == 0) {
    //                 xmin = spline.base_node[0];
    //                 xmax = spline.base_node[0];
    //                 ymin = spline.base_node[1];
    //                 ymax = spline.base_node[1];
    //             }
    //             for (int x = spline.base_node[0] - 1; x <= spline.base_node[0] + 2; ++x)
    //                 for (int y = spline.base_node[1] - 1; y <= spline.base_node[1] + 2; ++y) {
    //                     if (!cell_phi.find(x, y)) cell_phi(x, y) = std::numeric_limits<T>::max();
    //                     TV cell_pos = TV(x + 0.5, y + 0.5) * sim.dx;
    //                     TV dist_vec = cell_pos - Xp;
    //                     T dist = dist_vec.norm() - 0.4 * sim.dx;
    //                     cell_phi(x, y) = std::min(cell_phi(x, y), dist);
    //                     xmin = std::min(xmin, x);
    //                     xmax = std::max(xmax, x);
    //                     ymin = std::min(ymin, y);
    //                     ymax = std::max(ymax, y);
    //                 }
    //         }
    //         else {
    //             // TODO: 3D construct solid particles level-set
    //         }
    //     }
    //     ZIRAN_INFO("FAST SWEEP STARTS!");
    //     fastSweep(xmin, xmax, ymin, ymax, sim.dx, cell_phi, cell_sign, face_phi_x, face_phi_y);
    //     cell_phi.iterateGrid([&](int x, int y, int kind) {
    //         // cell_phi(x, y) += 0.8 * sim.dx;
    //     });
    //     ZIRAN_INFO("FAST SWEEP ENDS!");
    // }

    StdVector<TV> wall_particles = sim.boundary_positions;

    auto& Xarray = sim.particles.X.array;
    auto& Varray = sim.particles.V.array;
    // tbb::parallel_for(0, (int)sim.particles.count, [&](int i) {
    //     // T phi;
    //     // TV n;
    //     // bool colliding_with_solid = isInside(Xarray[i], phi, n, sim.dx, cell_phi, cell_sign, face_phi_x, face_phi_y);
    //     // if (colliding_with_solid) {
    //     //     Xarray[i] -= (phi)*n;
    //     //     if (Varray[i].dot(n) < 0)
    //     //         Varray[i] -= Varray[i].dot(n) * n;
    //     // }
    //     // else {
    //     const StdVector<std::unique_ptr<AnalyticCollisionObject<T, dim>>>& my_collisionObjects = sim.collision_objects;
    //     for (size_t k = 0; k < my_collisionObjects.size(); k++) {
    //         T phi;
    //         TV n;
    //         if (my_collisionObjects[k]->queryInside(Xarray[i], phi, n, -sim.dx)) {
    //             Xarray[i] -= (phi + sim.dx) * n;
    //             if (Varray[i].dot(n) < 0)
    //                 Varray[i] -= Varray[i].dot(n) * n;
    //         }
    //     }
    //     // }
    // });

    T particle_radii = sim.dx / std::pow((T)ppc, (T)1 / (T)dim);
    SpatialHash<T, dim> hash_fluid;
    SpatialHash<T, dim> hash_solid;
    T H_fluid = particle_radii;
    T H_solid = particle_radii;

    StdVector<TV> particles_tmpX(sim.particles.count);
    StdVector<TV> particles_tmpV(sim.particles.count);

    int N_fluids = sim.particles.count;
    StdVector<TV> all_fluid_points = sim.particles.X.array;
    int N_solids = sim_solid.particles.count + wall_particles.size();
    // int N_solids = wall_particles.size();
    StdVector<TV> all_solid_points = sim_solid.particles.X.array;
    // StdVector<TV> all_solid_points = wall_particles;
    all_solid_points.insert(all_solid_points.end(), wall_particles.begin(), wall_particles.end());

    // hash_fluid.rebuild(H_fluid, all_fluid_points);
    hash_solid.rebuild(H_solid, all_solid_points);

    // Compute solid particle normals
    StdVector<TV> solid_particle_normals(N_solids, TV::Zero());
    SpatialHash<T, dim> hash_solid_wide;
    T H_solid_wide = sim.dx;
    hash_solid_wide.rebuild(H_solid_wide, all_solid_points);
    for (int i = 0; i < N_solids; ++i) {
        TV& pi = all_solid_points[i];
        solid_particle_normals[i] = TV::Zero();
        StdVector<int> neighbors_solid;
        hash_solid_wide.oneLayerNeighbors(pi, neighbors_solid);
        for (auto j : neighbors_solid) {
            if (j != i) {
                TV& pj = all_solid_points[j];
                T dist = (pi - pj).norm();
                T w = (T)1 / dist;
                solid_particle_normals[i] += (w / dist) * (pi - pj);
            }
        }
        T length = solid_particle_normals[i].norm();
        if (length) solid_particle_normals[i] /= length;
    }

    for (int i = 0; i < sim.particles.count; ++i) {
        particles_tmpX[i] = Xarray[i];
        particles_tmpV[i] = Varray[i];
    }
    RandomNumber<T> random(123);

    StdVector<Matrix<T, dim, dim>>* Carray_pointer;
    Carray_pointer = ((sim.transfer_scheme == sim.APIC_blend_RPIC) && sim.interpolation_degree != 1) ? (&(sim.particles.DataManager::get(C_name<TM>()).array)) : NULL;

    for (size_t i = 0; i < (int)sim.particles.count; ++i) {
        TV& pi = Xarray[i];
        TV spring = TV::Zero();
        StdVector<int> neighbors_fluid;
        // hash_fluid.oneLayerNeighbors(pi, neighbors_fluid);
        // for (auto j : neighbors_fluid) {
        //     if (j != i) {
        //         TV pj = Xarray[j];
        //         T dist = (pi - pj).norm();
        //         T w = 50 * smooth_kernel(dist * dist, H_fluid);
        //         if (dist > 0.1 * H_fluid) {
        //             spring += w * (pi - pj) / dist * H_fluid;
        //         }
        //         else {
        //             for (int d = 0; d < dim; d++)
        //                 spring(d) += 0.1 * H_fluid / sim.dt * random.randReal(0, 1);
        //         }
        //     }
        // }
        StdVector<int> neighbors_solid;
        hash_solid.oneLayerNeighbors(pi, neighbors_solid);
        for (auto j : neighbors_solid) {
            TV pj = all_solid_points[j];
            T dist = (pi - pj).norm();
            T w = 25 * smooth_kernel(dist * dist, H_solid);
            spring += w * (pi - pj) / dist * H_solid;
            //
            //            if (dist > 0.1 * H_solid) {
            //                spring += w * (pi - pj) / dist * H_solid;
            //            }
            //            else {
            //                spring += 0.1 * H_solid / sim.dt * solid_particle_normals[j];
            //            }
        }
        if (spring.norm() > 0 && Carray_pointer) (*Carray_pointer)[i] = Matrix<T, dim, dim>::Zero();
        particles_tmpX[i] = Xarray[i] + sim.dt * spring;
    }
    //
    //     for (size_t i = 0; i < (int)sim.particles.count; ++i) {
    //         TV& pi = particles_tmpX[i];
    //         TV fz = TV::Zero();
    //         T fm = (T)0;
    //         StdVector<int> neighbors_fluid;
    //         hash_fluid.oneLayerNeighbors(pi, neighbors_fluid);
    //         for (auto j : neighbors_fluid) {
    //             TV pj = Xarray[j];
    //             TV vj = Varray[j];
    //             T dist = (pi - pj).norm();
    //             T dist2 = dist * dist;
    //             T w = sharp_kernel(dist2, H_fluid);
    //             fz += w * vj;
    //             fm += w;
    //         }
    //         if (fm) {
    //             particles_tmpV[i] = fz / fm;
    //         }
    //     }

    for (int i = 0; i < sim.particles.count; ++i) {
        Xarray[i] = particles_tmpX[i];
        Varray[i] = particles_tmpV[i];
    }
    return wall_particles;
}
}

} // namespace ZIRAN::TWO_PHASE_PARTICLE_CORRECTOR