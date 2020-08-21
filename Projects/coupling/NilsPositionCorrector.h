#pragma once

#include "../splitting/SplittingSimulation.h"
#include "BuildLevelset.h"
#include "ParticlesLevelSet.h"

namespace ZIRAN {

template <class T, int dim>
class NilsPositionCorrector {
    using TV = Vector<T, dim>;
    using IV = Vector<int, dim>;
    using TM = Matrix<T, dim, dim>;
    using TStack = Vector<T, Eigen::Dynamic>;

    SplittingSimulation<T, dim>& sim;
    MpmGrid<T, dim, 1> pressure_grid;
    MpmGrid<T, dim, 1> solid_grid;
    TStack rhs, solved_p;

public:
    ParticlesLevelSet<T, dim> levelset_builder;
    // BuildLevelset<T, dim> levelset_builder;

    NilsPositionCorrector(SplittingSimulation<T, dim>& sim)
        : sim(sim) {}

    bool isSolid(IV node)
    {
        TV xi = node.template cast<T>() * sim.dx + TV::Ones() * sim.dx;
        TV vi = TV::Zero();
        TM normal_basis = TM::Zero();
        bool inside_boundary = AnalyticCollisionObject<T, dim>::multiObjectCollision(sim.collision_objects, xi, vi, normal_basis);
        T phi;
        TV normal;
        bool inside_solid = levelset_builder.queryInside(xi, phi, normal, 0);
        return inside_boundary || inside_solid;
    }

    T getPressure(IV node)
    {
        return pressure_grid[node].idx < 0 ? (T)0 : solved_p(pressure_grid[node].idx);
    }

    void pushout()
    {
        ZIRAN_TIMER();

        StdVector<TV>& Xarray = sim.particles.X.array;
        solid_grid.pollute(Xarray, sim.dx, -1, 1);

        sim.parallel_for_updating_grid([&](int i) {
            TV Xp = sim.particles.X.array[i];
            BSplineWeights<T, dim, 1> spline(Xp - TV::Ones() * sim.dx * 0.5, sim.dx);
            IV node = spline.base_node;

            if (isSolid(node)) {
                TV Xt = Xp, normal;
                TV& Vt = sim.particles.V.array[i];
                TM& Ct = sim.scratch_gradV[i];
                T phi;
                for (int k = 0; k < (int)sim.collision_objects.size(); ++k) {
                    if (sim.collision_objects[k]->type == AnalyticCollisionObject<T, dim>::GHOST)
                        continue;
                    bool inside_boundary = sim.collision_objects[k]->queryInside(Xt, phi, normal, 0);
                    if (inside_boundary) {
                        Xt = Xt - phi * normal;
                        Vt -= normal * Vt.dot(normal);
                        Ct = TM::Zero();
                    }
                }
                {
                    bool inside_solid = levelset_builder.queryInside(Xt, phi, normal, 0);
                    if (inside_solid) {
                        Xt = Xt - phi * normal;
                        Vt -= normal * Vt.dot(normal);
                        Ct = TM::Zero();
                    }
                }
                if ((Xt - Xp).squaredNorm() > solid_grid[node].v[0]) {
                    solid_grid[node].v[0] = (Xt - Xp).squaredNorm();
                    solid_grid[node].new_v = Xt - Xp;
                    solid_grid[node].m = 1;
                }
            }
        });
        int solid_dof = solid_grid.getNumNodes();
        solid_grid.iterateGrid([&](IV node, GridState<T, dim>& g) {
            for (int d = 0; d < dim; ++d)
                MATH_TOOLS::clamp(g.new_v[d], -(T)0.5 * sim.dx, (T)0.5 * sim.dx);
        });
        solid_grid.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
            for (int d = 0; d < dim; ++d) {
                IV neighbor = node;
                neighbor[d] -= 1;
                pressure_grid[node].new_v[d] = (g.new_v[d] + (isSolid(neighbor) ? solid_grid[neighbor].new_v[d] : g.new_v[d])) / (T)2;
                neighbor = node;
                neighbor[d] += 1;
                pressure_grid[neighbor].new_v[d] = (g.new_v[d] + (isSolid(neighbor) ? solid_grid[neighbor].new_v[d] : g.new_v[d])) / (T)2;
            }
        });
        pressure_grid.iterateGrid([&](IV node, GridState<T, dim>& g) {
            T rho_rest = sim.particles.mass.array[0] / (sim.particles.DataManager::get(element_measure_name<T>())).array[0];
            IV neighbor;
            for (int d = 0; d < dim; ++d) {
                IV neighbor = node;
                neighbor[d] -= 1;
                if (isSolid(neighbor)) rhs(g.idx) -= solid_grid[node].new_v[d] * rho_rest * sim.dx / sim.dt / sim.dt;
                neighbor = node;
                neighbor[d] += 1;
                if (isSolid(neighbor)) rhs(g.idx) += solid_grid[neighbor].new_v[d] * rho_rest * sim.dx / sim.dt / sim.dt;
            }
        });
    }

    std::vector<T> detectIsland(const Eigen::SparseMatrix<T>& lhs, TStack& rhs)
    {
        int dof = lhs.rows();
        std::vector<int> sum(dof, 0);
        std::vector<bool> vis(dof, false);
        std::vector<std::vector<int>> edges(dof);
        for (int k = 0; k < lhs.outerSize(); ++k)
            for (typename Eigen::SparseMatrix<T>::InnerIterator it(lhs, k); it; ++it) {
                int row = it.row();
                int col = it.col();
                T val = it.value();
                sum[row] += (int)std::round(val);
                edges[row].push_back(col);
            }
        std::vector<T> null_spaces;
        for (int i = 0; i < dof; ++i) {
            if (!vis[i]) {
                bool island = true;
                std::vector<int> line(dof, 0);
                line[0] = i;
                vis[i] = true;
                if (sum[i] != 0) island = false;
                int l = 0, r = 1;
                for (; l < r; ++l) {
                    for (auto nxt : edges[line[l]])
                        if (!vis[nxt]) {
                            line[r++] = nxt;
                            vis[nxt] = true;
                            if (sum[nxt] != 0) island = false;
                        }
                }
                if (island) {
                    std::vector<T> null_space(dof, (T)0);
                    T dot_product = 0;
                    for (int j = 0; j < r; ++j) {
                        null_space[line[j]] = (T)1;
                        dot_product += rhs(line[j]) / (T)r;
                    }
                    for (int j = 0; j < r; ++j)
                        rhs(line[j]) -= dot_product;
                    for (auto v : null_space)
                        null_spaces.push_back(v);
                }
            }
        }

        return null_spaces;
    }

    void solve(SplittingSimulation<T, dim>& Ssim)
    {
        ZIRAN_TIMER();
        levelset_builder.build(Ssim, (T)0.4 * sim.dx, (T)0.5 * sim.dx, (T)0.4 * sim.dx);

        StdVector<TV>& Xarray = sim.particles.X.array;
        {
            ZIRAN_TIMER();
            pressure_grid.pollute(Xarray, sim.dx, -1, 1);
        }

        // compute m
        {
            ZIRAN_TIMER();
            sim.parallel_for_updating_grid([&](int i) {
                TV Xp = Xarray[i];
                BSplineWeights<T, dim, 1> spline(Xp - TV::Ones() * sim.dx * 0.5, sim.dx);
                if (!isSolid(spline.base_node))
                    pressure_grid[spline.base_node].m = 1;
            });
        }

        if (sim.particles.count > 0) {
            ZIRAN_TIMER();
            // compute rho
            T Vol = dim == 2 ? sim.dx * sim.dx : sim.dx * sim.dx * sim.dx;
            T rho = sim.particles.mass.array[0] / Vol;
            for (auto& Xp : sim.boundary_positions) {
                BSplineWeights<T, dim, 1> spline(Xp, sim.dx);
                uint64_t offset = SplittingSimulation<T, dim>::SparseMask::Linear_Offset(to_std_array(spline.base_node));
                pressure_grid.iterateKernelWithValidation(spline, offset, [&](IV node, T w, TV dw, GridState<T, dim>& g) {
                    g.v[0] += rho * w;
                });
            }
            sim.parallel_for_updating_grid([&](int i) {
                auto& Xp = sim.particles.X.array[i];
                BSplineWeights<T, dim, 1> spline(Xp, sim.dx);
                uint64_t offset = SplittingSimulation<T, dim>::SparseMask::Linear_Offset(to_std_array(spline.base_node));
                pressure_grid.iterateKernelWithValidation(spline, offset, [&](IV node, T w, TV dw, GridState<T, dim>& g) {
                    g.v[0] += rho * w;
                });
            });
            Ssim.parallel_for_updating_grid([&](int i) {
                auto& Xp = Ssim.particles.X.array[i];
                BSplineWeights<T, dim, 1> spline(Xp, sim.dx);
                uint64_t offset = SplittingSimulation<T, dim>::SparseMask::Linear_Offset(to_std_array(spline.base_node));
                pressure_grid.iterateKernelWithValidation(spline, offset, [&](IV node, T w, TV dw, GridState<T, dim>& g) {
                    g.v[0] += rho * w;
                });
            });
        }

        int pressure_dof = pressure_grid.getNumNodes();
        Eigen::SparseMatrix<T> lhs = Eigen::SparseMatrix<T>(pressure_dof, pressure_dof);
        rhs = TStack::Zero(pressure_dof);
        {
            ZIRAN_TIMER();
            std::vector<Eigen::Triplet<T>> lhs_tri(pressure_dof * dim * 2 * 2);
            pressure_grid.iterateGrid([&](IV node, GridState<T, dim>& g) {
                bool near_air = false;
                int row = g.idx;
                IV region = IV::Ones() * 3;
                iterateRegion(region, [&](const IV& offset) {
                    IV neighbor = node + offset - IV::Ones();
                    if (!isSolid(neighbor) && pressure_grid[neighbor].idx < 0)
                        near_air = true;
                });
                int cnt = row * dim * 2 * 2;
                for (int d = 0; d < dim; ++d)
                    for (int delta = -1; delta <= 1; delta += 2) {
                        IV neighbor = node;
                        neighbor[d] += delta;
                        if (!isSolid(neighbor)) {
                            int col = pressure_grid[neighbor].idx;
                            if (col < 0) {
                                near_air = true;
                                lhs_tri[cnt++] = Eigen::Triplet<T>(row, row, (T)1);
                            }
                            else {
                                lhs_tri[cnt++] = Eigen::Triplet<T>(row, row, (T)1);
                                lhs_tri[cnt++] = Eigen::Triplet<T>(row, col, (T)-1);
                            }
                        }
                    }
                T rho_rest = sim.particles.mass.array[0] / (sim.particles.DataManager::get(element_measure_name<T>())).array[0];
                T rho_star = pressure_grid[node + IV::Ones()].v[0];
                if (near_air) rho_star = std::max(rho_star, rho_rest);
                rho_star = std::max(rho_star, rho_rest * (T)0.5);
                rho_star = std::min(rho_star, rho_rest * (T)1.5);
                rhs(row) = -(rho_rest - rho_star) * sim.dx * sim.dx / sim.dt / sim.dt;
            });
            int cnt = 0;
            for (int i = 0; i < pressure_dof * dim * 2 * 2; ++i)
                if (lhs_tri[i].value() != 0)
                    lhs_tri[cnt++] = lhs_tri[i];
            lhs_tri.resize(cnt);
            lhs.setFromTriplets(lhs_tri.begin(), lhs_tri.end());
        }

        pushout();

        {
            ZIRAN_TIMER();
            std::vector<T> null_spaces = detectIsland(lhs, rhs);
            solved_p = DirectSolver<T>::solve(lhs, rhs, DirectSolver<T>::AMGCL, 1e-3, -1, 10000, -1, 1, false, false, (T)1e-4, null_spaces);
            pressure_grid.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
                T rho_rest = sim.particles.mass.array[0] / (sim.particles.DataManager::get(element_measure_name<T>())).array[0];
                T coeff = -sim.dt * sim.dt / rho_rest / sim.dx;
                T center_p = solved_p(g.idx);
                for (int d = 0; d < dim; ++d) {
                    IV neighbor = node;
                    neighbor[d] -= 1;
                    if (!isSolid(neighbor)) g.new_v[d] = coeff * (center_p - getPressure(neighbor));
                    MATH_TOOLS::clamp(g.new_v[d], -(T)0.5 * sim.dx, (T)0.5 * sim.dx);
                    neighbor = node;
                    neighbor[d] += 1;
                    if (!isSolid(neighbor)) pressure_grid[neighbor].new_v[d] = coeff * (getPressure(neighbor) - center_p);
                    MATH_TOOLS::clamp(pressure_grid[neighbor].new_v[d], -(T)0.5 * sim.dx, (T)0.5 * sim.dx);
                }
            });
        }

        {
            ZIRAN_TIMER();
            sim.parallel_for_updating_grid([&](int i) {
                TV& Xp = Xarray[i];
                for (int d = 0; d < dim; ++d) {
                    TV face_pos = Xp - TV::Ones() * sim.dx;
                    face_pos[d] += (T)0.5 * sim.dx;
                    BSplineWeights<T, dim, 1> spline(face_pos, sim.dx);
                    uint64_t offset = MpmSimulationBase<T, dim>::SparseMask::Linear_Offset(to_std_array(spline.base_node));
                    pressure_grid.iterateKernel(spline, offset, [&](const IV& node, T w, const TV& dw, GridState<T, dim>& g) { Xp[d] += w * g.new_v[d]; });
                }
            });
        }
    }
};

} // namespace ZIRAN