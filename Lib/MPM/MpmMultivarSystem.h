#pragma once

#include <tbb/tbb.h>
#include <MPM/MpmGrid.h>
#include <Ziran/CS/Util/Timer.h>
#include <Ziran/CS/Util/AttributeNamesForward.h>
#include <Eigen/SparseCholesky>
#include <Ziran/Math/Geometry/CollisionObject.h>
#include <Ziran/Math/Linear/DirectSolver.h>

namespace ZIRAN {

template <class T, int dim>
class MpmSimulationBase;

template <class T, int dim>
class MpmMultivarSystem {
    typedef Vector<T, dim> TV;
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<int, dim> IV;
    typedef Vector<T, Eigen::Dynamic> TStack;
    typedef typename MpmGrid<T, dim>::SparseMask SparseMask;

    MpmSimulationBase<T, dim>& sim;
    // respective data
    MpmGrid<T, dim>& grid1;
    StdVector<uint64_t>& particle_base_offset1;
    MpmGrid<T, dim, 1> grid2;
    StdVector<uint64_t> particle_base_offset2;
    std::vector<bool> particle_inextensibility;
    // linear system data
    int num_v, num_lambda;
    Eigen::SparseMatrix<T> A_inv, B;
    TStack v_star, zero;

    int collision_cnt;
    std::vector<T> B_rhs;

    void resetSparseMatrix(Eigen::SparseMatrix<T>& matrix, int rows, int cols)
    {
        matrix.resize(0, 0);
        matrix.setZero();
        matrix.data().squeeze();
        matrix.resize(rows, cols);
    }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MpmMultivarSystem(MpmSimulationBase<T, dim>& sim)
        : sim(sim)
        , grid1(sim.grid)
        , particle_base_offset1(sim.particle_base_offset)
    {
    }

    void polluteGrid2()
    {
        ZIRAN_TIMER();
        auto& Xarray = sim.particles.X.array;
        if (sim.particles.count != (int)particle_base_offset2.size())
            particle_base_offset2.resize(sim.particles.count);
        T one_over_dx = (T)1 / sim.dx;
        grid2.page_map->Clear();
        for (int i = 0; i < sim.particles.count; ++i) {
            uint64_t offset = SparseMask::Linear_Offset(to_std_array(baseNode<1, T, dim>((Xarray[i] - TV::Ones() * 0.5 * sim.dx) * one_over_dx)));
            particle_base_offset2[i] = offset;
            if (!i || (particle_base_offset2[i] != particle_base_offset2[i - 1])) {
                grid2.page_map->Set_Page(offset);
                if constexpr (dim == 2) {
                    auto x = 1 << SparseMask::block_xbits;
                    auto y = 1 << SparseMask::block_ybits;
                    for (int i = 0; i < 2; ++i)
                        for (int j = 0; j < 2; ++j)
                            grid2.page_map->Set_Page(SparseMask::Packed_Add(
                                offset, SparseMask::Linear_Offset(x * i, y * j)));
                }
                else {
                    auto x = 1 << SparseMask::block_xbits;
                    auto y = 1 << SparseMask::block_ybits;
                    auto z = 1 << SparseMask::block_zbits;
                    for (int i = 0; i < 2; ++i)
                        for (int j = 0; j < 2; ++j)
                            for (int k = 0; k < 2; ++k)
                                grid2.page_map->Set_Page(SparseMask::Packed_Add(
                                    offset, SparseMask::Linear_Offset(x * i, y * j, z * k)));
                }
            }
        }
        grid2.page_map->Update_Block_Offsets();

        auto grid_array = grid2.grid->Get_Array();
        auto blocks = grid2.page_map->Get_Blocks();
        for (int b = 0; b < (int)blocks.second; ++b) {
            auto base_offset = blocks.first[b];
            std::memset(&grid_array(base_offset), 0, (size_t)(1 << MpmGrid<T, dim, 1>::log2_page));
            GridState<T, dim>* g = reinterpret_cast<GridState<T, dim>*>(&grid_array(base_offset));
            for (int i = 0; i < (int)SparseMask::elements_per_block; ++i)
                g[i].idx = -1;
        }
    }

    void findDoF()
    {
        ZIRAN_TIMER();
        auto& Xarray = sim.particles.X.array;
        particle_inextensibility.assign(sim.particles.count, false);
        for (auto iter = sim.particles.iter(inextensibility_name<bool>()); iter; ++iter) {
            int p = iter.entryId(); // global index of this particle
            particle_inextensibility[p] = true;
        }
        sim.parallel_for_updating_grid([&](int i) {
            if (!particle_inextensibility[i])
                return;
            TV Xp = Xarray[i];
            BSplineWeights<T, dim, 1> spline(Xp - TV::Ones() * 0.5 * sim.dx, sim.dx);
            grid2.iterateKernel(spline, particle_base_offset2[i], [&](const IV& node, T w, const TV& dw, GridState<T, dim>& g) {
                g.m = (T)1;
            });
        });
        num_v = sim.num_nodes;
        num_lambda = grid2.getNumNodes();
        sim.parallel_for_updating_grid([&](int i) {
            if (!particle_inextensibility[i])
                return;
            TV Xp = Xarray[i];
            BSplineWeights<T, dim, 1> spline(Xp - TV::Ones() * 0.5 * sim.dx, sim.dx);
            grid2.iterateKernel(spline, particle_base_offset2[i], [&](const IV& node, T w, const TV& dw, GridState<T, dim>& g) {
                if (g.idx < 0) {
                    ZIRAN_ASSERT(false, "this is impossible!!!");
                }
            });
        });
    }

    void buildMatrix()
    {
        ZIRAN_TIMER();
        auto& Xarray = sim.particles.X.array;
        auto* F_pointer = &sim.particles.DataManager::get(F_name<T, dim>());
        auto* vol_pointer = &sim.particles.DataManager::get(element_measure_name<T>());
        // compute A_inv
        resetSparseMatrix(A_inv, num_v * dim, num_v * dim);
        grid1.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
            for (int d = 0; d < dim; ++d) {
                int idx = g.idx * dim + d;
                A_inv.coeffRef(idx, idx) = sim.dt / g.m;
            }
        });
        // compute B
        collision_cnt = 0;
        std::vector<Eigen::Triplet<T>> B_tri;
        B_rhs.clear();
        grid1.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
            for (int k = 0; k < (int)sim.collision_objects.size(); ++k) {
                TV xi = node.template cast<T>() * sim.dx;
                TV vi = g.new_v;
                TV n;
                TV v_object;
                if (sim.collision_objects[k]->type == AnalyticCollisionObject<T, dim>::GHOST)
                    continue;
                bool collided = sim.collision_objects[k]->detectAndResolveCollisionWithObjectSpeed(xi, vi, n, v_object);
                if (collided) {
                    if (sim.collision_objects[k]->type == AnalyticCollisionObject<T, dim>::STICKY) {
                        for (int d = 0; d < dim; ++d) {
                            B_tri.emplace_back(collision_cnt++, g.idx * dim + d, (T)1);
                            B_rhs.emplace_back(v_object[d]);
                        }
                    }
                    if (sim.collision_objects[k]->type == AnalyticCollisionObject<T, dim>::SLIP) {
                        for (int d = 0; d < dim; ++d)
                            B_tri.emplace_back(collision_cnt, g.idx * dim + d, n[d]);
                        collision_cnt++;
                        B_rhs.emplace_back(v_object.dot(n));
                    }
                }
            }
        });
        int B_size = B_tri.size();
        resetSparseMatrix(B, collision_cnt + num_lambda, num_v * dim);
        B_tri.resize(B_size + sim.particles.count * grid1.kernel_size * grid2.kernel_size * dim);
        tbb::parallel_for(0, sim.particles.count, [&](int i) {
            if (!particle_inextensibility[i])
                return;
            TV Xp = Xarray[i];
            TM F = (*F_pointer)[i];
            T J = F.determinant();
            T vol0 = (*vol_pointer)[i];
            int cnt = B_size + i * grid1.kernel_size * grid2.kernel_size * dim;
            BSplineWeights<T, dim> spline1(Xp, sim.dx);
            BSplineWeights<T, dim, 1> spline2(Xp - TV::Ones() * 0.5 * sim.dx, sim.dx);
            grid1.iterateKernel(spline1, particle_base_offset1[i], [&](IV node1, T w1, TV dw1, GridState<T, dim>& g1) {
                int idx1 = g1.idx;
                grid2.iterateKernel(spline2, particle_base_offset2[i], [&](IV node2, T w2, TV dw2, GridState<T, dim>& g2) {
                    int idx2 = g2.idx;
                    TV a = F.col(0);
                    for (int alpha = 0; alpha < dim; ++alpha)
                        B_tri[cnt++] = Eigen::Triplet<T>(collision_cnt + idx2, idx1 * dim + alpha, vol0 * J * w2 * a(alpha) * a.dot(dw1));
                });
            });
        });
        B.setFromTriplets(B_tri.begin(), B_tri.end());
    }

    void buildRhs()
    {
        ZIRAN_TIMER();
        v_star = TStack::Zero(num_v * dim);
        grid1.iterateGrid([&](IV node, GridState<T, dim>& g) {
            for (int d = 0; d < dim; ++d)
                v_star(g.idx * dim + d) = g.new_v[d] * g.m / sim.dt;
        });
        zero = TStack::Zero(collision_cnt + num_lambda);
        for (int i = 0; i < collision_cnt; ++i)
            zero(i) = B_rhs[i];
    }

    // A   B^T   D^T
    // B   0     0
    // D   0     0
    bool use_amgcl = false;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>> solverLDLT;
    TStack solved_lambda, solved_v;
    Eigen::ConjugateGradient<Eigen::SparseMatrix<T>, Eigen::Lower | Eigen::Upper, Eigen::IncompleteCholesky<T>> cg;
    void solveLinearSystem()
    {
        ZIRAN_TIMER();
        ZIRAN_WARN("S row : ", B.rows(), " , col : ", B.cols());
        ZIRAN_WARN("D row : ", A_inv.rows(), " , col : ", A_inv.cols());
        Eigen::SparseMatrix<T> left = B * A_inv * B.transpose();
        TStack right = B * A_inv * v_star - zero;

        // solverLDLT.compute(left);
        // solved_lambda = solverLDLT.solve(right);
        if (!use_amgcl) {
            ZIRAN_INFO("Solving with CG");
            size_t max_CG_iteration = 30000;
            T CG_tolerance = 1e-4;
            cg.setMaxIterations(max_CG_iteration);
            cg.setTolerance(CG_tolerance);
            cg.compute(left);
            solved_lambda = cg.solve(right);
            ZIRAN_INFO("#iterations:    ", cg.iterations());
            ZIRAN_INFO("estimated error:    ", cg.error());
            // std::cout << "#iterations:     " << cg.iterations() << std::endl;
            // std::cout << "estimated error: " << cg.error() << std::endl;
        }
        else {
            ZIRAN_INFO("Solving with AMGCL");
            solved_lambda = DirectSolver<T>::solve(left, right, DirectSolver<T>::AMGCL, 1e-3, -1, 10000);
        }

        solved_v = A_inv * v_star - A_inv * B.transpose() * solved_lambda;
        grid1.iterateGrid([&](IV node, GridState<T, dim>& g) {
            for (int d = 0; d < dim; ++d)
                g.new_v[d] = solved_v(g.idx * dim + d);
        });
        ZIRAN_WARN("After solve : ", solved_v.maxCoeff(), " ", solved_v.minCoeff(), " ", (B * solved_v).squaredNorm());
    }

    void solve()
    {
        polluteGrid2();
        findDoF();
        buildMatrix();
        buildRhs();
        solveLinearSystem();
    }
};

} // namespace ZIRAN
