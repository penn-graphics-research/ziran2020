#pragma once

#include <Ziran/Math/Geometry/Visualizer.h>
//#include "../aqua/fivepoint/ImplicitDensityProjection.h"
#include "LegoMarker.h"
#include "HeatSolver.h"
#include <MPM/MpmSimulationBase.h>
#include <MPM/Force/FJMixedMpmForceHelper.h>
#include <MPM/Force/MpmParallelPinningForceHelper.h>
#include <Ziran/Math/Linear/DirectSolver.h>
#include <Ziran/CS/DataStructure/SpatialHash.h>
#include <eigen3/unsupported/Eigen/src/SparseExtra/MarketIO.h>

#include <Partio.h>

namespace ZIRAN {

#define LINEAR_KERNEL

template <class T, int dim>
class SplittingSimulation : public MpmSimulationBase<T, dim> {
public:
    using Base = MpmSimulationBase<T, dim>;

    typedef Vector<T, dim> TV;
    typedef Vector<int, 3> IV3;
    typedef Vector<int, dim> IV;
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, 4> TV4;
    typedef Matrix<T, 4, 4> TM4;
    typedef Eigen::SparseMatrix<T> SM;
    typedef FJMixedMpmForceHelper<CorotatedElasticity<T, dim>, QuadraticVolumePenalty<T, dim>> FJHelpher;
    typedef Vector<T, Eigen::Dynamic> TStack;

    using Base::grid;
    using Base::particles;
    using Base::scene;

    T boundary_ppc = -1;
    T boundary_particle_vol;
    StdVector<TV> boundary_positions;
    StdVector<TV> boundary_normals;
    void sampleBoundaryParticlesInCollisionObjects(const AnalyticCollisionObject<T, dim>& object, TV min_corner, TV max_corner, T distance);

    // grid1: velocities affected by original particles
    // grid2: pressures affected by original particles
    // grid3: velocities affected by spreading particles
    // grid4: pressures affected by spreading particles
    // grid5: fluid-boundary
    // gird6: fluid-solid
    // m : 1
    // v[0] : z
    // v[1] : p
    MpmGrid<T, dim, 1> grid_comm;
    bool solid_Q2Q0 = true;
    bool fluid_Q1Q0 = true;
    static constexpr int aod = 0;

    Eigen::SparseMatrix<T> M_inv, G, D, S, W;
    TStack a, b;
    int num_v, num_p;

    T implicit_penalty_collision = 0;
    bool use_position_correction = false;
    bool use_laser = false;
    bool diag_scaling = false;
    HeatSolver<T, dim> heat_solver;
    std::vector<T> temperature;
    std::vector<std::pair<int, T>> phase;

    T relative_tol = -1;
    T abs_tol = -1;
    int max_iter = -1;

    MATERIAL_PHASE_ENUM category;

    enum SOLVER_TYPE { EIGEN,
        AMGCL,
        AMGCL_CUDA } solverType
        = AMGCL;

    SplittingSimulation() = default;

    void initialize() override
    {
        Base::initialize();
    }

    void particlesToGrid()
    {
        ZIRAN_TIMER();
        if (Base::symplectic) {
            Base::forcesUpdateParticleState();
            Base::particlesToGrid();
        }
        else {
            Base::particlesToGrid();
        }
    }

    void gridUpdate()
    {
        ZIRAN_TIMER();
        if (category == MATERIAL_PHASE_FLUID) {
            grid.iterateGrid([&](IV node, GridState<T, dim>& g) {
                g.new_v = g.v;
            });
            return;
        }
        FJHelpher::state = FJHelpher::SOLVE_F;
        TV tmp = Base::gravity;
        Base::gravity = TV::Zero();
        if (Base::symplectic) {
            Base::gridVelocityExplicitUpdate(Base::dt);
        }
        else {
            Base::backwardEulerStep();
        }
        Base::gravity = tmp;
    }

    void setSolver(const T relTol, const T absTol, int maxIter, SOLVER_TYPE type = AMGCL_CUDA)
    {
        relative_tol = relTol;
        abs_tol = absTol;
        max_iter = maxIter;
        solverType = type;
    }

    bool particleInSolid(const TV& Xp)
    {
        BSplineWeights<T, dim, 1> spline(Xp - TV::Ones() * Base::dx * 0.5, Base::dx);
        TV xi = (spline.base_node + IV::Ones()).template cast<T>() * Base::dx;
        TV vi;
        TM normal_basis;
        return AnalyticCollisionObject<T, dim>::multiObjectCollision(Base::collision_objects, xi, vi, normal_basis);
    }

    bool gridInSolid(const IV& node)
    {
        TV xi = (node + IV::Ones()).template cast<T>() * Base::dx;
        TV vi;
        TM normal_basis;
        return AnalyticCollisionObject<T, dim>::multiObjectCollision(Base::collision_objects, xi, vi, normal_basis);
    }

    SpatialHash<T, dim> spatial_hash;
    StdVector<TV> solid_positions;
    StdVector<TV> solid_normals;
    StdVector<T> solid_lens;
    StdVector<TV> interface_GQ;
    StdVector<TV> interface_N;
    StdVector<T> interface_L;

    void rebuildInterfaceQuad();

    template <int degree, int order>
    void fluidKernel(MpmGrid<T, dim, degree>& grid3, MpmGrid<T, dim, order>& grid4, MpmGrid<T, dim, order>& grid5, SM& M_inv, SM& G, SM& D, SM& S, TStack& a, TStack& b);
    template <int degree, int order>
    void solidKernel(MpmGrid<T, dim, degree>& grid3, MpmGrid<T, dim, order>& grid4, SM& M_inv, SM& G, SM& D, SM& S, TStack& a, TStack& b);

    template <int degree, int order>
    void buildFluidSystem();
    template <int order>
    void buildSolidSystem();

    void buildSystem()
    {
        if (category == MATERIAL_PHASE_SOLID)
            if (solid_Q2Q0)
                buildSolidSystem<0>();
            else
                buildSolidSystem<1>();
        else {
            if (fluid_Q1Q0)
                buildFluidSystem<1, 0>();
            else {
                ZIRAN_ASSERT(false, "not supported any more");
                // buildFluidSystem<2, 1>();
            }
        }
    }

    template <int degree>
    void buildWMatrix(MpmGrid<T, dim, degree>& gridv)
    {
        ZIRAN_TIMER();
        MpmGrid<T, dim, aod> grid6;
        grid6.pollute(interface_GQ, Base::dx, 0, 1, -TV::Ones() * Base::dx * 0.5);
        for (auto& Xp : interface_GQ) {
            BSplineWeights<T, dim, aod> spline6(Xp - TV::Ones() * Base::dx * 0.5, Base::dx);
            uint64_t offset6 = Base::SparseMask::Linear_Offset(to_std_array(spline6.base_node));
            grid6.iterateKernel(spline6, offset6, [&](IV node6, T w6, TV dw6, GridState<T, dim>& g6) {
                g6.m += w6;
            });
        }
        W = Eigen::SparseMatrix<T>(grid6.getNumNodes(), gridv.getNumNodes() * dim);
        std::vector<Eigen::Triplet<T>> W_tri(interface_GQ.size() * gridv.kernel_size * grid6.kernel_size * dim);
        tbb::parallel_for(0, (int)interface_GQ.size(), [&](int i) {
            TV Xp = interface_GQ[i];
            TV normal = interface_N[i];
            T len = interface_L[i];
            BSplineWeights<T, dim, degree> splinev(degree == 2 ? Xp : (Xp - TV::Ones() * Base::dx * 0.5), Base::dx);
            BSplineWeights<T, dim, aod> spline6(Xp - TV::Ones() * Base::dx * 0.5, Base::dx);
            uint64_t offsetv = Base::SparseMask::Linear_Offset(to_std_array(splinev.base_node));
            uint64_t offset6 = Base::SparseMask::Linear_Offset(to_std_array(spline6.base_node));
            int cnt = i * gridv.kernel_size * grid6.kernel_size * dim;
            gridv.iterateKernel(splinev, offsetv, [&](IV nodev, T wv, TV dwv, GridState<T, dim>& gv) {
                int idxv = gv.idx;
                if (idxv < 0) return;
                grid6.iterateKernel(spline6, offset6, [&](IV node6, T w6, TV dw6, GridState<T, dim>& g6) {
                    int idx6 = g6.idx;
                    if (idx6 < 0) return;
                    for (int alpha = 0; alpha < dim; ++alpha)
                        W_tri[cnt++] = Eigen::Triplet<T>(idx6, idxv * dim + alpha, len * wv * w6 * normal[alpha]);
                });
            });
        });
        W.setFromTriplets(W_tri.begin(), W_tri.end());
    }

    // M v + G p = a
    // D v + S p = b
    TStack solved_p, solved_v;
    int analytic_sample_condition_checking_serial_number;
    void solveSystem()
    {
        ZIRAN_TIMER();
        Eigen::SparseMatrix<T> left = S - D * M_inv * G;
        TStack right = b - D * M_inv * a;

        if (solverType == EIGEN)
            solved_p = DirectSolver<T>::solve(left, right, DirectSolver<T>::EIGEN);
        else if (solverType == AMGCL)
            solved_p = DirectSolver<T>::solve(left, right, DirectSolver<T>::AMGCL, relative_tol, abs_tol, max_iter, -1, 1, false, diag_scaling);
        else
            solved_p = DirectSolver<T>::solve(left, right, DirectSolver<T>::AMGCL_CUDA, relative_tol, abs_tol, max_iter, -1, 1, false, diag_scaling);

        constructVelocityFromSolvedP();
    }

    StdVector<TV> override_X;
    StdVector<TV> override_V;
    StdVector<TM> override_gradVp;
    StdVector<T> override_J;
    void constructVelocityFromSolvedP()
    {
        ZIRAN_TIMER();
        solved_v = M_inv * a - M_inv * G * solved_p;

        if (category == MATERIAL_PHASE_SOLID) {
            ZIRAN_ASSERT(grid.getNumNodes() * dim == solved_v.rows(), "dimension not match");
            grid.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
                for (int d = 0; d < dim; ++d)
                    g.new_v[d] = solved_v(g.idx * dim + d);
            });
            if (solid_Q2Q0) {
                MpmGrid<T, dim, 0> grid4;
                grid4.pollute(particles.X.array, Base::dx, 0, 1, -TV::Ones() * Base::dx * 0.5);
                for (auto Xp : particles.X.array) {
                    BSplineWeights<T, dim, 0> spline(Xp - TV::Ones() * Base::dx * 0.5, Base::dx);
                    uint64_t offset = Base::SparseMask::Linear_Offset(to_std_array(spline.base_node));
                    grid4.iterateKernel(spline, offset, [&](const IV& node, T w, const TV& dw, GridState<T, dim>& g) {
                        g.m += w;
                    });
                }
                int num4 = grid4.getNumNodes();
                ZIRAN_ASSERT(num4 == solved_p.rows(), "This is crazy.");
                override_J.resize(particles.count);
                auto* j_model = particles.DataManager::getPointer(FJHelpher::j_constitutive_model_name());
                for (int i = 0; i < particles.count; ++i) {
                    TV Xp = particles.X.array[i];
                    BSplineWeights<T, dim, 1> spline(Xp - TV::Ones() * Base::dx * 0.5, Base::dx);
                    ZIRAN_ASSERT(grid4[spline.base_node].idx >= 0, "This is crazy.");
                    T p = solved_p(grid4[spline.base_node].idx);
                    override_J[i] = -p / (*j_model)[i].lambda + 1;
                }
            }
        }
        else {
            if (fluid_Q1Q0) {
                MpmGrid<T, dim, 1>& grid3 = grid_comm;
                grid3.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
                    for (int d = 0; d < dim; ++d)
                        g.new_v[d] = solved_v(g.idx * dim + d);
                });
                override_X.resize(particles.count);
                override_V.resize(particles.count);
                override_gradVp.resize(particles.count);
                tbb::parallel_for(0, (int)particles.count, [&](int i) {
                    TV Xp = particles.X.array[i];
                    TV picV = TV::Zero();
                    TM gradVp = TM::Zero();
                    BSplineWeights<T, dim, 1> spline(Xp - TV::Ones() * Base::dx * 0.5, Base::dx);
                    uint64_t offset = Base::SparseMask::Linear_Offset(to_std_array(spline.base_node));
                    grid3.iterateKernel(spline, offset, [&](IV node, T w, TV dw, GridState<T, dim>& g) {
                        picV += w * g.new_v;
                        gradVp.noalias() += g.new_v * dw.transpose();
                    });
                    override_X[i] = Xp + picV * Base::dt;
                    override_V[i] = picV;
                    override_gradVp[i] = gradVp;
                });
            }
            else {
                MpmGrid<T, dim> grid3;
                grid3.pollute(lego.volume_GQ, Base::dx);
                for (int i = 0; i < lego.volume_GQ.size(); ++i) {
                    TV& Xp = lego.volume_GQ[i];
                    BSplineWeights<T, dim> spline(Xp, Base::dx);
                    uint64_t offset = MpmSimulationBase<T, dim>::SparseMask::Linear_Offset(to_std_array(spline.base_node));
                    grid3.iterateKernel(spline, offset, [&](const IV& node, T w, const TV& dw, GridState<T, dim>& g) {
                        g.m += w;
                    });
                }
                ZIRAN_ASSERT(grid3.getNumNodes() * dim == solved_v.rows(), "dimension not match");
                grid3.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
                    for (int d = 0; d < dim; ++d)
                        grid[node].new_v[d] = solved_v(g.idx * dim + d);
                });
            }
        }
    }

    void gridToParticles(double dt) override
    {
        Base::gridToParticles(dt);

        if (category == MATERIAL_PHASE_SOLID) {
            auto* F_pointer = particles.DataManager::getPointer(F_name<T, dim>());
            auto* J_pointer = particles.DataManager::getPointer(J_name<T>());
            for (int i = 0; i < particles.count; ++i) {
                // (*J_pointer)[i] = -p / (*j_model)[i].lambda + 1;
                (*J_pointer)[i] = (*F_pointer)[i].determinant();
                if (solid_Q2Q0) (*J_pointer)[i] = override_J[i];
                MATH_TOOLS::clamp((*J_pointer)[i], (T)0.3, (T)3);
            }
        }
        else {
            auto& Xarray = particles.X.array;
            auto& Varray = particles.V.array;
            tbb::parallel_for(0, (int)particles.count, [&](int i) {
                if (fluid_Q1Q0) {
                    Xarray[i] = override_X[i];
                    Varray[i] = override_V[i];
                    Base::scratch_gradV[i] = override_gradVp[i];
                }
                for (size_t k = 0; k < Base::collision_objects.size(); k++) {
                    T phi;
                    TV n;
                    if (Base::collision_objects[k]->queryInside(Xarray[i], phi, n, -3 * Base::dx)) {
                        Xarray[i] -= (phi + 2 * Base::dx) * n;
                        Varray[i] = TV::Zero();
                    }
                }
            });
        }
    }

    using TVStack = Matrix<T, dim, Eigen::Dynamic>;
    TVStack writeTVStack;
    LegoMarker<T, dim> lego;
    virtual void writeState(std::ostream& out) override
    {
        Base::writeState(out);
        visualize_points(*this, particles.X.array, particles.V.array, "xv");
        static bool first_run = true;
        if (first_run) {
            visualize_points(*this, boundary_positions, boundary_normals, "boundary");
            first_run = false;
        }
        return;
        visualize_points(*this, lego.volume_GQ, "volume_GQ");
        visualize_points(*this, lego.face_GQ, lego.face_N, "face_GQ");
        visualize_points(*this, interface_GQ, interface_N, "interface_GQ");
        // {
        //     auto* pin_pointer = particles.DataManager::getPointer(MpmPinningForceHelper<T, dim>::target_name());
        //     if (pin_pointer != nullptr) {
        //         StdVector<TV> positions;
        //         StdVector<TV> normals;
        //         for (int i = 0; i < particles.count; ++i)
        //             for (int c = 0; c < (*pin_pointer)[i].x.size(); ++c) {
        //                 positions.push_back(before_advect[i]);
        //                 normals.push_back((*pin_pointer)[i].x[c] - before_advect[i]);
        //             }
        //         visualize_points(*this, before_advect, "before_advect");
        //         visualize_points(*this, positions, normals, "spring");
        //     }
        // }
        return;
        visualize_points(*this, interface_GQ, interface_N, "interface");

        if constexpr (dim == 2) {
            visualize_points(*this, lego.face_GQ, "face_GQ");
            visualize_points(*this, lego.volume_GQ, "volume_GQ");
            StdVector<TV> points;
            StdVector<T> info;
            lego.cell_type.iterateGrid([&](int x, int y, int type) {
                points.emplace_back(((T)x + 1) * Base::dx, ((T)y + 1) * Base::dx);
                info.emplace_back((T)type);
            });
            visualize_points(*this, points, info, "cell_type");
            visualize_points(*this, before_advect, "before_advect");
        }

        if (use_laser) {
            std::string filename = SimulationBase::output_dir.absolutePath(SimulationBase::outputFileName("details", ".bgeo"));
            Partio::ParticlesDataMutable* parts = Partio::create();

            // visualize particles info
            Partio::ParticleAttribute posH, temperatureH, surfaceH, laserH, phaseH;
            posH = parts->addAttribute("position", Partio::VECTOR, 3);
            temperatureH = parts->addAttribute("temperature", Partio::VECTOR, 1);
            surfaceH = parts->addAttribute("surface", Partio::VECTOR, 1);
            laserH = parts->addAttribute("laser", Partio::VECTOR, 1);
            phaseH = parts->addAttribute("phase", Partio::VECTOR, 1);

            initializeTemperature();
            heat_solver.onSurface.resize(particles.count, false);
            heat_solver.inLaser.resize(particles.count, false);
            for (int k = 0; k < particles.count; k++) {
                int idx = parts->addParticle();
                float* posP = parts->dataWrite<float>(posH, idx);
                float* temperatureP = parts->dataWrite<float>(temperatureH, idx);
                float* surfaceP = parts->dataWrite<float>(surfaceH, idx);
                float* laserP = parts->dataWrite<float>(laserH, idx);
                float* phaseP = parts->dataWrite<float>(phaseH, idx);
                for (int d = 0; d < 3; ++d) posP[d] = 0;
                for (int d = 0; d < dim; ++d) posP[d] = particles.X.array[k](d);
                temperatureP[0] = temperature[k];
                surfaceP[0] = heat_solver.onSurface[k] ? (T)1 : (T)0;
                laserP[0] = heat_solver.inLaser[k] ? (T)1 : (T)0;
                phaseP[0] = (T)phase[k].first;
            }

            Partio::write(filename.c_str(), *parts);
            parts->release();
        }
    }

    bool use_vp = true;
    bool apply_heat_solver = false;
    bool apply_surface_tension = false;
    std::vector<T> old_m;
    std::vector<TV> old_p;

    void initializeTemperature()
    {
        if (temperature.size() == 0) {
            temperature.resize(particles.count, 298);
            for (int i = 0; i < particles.count; ++i)
                if (particles.X.array[i](1) > 0.0205) {
                    temperature[i] = 298;
                }
            phase.resize(particles.count, std::make_pair(0, 0));
        }
    }

    //    DensityProjection<T, dim> densitySolver;
    StdVector<TV> before_advect;
    void advanceOneTimeStep(double dt)
    {
        ZIRAN_TIMER();
        ZIRAN_INFO("Advance one time step from time ", std::setw(7), Base::step.time, " with                     dt = ", dt);
        Base::reinitialize();

        if (use_laser) {
            if (Base::symplectic) {
                Base::forcesUpdateParticleState();
                Base::particlesToGrid();
            }
            else {
                Base::particlesToGrid();
            }

            initializeTemperature();
            T center = 0.0202 + 0.4 * Base::step.time;
            if (center > 0.0208)
                center = 0;
            heat_solver.solve(*this, center, temperature, phase);
            return;
        }

        particlesToGrid();
        gridUpdate();

        {
            std::vector<T> particles_temperature(particles.count, 0);
            if (apply_heat_solver) {
                tbb::parallel_for(0, (int)particles.count, [&](int i) {
                    TV& Xp = particles.X.array[i];
                    particles_temperature[i] = (5.4 - Xp(0)) / 0.4;
                });
            }
        }

        if (use_vp) {
            buildSystem();
            solveSystem();
        }
        else {
            FJHelpher::state = FJHelpher::SOLVE_J;
            TV tmp_gravity = Base::gravity;
            Base::gravity = TV::Zero();
            Base::backwardEulerStep();
            Base::gravity = tmp_gravity;
        }

        before_advect = particles.X.array;

        for (size_t k = 0; k < Base::collision_objects.size(); ++k)
            if (Base::collision_objects[k]->updateState)
                Base::collision_objects[k]->updateState(Base::step.time + dt, *Base::collision_objects[k]);

        gridToParticles(dt);
        /*
        tbb::parallel_for(0, (int)particles.count, [&](int i) {
            TV& Xp = particles.X.array[i];
            TV& Vp = particles.V.array[i];
            const StdVector<std::unique_ptr<AnalyticCollisionObject<T, dim>>>& my_collisionObjects = Base::collision_objects;
            for (size_t k = 0; k < my_collisionObjects.size(); k++) {
                T phi;
                TV n;
                if (my_collisionObjects[k]->queryInside(Xp, phi, n, 0.5 * Base::dx)) {
                    Xp -= Vp * Base::dt;
                    if (Vp.dot(n) < 0) {
                        Vp -= 2 * Vp.dot(n) * n;
                    }
                    Xp += Vp * Base::dt;
                }
            }
        });
        */

        if (use_position_correction && particles.count > 0) {
            ZIRAN_TIMER();
            ZIRAN_ASSERT("No PC implemented here!");
            //            ZIRAN_ASSERT(false, "commented");
            //            densitySolver.solve(*this);
        }
    }

    const char* name() override { return "splitting"; }
};
} // namespace ZIRAN
