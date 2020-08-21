#pragma once

#include <Ziran/Sim/SimulationBase.h>
#include "../splitting/SplittingSimulation.h"
#include "TwoPhasePositionCorrector.h"
//#include "BuildLevelset.h"
#include "NilsPositionCorrector.h"
#include <Ziran/CS/Util/AttributeNamesForward.h>

namespace ZIRAN {

template <class T, int dim>
class CouplingSimulation : public SimulationBase {
public:
    using Base = SimulationBase;

    typedef Vector<T, dim> TV;
    typedef Vector<int, dim> IV;
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, Eigen::Dynamic> TStack;
    typedef Matrix<T, dim, Eigen::Dynamic> TVStack;

    SplittingSimulation<T, dim> e1;
    SplittingSimulation<T, dim> e2;

    TVStack surface_tension_delta1;
    TVStack surface_tension_delta2;
    NilsPositionCorrector<T, dim> nils_position_corrector1;
    NilsPositionCorrector<T, dim> nils_position_corrector2;

    /////////////////////////////////////////////////////////////////
    //////////////////////// INTERFACE START ////////////////////////
    /////////////////////////////////////////////////////////////////
    bool use_position_correction = true;
    bool use_surface_tension = false;

    T kkt_relax = 0; // this should be set to 1/fluid_rho * penalty_stiffness_inverse.
    T relative_tol, abs_tol;
    int max_iter;
    StdVector<TV> wall_particles;
    enum SOLVER_TYPE { EIGEN,
        AMGCL,
        AMGCL_CUDA } solverType
        = AMGCL_CUDA;
    void setSolver(const T relTol, const T absTol, int maxIter, SOLVER_TYPE type = AMGCL_CUDA)
    {
        relative_tol = relTol;
        abs_tol = absTol;
        max_iter = maxIter;
        solverType = type;
    }
    /////////////////////////////////////////////////////////////////
    //////////////////////// INTERFACE END //////////////////////////
    /////////////////////////////////////////////////////////////////

    CouplingSimulation()
        : Base()
        , nils_position_corrector1(e1)
        , nils_position_corrector2(e2)
    {
    }

    double calculateDt() override
    {
        return std::min(e1.calculateDt(), e2.calculateDt());
    }

    void restart(int new_restart_frame) override
    {
        e1.restart(new_restart_frame);
        e2.restart(new_restart_frame);
        Base::restart(new_restart_frame);
    }

    void initialize() override
    {
        e1.autorestart = e2.autorestart = false;
        e1.initialize();
        e2.initialize();
        Base::initialize();
    }

    void reinitialize() override
    {
        e1.reinitialize();
        e2.reinitialize();
        Base::reinitialize();
    }

    int interface_cnt;
    void buildSystem()
    {
        ZIRAN_TIMER();
        SplittingSimulation<T, dim>& e_solid = (e1.category == MATERIAL_PHASE_SOLID) ? e1 : e2;
        SplittingSimulation<T, dim>& e_fluid = (e1.category == MATERIAL_PHASE_FLUID) ? e1 : e2;
        e_fluid.solid_positions = e_solid.particles.X.array;
        e_fluid.solid_normals.resize(e_solid.particles.count);
        e_fluid.solid_lens.resize(e_solid.particles.count);
        auto* J_pointer = e_solid.particles.DataManager::getPointer(J_name<T>());
        auto* vol_pointer = e_solid.particles.DataManager::getPointer(element_measure_name<T>());
        tbb::parallel_for(0, (int)e_solid.particles.count, [&](int i) {
            TV normal = TV::Zero();
            T vol = (*vol_pointer)[i];
            T len = (T)2 * std::sqrt(std::max(vol, (T)0) / M_PI);
            BSplineWeights<T, dim> spline(e_solid.particles.X.array[i], e_solid.dx);
            e_solid.grid.iterateKernel(spline, e_solid.particle_base_offset[i], [&](IV node, T w, TV dw, GridState<T, dim>& g) {
                normal += g.m * dw;
            });
            e_fluid.solid_normals[i] = normal.normalized();
            e_fluid.solid_lens[i] = len;
        });
        e_fluid.buildSystem();
        e_solid.interface_GQ = e_fluid.interface_GQ;
        e_solid.interface_N = e_fluid.interface_N;
        e_solid.interface_L = e_fluid.interface_L;
        e_solid.buildSystem();
        interface_cnt = e1.W.rows();
    }

    std::vector<Eigen::Triplet<T>> coupling_tri;
    void fillInCouplingTri(int offset0, int offset1, const Eigen::SparseMatrix<T>& mat)
    {
        ZIRAN_TIMER();
        for (int k = 0; k < mat.outerSize(); ++k)
            for (typename Eigen::SparseMatrix<T>::InnerIterator it(mat, k); it; ++it)
                coupling_tri.emplace_back(offset0 + it.row(), offset1 + it.col(), it.value());
    }

    TStack coupling_rhs;
    void fillInCouplingRhs(int offset, const TStack& rhs)
    {
        ZIRAN_TIMER();
        for (int i = 0; i < rhs.rows(); ++i)
            coupling_rhs(offset + i) = rhs(i);
    }

    void solveSystem()
    {
        ZIRAN_TIMER();
        Eigen::SparseMatrix<T>& Minv = e1.M_inv;
        Eigen::SparseMatrix<T>& G = e1.G;
        Eigen::SparseMatrix<T>& D = e1.D;
        Eigen::SparseMatrix<T>& S = e1.S;
        Eigen::SparseMatrix<T>& W1 = e1.W;
        Eigen::SparseMatrix<T>& hat_Minv = e2.M_inv;
        Eigen::SparseMatrix<T>& hat_G = e2.G;
        Eigen::SparseMatrix<T>& hat_D = e2.D;
        Eigen::SparseMatrix<T>& hat_S = e2.S;
        Eigen::SparseMatrix<T>& W2 = e2.W;
        TStack& a = e1.a;
        TStack& b = e1.b;
        TStack& hat_a = e2.a;
        TStack& hat_b = e2.b;
        // build system
        coupling_tri.clear();
        int num_p1 = e1.num_p;
        int num_p2 = e2.num_p;
        {
            ZIRAN_TIMER();
            fillInCouplingTri(0, 0, S - D * Minv * G);
            fillInCouplingTri(0, num_p1, -D * Minv * W1.transpose());
            fillInCouplingTri(num_p1, 0, W1 * Minv * G);
            fillInCouplingTri(num_p1, num_p1, W1 * Minv * W1.transpose() + W2 * hat_Minv * W2.transpose());
            fillInCouplingTri(num_p1, num_p1 + interface_cnt, -W2 * hat_Minv * hat_G);
            fillInCouplingTri(num_p1 + interface_cnt, num_p1, hat_D * hat_Minv * W2.transpose());
            fillInCouplingTri(num_p1 + interface_cnt, num_p1 + interface_cnt, hat_S - hat_D * hat_Minv * hat_G);
        }
        Eigen::SparseMatrix<T> coupling_m(num_p1 + interface_cnt + num_p2, num_p1 + interface_cnt + num_p2);
        coupling_m.setFromTriplets(coupling_tri.begin(), coupling_tri.end());
        // build rhs
        coupling_rhs = TStack::Zero(num_p1 + interface_cnt + num_p2);
        fillInCouplingRhs(0, b - D * Minv * a);
        fillInCouplingRhs(num_p1, W1 * Minv * a - W2 * hat_Minv * hat_a);
        fillInCouplingRhs(num_p1 + interface_cnt, hat_b - hat_D * hat_Minv * hat_a);
        TStack solved_pap;

        ZIRAN_ASSERT(e1.dt == e2.dt);
        ZIRAN_ASSERT(e1.dx == e2.dx);
        T kkt_relax_local = kkt_relax * std::pow(e1.dx, (T)dim) / e1.dt;

        if (solverType == EIGEN)
            solved_pap = DirectSolver<T>::solve(coupling_m, coupling_rhs, DirectSolver<T>::EIGEN);
        else if (solverType == AMGCL)
            solved_pap = DirectSolver<T>::solve(coupling_m, coupling_rhs, DirectSolver<T>::AMGCL, relative_tol, abs_tol, max_iter, -1, 1, false, false, kkt_relax_local);
        else
            solved_pap = DirectSolver<T>::solve(coupling_m, coupling_rhs, DirectSolver<T>::AMGCL_CUDA, relative_tol, abs_tol, max_iter, -1, 1, false, false, kkt_relax_local);
        e1.solved_p = TStack(num_p1);
        for (int i = 0; i < num_p1; ++i) e1.solved_p(i) = solved_pap(i);
        e2.solved_p = TStack(num_p2);
        for (int i = 0; i < num_p2; ++i) e2.solved_p(i) = solved_pap(num_p1 + interface_cnt + i);
        TStack solved_alpha = TStack(interface_cnt);
        for (int i = 0; i < interface_cnt; ++i) solved_alpha(i) = solved_pap(num_p1 + i);
        ZIRAN_ASSERT(W1.rows() == solved_alpha.rows(), "wrong");
        ZIRAN_ASSERT(W1.cols() == e1.a.rows(), "wrong");
        e1.a -= W1.transpose() * solved_alpha;
        e1.constructVelocityFromSolvedP();
        ZIRAN_ASSERT(W2.rows() == solved_alpha.rows(), "wrong");
        ZIRAN_ASSERT(W2.cols() == e2.a.rows(), "wrong");
        e2.a += W2.transpose() * solved_alpha;
        e2.constructVelocityFromSolvedP();
    }

    void addBoundarySpring(SplittingSimulation<T, dim>& sim)
    {
        auto* pin_pointer = sim.particles.DataManager::getPointer(MpmParallelPinningForceHelper<T, dim>::target_name());
        if (pin_pointer == nullptr)
            return;
        sim.before_advect = sim.particles.X.array;
        tbb::parallel_for(0, sim.particles.count, [&](int i) {
            (*pin_pointer)[i].x.clear();
            (*pin_pointer)[i].v.clear();
            (*pin_pointer)[i].k = 0;
        });
        // addSpringWithSolid(another_sim);
        tbb::parallel_for(0, sim.particles.count, [&](int i) {
            TV& Xp = sim.particles.X.array[i];
            for (int k = 0; k < (int)sim.collision_objects.size(); ++k) {
                T phi;
                TV n;
                if (sim.collision_objects[k]->type != AnalyticCollisionObject<T, dim>::GHOST)
                    continue;
                bool collided = sim.collision_objects[k]->queryInside(Xp, phi, n, 0);
                if (collided) {
                    TV Xs = Xp - phi * n;
                    (*pin_pointer)[i].x.push_back(Xs);
                    (*pin_pointer)[i].v.push_back(TV::Zero());
                    (*pin_pointer)[i].k = sim.implicit_penalty_collision;
                }
            }
        });
    }

    StdVector<TV> points;
    StdVector<TV> solid_particles;
    StdVector<TV> before_advect;
    StdVector<TV> trajectory;
    StdVector<TV> normals;
    StdVector<Vector<int, 2>> segments;
    void advanceOneTimeStep(double dt) override
    {
        ZIRAN_INFO("Advance one time step with dt = ", dt);

        ZIRAN_ASSERT((e1.category == MATERIAL_PHASE_FLUID && e2.category == MATERIAL_PHASE_SOLID) || (e1.category == MATERIAL_PHASE_SOLID && e2.category == MATERIAL_PHASE_FLUID));

        e1.dt = dt;
        e1.setFrame(frame);
        e2.dt = dt;
        e2.setFrame(frame);

        ZIRAN_TIMER();
        e1.reinitialize();
        e2.reinitialize();

        e1.particlesToGrid();
        e2.particlesToGrid();

        if (e1.category == MATERIAL_PHASE_SOLID) addBoundarySpring(e1);
        if (e2.category == MATERIAL_PHASE_SOLID) addBoundarySpring(e2);

        e1.gridUpdate();
        e2.gridUpdate();

        ZIRAN_ASSERT(e1.use_vp && e2.use_vp, "only solve coupling system here");

        buildSystem();
        solveSystem();

        for (size_t k = 0; k < e1.collision_objects.size(); ++k)
            if (e1.collision_objects[k]->updateState)
                e1.collision_objects[k]->updateState(Base::step.time + dt, *e1.collision_objects[k]);
        for (size_t k = 0; k < e2.collision_objects.size(); ++k)
            if (e2.collision_objects[k]->updateState)
                e2.collision_objects[k]->updateState(Base::step.time + dt, *e2.collision_objects[k]);

        e1.gridToParticles(dt);
        e2.gridToParticles(dt);

        if (use_position_correction) {
            if (e1.category == MATERIAL_PHASE_FLUID) {
                nils_position_corrector1.solve(e2);
                nils_position_corrector1.levelset_builder.marchingSquare(points, segments);
            }
            if (e2.category == MATERIAL_PHASE_FLUID) {
                nils_position_corrector2.solve(e1);
                nils_position_corrector2.levelset_builder.marchingSquare(points, segments);
            }
        }

        for (int i = e1.particles.count - 1; i >= 0; --i) {
            for (int d = 0; d < dim; d++) {
                if (e1.particles.X.array[i](d) < 10 * e1.dx || e1.particles.X.array[i](d) > 2028. * e1.dx) {
                    e1.particles.remove(i);
                    break;
                }
            }
        }
        for (int i = e2.particles.count - 1; i >= 0; --i) {
            for (int d = 0; d < dim; d++) {
                if (e2.particles.X.array[i](d) < 10 * e2.dx || e2.particles.X.array[i](d) > 2028. * e2.dx) {
                    e2.particles.remove(i);
                    break;
                }
            }
        }
    }

    void write(const std::string& filename) override
    {
        e1.setFrame(frame);
        e2.setFrame(frame);
        e1.write(e1.outputFileName());
        e2.write(e2.outputFileName());
        return;
        visualize_points(e1, before_advect, trajectory, "trajectory");
        visualize_points(e1, solid_particles, normals, "solid_normal");
        visualize_segments(e2, points, segments, "marching_square");
        visualize_points(e1, wall_particles, "wall_points");
        visualize_g2p_vec(e1, surface_tension_delta1, "st");
        visualize_g2p_vec(e2, surface_tension_delta2, "st");
        // if (e1.quad_particles.size()) visualize_points(e1, e1.quad_particles, "quad_particles");
        // if (e2.quad_particles.size()) visualize_points(e2, e2.quad_particles, "quad_particles");
    }

    void read(const std::string& filename) override
    {
        e1.setFrame(frame);
        e2.setFrame(frame);
        e1.read(e1.outputFileName());
        e2.read(e2.outputFileName());
    }

    void writeState(std::ostream&) override {}
    void readState(std::istream&) override {}
    bool useDouble() override { return std::is_same<T, double>::value; }
    int dimension() override { return dim; }
    const char* name() override { return "coupling"; }
};
} // namespace ZIRAN
