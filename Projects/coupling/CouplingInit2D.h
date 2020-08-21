#pragma once

#include <Ziran/CS/Util/RandomNumber.h>
#include <Ziran/Math/Geometry/MeshConstruction.h>
#include <Ziran/Math/Geometry/AnalyticLevelSet.h>
#include <Ziran/Math/Geometry/CollisionObject.h>
#include <Ziran/Math/MathTools.h>

#include <Ziran/Sim/MeshHandle.h>
#include <Ziran/Sim/SceneInitializationCore.h>

#include "CouplingSimulation.h"
#include "CouplingInit.h"
#include <float.h>

namespace ZIRAN {

template <class T, int dim>
class CouplingInitBase;

template <class T>
class CouplingInit2D : public CouplingInitBase<T, 2> {
public:
    static const int dim = 2;
    using Base = CouplingInitBase<T, dim>;
    using TV = Vector<T, dim>;
    using TVI = Vector<int, dim>;

    using Base::init_helper1;
    using Base::init_helper2;
    using Base::scene1;
    using Base::scene2;
    using Base::sim1;
    using Base::sim2;
    using Base::sim_coordinator;
    using Base::test_number;
    CouplingInit2D(CouplingSimulation<T, dim>& sim, const int test_number)
        : Base(sim, test_number)
    {
    }

#define SET_SIM_VALUE(A, B) sim_coordinator.A = sim1.A = sim2.A = B
#define SET_MPM_VALUE(A, B) sim1.A = sim2.A = B

    void set_path(std::string path)
    {
        sim_coordinator.output_dir.path = path;
        sim1.output_dir.path = path + "/material1";
        sim2.output_dir.path = path + "/material2";
    }

    void set_simulator(T target_dx)
    {
        SET_SIM_VALUE(end_frame, 1000);
        SET_SIM_VALUE(step.max_dt, 0.001);
        SET_MPM_VALUE(dx, target_dx);
        sim_coordinator.e1.apic_rpic_ratio = 0.99;
        sim_coordinator.e2.apic_rpic_ratio = 0;
        SET_MPM_VALUE(gravity, -9.8 * TV::Unit(1));
        SET_MPM_VALUE(symplectic, false);
        SET_MPM_VALUE(objective.matrix_free, true);
        SET_MPM_VALUE(cfl, 0.4);
        SET_MPM_VALUE(mls_mpm, false);
        SET_MPM_VALUE(ignoreCollisionObject, true);
        sim_coordinator.setSolver(1e-3, -1, 10000, CouplingSimulation<T, dim>::AMGCL_CUDA);
        sim_coordinator.use_position_correction = true;
        sim_coordinator.e1.category = MATERIAL_PHASE_FLUID;
        sim_coordinator.e1.newton.max_iterations = 20;
        sim_coordinator.e1.newton.tolerance = 1e-3;
        sim_coordinator.e1.objective.minres.tolerance = 1e-6;
        sim_coordinator.e1.objective.minres.max_iterations = 1000;
        sim_coordinator.e2.category = MATERIAL_PHASE_SOLID;
        sim_coordinator.e2.newton.max_iterations = 50;
        sim_coordinator.e2.newton.tolerance = 1e-3;
        sim_coordinator.e2.newton.linesearch = true;
        sim_coordinator.e2.objective.minres.tolerance = 1e-6;
        sim_coordinator.e2.objective.minres.max_iterations = 1000;
        sim_coordinator.e1.boundary_ppc = 9;
        sim_coordinator.e2.boundary_ppc = 9;
        sim_coordinator.kkt_relax = ((T)1) / (T)1000 / (T)(1e5);
        sim_coordinator.e2.implicit_penalty_collision = (T)1000 * std::pow((T)sim_coordinator.e2.dx, (T)dim) / (T)9 * 1000000;

        ///////////////// SIM1 (FLUID)
        {
            T half_dx = (T)0.5 * sim1.dx;
            HalfSpace<T, dim> ground_ls(TV(1 - half_dx, 1 - half_dx), TV(0, 1));
            AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::SLIP);
            sim1.sampleBoundaryParticlesInCollisionObjects(ground_object, TV(0.9, 0.9), TV(2.1, 1.1), sim1.dx * 2);
            init_helper1.addAnalyticCollisionObject(ground_object);
            HalfSpace<T, dim> left_wall_ls(TV(1 - half_dx, 1 - half_dx), TV(1, 0));
            AnalyticCollisionObject<T, dim> left_wall_object(left_wall_ls, AnalyticCollisionObject<T, dim>::SLIP);
            sim1.sampleBoundaryParticlesInCollisionObjects(left_wall_object, TV(0.9, 0.9), TV(1.1, 4.1), sim1.dx * 2);
            init_helper1.addAnalyticCollisionObject(left_wall_object);
            HalfSpace<T, dim> right_wall_ls(TV(2 + half_dx, 1 - half_dx), TV(-1, 0));
            AnalyticCollisionObject<T, dim> right_wall_object(right_wall_ls, AnalyticCollisionObject<T, dim>::SLIP);
            sim1.sampleBoundaryParticlesInCollisionObjects(right_wall_object, TV(1.9, 0.9), TV(2.1, 4.1), sim1.dx * 2);
            init_helper1.addAnalyticCollisionObject(right_wall_object);
        }
        ///////////////// SIM2 (SOLID)
        {
            HalfSpace<T, dim> ground_ls(TV(1, 1), TV(0, 1));
            AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::GHOST);
            init_helper2.addAnalyticCollisionObject(ground_object);
            HalfSpace<T, dim> left_wall_ls(TV(1, 1), TV(1, 0));
            AnalyticCollisionObject<T, dim> left_wall_object(left_wall_ls, AnalyticCollisionObject<T, dim>::GHOST);
            init_helper2.addAnalyticCollisionObject(left_wall_object);
            HalfSpace<T, dim> right_wall_ls(TV(2, 1), TV(-1, 0));
            AnalyticCollisionObject<T, dim> right_wall_object(right_wall_ls, AnalyticCollisionObject<T, dim>::GHOST);
            init_helper2.addAnalyticCollisionObject(right_wall_object);
        }
    }

    void reload() override
    {
    } // namespace ZIRAN

#undef SET_SIM_VALUE
};
} // namespace ZIRAN
