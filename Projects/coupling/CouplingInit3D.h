#pragma once

#include <Ziran/CS/Util/RandomNumber.h>
#include <Ziran/Math/Geometry/MeshConstruction.h>
#include <Ziran/Physics/SoundSpeedCfl.h>
#include <Ziran/Sim/MeshHandle.h>
#include <Ziran/Sim/SceneInitializationCore.h>
#include "CouplingSimulation.h"
#include "CouplingInit.h"

namespace ZIRAN {

template <class T, int dim>
class CouplingInitBase;

template <class T>
class CouplingInit3D : public CouplingInitBase<T, 3> {
public:
    static const int dim = 3;
    using Base = CouplingInitBase<T, dim>;
    using TV2 = Vector<T, 2>;
    using TVI2 = Vector<int, 2>;
    using TV = Vector<T, dim>;
    using TM = Eigen::Matrix<T, dim, dim>;
    using TVI = Vector<int, dim>;

    using Base::init_helper1;
    using Base::init_helper2;
    using Base::scene1;
    using Base::scene2;
    using Base::sim1;
    using Base::sim2;
    using Base::sim_coordinator;
    using Base::test_number;
    CouplingInit3D(CouplingSimulation<T, dim>& sim, const int test_number)
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

    void set_simulator(T target_dx, T dx, T dy, T dz, T h = -1, bool bb_box = true)
    {
        if (h < 0) {
            h = 1 - dy - dy;
        }
        SET_SIM_VALUE(end_frame, 1000);
        SET_SIM_VALUE(step.max_dt, 0.004);
        SET_MPM_VALUE(dx, target_dx);
        sim_coordinator.e1.apic_rpic_ratio = 0.99;
        sim_coordinator.e2.apic_rpic_ratio = 0;
        SET_MPM_VALUE(gravity, -9.8 * TV::Unit(1));
        SET_MPM_VALUE(symplectic, false);
        SET_MPM_VALUE(objective.matrix_free, true);
        SET_MPM_VALUE(cfl, 0.4);
        SET_MPM_VALUE(mls_mpm, false);
        SET_MPM_VALUE(ignoreCollisionObject, true);
        sim_coordinator.setSolver(1e-3, -1, 10000, CouplingSimulation<T, dim>::AMGCL);
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
        sim_coordinator.e1.boundary_ppc = 27;
        sim_coordinator.e2.boundary_ppc = 27;
        sim_coordinator.kkt_relax = ((T)1) / (T)1000 / (T)(1e5);
        sim_coordinator.e2.implicit_penalty_collision = (T)1000 * std::pow((T)sim_coordinator.e2.dx, (T)dim) / (T)9 * 1000000;
        sim_coordinator.e2.dump_F_for_meshing = true;

        ///////////////// SIM1 (FLUID)
        if (bb_box) {
            T half_dx = (T)0.5 * sim1.dx;
            T three_dx = (T)3 * sim1.dx;
            HalfSpace<T, dim> ground_ls(TV(1 + dx - half_dx, 1 + dy - half_dx, 1 + dz - half_dx), TV(0, 1, 0));
            AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::SLIP);
            sim1.sampleBoundaryParticlesInCollisionObjects(ground_object, TV(1 + dx - three_dx, 1 + dy - three_dx, 1 + dz - three_dx), TV(2 - dx + three_dx, 1 + dy + three_dx, 2 - dz + three_dx), sim1.dx * 2);
            init_helper1.addAnalyticCollisionObject(ground_object);
            HalfSpace<T, dim> left_wall_ls(TV(1 + dx - half_dx, 1 + dy - half_dx, 1 + dz - half_dx), TV(1, 0, 0));
            AnalyticCollisionObject<T, dim> left_wall_object(left_wall_ls, AnalyticCollisionObject<T, dim>::SLIP);
            sim1.sampleBoundaryParticlesInCollisionObjects(left_wall_object, TV(1 + dx - three_dx, 1 + dy - three_dx, 1 + dz - three_dx), TV(1 + dx + three_dx, 1 + dy + h, 2 - dz + three_dx), sim1.dx * 2);
            init_helper1.addAnalyticCollisionObject(left_wall_object);
            HalfSpace<T, dim> back_wall_ls(TV(1 + dx - half_dx, 1 + dy - half_dx, 1 + dz - half_dx), TV(0, 0, 1));
            AnalyticCollisionObject<T, dim> back_wall_object(back_wall_ls, AnalyticCollisionObject<T, dim>::SLIP);
            sim1.sampleBoundaryParticlesInCollisionObjects(back_wall_object, TV(1 + dx - three_dx, 1 + dy - three_dx, 1 + dz - three_dx), TV(2 - dx + three_dx, 1 + dy + h, 1 + dz + three_dx), sim1.dx * 2);
            init_helper1.addAnalyticCollisionObject(back_wall_object);
            HalfSpace<T, dim> right_wall_ls(TV(2 - dx + half_dx, 2 - dy + half_dx, 2 - dz + half_dx), TV(-1, 0, 0));
            AnalyticCollisionObject<T, dim> right_wall_object(right_wall_ls, AnalyticCollisionObject<T, dim>::SLIP);
            sim1.sampleBoundaryParticlesInCollisionObjects(right_wall_object, TV(2 - dx - three_dx, 1 + dy - three_dx, 1 + dz - three_dx), TV(2 - dx + three_dx, 1 + dy + h, 2 - dz + three_dx), sim1.dx * 2);
            init_helper1.addAnalyticCollisionObject(right_wall_object);
            HalfSpace<T, dim> front_wall_ls(TV(2 - dx + half_dx, 2 - dy + half_dx, 2 - dz + half_dx), TV(0, 0, -1));
            AnalyticCollisionObject<T, dim> front_wall_object(front_wall_ls, AnalyticCollisionObject<T, dim>::SLIP);
            sim1.sampleBoundaryParticlesInCollisionObjects(front_wall_object, TV(1 + dx - three_dx, 1 + dy - three_dx, 2 - dz - three_dx), TV(2 - dx + three_dx, 1 + dy + h, 2 - dz + three_dx), sim1.dx * 2);
            init_helper1.addAnalyticCollisionObject(front_wall_object);
        }
        ///////////////// SIM2 (SOLID)
        if (bb_box) {
            HalfSpace<T, dim> ground_ls(TV(1 + dx, 1 + dy, 1 + dz), TV(0, 1, 0));
            AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::GHOST);
            init_helper2.addAnalyticCollisionObject(ground_object);
            HalfSpace<T, dim> left_wall_ls(TV(1 + dx, 1 + dy, 1 + dz), TV(1, 0, 0));
            AnalyticCollisionObject<T, dim> left_wall_object(left_wall_ls, AnalyticCollisionObject<T, dim>::GHOST);
            init_helper2.addAnalyticCollisionObject(left_wall_object);
            HalfSpace<T, dim> back_wall_ls(TV(1 + dx, 1 + dy, 1 + dz), TV(0, 0, 1));
            AnalyticCollisionObject<T, dim> back_wall_object(back_wall_ls, AnalyticCollisionObject<T, dim>::GHOST);
            init_helper2.addAnalyticCollisionObject(back_wall_object);
            HalfSpace<T, dim> right_wall_ls(TV(2 - dx, 2 - dy, 2 - dz), TV(-1, 0, 0));
            AnalyticCollisionObject<T, dim> right_wall_object(right_wall_ls, AnalyticCollisionObject<T, dim>::GHOST);
            init_helper2.addAnalyticCollisionObject(right_wall_object);
            HalfSpace<T, dim> front_wall_ls(TV(2 - dx, 2 - dy, 2 - dz), TV(0, 0, -1));
            AnalyticCollisionObject<T, dim> front_wall_object(front_wall_ls, AnalyticCollisionObject<T, dim>::GHOST);
            init_helper2.addAnalyticCollisionObject(front_wall_object);
        }
    }

    void reload() override
    {
        if (test_number == 1) {
            set_path("output/bear_bath");
            T dx = 0, dy = 0, dz = 0;
            set_simulator(0.02, dx, dy, dz);
            T E = 1e5, nu = 0.4;
            ///////////////// SIM1 (FLUID)
            {
                T fluid_density = 1000;
                T ppc = 27;
                TV source_center1(1.2, 1.3, 1.5);
                Sphere<T, dim> source_ls1(source_center1, 0.01);
                MpmParticleHandleBase<T, dim> p_handle1 = init_helper1.sampleInAnalyticLevelSet(source_ls1, fluid_density, ppc);
                LinearCorotated<T, dim> model1(E, nu);
                QuadraticVolumePenalty<T, dim> model2(E, nu);
                p_handle1.addFJMixedMpmForce(model1, model2, MATERIAL_PHASE_FLUID, MATERIAL_PROPERTY_INCOMPRESSIBLE);

                TV source_center(1.2, 1.3, 1.5);
                Sphere<T, dim> source_ls(source_center, 0.05);
                TV material_speed(3.5, 0, 0);
                SourceCollisionObject<T, dim> sphere_source(source_ls, material_speed);
                int source_id = init_helper1.addSourceCollisionObject(sphere_source);
                init_helper1.sampleSourceAtTheBeginning(source_id, 1000, 27);
                sim_coordinator.end_time_step_callbacks.push_back(
                    [this, source_id, fluid_density, ppc, model1, model2](int frame, int substep) {
                        if (frame < 20) {
                            // add more particles from source Collision object
                            int N = init_helper1.sourceSampleAndPrune(source_id, fluid_density, ppc);
                            if (N) {
                                MpmParticleHandleBase<T, dim> source_particles_handle = init_helper1.getParticlesFromSource(source_id, fluid_density, ppc);
                                source_particles_handle.addFJMixedMpmForce(model1, model2, MATERIAL_PHASE_FLUID, MATERIAL_PROPERTY_INCOMPRESSIBLE, /*linear corotated*/ false);
                            }
                        }
                    });
            }
            ///////////////// SIM2
            {
                StdVector<TV> meshed_points;
                std::string absolute_path = DataDir().absolutePath("TriMesh/bear.obj");
                readPositionObj(absolute_path, meshed_points);
                MpmParticleHandleBase<T, dim> p_handle22 = init_helper2.sampleFromVdbFileWithExistingPoints(meshed_points, "LevelSets/bear.vdb", 1000, 27);
                LinearCorotated<T, dim> model1(E, nu);
                QuadraticVolumePenalty<T, dim> model2(E, nu);
                p_handle22.addFJMixedMpmForce(model1, model2, MATERIAL_PHASE_SOLID, MATERIAL_PROPERTY_COMPRESSIBLE, /*linear corotated*/ true);
                SnowPlasticity<T> p(0, 0.9, 5); //restrict F to [1-0.9, 1+5]
                p_handle22.addPlasticity(model1, p, "F");
                Sphere<T, dim> dummy(TV(0, 0, 0), 100);
                p_handle22.parallelPinParticlesInLevelSet(dummy, 0, 0, /*collider spring mode*/ true);
            }
        }

        if (test_number == 2) {
            set_path("output/balls");
            T dx = 0, dy = 0, dz = 0;
            set_simulator(0.008, dx, dy, dz, -1, false);
            SET_SIM_VALUE(step.frame_dt, 1. / 48.);
            sim_coordinator.e1.apic_rpic_ratio = 1;
            sim_coordinator.e2.apic_rpic_ratio = 1;
            T E = 5e5, nu = 0.3;
            ///////////////// SIM1 (FLUID)
            {
                T half_dx = (T)0.5 * sim1.dx;
                T three_dx = (T)3 * sim1.dx;
                HalfSpace<T, dim> ground_ls(TV(1 + dx - half_dx, 1 + dy - half_dx, 1 + dz - half_dx), TV(0, 1, 0));
                AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::SLIP);
                sim1.sampleBoundaryParticlesInCollisionObjects(ground_object, TV(1 + dx - three_dx, 1 + dy - three_dx, 1 + dz - three_dx), TV(2 - dx + three_dx, 1 + dy + three_dx, 2 - dz + three_dx), sim1.dx * 2);
                init_helper1.addAnalyticCollisionObject(ground_object);

                Sphere<T, dim> sphere(TV(6.5, 1.5, 6.5), 0.119);
                MpmParticleHandleBase<T, dim> p_handle1 = init_helper1.sampleInAnalyticLevelSet(sphere, /* fluid_density */ 1000, /* particles per dimension */ 27);
                LinearCorotated<T, dim> model1(E, nu);
                QuadraticVolumePenalty<T, dim> model2(E, nu);
                p_handle1.addFJMixedMpmForce(model1, model2, MATERIAL_PHASE_FLUID, MATERIAL_PROPERTY_INCOMPRESSIBLE, /*linear corotated*/ true);

                sim_coordinator.end_frame_callbacks.push_back(
                    [this, model1, model2](int frame) {
                        if (frame == 20) {
                            Sphere<T, dim> sphere(TV(6.4, 1.55, 6.4), 0.119);
                            MpmParticleHandleBase<T, dim> p_handle1 = init_helper1.sampleInAnalyticLevelSet(sphere, /* fluid_density */ 1000, /* particles per dimension */ 27);
                            p_handle1.addFJMixedMpmForce(model1, model2, MATERIAL_PHASE_FLUID, MATERIAL_PROPERTY_INCOMPRESSIBLE, /*linear corotated*/ true);
                        }
                        if (frame == 40) {
                            Sphere<T, dim> sphere(TV(6.6, 1.55, 6.6), 0.119);
                            MpmParticleHandleBase<T, dim> p_handle1 = init_helper1.sampleInAnalyticLevelSet(sphere, /* fluid_density */ 1000, /* particles per dimension */ 27);
                            p_handle1.addFJMixedMpmForce(model1, model2, MATERIAL_PHASE_FLUID, MATERIAL_PROPERTY_INCOMPRESSIBLE, /*linear corotated*/ true);
                        }
                    });
            }
            ///////////////// SIM2
            {
                HalfSpace<T, dim> ground_ls(TV(1 + dx, 1 + dy, 1 + dz), TV(0, 1, 0));
                AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::GHOST);
                init_helper2.addAnalyticCollisionObject(ground_object);

                LinearCorotated<T, dim> model_linear(E, nu);
                CorotatedIsotropic<T, dim> model_corotated(E, nu);
                NeoHookeanIsotropic<T, dim> model_neo(E, nu);
                StvkWithHenckyIsotropic<T, dim> model_stvk(E, nu);
                QuadraticVolumePenalty<T, dim> model2(E, nu);
                Sphere<T, dim> dummy(TV(0, 0, 0), 100);

                StdVector<TV> meshed_points;
                std::string absolute_path = DataDir().absolutePath("TriMesh/ball.obj");
                readPositionObj(absolute_path, meshed_points);
                MpmParticleHandleBase<T, dim> p_handle22 = init_helper2.sampleFromVdbFileWithExistingPoints(meshed_points, "LevelSets/ball.vdb", 800, 27);
                p_handle22.transform([&](int index, Ref<T> mass, TV& X, TV& V) { X += TV(6.5, 1.5, 6.5); });
                p_handle22.addFJMixedMpmForce(model_corotated, model2, MATERIAL_PHASE_SOLID, MATERIAL_PROPERTY_COMPRESSIBLE, /*linear corotated*/ false);
                p_handle22.parallelPinParticlesInLevelSet(dummy, 0, 0, /*collider spring mode*/ true);

                sim_coordinator.end_frame_callbacks.push_back(
                    [this, model_neo, model_stvk, model_linear, model2](int frame) {
                        Sphere<T, dim> dummy(TV(0, 0, 0), 100);
                        if (frame == 20) {
                            StdVector<TV> meshed_points;
                            std::string absolute_path = DataDir().absolutePath("TriMesh/ball.obj");
                            readPositionObj(absolute_path, meshed_points);
                            MpmParticleHandleBase<T, dim> p_handle22 = init_helper2.sampleFromVdbFileWithExistingPoints(meshed_points, "LevelSets/ball.vdb", 800, 27);
                            p_handle22.transform([&](int index, Ref<T> mass, TV& X, TV& V) { X += TV(6.4, 1.55, 6.4); });
                            p_handle22.addFJMixedMpmForce(model_neo, model2, MATERIAL_PHASE_SOLID, MATERIAL_PROPERTY_COMPRESSIBLE, /*linear corotated*/ false);
                            p_handle22.parallelPinParticlesInLevelSet(dummy, 0, 0, /*collider spring mode*/ true);
                        }

                        if (frame == 40) {
                            StdVector<TV> meshed_points;
                            std::string absolute_path = DataDir().absolutePath("TriMesh/ball.obj");
                            readPositionObj(absolute_path, meshed_points);
                            MpmParticleHandleBase<T, dim> p_handle22 = init_helper2.sampleFromVdbFileWithExistingPoints(meshed_points, "LevelSets/ball.vdb", 800, 27);
                            p_handle22.transform([&](int index, Ref<T> mass, TV& X, TV& V) { X += TV(6.6, 1.55, 6.6); });
                            p_handle22.addFJMixedMpmForce(model_stvk, model2, MATERIAL_PHASE_SOLID, MATERIAL_PROPERTY_COMPRESSIBLE, /*linear corotated*/ false);
                            p_handle22.parallelPinParticlesInLevelSet(dummy, 0, 0, /*collider spring mode*/ true);
                        }
                    });
            }
        }

        if (test_number == 3) {
            set_path("output/flush_rubber");
            T dx = 0, dy = 0, dz = 0;
            set_simulator(0.008, dx, dy, dz);
            SET_SIM_VALUE(step.frame_dt, 1. / 48.);
            sim_coordinator.e1.apic_rpic_ratio = 0;
            sim_coordinator.e2.apic_rpic_ratio = 0;
            T E = 1e5, nu = 0.3;
            ///////////////// SIM1 (FLUID)
            {
                T ppc = 27;
                T fluid_density = 1000;
                LinearCorotated<T, dim> model1(E, nu);
                QuadraticVolumePenalty<T, dim> model2(E, nu);

                Sphere<T, dim> ball(TV(2.5, 1.5, 2.5), 0.004);
                MpmParticleHandleBase<T, dim> p_handle21 = init_helper1.sampleInAnalyticLevelSet(ball, 1000, ppc);
                p_handle21.addFJMixedMpmForce(model1, model2, MATERIAL_PHASE_FLUID, MATERIAL_PROPERTY_INCOMPRESSIBLE, /*linear corotated*/ true);

                TV source_center(1.5, 2, 1.5);
                Sphere<T, dim> source_ls(source_center, 0.05);
                TV material_speed(0, -1, 0);
                SourceCollisionObject<T, dim> sphere_source(source_ls, material_speed);
                int source_id = init_helper1.addSourceCollisionObject(sphere_source);
                init_helper1.sampleSourceAtTheBeginning(source_id, fluid_density, ppc);
                sim_coordinator.end_time_step_callbacks.push_back(
                    [this, source_id, fluid_density, ppc, model1, model2](int frame, int substep) {
                        if (frame < 130 && frame > 70) {
                            // add more particles from source Collision object
                            int N = init_helper1.sourceSampleAndPrune(source_id, fluid_density, ppc);
                            if (N) {
                                MpmParticleHandleBase<T, dim> source_particles_handle = init_helper1.getParticlesFromSource(source_id, 1500, ppc);
                                source_particles_handle.addFJMixedMpmForce(model1, model2, MATERIAL_PHASE_FLUID, MATERIAL_PROPERTY_INCOMPRESSIBLE, /*linear corotated*/ true);
                            }
                        }
                    });
            }
            ///////////////// SIM2
            {
                T solid_density = 1000;
                LinearCorotated<T, dim> model_linear(E, nu);
                CorotatedIsotropic<T, dim> model_corotated(E, nu);
                NeoHookeanIsotropic<T, dim> model_neo(E, nu);
                StvkWithHenckyIsotropic<T, dim> model_stvk(E, nu);
                QuadraticVolumePenalty<T, dim> model2(E, nu);
                Sphere<T, dim> dummy(TV(0, 0, 0), 100);

                StdVector<TV> meshed_points;
                std::string absolute_path = DataDir().absolutePath("TriMesh/sheet.obj");
                readPositionObj(absolute_path, meshed_points);
                MpmParticleHandleBase<T, dim> p_handle22 = init_helper2.sampleFromVdbFileWithExistingPoints(meshed_points, "LevelSets/sheet.vdb", solid_density, 27);
                p_handle22.addFJMixedMpmForce(model_linear, model2, MATERIAL_PHASE_SOLID, MATERIAL_PROPERTY_COMPRESSIBLE, /*linear corotated*/ true);
                p_handle22.parallelPinParticlesInLevelSet(dummy, 0, 0, /*collider spring mode*/ true);
                SnowPlasticity<T> p(0, 0.9, 5); //restrict F to [1-0.9, 1+5]
                p_handle22.addPlasticity(model_linear, p, "F");
                sim_coordinator.end_time_step_callbacks.emplace_back(
                    [& particles = sim_coordinator.e2.particles, &dt = sim_coordinator.e2.dt, &dx = sim_coordinator.e2.dx, solid_density](int frame, int substep) {
                        if (frame < 40) {
                            auto* phase_pointer = particles.DataManager::getPointer(material_phase_name());
                            for (int i = 0; i < particles.count; i++)
                                if ((*phase_pointer)[i] == MATERIAL_PHASE_SOLID) {
                                    T coeff = 0.3;
                                    T area = dx * dx * M_PI;
                                    TV resistant_f = -0.5 * solid_density * particles.V(i).norm() * particles.V(i).norm() * coeff * area * particles.V(i).normalized();
                                    particles.V(i) += resistant_f / particles.mass(i) * dt;
                                }
                        }
                        if (frame > 40 && frame < 55) {
                            auto* phase_pointer = particles.DataManager::getPointer(material_phase_name());
                            for (int i = 0; i < particles.count; i++)
                                if ((*phase_pointer)[i] == MATERIAL_PHASE_SOLID) {
                                    T coeff = 0.1;
                                    T area = dx * dx * M_PI;
                                    TV resistant_f = -0.5 * solid_density * particles.V(i).norm() * particles.V(i).norm() * coeff * area * particles.V(i).normalized();
                                    particles.V(i) += resistant_f / particles.mass(i) * dt;
                                }
                        }
                    });

                Sphere<T, dim> sphere1(TV(1.25, 1.65, 1.25), 0.08);
                Sphere<T, dim> sphere2(TV(1.25, 1.65, 1.75), 0.08);
                Sphere<T, dim> sphere3(TV(1.75, 1.35, 1.75), 0.08);
                Sphere<T, dim> sphere4(TV(1.75, 1.35, 1.25), 0.08);
                DisjointUnionLevelSet<T, dim> stickers;
                stickers.add(sphere1);
                stickers.add(sphere2);
                stickers.add(sphere3);
                stickers.add(sphere4);
                p_handle22.pinParticlesInLevelSet(stickers, 2e5, 200);
            }
        }

        if (test_number == 4) {
            set_path("output/dam_jello_ball");
            SET_SIM_VALUE(end_frame, 480);
            SET_SIM_VALUE(step.max_dt, 0.003);
            SET_MPM_VALUE(dx, (T)2 / (T)128);
            sim_coordinator.e1.apic_rpic_ratio = 1;
            sim_coordinator.e2.apic_rpic_ratio = 0;
            T fluid_density = 1e3;
            T solid_density = 500;
            T ppc = 16;
            T E = 50000, nu = 0.4;
            SET_MPM_VALUE(gravity, -3 * TV::Unit(1));
            SET_MPM_VALUE(symplectic, false);
            SET_MPM_VALUE(objective.matrix_free, true);
            SET_MPM_VALUE(cfl, 0.4);
            SET_MPM_VALUE(mls_mpm, false);
            SET_MPM_VALUE(ignoreCollisionObject, true);
            sim_coordinator.setSolver(1e-3, -1, 10000, CouplingSimulation<T, dim>::AMGCL);
            sim_coordinator.use_position_correction = true;
            sim_coordinator.e1.category = MATERIAL_PHASE_FLUID;
            sim_coordinator.e1.newton.max_iterations = 20;
            sim_coordinator.e1.newton.tolerance = 1e-1;
            sim_coordinator.e1.objective.minres.tolerance = 1e-6;
            sim_coordinator.e1.objective.minres.max_iterations = 1000;
            sim_coordinator.e2.category = MATERIAL_PHASE_SOLID;
            sim_coordinator.e2.newton.max_iterations = 1;
            sim_coordinator.e2.newton.tolerance = 1e-2;
            sim_coordinator.e2.newton.linesearch = true;
            sim_coordinator.e2.objective.switchLinearSolver(ZIRAN::SimulationBase::LinearSolverType(1));
            sim_coordinator.e2.objective.linear_solver->max_iterations = 10000;
            sim_coordinator.e2.objective.linear_solver->tolerance = 1e-6;
            sim_coordinator.e2.implicit_penalty_collision = solid_density * std::pow((T)sim_coordinator.e2.dx, (T)dim) / (T)ppc * 1000000;
            sim_coordinator.e1.boundary_ppc = ppc;
            sim_coordinator.e2.boundary_ppc = ppc;
            sim_coordinator.kkt_relax = ((T)1) / fluid_density / (T)(1e5);
            CorotatedIsotropic<T, dim> model1(E, nu);
            QuadraticVolumePenalty<T, dim> model2(E, nu);
            TV scene_shift = 11.25 * sim_coordinator.e1.dx * TV::Ones();

            ///////////////// SIM1 (FLUID)
            {
                {
                    AxisAlignedAnalyticBox<T, dim> box(TV(0, 0, 0) + scene_shift, TV(2, 0.2, 1) + scene_shift);
                    MpmParticleHandleBase<T, dim> p_handle1 = init_helper1.sampleInAnalyticLevelSet(box, fluid_density, ppc);
                    p_handle1.addFJMixedMpmForce(model1, model2, MATERIAL_PHASE_FLUID, MATERIAL_PROPERTY_INCOMPRESSIBLE, /*linear corotated*/ false);
                }
                {
                    AxisAlignedAnalyticBox<T, dim> box(TV(0, 0.2, 0.3) + scene_shift, TV(0.5, 1, 0.7) + scene_shift);
                    MpmParticleHandleBase<T, dim> p_handle1 = init_helper1.sampleInAnalyticLevelSet(box, fluid_density, ppc);
                    p_handle1.addFJMixedMpmForce(model1, model2, MATERIAL_PHASE_FLUID, MATERIAL_PROPERTY_INCOMPRESSIBLE, /*linear corotated*/ false);
                }

                AxisAlignedAnalyticBox<T, dim> box1(TV(0, 0, 0) - 4 * sim_coordinator.e1.dx * TV::Ones() + scene_shift, TV(2, 4, 1) + 4 * sim_coordinator.e1.dx * TV::Ones() + scene_shift);
                AxisAlignedAnalyticBox<T, dim> box2(TV(0, 0, 0) + scene_shift, TV(2, 4, 1) + scene_shift);
                DifferenceLevelSet<T, dim> ls1, ls2;
                ls1.add(box1, box2);
                AnalyticCollisionObject<T, dim> container_object(ls1, AnalyticCollisionObject<T, dim>::SLIP);
                sim1.sampleBoundaryParticlesInCollisionObjects(container_object, TV(0, 0, 0) - 4 * sim_coordinator.e1.dx * TV::Ones() + scene_shift, TV(2, 4, 2) + 4 * sim_coordinator.e1.dx * TV::Ones() + scene_shift, sim1.dx * 2);
                init_helper1.addAnalyticCollisionObject(container_object);
            }

            ///////////////// SIM2
            {
                {
                    Sphere<T, dim> sphere(TV(1, 0.8, 0.3) + scene_shift, 0.2);
                    MpmParticleHandleBase<T, dim> p_handle = init_helper2.sampleInAnalyticLevelSet(sphere, solid_density, ppc);
                    p_handle.addFJMixedMpmForce(model1, model2, MATERIAL_PHASE_SOLID, MATERIAL_PROPERTY_COMPRESSIBLE, /*linear corotated*/ false);
                    Sphere<T, dim> dummy(TV(0, 0), 100);
                    p_handle.parallelPinParticlesInLevelSet(dummy, 0, 0, /*collider spring mode*/ true);
                }
                AxisAlignedAnalyticBox<T, dim> box1(TV(0, 0, 0) - 4 * sim_coordinator.e1.dx * TV::Ones() + scene_shift, TV(2, 4, 1) + 4 * sim_coordinator.e1.dx * TV::Ones() + scene_shift);
                AxisAlignedAnalyticBox<T, dim> box2(TV(0, 0, 0) + sim2.dx * TV::Ones() + scene_shift, TV(2, 4, 1) - sim2.dx * TV::Ones() + scene_shift);
                DifferenceLevelSet<T, dim> ls1, ls2;
                ls1.add(box1, box2);
                AnalyticCollisionObject<T, dim> container_object(ls1, AnalyticCollisionObject<T, dim>::GHOST);
                init_helper2.addAnalyticCollisionObject(container_object);
            }
        }

        if (test_number == 5) {
            set_path("output/bunny");
            T ratio = 0.2;
            SET_SIM_VALUE(end_frame, 300);
            SET_SIM_VALUE(step.frame_dt, 1. / 48.);
            SET_SIM_VALUE(step.max_dt, 0.005);
            SET_MPM_VALUE(dx, 0.01);
            sim_coordinator.e1.apic_rpic_ratio = 1;
            sim_coordinator.e2.apic_rpic_ratio = 0;
            T fluid_density = 1e3 * ratio;
            T solid_density = 1e3;
            T ppc = 8;
            T E = 1e5, nu = 0.4;
            SET_MPM_VALUE(gravity, -9.8 * TV::Unit(1));
            SET_MPM_VALUE(symplectic, false);
            SET_MPM_VALUE(objective.matrix_free, true);
            SET_MPM_VALUE(cfl, 0.5);
            SET_MPM_VALUE(mls_mpm, false);
            SET_MPM_VALUE(ignoreCollisionObject, true);
            sim_coordinator.setSolver(1e-3, -1, 10000, CouplingSimulation<T, dim>::AMGCL);
            sim_coordinator.use_position_correction = true;
            sim_coordinator.e1.category = MATERIAL_PHASE_FLUID;
            sim_coordinator.e1.newton.max_iterations = 20;
            sim_coordinator.e1.newton.tolerance = 1e-3;
            sim_coordinator.e1.objective.minres.tolerance = 1e-6;
            sim_coordinator.e1.objective.minres.max_iterations = 1000;
            sim_coordinator.e2.category = MATERIAL_PHASE_SOLID;
            sim_coordinator.e2.newton.max_iterations = 3;
            sim_coordinator.e2.newton.tolerance = 1e-2;
            sim_coordinator.e2.newton.linesearch = true;
            sim_coordinator.e2.objective.switchLinearSolver(ZIRAN::SimulationBase::LinearSolverType(1));
            sim_coordinator.e2.objective.linear_solver->max_iterations = 10000;
            sim_coordinator.e2.objective.linear_solver->tolerance = 1e-3;
            sim_coordinator.e2.implicit_penalty_collision = solid_density * std::pow((T)sim_coordinator.e2.dx, (T)dim) / (T)ppc * 100000;
            sim_coordinator.e2.dump_F_for_meshing = true;
            sim_coordinator.e1.boundary_ppc = ppc;
            sim_coordinator.e2.boundary_ppc = ppc;
            sim_coordinator.kkt_relax = ((T)1) / fluid_density / (T)(1e5);
            LinearCorotated<T, dim> model1(E, nu);
            //            CorotatedIsotropic<T, dim> model1(E, nu);
            QuadraticVolumePenalty<T, dim> model2(E, nu);
            model1.lambda /= 2;
            model2.lambda /= 2;
            T scene_shift = 3 * sim_coordinator.e1.dx;

            ///////////////// SIM1
            {
                TV source_center1(1 + scene_shift, 0.5 + scene_shift, 1 + scene_shift);
                Sphere<T, dim> source_ls1(source_center1, 0.01);
                MpmParticleHandleBase<T, dim> p_handle1 = init_helper1.sampleInAnalyticLevelSet(source_ls1, fluid_density, ppc);
                p_handle1.addFJMixedMpmForce(model1, model2, MATERIAL_PHASE_FLUID, MATERIAL_PROPERTY_INCOMPRESSIBLE);

                sim_coordinator.end_frame_callbacks.push_back(
                    [this, scene_shift, fluid_density, ppc, model1, model2](int frame) {
                        if (frame == 10) {
                            TV source_center1(1 + scene_shift, 2 + scene_shift, 1 + scene_shift);
                            Sphere<T, dim> source_ls1(source_center1, 0.25);
                            MpmParticleHandleBase<T, dim> p_handle1 = init_helper1.sampleInAnalyticLevelSet(source_ls1, fluid_density, ppc);
                            p_handle1.addFJMixedMpmForce(model1, model2, MATERIAL_PHASE_FLUID, MATERIAL_PROPERTY_INCOMPRESSIBLE);
                        }
                    });
            }

            ///////////////// SIM2
            {
                StdVector<TV> samples;
                StdVector<Vector<int, 4>> indices;
                std::string absolute_path = DataDir().absolutePath("TetMesh/twohead500k.mesh");
                readTetMeshTetWild(absolute_path, samples, indices);
                std::string vtk_path = DataDir().path + "/TetMesh/twohead500k.vtk";
                writeTetmeshVtk(vtk_path, samples, indices);
                using TMesh = SimplexMesh<3>;
                auto reader = MeshReader<T, dim, TMesh>("TetMesh/twohead500k.vtk");
                MeshHandle<T, dim, TMesh> mesh = scene2.createMesh((std::function<void(TMesh&, StdVector<TV>&)>)reader);
                auto& x_array = mesh.particles.X.array;
                DeformableObjectHandle<T, dim, TMesh> deformable = scene2.addDeformableObject(mesh);
                deformable.setMassFromDensity(fluid_density);
                deformable.addFemHyperelasticForce(model1);
                auto position_shift_to_right = [&](int index, Ref<T> mass, TV& X, TV& V) {
                    X += TV(1, 1, 1);
                };
                deformable.transform(position_shift_to_right);
                MpmParticleHandleBase<T, dim> p_handle22 = init_helper2.makeParticleHandle(deformable, (T)1);
                p_handle22.addFJMixedMpmForce(model1, model2, MATERIAL_PHASE_SOLID, MATERIAL_PROPERTY_COMPRESSIBLE, /*linear corotated*/ true);
                SnowPlasticity<T> p(0, 0.9, 5); //restrict F to [1-0.9, 1+5]
                p_handle22.addPlasticity(model1, p, "F");
                sim_coordinator.end_time_step_callbacks.emplace_back(
                    [& particles = sim_coordinator.e2.particles, &dt = sim_coordinator.e2.dt, &dx = sim_coordinator.e2.dx, solid_density](int frame, int substep) {
                        if (frame < 10) {
                            auto* phase_pointer = particles.DataManager::getPointer(material_phase_name());
                            for (int i = 0; i < particles.count; i++)
                                if ((*phase_pointer)[i] == MATERIAL_PHASE_SOLID) {
                                    T coeff = 0.5;
                                    T area = dx * dx * M_PI;
                                    TV resistant_f = -0.5 * solid_density * particles.V(i).norm() * particles.V(i).norm() * coeff * area * particles.V(i).normalized();
                                    particles.V(i) += resistant_f / particles.mass(i) * dt;
                                }
                        }
                    });
                Sphere<T, dim> sphere_ls(TV(1.1, 1.18, 0.985), 0.18);
                AnalyticCollisionObject<T, dim> container_object(sphere_ls, AnalyticCollisionObject<T, dim>::GHOST);
                p_handle22.pinParticlesInLevelSet(sphere_ls, (T)10000, (T)1000);
            }
        }
    }
#undef SET_SIM_VALUE
};
} // namespace ZIRAN
