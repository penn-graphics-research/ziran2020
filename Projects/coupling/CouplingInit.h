#pragma once

#include <Ziran/CS/Util/RandomNumber.h>
#include <Ziran/Math/Geometry/MeshConstruction.h>
#include <Ziran/Sim/MeshHandle.h>
#include <Ziran/Sim/SceneInitializationCore.h>
#include <MPM/MpmInitializationHelper.h>

#include "CouplingSimulation.h"

namespace ZIRAN {

template <class T, int dim>
class CouplingInitBase {
public:
    using TV = Vector<T, dim>;
    using TVI = Vector<int, dim>;

    CouplingSimulation<T, dim>& sim_coordinator;
    SplittingSimulation<T, dim>& sim1;
    SplittingSimulation<T, dim>& sim2;
    MpmInitializationHelper<T, dim> init_helper1;
    MpmInitializationHelper<T, dim> init_helper2;
    const int test_number;
    SceneInitialization<T, dim> scene1; // This stores a reference to the scene.
    SceneInitialization<T, dim> scene2; // This stores a reference to the scene.

    CouplingInitBase(CouplingSimulation<T, dim>& sim, const int test_number)
        : sim_coordinator(sim)
        , sim1(sim.e1)
        , sim2(sim.e2)
        , init_helper1(sim.e1)
        , init_helper2(sim.e2)
        , test_number(test_number)
        , scene1(sim1.scene)
        , scene2(sim2.scene)
    {
    }

    // Normal start call.
    void start()
    {
        printBasicInfo();
        initialize();
        sim_coordinator.simulate();
    }

    // Restart call.
    void restart(int frame)
    {
        initialize();
        sim_coordinator.restart(frame);
        sim_coordinator.simulate();
    }

    void initialize()
    {
        reload();
        sim_coordinator.initialize();
        sim_coordinator.reinitialize();
    }

    virtual void reload() = 0;

    void printBasicInfo()
    {
        bool is_double = std::is_same<T, double>();
        ZIRAN_INFO("Simulation using double:", is_double);
        ZIRAN_INFO("Simulation dimension:", dim);
    }
};
} // namespace ZIRAN
