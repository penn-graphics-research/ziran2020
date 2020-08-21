#pragma once

#include <Ziran/CS/Util/RandomNumber.h>
#include <Ziran/Math/Geometry/MeshConstruction.h>
#include <Ziran/Math/Geometry/AnalyticLevelSet.h>
#include <Ziran/Math/Geometry/CollisionObject.h>
#include <Ziran/Math/MathTools.h>

#include <Ziran/Sim/MeshHandle.h>
#include <Ziran/Sim/SceneInitializationCore.h>

#include "SplittingSimulation.h"
#include "SplittingInit.h"
#include <float.h>

namespace ZIRAN {

template <class T, int dim>
class SplittingInitBase;

template <class T>
class SplittingInit2D : public SplittingInitBase<T, 2> {
public:
    static const int dim = 2;
    using Base = SplittingInitBase<T, dim>;
    using TV = Vector<T, dim>;
    using TVI = Vector<int, dim>;

    using Base::init_helper;
    using Base::scene;
    using Base::sim;
    using Base::test_number;
    SplittingInit2D(SplittingSimulation<T, dim>& sim, const int test_number)
        : Base(sim, test_number)
    {
    }

    void reload() override
    {
    }
};
} // namespace ZIRAN
