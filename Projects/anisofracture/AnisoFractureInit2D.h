#ifndef MPM_INIT_2D_H
#define MPM_INIT_2D_H

#include <Ziran/CS/Util/RandomNumber.h>
#include <Ziran/CS/Util/AttributeNamesForward.h>
#include <Ziran/Math/Geometry/PartioIO.h>
#include <Ziran/Math/Geometry/MeshConstruction.h>
#include <Ziran/Math/Geometry/AnalyticLevelSet.h>
#include <Ziran/Math/Geometry/CollisionObject.h>
#include <Ziran/Math/MathTools.h>
#include <Ziran/Physics/SoundSpeedCfl.h>

#include <Ziran/Sim/MeshHandle.h>
#include <Ziran/Sim/SceneInitializationCore.h>

#include <MPM/MpmInitializationHelper.h>

#include "AnisoFractureSimulation.h"
#include "AnisoFractureInit.h"
#include <float.h>

namespace Python {
extern double ax, ay, az, bx, by, bz, alpha1, alpha2, fiberScale, residual, percent, eta;
extern bool useRadial, useLongitudinal, isotropic;
} // namespace Python

namespace ZIRAN {

template <class T>
class AnisoFractureInit2D : public AnisoFractureInitBase<T, 2> {
public:
    static const int dim = 2;
    using Base = AnisoFractureInitBase<T, dim>;
    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;
    using TVI = Vector<int, dim>;

    using Base::init_helper;
    using Base::scene;
    using Base::sim;
    using Base::test_number;
    AnisoFractureInit2D(AnisoFractureSimulation<T, dim>& sim, const int test_number)
        : Base(sim, test_number)
    {
    }

    void reload() override
    {
        if (test_number == 6) {
#include "examples/2DdiskTear.h"
        }
    }
};
} // namespace ZIRAN
#endif
