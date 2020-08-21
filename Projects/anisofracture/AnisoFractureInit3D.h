#ifndef MPM_INIT_3D_H
#define MPM_INIT_3D_H

#include <Ziran/CS/Util/RandomNumber.h>
#include <Ziran/Math/Geometry/MeshConstruction.h>
#include <Ziran/Math/Geometry/VoronoiNoise.h>
#include <Ziran/Physics/SoundSpeedCfl.h>
#include <Ziran/Physics/FiberGen.h>
#include <Ziran/Sim/MeshHandle.h>
#include <Ziran/Sim/SceneInitializationCore.h>
#include "AnisoFractureSimulation.h"
#include "AnisoFractureInit.h"

namespace Python {
extern double ax, ay, az, bx, by, bz, alpha1, alpha2, fiberScale, residual, percent, eta, E, rho, tau;
extern bool useRadial, useLongitudinal, isotropic, orthotropic, inextensible, implicit;
} // namespace Python

namespace CmdHelper {
extern int helper;
}

namespace ZIRAN {

template <class T>
class AnisoFractureInit3D : public AnisoFractureInitBase<T, 3> {
public:
    static const int dim = 3;
    using Base = AnisoFractureInitBase<T, dim>;
    using TV2 = Vector<T, 2>;
    using TVI2 = Vector<int, 2>;
    using TV = Vector<T, dim>;
    using TM = Eigen::Matrix<T, dim, dim>;
    using TVI = Vector<int, dim>;

    using Base::init_helper;
    using Base::scene;
    using Base::sim;
    using Base::test_number;

    AnisoFractureInit3D(AnisoFractureSimulation<T, dim>& sim, const int test_number)
        : Base(sim, test_number)
    {
    }

    void reload() override
    {

        if (test_number == 2) {
#include "examples/tubepull.h"
        }
        else if (test_number == 3) {
#include "examples/diskshoot.h"
        }
        else if (test_number == 5) {
#include "examples/meatproxy.h"
        }
        else if (test_number == 8) {
#include "examples/orange.h"
        }
        else if (test_number == 9) {
#include "examples/cheesePull.h"
        }
        else if (test_number == 12) {
#include "examples/fishskin.h"
        }
        else if (test_number == 13) {
#include "examples/paramEffects.h"
        }
        else if (test_number == 14) {
#include "examples/pork.h"
        }
        else if (test_number == 16) {
#include "examples/hangingTorus.h"
        }
        else if (test_number == 19) {
#include "examples/3DtestImplicit.h"
        }
        else if (test_number == 21) {
#include "examples/3DTubeSmash.h"
        }
        else if (test_number == 22) {
#include "examples/vonMisesTear.h"
        }
        else if (test_number == 25) {
#include "examples/boneTwist.h"
        }
    }
};
} // namespace ZIRAN
#endif
