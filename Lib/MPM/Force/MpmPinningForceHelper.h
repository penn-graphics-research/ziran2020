#ifndef MPM_PINNING_FORCE_HELPER_H
#define MPM_PINNING_FORCE_HELPER_H

#include <Ziran/Physics/ConstitutiveModel/PinningTarget.h>
#include <Ziran/CS/DataStructure/DataArray.h>
#include <Ziran/CS/DataStructure/DataManager.h>
#include <Ziran/CS/DataStructure/DisjointRanges.h>
#include <Ziran/CS/Util/Forward.h>
#include <Ziran/Math/Geometry/Particles.h>
#include <MPM/Force/MpmForceHelperBase.h>

namespace ZIRAN {

/** 
   (1) Implicit Case
   x = xn + dt * v
   xt = xtn + dt * vt (target motion. position update should be done after the solve.)
   Force is
   f = -k (x-xt) - d (v -vt)  (k is stiffness, d is damping coeff)
     = - (k+d/dt) x + (k xt - (d/dt)(-xn-xt+xtn) )
    := - A x + b
   where
   A = k + d/dt
   b = k xt - (d/dt)(-xn-xt+xtn).
   Energy is
   E = (A/2) x^2 - b' x + something (so that it is non negative)
   Force derivative is
   dfdx = -A

   (2) Explicit Case
   x = xn, v = vn
   xt = xtn + dt * vt    (target motion)
   Force is 
   f = -k(x-xt) - d(v-vt)
     = -k(xn-xt) - d(vn-vt)
*/
template <class T, int dim>
class MpmPinningForceHelper : public MpmForceHelperBase<T, dim> {
public:
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    typedef Matrix<T, dim, Eigen::Dynamic> TVStack;

    typedef PinningTarget<T, dim> Target;
    typedef PinningTargetScratch<T, dim> Scratch;

    Particles<T, dim>& particles; // contains Xn and Vn
    const StdVector<TV>& xp; // this is particle position during implicit solve
    const T& dt;
    DataArray<Target>& target; // contains k, d, xt, vt. readwrite state.
    DataArray<Scratch>& scratch; // contains A and b. no write.

    MpmPinningForceHelper(Particles<T, dim>& particles, const StdVector<TV>& xp, const T& dt)
        : particles(particles)
        , xp(xp) // this is particle position during implicit solve. use scratch_xp
        , dt(dt)
        , target(particles.add(target_name()))
        , scratch(particles.add(target_scratch_name()))
    {
    }

    virtual ~MpmPinningForceHelper() {}

    void reinitialize() override;

    // add to fp for explicit
    void updateState(const DisjointRanges& subrange, StdVector<TM>& vtau, TVStack& fp) override;

    // add to fp for implicit
    void updateImplicitState(const DisjointRanges& subrange, StdVector<TM>& vPFnT, TVStack& fp) override;

    double totalEnergy(const DisjointRanges& subrange) override;

    void computeStressDifferential(const DisjointRanges& subrange, const StdVector<TM>& gradDv, StdVector<TM>& dstress, const TVStack& dvp, TVStack& dfp) override;

    inline static AttributeName<Target> target_name()
    {
        return AttributeName<Target>(Target::name());
    }

    inline static AttributeName<Scratch> target_scratch_name()
    {
        return AttributeName<Scratch>(Scratch::name());
    }
};
} // namespace ZIRAN
#endif
