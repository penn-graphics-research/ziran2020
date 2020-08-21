#ifndef MPM_PARALLEL_PINNING_FORCE_HELPER_H
#define MPM_PARALLEL_PINNING_FORCE_HELPER_H

#include <Ziran/Physics/ConstitutiveModel/ParallelPinningTarget.h>
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

   (3) Special: collider spring mode-- see A. McAdams, et al, Efficient Elasticity for Character Skinning with Contact and Collisions. 2011
   We implement the slidable spring further with unilateral energy (allow separation).
   No damping.
   x = xn + dt * v;
   xt = xtn; (lagged target)
   n = (xtn - xn).normalize(); 
   M = n n^T
   Psi = 0 if (x-xtn).dot(n) > 0
    otherwise Psi  = 1/2 k (x-xtn)^T M (x-xtn) 
   force = 0 if (x-xtn).dot(n) > 0
    otherwise force = -dPsidx = - k M (x-xtn)
   force differential = 0 if (x-xtn).dot(n) > 0
    otherwise force differential  = dfdx:dv = - k M dv
*/
template <class T, int dim>
class MpmParallelPinningForceHelper : public MpmForceHelperBase<T, dim> {
public:
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    typedef Matrix<T, dim, Eigen::Dynamic> TVStack;

    typedef ParallelPinningTarget<T, dim> Target;
    typedef ParallelPinningTargetScratch<T, dim> Scratch;

    Particles<T, dim>& particles; // contains Xn and Vn
    const StdVector<TV>& xp; // this is particle position during implicit solve
    const T& dt;
    const bool collider_spring;
    DataArray<Target>& target; // contains k, d, xt, vt. readwrite state.
    DataArray<Scratch>& scratch; // contains A and b. no write.

    MpmParallelPinningForceHelper(Particles<T, dim>& particles, const StdVector<TV>& xp, const T& dt, const bool is_collider_spring = false)
        : particles(particles)
        , xp(xp) // this is particle position during implicit solve. use scratch_xp
        , dt(dt)
        , collider_spring(is_collider_spring)
        , target(particles.add(target_name()))
        , scratch(particles.add(target_scratch_name()))
    {
    }

    virtual ~MpmParallelPinningForceHelper() {}

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
