#ifndef MPM_ETHER_DRAG_FORCE_HELPER_H
#define MPM_ETHER_DRAG_FORCE_HELPER_H

#include <Ziran/Math/Geometry/Particles.h>
#include <Ziran/CS/Util/AttributeNamesForward.h>
#include <MPM/Force/MpmForceHelperBase.h>

namespace ZIRAN {

template <class T, int dim>
class MpmEtherDragForceHelper : public MpmForceHelperBase<T, dim> {
public:
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;

    Particles<T, dim>& particles; // contains Xn and Vn
    const StdVector<TV>& xp; // this is particle position during implicit solve
    const T& dt;
    DataArray<T>& ether_drag; // drag coefficients

    MpmEtherDragForceHelper(Particles<T, dim>& particles, const StdVector<TV>& xp, const T& dt)
        : particles(particles)
        , xp(xp) // this is particle position during implicit solve. use scratch_xp
        , dt(dt)
        , ether_drag(particles.add(ether_drag_name<T>()))
    {
    }

    void reinitialize() override;

    // add to fp for explicit
    void updateState(const DisjointRanges& subrange, StdVector<TM>& vtau, TVStack& fp) override;

    // add to fp for implicit
    void updateImplicitState(const DisjointRanges& subrange, StdVector<TM>& vPFnT, TVStack& fp) override;

    double totalEnergy(const DisjointRanges& subrange) override;

    void computeStressDifferential(const DisjointRanges& subrange, const StdVector<TM>& gradDv, StdVector<TM>& dstress, const TVStack& dvp, TVStack& dfp) override;
};
} // namespace ZIRAN

#endif
