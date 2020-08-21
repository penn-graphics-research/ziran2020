#include "MpmPinningForceHelper.h"
#include <Ziran/Math/Geometry/Particles.h>

namespace ZIRAN {

template <class T, int dim>
void MpmPinningForceHelper<T, dim>::
    reinitialize()
{
    // make sure scratch has the same size with target
    scratch.lazyResize(target.ranges);
}

// add to fp for explicit
template <class T, int dim>
void MpmPinningForceHelper<T, dim>::
    updateState(const DisjointRanges& subrange, StdVector<TM>& vtau, TVStack& fp)
{
    DisjointRanges subset(subrange,
        particles.commonRanges(target_name(),
            target_scratch_name()));
    for (auto iter = particles.subsetIter(subset, target_name(), target_scratch_name()); iter; ++iter) {
        auto& t = iter.template get<0>();
        int p = iter.entryId();
        const auto& xn = particles.X[p];
        const auto& vn = particles.V[p];
        // f = -k(x - xt) - d(v - vt);
        fp.col(p) += (-t.k * (xn - t.x) - t.d * (vn - t.v));
    }
}

// add to fp for implicit
template <class T, int dim>
void MpmPinningForceHelper<T, dim>::
    updateImplicitState(const DisjointRanges& subrange, StdVector<TM>& vPFnT, TVStack& fp)
{
    DisjointRanges subset(subrange,
        particles.commonRanges(target_name(),
            target_scratch_name()));
    for (auto iter = particles.subsetIter(subset, target_name(), target_scratch_name()); iter; ++iter) {
        auto& t = iter.template get<0>(); // target
        auto& s = iter.template get<1>(); // scratch
        int p = iter.entryId();
        const TV& xn = particles.X[p];
        const TV& x = xp[p]; // from scratch_xp
        const TV& xtn = t.x;
        const TV& vtn = t.v;
        const TV& xt = xtn + vtn * dt; // candidate target n+1 position
        s.A = t.k + t.d / dt;
        s.b = t.k * xt - (t.d / dt) * (-xn - xt + xtn);
        fp.col(p) += (-s.A * x + s.b);
    }
}

template <class T, int dim>
double MpmPinningForceHelper<T, dim>::
    totalEnergy(const DisjointRanges& subrange)
{
    double e = 0.0;
    DisjointRanges subset(subrange,
        particles.commonRanges(
            target_scratch_name()));
    for (auto iter = particles.subsetIter(subset, target_scratch_name()); iter; ++iter) {
        auto& s = iter.template get<0>(); // scratch
        int p = iter.entryId();
        auto& x = xp[p];
        e += (s.A / 2) * (x - s.b / s.A).squaredNorm();
    }
    return e;
}

template <class T, int dim>
void MpmPinningForceHelper<T, dim>::
    computeStressDifferential(const DisjointRanges& subrange, const StdVector<TM>& gradDv, StdVector<TM>& dstress, const TVStack& dvp, TVStack& dfp)
{
    DisjointRanges subset(subrange,
        particles.commonRanges(
            target_scratch_name()));
    for (auto iter = particles.subsetIter(subset, target_scratch_name()); iter; ++iter) {
        auto& s = iter.template get<0>(); // scratch
        int p = iter.entryId();
        dfp.col(p) += (-s.A * dvp.col(p));
    }
}

template class MpmPinningForceHelper<float, 2>;
template class MpmPinningForceHelper<float, 3>;
template class MpmPinningForceHelper<double, 2>;
template class MpmPinningForceHelper<double, 3>;
} // namespace ZIRAN
