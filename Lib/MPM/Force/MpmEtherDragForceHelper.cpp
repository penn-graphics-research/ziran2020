#include "MpmEtherDragForceHelper.h"

namespace ZIRAN {

template <class T, int dim>
void MpmEtherDragForceHelper<T, dim>::
    reinitialize()
{
}

// add to fp for explicit
template <class T, int dim>
void MpmEtherDragForceHelper<T, dim>::
    updateState(const DisjointRanges& subrange, StdVector<TM>& vtau, TVStack& fp)
{
    DisjointRanges subset(subrange, particles.commonRanges(ether_drag_name<T>()));
    for (auto iter = particles.subsetIter(subset, ether_drag_name<T>()); iter; ++iter) {
        T& drag = iter.template get<0>();
        int p = iter.entryId();
        const auto& vn = particles.V[p];
        const T mass = particles.mass[p];
        // f = -drag * v
        fp.col(p) += (-drag * mass * vn);
    }
}

// add to fp for implicit
// f = -k (x-xn)/dt
template <class T, int dim>
void MpmEtherDragForceHelper<T, dim>::
    updateImplicitState(const DisjointRanges& subrange, StdVector<TM>& vPFnT, TVStack& fp)
{
    DisjointRanges subset(subrange, particles.commonRanges(ether_drag_name<T>()));
    for (auto iter = particles.subsetIter(subset, ether_drag_name<T>()); iter; ++iter) {
        T& drag = iter.template get<0>();
        int p = iter.entryId();
        const TV& xn = particles.X[p];
        const TV& x = xp[p]; // from scratch_xp
        const T mass = particles.mass[p];
        TV v = (x - xn) / dt;
        fp.col(p) += (-drag * mass * v);
    }
}

template <class T, int dim>
double MpmEtherDragForceHelper<T, dim>::
    totalEnergy(const DisjointRanges& subrange)
{
    double e = 0.0;
    // TODO
    return e;
}

// f = -k (x-xn)/dt
// dfdx = -k/dt
template <class T, int dim>
void MpmEtherDragForceHelper<T, dim>::
    computeStressDifferential(const DisjointRanges& subrange, const StdVector<TM>& gradDv, StdVector<TM>& dstress, const TVStack& dvp, TVStack& dfp)
{
    DisjointRanges subset(subrange, particles.commonRanges(ether_drag_name<T>()));
    for (auto iter = particles.subsetIter(subset, ether_drag_name<T>()); iter; ++iter) {
        T& drag = iter.template get<0>();
        int p = iter.entryId();
        const T mass = particles.mass[p];
        dfp.col(p) += (-drag * mass / dt) * dvp.col(p);
    }
}

template class MpmEtherDragForceHelper<float, 2>;
template class MpmEtherDragForceHelper<float, 3>;
template class MpmEtherDragForceHelper<double, 2>;
template class MpmEtherDragForceHelper<double, 3>;
} // namespace ZIRAN
