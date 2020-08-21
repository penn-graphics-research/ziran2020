#include "MpmParallelPinningForceHelper.h"
#include <Ziran/Math/Geometry/Particles.h>

namespace ZIRAN {

template <class T, int dim>
void MpmParallelPinningForceHelper<T, dim>::
    reinitialize()
{
    // make sure scratch has the same size with target
    scratch.lazyResize(target.ranges);
}

// add to fp for explicit
template <class T, int dim>
void MpmParallelPinningForceHelper<T, dim>::
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

        for (int i = 0; i < (int)t.x.size(); ++i) {
            if (collider_spring) {
                TV n = (t.x[i] - xn);
                if (n.squaredNorm() < (T)1e-12) continue;
                n.normalize();
                TM M = n * n.transpose();
                fp.col(p) += -t.k * M * (xn - t.x[i]);
            }
            else {
                // f = -k(x - xt) - d(v - vt);
                fp.col(p) += (-t.k * (xn - t.x[i]) - t.d * (vn - t.v[i]));
            }
        }
    }
}

// add to fp for implicit
template <class T, int dim>
void MpmParallelPinningForceHelper<T, dim>::
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
        for (int i = 0; i < (int)t.x.size(); ++i) {
            const TV& xtn = t.x[i];
            if (collider_spring) {
                TV n = (xtn - xn);
                if (n.squaredNorm() < (T)1e-12) continue;
                n.normalize();
                TM M = n * n.transpose();
                if ((x - xtn).dot(n) < 0)
                    fp.col(p) += -t.k * M * (x - xtn);
            }
            else {
                const TV& vtn = t.v[i];
                const TV& xt = xtn + vtn * dt; // candidate target n+1 position
                s.A = t.k + t.d / dt;
                s.b = t.k * xt - (t.d / dt) * (-xn - xt + xtn);
                fp.col(p) += (-s.A * x + s.b);
            }
        }
    }
}

template <class T, int dim>
double MpmParallelPinningForceHelper<T, dim>::
    totalEnergy(const DisjointRanges& subrange)
{
    double e = 0.0;
    DisjointRanges subset(subrange,
        particles.commonRanges(target_name(),
            target_scratch_name()));
    for (auto iter = particles.subsetIter(subset, target_name(), target_scratch_name()); iter; ++iter) {
        auto& t = iter.template get<0>(); // target
        auto& s = iter.template get<1>(); // scratch
        int p = iter.entryId();
        auto& x = xp[p];
        for (int i = 0; i < (int)t.x.size(); ++i) {
            if (collider_spring) {
                const TV& xn = particles.X[p];
                const TV& xtn = t.x[i];
                TV n = (xtn - xn);
                if (n.squaredNorm() < (T)1e-12) continue;
                n.normalize();
                TM M = n * n.transpose();
                if ((x - xtn).dot(n) < 0)
                    e += t.k / 2 * (x - xtn).transpose() * M * (x - xtn);
            }
            else
                e += (s.A / 2) * (x - s.b / s.A).squaredNorm();
        }
    }
    return e;
}

template <class T, int dim>
void MpmParallelPinningForceHelper<T, dim>::
    computeStressDifferential(const DisjointRanges& subrange, const StdVector<TM>& gradDv, StdVector<TM>& dstress, const TVStack& dvp, TVStack& dfp)
{
    DisjointRanges subset(subrange,
        particles.commonRanges(target_name(),
            target_scratch_name()));
    for (auto iter = particles.subsetIter(subset, target_name(), target_scratch_name()); iter; ++iter) {
        auto& t = iter.template get<0>(); // target
        auto& s = iter.template get<1>(); // scratch
        int p = iter.entryId();
        for (int i = 0; i < (int)t.x.size(); ++i) {
            if (collider_spring) {
                const TV& xn = particles.X[p];
                const TV& x = xp[p]; // from scratch_xp
                const TV& xtn = t.x[i];
                TV n = (xtn - xn);
                if (n.squaredNorm() < (T)1e-12) continue;
                n.normalize();
                TM M = n * n.transpose();
                if ((x - xtn).dot(n) < 0)
                    dfp.col(p) += -t.k * M * dvp.col(p);
            }
            else
                dfp.col(p) += (-s.A * dvp.col(p));
        }
    }
}

template class MpmParallelPinningForceHelper<float, 2>;
template class MpmParallelPinningForceHelper<float, 3>;
template class MpmParallelPinningForceHelper<double, 2>;
template class MpmParallelPinningForceHelper<double, 3>;
} // namespace ZIRAN
