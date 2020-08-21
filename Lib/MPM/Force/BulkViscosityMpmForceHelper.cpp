#include "BulkViscosityMpmForceHelper.h"

#include <Ziran/Math/MathTools.h>
#include <Eigen/LU>

namespace ZIRAN {

template <class T, int dim>
bool BulkViscosityMpmForceHelper<T, dim>::
    needGradVn()
{
    return true;
}

template <class T, int dim>
void BulkViscosityMpmForceHelper<T, dim>::
    getGradVn(const StdVector<TM>& gradVn)
{
    div_vn.resize(particles.count);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, particles.count),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t b = range.begin(), b_end = range.end(); b < b_end; ++b) {
                div_vn[b] = gradVn[b].trace();
            }
        });
}

template <class T, int dim>
void BulkViscosityMpmForceHelper<T, dim>::
    updateStateHelper(const BulkViscosity<T, dim>& bulk, const T element_measure, const T& J, const T rate, TM& vtau)
{
    using MATH_TOOLS::sqr;
    if (J < std::numeric_limits<T>::epsilon()) // skip inverted particles
        return;
    T p = bulk.b * rate;
    vtau += p * TM::Identity() * (J * element_measure);
}

// add stress to V^0 tau = V^n sigma
template <class T, int dim>
void BulkViscosityMpmForceHelper<T, dim>::
    updateState(const DisjointRanges& subrange, StdVector<TM>& vtau, TVStack& fp)
{
    if (particles.exist(J_name())) {
        DisjointRanges subset_J(subrange,
            particles.commonRanges(bulk_name(), element_measure_name(), J_name()));
        for (auto iter = particles.subsetIter(subset_J, bulk_name(), element_measure_name(), J_name()); iter; ++iter) {
            const auto& bulk = iter.template get<0>();
            const auto& element_measure = iter.template get<1>();
            const auto& J = iter.template get<2>();
            int p = iter.entryId();
            updateStateHelper(bulk, element_measure, J, div_vn[p], vtau[p]);
        }
    }

    else if (particles.exist(F_name())) {
        DisjointRanges subset_F(subrange,
            particles.commonRanges(bulk_name(), element_measure_name(), F_name()));
        for (auto iter = particles.subsetIter(subset_F, bulk_name(), element_measure_name(), F_name()); iter; ++iter) {
            const auto& bulk = iter.template get<0>();
            const auto& element_measure = iter.template get<1>();
            const auto& F = iter.template get<2>();
            int p = iter.entryId();
            updateStateHelper(bulk, element_measure, F.determinant(), div_vn[p], vtau[p]);
        }
    }
    else if (particles.exist(cotangent_name())) {
        DisjointRanges subset_cotangent(subrange,
            particles.commonRanges(bulk_name(), element_measure_name(), cotangent_name()));
        for (auto iter = particles.subsetIter(subset_cotangent, bulk_name(), element_measure_name(), cotangent_name()); iter; ++iter) {
            const auto& bulk = iter.template get<0>();
            const auto& element_measure = iter.template get<1>();
            const auto& cotangent = iter.template get<2>();
            int p = iter.entryId();
            updateStateHelper(bulk, element_measure, cotangent.determinant(), div_vn[p], vtau[p]);
        }
    }
    else
        ZIRAN_ASSERT(false, "Attempt to use BulkViscosity without F or cotangent");
}

template class BulkViscosityMpmForceHelper<double, 2>;
template class BulkViscosityMpmForceHelper<double, 3>;
template class BulkViscosityMpmForceHelper<float, 2>;
template class BulkViscosityMpmForceHelper<float, 3>;
} // namespace ZIRAN
