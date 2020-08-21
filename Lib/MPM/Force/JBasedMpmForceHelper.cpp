#include <Ziran/Physics/ConstitutiveModel/EquationOfState.h>
#include <Ziran/CS/Util/AttributeNamesForward.h>
#include "JBasedMpmForceHelper.h"

namespace ZIRAN {

template <class TCONST>
void JBasedMpmForceHelper<TCONST>::reinitialize()
{
}

template <class TCONST>
void JBasedMpmForceHelper<TCONST>::
    backupStrain()
{
    //TODO: parallellize
    Jn = J.array;
}

template <class TCONST>
void JBasedMpmForceHelper<TCONST>::
    restoreStrain()
{
    //TODO: parallellize
    J.array = Jn;
}

// add stress to vtau
template <class TCONST>
void JBasedMpmForceHelper<TCONST>::
    updateState(const DisjointRanges& subrange, StdVector<TM>& vtau, TVStack& fp)
{
    DisjointRanges subset(subrange,
        particles.commonRanges(constitutive_model_name(),
            constitutive_model_scratch_name(),
            element_measure_name<T>(),
            J_name<T>()));
    for (auto iter = particles.subsetIter(subset, constitutive_model_name(), constitutive_model_scratch_name(), element_measure_name<T>(), J_name<T>()); iter; ++iter) {
        auto& constitutive_model = iter.template get<0>();
        auto& scratch = iter.template get<1>();
        const auto& element_measure = iter.template get<2>();
        const auto& J = iter.template get<3>();
        int p = iter.entryId();
        constitutive_model.updateScratch(J, scratch);
        T dpsi_dJ;
        constitutive_model.firstDerivative(scratch, dpsi_dJ);
        assert(dpsi_dJ == dpsi_dJ);
        vtau[p] += element_measure * dpsi_dJ * J * TM::Identity();
    }
}

// add stress to vPFnT
template <class TCONST>
void JBasedMpmForceHelper<TCONST>::
    updateImplicitState(const DisjointRanges& subrange, StdVector<TM>& vPFnT, TVStack& fp)
{
    DisjointRanges subset(subrange,
        particles.commonRanges(constitutive_model_name(),
            constitutive_model_scratch_name(),
            element_measure_name<T>(),
            J_name<T>()));
    if (subset.size() == 0)
        return;
    for (auto iter = particles.subsetIter(subset, constitutive_model_name(), constitutive_model_scratch_name(), element_measure_name<T>(), J_name<T>(), valueIdOnly(J_name<T>())); iter; ++iter) {
        auto& constitutive_model = iter.template get<0>();
        auto& scratch = iter.template get<1>();
        const auto& element_measure = iter.template get<2>();
        const auto& J = iter.template get<3>();
        const int J_id = iter.template get<4>();
        int p = iter.entryId();
        constitutive_model.updateScratch(J, scratch);
        T dpsi_dJ;
        constitutive_model.firstDerivative(scratch, dpsi_dJ);
        assert(dpsi_dJ == dpsi_dJ);
        vPFnT[p] += element_measure * dpsi_dJ * Jn[J_id] * TM::Identity();
    }
}

template <class TCONST>
void JBasedMpmForceHelper<TCONST>::
    evolveStrain(const DisjointRanges& subrange, T dt, const StdVector<TM>& gradV)
{
    DisjointRanges subset(subrange,
        particles.commonRanges(constitutive_model_name(),
            constitutive_model_scratch_name(),
            element_measure_name<T>(),
            J_name<T>()));
    for (auto iter = particles.subsetIter(subset, constitutive_model_name(), J_name<T>()); iter; ++iter) {
        auto& J = iter.template get<1>();
        int p = iter.entryId();
        J = (1 + ((T)dt) * gradV[p].trace()) * J;
        assert(gradV[p] == gradV[p]);
    }
}

//total potential energy
template <class TCONST>
double JBasedMpmForceHelper<TCONST>::
    totalEnergy(const DisjointRanges& subrange)
{
    DisjointRanges subset(subrange,
        particles.commonRanges(constitutive_model_name(),
            constitutive_model_scratch_name(),
            element_measure_name<T>(),
            J_name<T>()));
    double e = 0.0;
    for (auto iter = particles.subsetIter(subset, constitutive_model_name(), constitutive_model_scratch_name(), element_measure_name<T>(), J_name<T>()); iter; ++iter) {
        auto& constitutive_model = iter.template get<0>();
        auto& scratch = iter.template get<1>();
        const auto& element_measure = iter.template get<2>();
        const T& J = iter.template get<3>();
        constitutive_model.updateScratch(J, scratch);
        e += element_measure * constitutive_model.psi(scratch);
    }
    return e;
}

template <class TCONST>
void JBasedMpmForceHelper<TCONST>::
    computeStressDifferential(const DisjointRanges& subrange, const StdVector<TM>& gradDv, StdVector<TM>& dstress, const TVStack& dvp, TVStack& dfp)
{
    DisjointRanges subset(subrange,
        particles.commonRanges(constitutive_model_name(),
            constitutive_model_scratch_name(),
            element_measure_name<T>()));
    if (subset.size() == 0)
        return;
    for (auto iter = particles.subsetIter(subset, constitutive_model_name(), constitutive_model_scratch_name(), element_measure_name<T>(), valueIdOnly(J_name<T>())); iter; ++iter) {
        const auto& constitutive_model = iter.template get<0>();
        auto& scratch = iter.template get<1>();
        const auto& element_measure = iter.template get<2>();
        const int J_index = iter.template get<3>();
        int p = iter.entryId();
        T d2psi_dJ2;
        constitutive_model.secondDerivative(scratch, d2psi_dJ2);
        assert(d2psi_dJ2 == d2psi_dJ2);
        dstress[p] += (element_measure * d2psi_dJ2 * Jn[J_index] * gradDv[p].trace() * Jn[J_index]) * TM::Identity();
    }
}

template class JBasedMpmForceHelper<EquationOfState<double, 2>>;
template class JBasedMpmForceHelper<EquationOfState<double, 3>>;
template class JBasedMpmForceHelper<EquationOfState<float, 2>>;
template class JBasedMpmForceHelper<EquationOfState<float, 3>>;
template class JBasedMpmForceHelper<QuadraticVolumePenalty<double, 2>>;
template class JBasedMpmForceHelper<QuadraticVolumePenalty<double, 3>>;
template class JBasedMpmForceHelper<QuadraticVolumePenalty<float, 2>>;
template class JBasedMpmForceHelper<QuadraticVolumePenalty<float, 3>>;
} // namespace ZIRAN
