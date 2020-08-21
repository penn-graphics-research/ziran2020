#include "MpmForceBase.h"
#include "LinearCorotatedMpmForceHelper.h"
#include <Ziran/CS/DataStructure/HashTable.h>
#include <Ziran/CS/DataStructure/DataManager.h>
#include <Ziran/Math/Geometry/SimplexMesh.h>
#include <Ziran/Math/Geometry/Particles.h>
#include <Ziran/Physics/ConstitutiveModel/ConstitutiveModel.h>
#include <tbb/tbb.h>
#include <tick/requires.h>
#include <Ziran/CS/Util/AttributeNamesForward.h>

namespace ZIRAN {

template <class T, int dim>
void LinearCorotatedMpmForceHelper<T, dim>::
    reinitialize()
{
}

template <class T, int dim>
void LinearCorotatedMpmForceHelper<T, dim>::
    refreshRotations()
{
    auto ranges = particles.X.ranges;
    tbb::parallel_for(ranges,
        [&](DisjointRanges& subrange) {
            DisjointRanges subset(subrange,
                particles.commonRanges(constitutive_model_name(),
                    F_name()));
            for (auto iter = particles.subsetIter(subset, constitutive_model_name(), F_name()); iter; ++iter) {
                auto& constitutive_model = iter.template get<0>();
                const auto& F = iter.template get<1>();
                constitutive_model.rebuildR(F); // specific to LinearCorotated
            }
        });
}

template <class T, int dim>
void LinearCorotatedMpmForceHelper<T, dim>::
    backupStrain()
{
    Fn.resize(F.array.size());
    particles.parallel_for([&](auto& FF, int FF_index) {
        Fn[FF_index] = FF;
    },
        F_name(), valueIdOnly(F_name()));

    refreshRotations();
}

template <class T, int dim>
void LinearCorotatedMpmForceHelper<T, dim>::
    restoreStrain()
{
    particles.parallel_for([&](auto& FF, int FF_index) {
        FF = Fn[FF_index];
    },
        F_name(), valueIdOnly(F_name()));
}

// add stress to vtau
template <class T, int dim>
void LinearCorotatedMpmForceHelper<T, dim>::updateState(const DisjointRanges& subrange, StdVector<TM>& vtau, TVStack& fp)
{
    DisjointRanges subset(subrange,
        particles.commonRanges(constitutive_model_name(),
            constitutive_model_scratch_name(),
            element_measure_name(),
            F_name()));
    for (auto iter = particles.subsetIter(subset, constitutive_model_name(), constitutive_model_scratch_name(), element_measure_name(), F_name()); iter; ++iter) {
        auto& constitutive_model = iter.template get<0>();
        auto& scratch = iter.template get<1>();
        const auto& element_measure = iter.template get<2>();
        const auto& F = iter.template get<3>();
        int p = iter.entryId();
        constitutive_model.rebuildR(F); // specific to LinearCorotated
        constitutive_model.updateScratch(F, scratch);
        TM vtau_local;
        constitutive_model.kirchhoff(scratch, vtau_local);
        vtau_local *= element_measure;
        assert(vtau_local == vtau_local);
        vtau[p] += vtau_local;
    }
}

// add stress to vPFnT
template <class T, int dim>
void LinearCorotatedMpmForceHelper<T, dim>::
    updateImplicitState(const DisjointRanges& subrange, StdVector<TM>& vPFnT, TVStack& fp)
{
    DisjointRanges subset(subrange,
        particles.commonRanges(constitutive_model_name(),
            constitutive_model_scratch_name(),
            element_measure_name(),
            F_name()));
    if (subset.size() == 0)
        return;
    for (auto iter = particles.subsetIter(subset, constitutive_model_name(), constitutive_model_scratch_name(), element_measure_name(), F_name(), valueIdOnly(F_name())); iter; ++iter) {
        auto& constitutive_model = iter.template get<0>();
        auto& scratch = iter.template get<1>();
        const auto& element_measure = iter.template get<2>();
        const auto& F = iter.template get<3>();
        const int F_index = iter.template get<4>();
        int p = iter.entryId();
        constitutive_model.updateScratch(F, scratch);
        TM vPFnT_local;
        constitutive_model.firstPiola(scratch, vPFnT_local);
        vPFnT_local = element_measure * vPFnT_local * Fn[F_index].transpose();
        assert(vPFnT_local == vPFnT_local);
        vPFnT[p] += vPFnT_local;
    }
}

template <class T, int dim>
void LinearCorotatedMpmForceHelper<T, dim>::
    evolveStrain(const DisjointRanges& subrange, T dt, const StdVector<TM>& gradV)
{
    DisjointRanges subset(subrange,
        particles.commonRanges(constitutive_model_name(),
            constitutive_model_scratch_name(),
            element_measure_name(),
            F_name()));
    for (auto iter = particles.subsetIter(subset, constitutive_model_name(), F_name()); iter; ++iter) {
        auto& F = iter.template get<1>();
        int p = iter.entryId();
        F = (TM::Identity() + ((T)dt) * gradV[p]) * F;
        assert(gradV[p] == gradV[p]);
    }
}

template <class T, int dim>
double LinearCorotatedMpmForceHelper<T, dim>::
    totalEnergy(const DisjointRanges& subrange)
{
    DisjointRanges subset(subrange,
        particles.commonRanges(constitutive_model_name(),
            constitutive_model_scratch_name(),
            element_measure_name()));
    double e = 0.0;
    for (auto iter = particles.subsetIter(subset, constitutive_model_name(), constitutive_model_scratch_name(), F_name(), element_measure_name()); iter; ++iter) {
        auto& constitutive_model = iter.template get<0>();
        auto& scratch = iter.template get<1>();
        const TM& F = iter.template get<2>();
        const auto& element_measure = iter.template get<3>();
        constitutive_model.updateScratch(F, scratch);
        e += element_measure * constitutive_model.psi(scratch);
    }
    return e;
}

template <class T, int dim>
void LinearCorotatedMpmForceHelper<T, dim>::
    computeStressDifferential(const DisjointRanges& subrange, const StdVector<TM>& gradDv, StdVector<TM>& dstress, const TVStack& dvp, TVStack& dfp)
{
    DisjointRanges subset(subrange,
        particles.commonRanges(constitutive_model_name(),
            constitutive_model_scratch_name(),
            element_measure_name(),
            F_name()));
    if (subset.size() == 0)
        return;
    for (auto iter = particles.subsetIter(subset, constitutive_model_name(), constitutive_model_scratch_name(), element_measure_name(), valueIdOnly(F_name())); iter; ++iter) {
        const auto& constitutive_model = iter.template get<0>();
        auto& scratch = iter.template get<1>();
        const auto& element_measure = iter.template get<2>();
        int F_index = iter.template get<3>();
        int p = iter.entryId();
        TM dP;
        const auto& Fn_local = Fn[F_index];
        constitutive_model.firstPiolaDifferential(scratch, gradDv[p] * Fn_local, dP);
        assert(dP == dP);
        dstress[p] += dP * element_measure * Fn_local.transpose();
    }
}

template class LinearCorotatedMpmForceHelper<double, 2>;
template class LinearCorotatedMpmForceHelper<double, 3>;
template class LinearCorotatedMpmForceHelper<float, 2>;
template class LinearCorotatedMpmForceHelper<float, 3>;
} // namespace ZIRAN
