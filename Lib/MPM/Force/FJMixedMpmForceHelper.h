#pragma once

#include "MpmForceHelperBase.h"

#include <Ziran/CS/Util/Forward.h>
#include <Ziran/CS/DataStructure/DataArray.h>
#include <Ziran/CS/DataStructure/DataManager.h>
#include <Ziran/Math/Geometry/Particles.h>

namespace ZIRAN {

template <class T, int dim>
class MpmForceHelperBase;

template <class FCONST, class JCONST>
class FJMixedMpmForceHelper : public MpmForceHelperBase<typename FCONST::Scalar, FCONST::TM::RowsAtCompileTime> {
public:
    static const int dim = FCONST::TM::RowsAtCompileTime;
    using T = typename FCONST::Scalar;
    using FScratch = typename FCONST::Scratch;
    using JScratch = typename JCONST::Scratch;
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;

    Particles<T, dim>& particles;
    DataArray<T>& element_measure;

    DataArray<FCONST>& f_constitutive_model;
    DataArray<FScratch>& f_scratch;
    DataArray<TM>& F;
    StdVector<TM> Fn;

    DataArray<JCONST>& j_constitutive_model;
    DataArray<JScratch>& j_scratch;
    DataArray<T>& J;
    StdVector<T> Jn;

    bool linear_corotated = false;

    inline static enum { SOLVE_F,
        SOLVE_J } state
        = SOLVE_F;

    explicit FJMixedMpmForceHelper(Particles<T, dim>& particles)
        : particles(particles)
        , element_measure(particles.add(element_measure_name()))
        , f_constitutive_model(particles.add(f_constitutive_model_name()))
        , f_scratch(particles.add(f_constitutive_model_scratch_name()))
        , F(particles.add(F_name()))
        , j_constitutive_model(particles.add(j_constitutive_model_name()))
        , j_scratch(particles.add(j_constitutive_model_scratch_name()))
        , J(particles.add(J_name()))
    {
    }
    virtual ~FJMixedMpmForceHelper() {}

    void reinitialize() override {}

    void backupStrain() override
    {
        Fn.resize(F.array.size());
        particles.parallel_for([&](auto& FF, int FF_index) { Fn[FF_index] = FF; }, F_name(), valueIdOnly(F_name()));
        Jn = J.array;
        if (linear_corotated) {
            auto ranges = particles.X.ranges;
            tbb::parallel_for(ranges,
                [&](DisjointRanges& subrange) {
                    DisjointRanges subset(subrange, particles.commonRanges(f_constitutive_model_name(), F_name()));
                    for (auto iter = particles.subsetIter(subset, f_constitutive_model_name(), F_name()); iter; ++iter) {
                        auto& constitutive_model = iter.template get<0>();
                        const auto& F = iter.template get<1>();
                        constitutive_model.rebuildR(F);
                    }
                });
        }
    }

    void restoreStrain()
    {
        particles.parallel_for([&](auto& FF, int FF_index) { FF = Fn[FF_index]; }, F_name(), valueIdOnly(F_name()));
        J.array = Jn;
    }

    // add stress to vtau (called only by symplectic)
    void updateState(const DisjointRanges& subrange, StdVector<TM>& vtau, TVStack& fp)
    {
        DisjointRanges subset(subrange,
            particles.commonRanges(f_constitutive_model_name(),
                j_constitutive_model_name(),
                element_measure_name(),
                J_name()));
        if (state == SOLVE_F) {
            for (auto iter = particles.subsetIter(subset, f_constitutive_model_name(), f_constitutive_model_scratch_name(), element_measure_name(), F_name()); iter; ++iter) {
                auto& constitutive_model = iter.template get<0>();
                auto& scratch = iter.template get<1>();
                const auto& element_measure = iter.template get<2>();
                const auto& F = iter.template get<3>();
                int p = iter.entryId();
                if (linear_corotated) constitutive_model.rebuildR(F);
                constitutive_model.updateScratch(F, scratch);
                TM vtau_local;
                constitutive_model.kirchhoff(scratch, vtau_local);
                vtau_local *= element_measure;
                assert(vtau_local == vtau_local);
                vtau[p] += vtau_local;
            }
        }
        else {
            for (auto iter = particles.subsetIter(subset, j_constitutive_model_name(), j_constitutive_model_scratch_name(), element_measure_name(), J_name()); iter; ++iter) {
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
    }

    // add stress to vPFnT (called only by implicit)
    void updateImplicitState(const DisjointRanges& subrange, StdVector<TM>& vPFnT, TVStack& fp)
    {
        DisjointRanges subset(subrange,
            particles.commonRanges(f_constitutive_model_name(),
                j_constitutive_model_name(),
                element_measure_name(),
                J_name()));
        if (subset.size() == 0)
            return;
        if (state == SOLVE_F) {
            for (auto iter = particles.subsetIter(subset, f_constitutive_model_name(), f_constitutive_model_scratch_name(), element_measure_name(), F_name(), valueIdOnly(F_name())); iter; ++iter) {
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
        else {
            for (auto iter = particles.subsetIter(subset, j_constitutive_model_name(), j_constitutive_model_scratch_name(), element_measure_name(), J_name(), valueIdOnly(J_name())); iter; ++iter) {
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
    }

    void evolveStrain(const DisjointRanges& subrange, T dt, const StdVector<TM>& gradV)
    {
        DisjointRanges subset(subrange,
            particles.commonRanges(f_constitutive_model_name(),
                f_constitutive_model_scratch_name(),
                j_constitutive_model_name(),
                j_constitutive_model_scratch_name(),
                element_measure_name(),
                F_name(),
                material_phase_name()));
        for (auto iter = particles.subsetIter(subset, f_constitutive_model_name(), F_name(), J_name(), material_phase_name(), j_constitutive_model_name()); iter; ++iter) {
            auto& constitutive_model_F = iter.template get<0>();
            auto& F = iter.template get<1>();
            auto& J = iter.template get<2>();
            auto& phase = iter.template get<3>();
            auto& constitutive_model_J = iter.template get<4>();
            int p = iter.entryId();
            if (phase == MATERIAL_PHASE_FLUID)
                continue;
            F = (TM::Identity() + ((T)dt) * gradV[p]) * F;
            J = (1 + ((T)dt) * gradV[p].trace()) * J;
            assert(gradV[p] == gradV[p]);
            constitutive_model_J.lambda = constitutive_model_F.lambda; // some plasticity modifies lambda in the F model
        }
    }

    double totalEnergy(const DisjointRanges& subrange)
    {
        DisjointRanges subset(subrange,
            particles.commonRanges(f_constitutive_model_name(),
                j_constitutive_model_name(),
                element_measure_name()));
        double e = 0.0;
        for (auto iter = particles.subsetIter(subset, f_constitutive_model_name(), f_constitutive_model_scratch_name(), F_name(), element_measure_name()); iter; ++iter) {
            auto& constitutive_model = iter.template get<0>();
            auto& scratch = iter.template get<1>();
            const TM& F = iter.template get<2>();
            const auto& element_measure = iter.template get<3>();
            constitutive_model.updateScratch(F, scratch);
            e += element_measure * constitutive_model.psi(scratch);
        }
        return e;
    }

    void computeStressDifferential(const DisjointRanges& subrange, const StdVector<TM>& gradDv, StdVector<TM>& dstress, const TVStack& dvp, TVStack& dfp)
    {
        DisjointRanges subset(subrange,
            particles.commonRanges(f_constitutive_model_name(),
                j_constitutive_model_name(),
                element_measure_name()));
        if (subset.size() == 0)
            return;
        if (state == SOLVE_F) {
            for (auto iter = particles.subsetIter(subset, f_constitutive_model_name(), f_constitutive_model_scratch_name(), element_measure_name(), valueIdOnly(F_name())); iter; ++iter) {
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
        else {
            for (auto iter = particles.subsetIter(subset, j_constitutive_model_name(), j_constitutive_model_scratch_name(), element_measure_name(), valueIdOnly(J_name())); iter; ++iter) {
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
    }

    inline static AttributeName<FCONST> f_constitutive_model_name()
    {
        return AttributeName<FCONST>(FCONST::name());
    }
    inline static AttributeName<FScratch> f_constitutive_model_scratch_name()
    {
        return AttributeName<FScratch>(FCONST::scratch_name());
    }
    inline static AttributeName<JCONST> j_constitutive_model_name()
    {
        return AttributeName<JCONST>(JCONST::name());
    }
    inline static AttributeName<JScratch> j_constitutive_model_scratch_name()
    {
        return AttributeName<JScratch>(JCONST::scratch_name());
    }
    inline static AttributeName<T> element_measure_name()
    {
        return AttributeName<T>("element measure");
    }
    inline static AttributeName<TM> F_name()
    {
        return AttributeName<TM>("F");
    }
    inline static AttributeName<T> J_name()
    {
        return AttributeName<T>("J");
    }
};
} // namespace ZIRAN
