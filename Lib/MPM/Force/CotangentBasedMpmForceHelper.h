#ifndef COTANGENT_BASED_MPM_FORCE_HELPER_H
#define COTANGENT_BASED_MPM_FORCE_HELPER_H

#include <Ziran/Math/Geometry/Particles.h>
#include <Ziran/Math/Geometry/Elements.h>

namespace ZIRAN {

template <class T, int dim>
class MpmForceHelperBase;

template <class TCONST, int manifold_dim>
class CotangentBasedMpmForceHelper : public MpmForceHelperBase<typename TCONST::Scalar, TCONST::TM::RowsAtCompileTime> {
public:
    static const int dim = TCONST::TM::RowsAtCompileTime;
    using T = typename TCONST::Scalar;
    using Scratch = typename TCONST::Scratch;
    typedef Matrix<T, dim, dim> TM;
    typedef Matrix<T, dim, dim - manifold_dim> CTM;
    typedef Matrix<T, dim, manifold_dim> TTM;
    typedef Vector<T, dim> TV;
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;

    Particles<T, dim>& particles;
    SimplexElements<T, manifold_dim, dim>& elements;
    DataArray<T>& element_measure;
    DataArray<TCONST>& constitutive_model;
    DataArray<Scratch>& scratch;

    DataArray<TM>& cotangent; // this is the strain measure

    DataArray<TTM>& tangent_F; // this belongs to the element manager, resized and set by the tangent force
    const DataArray<TTM>& tangent_dF; // this belongs to the element manager, resized and set by the tangent force
    DataArray<int>& parent_element; // initialization should take care of the size of this.

    // !!!!!!!!!!!!!!!!!!! this index into the local data array of tangent_F, not the global element index!

    DataArray<TM>& VP; // volume scaled first Piola Kichhoff stress
    DataArray<TM>& VdP; // volume scaled first Piola Kichhoff stress differential

    StdVector<CTM> cotangent_n;

    explicit CotangentBasedMpmForceHelper(Particles<T, dim>& particles, SimplexElements<T, manifold_dim, dim>& elements)
        : particles(particles)
        , elements(elements)
        , element_measure(particles.add(element_measure_name()))
        , constitutive_model(particles.add(constitutive_model_name()))
        , scratch(particles.add(constitutive_model_scratch_name()))
        , cotangent(particles.add(cotangent_name()))
        , tangent_F(elements.get(tangent_F_name()))
        , tangent_dF(elements.get(tangent_dF_name()))
        , parent_element(particles.get(parent_element_name()))
        , VP(particles.get(VP_name()))
        , VdP(particles.add(VdP_name()))
    {
        tangent_F.registerReorderingCallback(
            [&](const DataArrayBase::ValueIDToNewValueID& old_to_new) {
                for (int i = 0; i < (int)parent_element.size(); i++)
                    parent_element[i] = old_to_new(parent_element[i]); });
    }
    virtual ~CotangentBasedMpmForceHelper() {}

    void reinitialize() override
    {
        // DisjointRanges dr = cotangent.ranges;
        // dr.merge(constitutive_model.ranges);
        // scratch.lazyResize(dr);
        // VP.lazyResize(dr);
        // VdP.lazyResize(dr);
        // the user should set the initial cotangents (using mesh)
    }

    void backupStrain() override
    {
        cotangent_n.resize(cotangent.array.size());
        particles.parallel_for([&](auto& ct, int ct_index) {
            cotangent_n[ct_index] = ct.template topRightCorner<dim, dim - manifold_dim>();
        },
            cotangent_name(), valueIdOnly(cotangent_name()));
    }

    void restoreStrain() override
    {
        particles.parallel_for([&](auto& ct, int ct_index) {
            ct.template topRightCorner<dim, dim - manifold_dim>() = cotangent_n[ct_index];
        },
            cotangent_name(), valueIdOnly(cotangent_name()));
    }

    // build the full F, compute full VP, and vtau stores Vp_0 * P.col(2) * d^T
    void updateState(const DisjointRanges& subrange, StdVector<TM>& vtau, TVStack& fp) override
    {
        DisjointRanges subset(subrange,
            particles.commonRanges(constitutive_model_name(),
                constitutive_model_scratch_name(),
                element_measure_name(),
                parent_element_name(),
                cotangent_name(),
                VP_name()));

        for (auto iter = particles.subsetIter(subset, constitutive_model_name(), constitutive_model_scratch_name(), element_measure_name(), parent_element_name(), cotangent_name(), VP_name()); iter; ++iter) {
            auto& constitutive_model = iter.template get<0>();
            auto& scratch = iter.template get<1>();
            const auto& element_measure = iter.template get<2>();
            const int parent = iter.template get<3>(); // parent index into the data array of tangent_F, not the global element index
            const TTM& tF = tangent_F(parent);
            TM& ctF = iter.template get<4>();
            TM& VP_local = iter.template get<5>();
            ctF.template topLeftCorner<dim, manifold_dim>() = tF;
            TM& F = ctF;
            int p = iter.entryId();
            constitutive_model.updateScratch(F, scratch);
            TM P_local;
            constitutive_model.firstPiola(scratch, P_local);
            assert(P_local == P_local);
            P_local *= element_measure;
            VP_local = P_local;
            vtau[p] += P_local.template bottomRightCorner<dim, dim - manifold_dim>() * ctF.template topRightCorner<dim, dim - manifold_dim>().transpose();
        }
    }

    void updateImplicitState(const DisjointRanges& subrange, StdVector<TM>& vPFnT, TVStack& fp) override
    {
        DisjointRanges subset(subrange,
            particles.commonRanges(constitutive_model_name(),
                constitutive_model_scratch_name(),
                element_measure_name(),
                parent_element_name(),
                cotangent_name(),
                VP_name()));
        if (subset.size() == 0)
            return;
        for (auto iter = particles.subsetIter(subset, constitutive_model_name(), constitutive_model_scratch_name(), element_measure_name(), parent_element_name(), cotangent_name(), VP_name(), valueIdOnly(cotangent_name())); iter; ++iter) {
            auto& constitutive_model = iter.template get<0>();
            auto& scratch = iter.template get<1>();
            const auto& element_measure = iter.template get<2>();
            const int parent = iter.template get<3>(); // parent index into the data array of tangent_F, not the global element index
            const TTM& tF = tangent_F(parent);
            TM& ctF = iter.template get<4>();
            TM& VP_local = iter.template get<5>();
            int ctF_index = iter.template get<6>();
            ctF.template topLeftCorner<dim, manifold_dim>() = tF;
            TM& F = ctF;
            int p = iter.entryId();
            constitutive_model.updateScratch(F, scratch);
            TM P_local;
            constitutive_model.firstPiola(scratch, P_local);
            assert(P_local == P_local);
            P_local *= element_measure;
            VP_local = P_local;
            vPFnT[p] += P_local.template bottomRightCorner<dim, dim - manifold_dim>() * cotangent_n[ctF_index].template topRightCorner<dim, dim - manifold_dim>().transpose();
        }
    }

    void evolveStrain(const DisjointRanges& subrange, T dt, const StdVector<TM>& gradV) override
    {
        DisjointRanges subset(subrange,
            particles.commonRanges(constitutive_model_name(),
                constitutive_model_scratch_name(), // Do not delete (scratch name has Cotangent keyword afterward)
                element_measure_name(),
                cotangent_name()));
        for (auto iter = particles.subsetIter(subset, constitutive_model_name(), cotangent_name()); iter; ++iter) {
            auto& cotangent = iter.template get<1>();
            int p = iter.entryId();
            cotangent.template topRightCorner<dim, dim - manifold_dim>() = (TM::Identity() + ((T)dt) * gradV[p]) * cotangent.template topRightCorner<dim, dim - manifold_dim>();
            assert(gradV[p] == gradV[p]);
        }
    }

    double totalEnergy(const DisjointRanges& subrange) override
    {
        DisjointRanges subset(subrange,
            particles.commonRanges(constitutive_model_name(),
                constitutive_model_scratch_name(),
                element_measure_name()));
        double e = 0.0;
        for (auto iter = particles.subsetIter(subset, constitutive_model_name(), constitutive_model_scratch_name(), element_measure_name()); iter; ++iter) {
            const auto& constitutive_model = iter.template get<0>();
            auto& scratch = iter.template get<1>();
            const auto& element_measure = iter.template get<2>();
            e += element_measure * constitutive_model.psi(scratch);
        }
        return e;
    }

    void computeStressDifferential(const DisjointRanges& subrange, const StdVector<TM>& gradDv, StdVector<TM>& dstress, const TVStack& dvp, TVStack& dfp) override
    {
        DisjointRanges subset(subrange,
            particles.commonRanges(constitutive_model_name(),
                constitutive_model_scratch_name(),
                element_measure_name(),
                parent_element_name(),
                VdP_name(),
                cotangent_name()));
        if (subset.size() == 0)
            return;
        for (auto iter = particles.subsetIter(subset, constitutive_model_name(), constitutive_model_scratch_name(), element_measure_name(), parent_element_name(), VdP_name(), valueIdOnly(cotangent_name())); iter; ++iter) {
            auto& constitutive_model = iter.template get<0>();
            auto& scratch = iter.template get<1>();
            const auto& element_measure = iter.template get<2>();
            const int parent = iter.template get<3>(); // parent index into the data array of tangent_F and tangent_dF, not the global element index
            TM& VdP_local = iter.template get<4>();
            int ctF_index = iter.template get<5>();
            int p = iter.entryId();
            const TTM& t_dF = tangent_dF(parent);
            const auto& ctF_n = cotangent_n[ctF_index];
            const CTM& ct_dF = gradDv[p] * ctF_n.template topRightCorner<dim, dim - manifold_dim>();
            TM dF;
            dF << t_dF, ct_dF;
            TM dP_local;
            constitutive_model.firstPiolaDifferential(scratch, dF, dP_local);
            assert(dP_local == dP_local);
            dP_local *= element_measure;
            VdP_local = dP_local;
            dstress[p] += dP_local.template bottomRightCorner<dim, dim - manifold_dim>() * ctF_n.template topRightCorner<dim, dim - manifold_dim>().transpose();
        }
    }

    static AttributeName<TCONST> constitutive_model_name()
    {
        return AttributeName<TCONST>(std::string(TCONST::name()));
    }
    static AttributeName<Scratch> constitutive_model_scratch_name()
    {
        return AttributeName<Scratch>(std::string(TCONST::scratch_name()) + "Cotangent");
    }
    static AttributeName<T> element_measure_name()
    {
        return AttributeName<T>("element measure");
    }
    static AttributeName<TM> cotangent_name()
    {
        return AttributeName<TM>("cotangent");
    }

    static AttributeName<TTM> tangent_F_name()
    {
        return AttributeName<TTM>("tangent F");
    }

    static AttributeName<TTM> tangent_dF_name()
    {
        return AttributeName<TTM>("tangent dF");
    }

    static AttributeName<int> parent_element_name()
    {
        return AttributeName<int>("parent element" + std::to_string(manifold_dim));
    }

    static AttributeName<TM> VP_name() // volume scaled P
    {
        return AttributeName<TM>("VP");
    }

    static AttributeName<TM> VdP_name() // volume scaled dP
    {
        return AttributeName<TM>("VdP");
    }
};
} // namespace ZIRAN

#endif
