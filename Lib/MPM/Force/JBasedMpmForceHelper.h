#ifndef J_BASED_MPM_FORCE_HELPER_H
#define J_BASED_MPM_FORCE_HELPER_H

#include <Ziran/Math/Geometry/Particles.h>
#include <Ziran/CS/Util/AttributeNamesForward.h>
#include <MPM/Force/MpmForceHelperBase.h>

namespace ZIRAN {

/**
 This is for fluid equation of state.

 Energy density is in terms of J:
 Phi = \sum_p V_p^0 \Psi(J_p(x))

 Strain evolution:
 J(x) = ( 1 + tr( \sum_i (xi-xi^n) (grad w_ip^n)^T ) ) J^n
 */
template <class TCONST>
class JBasedMpmForceHelper : public MpmForceHelperBase<typename TCONST::Scalar, TCONST::TM::RowsAtCompileTime> {
public:
    static const int dim = TCONST::TM::RowsAtCompileTime;
    using T = typename TCONST::Scalar;
    using Scratch = typename TCONST::Scratch;
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;

    Particles<T, dim>& particles;
    DataArray<T>& element_measure;
    DataArray<TCONST>& constitutive_model;
    DataArray<Scratch>& scratch;
    DataArray<T>& J;

    StdVector<T> Jn;

    explicit JBasedMpmForceHelper(Particles<T, dim>& particles)
        : particles(particles)
        , element_measure(particles.add(element_measure_name<T>()))
        , constitutive_model(particles.add(constitutive_model_name()))
        , scratch(particles.add(constitutive_model_scratch_name()))
        , J(particles.add(J_name<T>()))
    {
    }
    virtual ~JBasedMpmForceHelper() {}

    void reinitialize() override;

    void backupStrain() override;

    void restoreStrain() override;

    // add stress to vtau
    void updateState(const DisjointRanges& subrange, StdVector<TM>& vtau, TVStack& fp) override;

    // add stress to vPFnT
    void updateImplicitState(const DisjointRanges& subrange, StdVector<TM>& vPFnT, TVStack& fp) override;

    void evolveStrain(const DisjointRanges& subrange, T dt, const StdVector<TM>& gradV) override;

    double totalEnergy(const DisjointRanges& subrange) override;

    void computeStressDifferential(const DisjointRanges& subrange, const StdVector<TM>& gradDv, StdVector<TM>& dstress, const TVStack& dvp, TVStack& dfp) override;

    using Hessian = Eigen::Matrix<T, dim * dim, dim * dim>;
    void runLambdaWithDifferential(
        const StdVector<int>& particle_order,
        const std::vector<std::pair<int, int>>& particle_group,
        const std::vector<uint64_t>& block_offset,
        std::function<void(int, const Hessian&, const TM&, const T&, const T&, bool)> func, int opt = 0) override
    {
        // TODO: model may not match
        std::vector<int> model_local_idx(particles.count, -1);
        std::vector<int> J_local_idx(particles.count, -1);
        static std::vector<T> dPdJ(particles.count);
        if (opt == 1) {
            if ((int)dPdJ.size() != particles.count)
                dPdJ.resize(particles.count);
        }
        tbb::parallel_for(particles.X.ranges, [&](DisjointRanges& subrange) {
            DisjointRanges subset(subrange, particles.commonRanges(constitutive_model_name(), J_name<T>()));
            for (auto iter = particles.subsetIter(subset, constitutive_model_name(), valueIdOnly(constitutive_model_name()), J_name<T>(), valueIdOnly(J_name<T>())); iter; ++iter) {
                model_local_idx[iter.entryId()] = iter.template get<1>();
                J_local_idx[iter.entryId()] = iter.template get<3>();
            }
        });
        auto* J_pointer = &particles.DataManager::get(J_name<T>());
        auto* model_pointer = &particles.DataManager::get(constitutive_model_name());
        for (uint64_t color = 0; color < (1 << dim); ++color)
            tbb::parallel_for(0, (int)particle_group.size(), [&](int group_idx) {
                if ((block_offset[group_idx] & ((1 << dim) - 1)) != color)
                    return;
                for (int idx = particle_group[group_idx].first; idx <= particle_group[group_idx].second; ++idx) {
                    int i = particle_order[idx];
                    // not the same model
                    int model_idx = model_local_idx[i];
                    int J_idx = J_local_idx[i];
                    if (model_idx < 0)
                        continue;
                    auto& J = (*J_pointer)[J_idx];
                    auto& model = (*model_pointer)[model_idx];
                    auto& Jn_local = Jn[J_idx];
                    typename TCONST::Scratch s;
                    T ddJ;
                    if (opt < 2) {
                        model.updateScratch(J, s);
                        model.secondDerivative(s, ddJ);
                        if (opt == 1)
                            dPdJ[i] = ddJ;
                    }
                    else if (opt == 2) {
                        ddJ = dPdJ[i];
                    }
                    func(i, Hessian::Zero(), TM::Zero(), ddJ, Jn_local, true);
                }
            });
    }

    inline static AttributeName<TCONST> constitutive_model_name()
    {
        return AttributeName<TCONST>(TCONST::name());
    }
    inline static AttributeName<Scratch> constitutive_model_scratch_name()
    {
        return AttributeName<Scratch>(TCONST::scratch_name());
    }
};
} // namespace ZIRAN
#endif
