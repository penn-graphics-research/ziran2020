#ifndef LINEAR_COROTATED_MPM_FORCE_HELPER_H
#define LINEAR_COROTATED_MPM_FORCE_HELPER_H

#include "MpmForceHelperBase.h"

#include <Ziran/CS/Util/Forward.h>
#include <Ziran/CS/DataStructure/DataArray.h>
#include <Ziran/CS/DataStructure/DataManager.h>

namespace ZIRAN {

template <class T, int dim>
class MpmForceHelperBase;

template <class T, int dim>
class LinearCorotated;

template <class T, int dim>
class LinearCorotatedMpmForceHelper : public MpmForceHelperBase<typename LinearCorotated<T, dim>::Scalar, LinearCorotated<T, dim>::TM::RowsAtCompileTime> {
public:
    using Scratch = typename LinearCorotated<T, dim>::Scratch;
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;

    Particles<T, dim>& particles;
    DataArray<T>& element_measure;
    DataArray<LinearCorotated<T, dim>>& constitutive_model;
    DataArray<Scratch>& scratch;
    DataArray<TM>& F;

    StdVector<TM> Fn;

    explicit LinearCorotatedMpmForceHelper(Particles<T, dim>& particles)
        : particles(particles)
        , element_measure(particles.add(element_measure_name()))
        , constitutive_model(particles.add(constitutive_model_name()))
        , scratch(particles.add(constitutive_model_scratch_name()))
        , F(particles.add(F_name()))
    {
    }
    virtual ~LinearCorotatedMpmForceHelper() {}

    void reinitialize() override;

    void refreshRotations();

    void backupStrain() override;

    void restoreStrain() override;

    // add stress to vtau (called only by symplectic)
    void updateState(const DisjointRanges& subrange, StdVector<TM>& vtau, TVStack& fp) override;

    // add stress to vPFnT (called only by implicit)
    void updateImplicitState(const DisjointRanges& subrange, StdVector<TM>& vPFnT, TVStack& fp) override;

    void evolveStrain(const DisjointRanges& subrange, T dt, const StdVector<TM>& gradV) override;

    double totalEnergy(const DisjointRanges& subrange) override;

    void computeStressDifferential(const DisjointRanges& subrange, const StdVector<TM>& gradDv, StdVector<TM>& dstress, const TVStack& dvp, TVStack& dfp) override;

    inline static AttributeName<LinearCorotated<T, dim>> constitutive_model_name()
    {
        return AttributeName<LinearCorotated<T, dim>>(LinearCorotated<T, dim>::name());
    }
    inline static AttributeName<Scratch> constitutive_model_scratch_name()
    {
        return AttributeName<Scratch>(LinearCorotated<T, dim>::scratch_name());
    }
    inline static AttributeName<T> element_measure_name()
    {
        return AttributeName<T>("element measure");
    }
    inline static AttributeName<TM> F_name()
    {
        return AttributeName<TM>("F");
    }
};
} // namespace ZIRAN
#endif /* ifndef F_BASED_MPM_FORCE_HELPER */
