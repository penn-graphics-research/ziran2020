#ifndef IMPLICIT_VISCOSITY_HELPER_H
#define IMPLICIT_VISCOSITY_HELPER_H

#include <Ziran/Math/Geometry/Particles.h>
#include <MPM/Force/MpmForceBase.h>

namespace ZIRAN {

template <class T, int dim>
class ImplicitViscosityHelper : public MpmForceHelperBase<T, dim> {
public:
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;

    Particles<T, dim>& particles; // contains Xn and Vn
    T dt;

    StdVector<TM> cauchy_strain;

    DataArray<T>& viscosity;
    DataArray<T>& element_measure;

    ImplicitViscosityHelper(Particles<T, dim>& particles)
        : particles(particles)
        , dt(NAN)
        , viscosity(particles.get(viscosity_name())) // Make sure add this attribute to particles in initialization
        , element_measure(particles.get(element_measure_name()))
    {
    }

    virtual ~ImplicitViscosityHelper() {}

    void reinitialize() override
    {
        cauchy_strain.resize(viscosity.size());
    }

    // add stress to vtau (called only by symplectic)
    void updateState(const DisjointRanges& subrange, StdVector<TM>& vtau, TVStack& fp) override
    {
        ZIRAN_ASSERT(false, "This helper only works for implicit.");
    }

    // add stress to vPFnT (called only by implicit)
    void updateImplicitState(const DisjointRanges& subrange, StdVector<TM>& vPFnT, TVStack& fp) override
    {
        DisjointRanges subset(subrange,
            particles.commonRanges(viscosity_name(),
                element_measure_name(), J_name()));
        if (subset.size() == 0)
            return;
        for (auto iter = particles.subsetIter(subset, viscosity_name(), element_measure_name(), J_name(), valueIdOnly(viscosity_name())); iter; ++iter) {
            auto& mu = iter.template get<0>();
            const auto& element_measure = iter.template get<1>();
            const auto& J = iter.template get<2>();
            int mu_id = iter.template get<3>();
            T Vn = element_measure * J;
            int p = iter.entryId();
            TM& epsilon = cauchy_strain[mu_id];
            assert(dt == dt);
            vPFnT[p] += (2 * Vn * mu / dt) * epsilon;
        }
    }

    void evolveStrain(const DisjointRanges& subrange, T dt_in, const StdVector<TM>& gradV) override
    {
        dt = dt_in;
        DisjointRanges subset(subrange,
            particles.commonRanges(viscosity_name()));
        for (auto iter = particles.subsetIter(subset, valueIdOnly(viscosity_name())); iter; ++iter) {
            int mu_id = iter.template get<0>();
            int p = iter.entryId();
            const TM& gV = gradV[p];
            cauchy_strain[mu_id] = (gV + gV.transpose()) * (T)0.5;
        }
    }

    double totalEnergy(const DisjointRanges& subrange) override
    {
        DisjointRanges subset(subrange,
            particles.commonRanges(viscosity_name(),
                element_measure_name(), J_name()));
        if (subset.size() == 0)
            return 0;
        double e = 0.0;
        for (auto iter = particles.subsetIter(subset, viscosity_name(), element_measure_name(), J_name(), valueIdOnly(viscosity_name())); iter; ++iter) {
            auto& mu = iter.template get<0>();
            const auto& element_measure = iter.template get<1>();
            const auto& J = iter.template get<2>();
            const int mu_id = iter.template get<3>();
            T Vn = element_measure * J;
            TM& epsilon = cauchy_strain[mu_id];
            assert(dt == dt);
            e += mu * Vn * epsilon.squaredNorm();
        }
        return e;
    }

    void computeStressDifferential(const DisjointRanges& subrange, const StdVector<TM>& gradDv, StdVector<TM>& dstress, const TVStack& dvp, TVStack& dfp) override
    {
        DisjointRanges subset(subrange,
            particles.commonRanges(viscosity_name(),
                element_measure_name(), J_name()));
        if (subset.size() == 0)
            return;
        for (auto iter = particles.subsetIter(subset, viscosity_name(), element_measure_name(), J_name()); iter; ++iter) {
            auto& mu = iter.template get<0>();
            const auto& element_measure = iter.template get<1>();
            const auto& J = iter.template get<2>();
            T Vn = element_measure * J;
            int p = iter.entryId();
            const TM& gdv = gradDv[p];
            TM depsilon = (gdv + gdv.transpose()) * (T)0.5;
            assert(dt == dt);
            dstress[p] += (2 * Vn * mu) / (dt * dt) * depsilon;
        }
    }

    inline static AttributeName<T> viscosity_name()
    {
        return AttributeName<T>("viscosity");
    }

    inline static AttributeName<T> element_measure_name()
    {
        return AttributeName<T>("element measure");
    }

    // "F" is a possible strain measures for computing J
    inline static AttributeName<TM> F_name()
    {
        return AttributeName<TM>("F");
    }

    // "cotangent" is a possible strain measure for computing J
    inline static AttributeName<TM> cotangent_name()
    {
        return AttributeName<TM>("cotangent");
    }

    // "J" is a possible strain measure for computing J
    inline static AttributeName<T> J_name()
    {
        return AttributeName<T>("J");
    }
};
} // namespace ZIRAN

#endif
