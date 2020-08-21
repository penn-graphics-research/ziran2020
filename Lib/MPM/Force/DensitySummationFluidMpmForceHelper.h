#ifndef DENSITY_SUMMATION_FLUID_MPM_FORCE_HELPER_H
#define DENSITY_SUMMATION_FLUID_MPM_FORCE_HELPER_H

#include <unsupported/Eigen/MatrixFunctions>
#include <Ziran/CS/Util/Forward.h>
#include <Ziran/CS/DataStructure/HashTable.h>
#include <Ziran/CS/DataStructure/DataManager.h>
#include <Ziran/Math/Geometry/SimplexMesh.h>
#include <Ziran/Math/Geometry/Particles.h>
#include <Ziran/Physics/ConstitutiveModel/ConstitutiveModel.h>
#include <Ziran/CS/Util/AttributeNamesForward.h>
#include <tbb/tbb.h>
#include <tick/requires.h>
#include <algorithm>
#include <MPM/Forward/MpmForward.h>
#include <MPM/Force/MpmForceBase.h>

namespace ZIRAN {

template <class T, int dim>
class MpmForceHelperBase;

template <class T, int dim>
class DensitySummationFluidMpmForceHelper : public MpmForceHelperBase<T, dim> {
public:
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;

    T kappa;

    Particles<T, dim>& particles;

    DataArray<TM>& F_Distortional; // F-isochoric (distortional)

    T h; // SPH kernel h
    T kernel_normalization; // SPH kernal normalization scaling factor
    T rho0;

    SpatialHash<T, dim> hash;

    explicit DensitySummationFluidMpmForceHelper(Particles<T, dim>& particles)
        : particles(particles)
        , F_Distortional(particles.add(F_Distortional_name<TM>()))
        , h(-1)
        , kernel_normalization(-1)
        , rho0(0)
    {
        kappa = 2e6;
    }
    virtual ~DensitySummationFluidMpmForceHelper() {}

    void preheat(const T h_in) // called when adding force helper
    {
        h = h_in;

        // kernel normalization factors for cubic SPH kernel
        if constexpr (dim == 1)
            kernel_normalization = (T)1 / (h);
        else if constexpr (dim == 2)
            kernel_normalization = (T)15 / (7 * M_PI * h * h);
        else if constexpr (dim == 3)
            kernel_normalization = (T)3 / (2 * M_PI * h * h * h);
        else
            ZIRAN_ERR("wrong dimension");
    }

    void reinitialize() override
    {
        hash.rebuild(2 * h, particles.X.array);
        evalDensity();
        if (!rho0) { // this will only be called at the first time step
            for (auto iter = particles.iter(density_name<T>()); iter; ++iter) {
                auto& rho = iter.template get<0>();
                rho0 = std::max(rho, rho0);
            }
        }
    }

    void backupStrain() override
    {
        ZIRAN_ERR("Implicit force not implemented.");
    }

    void restoreStrain() override
    {
        ZIRAN_ERR("Implicit force not implemented.");
    }

    void updateState(const DisjointRanges& subrange, StdVector<TM>& vtau, TVStack& fp) override
    {
        T one_over_h = 1 / h;

        DisjointRanges subset1(subrange, particles.commonRanges(density_name<T>()));
        for (auto iter = particles.subsetIter(subset1, density_name<T>()); iter; ++iter) {
            int p = iter.entryId();
            const auto& density_p = iter.template get<0>();
            T pressure = kappa / 2 * (density_p / rho0 - rho0 / density_p);
            if (pressure < 0) pressure = 0;
            vtau[p] -= particles.mass[p] / density_p * pressure * TM::Identity();
        }
    }

    void updateImplicitState(const DisjointRanges& subrange, StdVector<TM>& vPFnT, TVStack& fp) override
    {
        ZIRAN_ERR("Implicit force not implemented.");
    }

    double totalEnergy(const DisjointRanges& subrange) override
    {
        ZIRAN_ERR("total energy not implemented.");
        return 0;
    }

    void computeStressDifferential(const DisjointRanges& subrange, const StdVector<TM>& gradDv, StdVector<TM>& dstress, const TVStack& dvp, TVStack& dfp) override
    {
        ZIRAN_ERR("Implicit force not implemented.");
    }

private:
    T W(T R)
    {
        ZIRAN_ASSERT(R >= 0);

        // cubic spline
        if (R < 1)
            return (R * R * R / 2 - R * R + 2 / (T)3) * kernel_normalization;
        else if (R < 2)
            return MATH_TOOLS::cube(2 - R) / (T)6 * kernel_normalization;
        else
            return 0;
    }

    void evalDensity()
    {
        tbb::parallel_for(particles.X.ranges, [&](DisjointRanges& subrange) {
            DisjointRanges subset(subrange, particles.commonRanges(density_name<T>()));
            for (auto iter = particles.subsetIter(subset, density_name<T>()); iter; ++iter) {
                auto& density_p = iter.template get<0>();
                int p = iter.entryId();
                const TV& Xp = particles.X[p];
                StdVector<int> neighbors;
                hash.oneLayerNeighbors(Xp, neighbors);
                T rhop = 0;
                for (auto q : neighbors) {
                    const TV& Xq = particles.X[q];
                    T R = (Xp - Xq).norm() / h;
                    rhop += particles.mass[q] * W(R);
                }
                density_p = rhop;
            }
        });
    }
};
} // namespace ZIRAN
#endif
