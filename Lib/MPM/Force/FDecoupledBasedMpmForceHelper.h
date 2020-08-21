#ifndef F_DECOUPLED_BASED_MPM_FORCE_HELPER_H
#define F_DECOUPLED_BASED_MPM_FORCE_HELPER_H

#include <unsupported/Eigen/MatrixFunctions>
#include <Ziran/CS/DataStructure/DisjointRanges.h>
#include <Ziran/CS/DataStructure/HashTable.h>
#include <Ziran/CS/DataStructure/SpatialHash.h>
#include <Ziran/CS/Util/Forward.h>
#include <MPM/Forward/MpmForward.h>
#include "FBasedMpmForceHelper.h"

#include <math.h>

namespace ZIRAN {

template <class T, int dim>
class MpmForceHelperBase;

template <class TCONST>
class FDecoupledBasedMpmForceHelper : public FBasedMpmForceHelper<TCONST> {
public:
    static const int dim = TCONST::TM::RowsAtCompileTime;
    using T = typename TCONST::Scalar;
    using Scratch = typename TCONST::Scratch;
    using TM = Matrix<T, dim, dim>;
    using TV = Vector<T, dim>;
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;
    using Base = FBasedMpmForceHelper<TCONST>;

    using Base::constitutive_model;
    using Base::element_measure;
    using Base::F;
    using Base::Fn;
    using Base::particles;
    using Base::scratch;

    using Base::constitutive_model_name;
    using Base::constitutive_model_scratch_name;
    using Base::element_measure_name;
    using Base::F_name;

    T rho0;
    T neighbor_search_h;
    SpatialHash<T, dim> hash;
    size_t strain_evolution_count;
    StdVector<T> shepard;

    explicit FDecoupledBasedMpmForceHelper(Particles<T, dim>& particles)
        : Base(particles)
        , rho0(0)
        , neighbor_search_h(0)
        , strain_evolution_count(0)
    {
    }

    virtual ~FDecoupledBasedMpmForceHelper() {}

    void rebuildShepard()
    {
        T h2 = MATH_TOOLS::sqr(neighbor_search_h);
        T scaling = 315 / (64 * 3.1415 * std::pow(neighbor_search_h, 9));

        shepard.resize(particles.count);

        auto& F_dilational_array = particles.get(F_Dilational_name<T>());

        auto ranges = particles.X.ranges;
        tbb::parallel_for(ranges,
            [&](DisjointRanges& subrange) {
                DisjointRanges subset(subrange,
                    particles.commonRanges(F_Dilational_name<T>()));
                for (auto iter = particles.subsetIter(subset, F_Dilational_name<T>()); iter; ++iter) {
                    int p = iter.entryId();
                    const TV& Xp = particles.X[p];
                    shepard[p] = 0;

                    StdVector<int> neighbors;
                    hash.oneLayerNeighbors(Xp, neighbors);
                    for (auto q : neighbors) {
                        const TV& Xq = particles.X[q];
                        T r2 = (Xp - Xq).squaredNorm();
                        if (r2 < h2) {
                            T w = scaling * std::pow(h2 - r2, 3);
                            shepard[p] += F_dilational_array[q] * element_measure[q] * w;
                        }
                    }
                }
            });
    }

    void reinitialize() override
    {
        Base::reinitialize();

        hash.rebuild(neighbor_search_h, particles.X.array);

        auto ranges = particles.X.ranges;
        tbb::parallel_for(ranges,
            [&](DisjointRanges& subrange) {
                DisjointRanges subset(subrange,
                    particles.commonRanges(F_name(),
                        F_Distortional_name<TM>(),
                        F_Dilational_name<T>()));
                for (auto iter = particles.subsetIter(subset, F_name(), F_Distortional_name<TM>(), F_Dilational_name<T>()); iter; ++iter) {
                    const TM& F = iter.template get<0>();
                    TM& F_Distortional = iter.template get<1>();
                    T& F_Dilational = iter.template get<2>();
                    F_Dilational = F.determinant();
                    F_Distortional = std::pow(F_Dilational, -(T)1 / dim) * F;
                }
            });

        rebuildShepard();

        if (!rho0)
            evaluateInitialDensity();
    }

    void backupStrain() override
    {
        ZIRAN_ASSERT(false, "FDecoupledBasedMpmForceHelper::backupStrain() not implemented");
    }

    void restoreStrain() override
    {
        ZIRAN_ASSERT(false, "FDecoupledBasedMpmForceHelper::restoreStrain() not implemented");
    }

    // add stress to vtau (called only by symplectic)
    void updateState(const DisjointRanges& subrange, StdVector<TM>& vtau, TVStack& fp) override
    {
        Base::updateState(subrange, vtau, fp);
    }

    // add stress to vPFnT (called only by implicit)
    void updateImplicitState(const DisjointRanges& subrange, StdVector<TM>& vPFnT, TVStack& fp) override
    {
        ZIRAN_ASSERT(false, "FDecoupledBasedMpmForceHelper::updateImplicitState() not implemented");
    }

    void evolveStrain(const DisjointRanges& subrange, T dt, const StdVector<TM>& gradV) override
    {
        ZIRAN_ASSERT(std::abs(neighbor_search_h) > 1e-10, "Neighbor search h is close to 0");
        T h2 = MATH_TOOLS::sqr(neighbor_search_h);
        T scaling = 315 / (64 * 3.1415 * std::pow(neighbor_search_h, 9));

        DisjointRanges subset(subrange,
            particles.commonRanges(F_Distortional_name<TM>(),
                F_Dilational_name<T>(),
                F_name()));
        for (auto iter = particles.subsetIter(subset, F_Distortional_name<TM>(), F_Dilational_name<T>(), F_name()); iter; ++iter) {
            int p = iter.entryId();

            TM& F_Distortional = iter.template get<0>();
            TM Ldev = gradV[p] - (T)1 / (T)dim * gradV[p].trace() * TM::Identity(); // get the deviatoric component of grad v
            TM dtLdev = ((T)dt) * Ldev;
            F_Distortional = dtLdev.exp() * F_Distortional;
            assert(gradV[p] == gradV[p]);

            T& F_Dilational = iter.template get<1>();

            if (strain_evolution_count % 1 == 0) {

                const TV& Xp = particles.X[p];
                T rhop = 0;

                StdVector<int> neighbors;
                hash.oneLayerNeighbors(Xp, neighbors);
                for (auto q : neighbors) {
                    const TV& Xq = particles.X[q];
                    T r2 = (Xp - Xq).squaredNorm();
                    if (r2 < h2) {
                        T w = scaling * std::pow(h2 - r2, 3);
                        rhop += particles.mass[q] * w / shepard[p];
                    }
                }

                //            for (int q = 0; q < particles.count; q++) {
                //                const TV& Xq = particles.X[q];
                //                T r2 = (Xp - Xq).squaredNorm();
                //                if (r2 < h2) {
                //                    T w = scaling * std::pow(h2 - r2, 3);
                //                    rhop += particles.mass[q] * w;
                //                }
                //            }

                F_Dilational = rho0 / rhop;
            }
            else
                F_Dilational = (1 + gradV[p].trace() * dt) * F_Dilational;

            TM& F = iter.template get<2>();
            F = std::pow(F_Dilational, (T)1 / dim) * F_Distortional;
        }

        strain_evolution_count++;
    }

    void evaluateInitialDensity()
    {
        StdVector<T> density(particles.count, 0);

        T h2 = MATH_TOOLS::sqr(neighbor_search_h);
        T scaling = 315 / (64 * 3.1415 * std::pow(neighbor_search_h, 9));
        auto ranges = particles.X.ranges;
        tbb::parallel_for(ranges,
            [&](DisjointRanges& subrange) {
                DisjointRanges subset(subrange,
                    particles.commonRanges(element_measure_name(),
                        F_name()));
                for (auto iter = particles.subsetIter(subset, F_name()); iter; ++iter) {
                    int p = iter.entryId();

                    const TV& Xp = particles.X[p];
                    T rhop = 0;
                    for (int q = 0; q < particles.count; q++) {
                        const TV& Xq = particles.X[q];
                        T r2 = (Xp - Xq).squaredNorm();
                        if (r2 < h2) {
                            T w = scaling * std::pow(h2 - r2, 3);
                            rhop += particles.mass[q] * w / shepard[p];
                        }
                    }
                    density[p] = rhop;
                }
            });

        // compute rho0 using max rho
        T max_rho = 0;
        for (size_t i = 0; i < density.size(); i++)
            if (density[i] > max_rho)
                max_rho = density[i];
        rho0 = max_rho;

        //        // compute rho0 using mean rho
        //                T sum = std::accumulate(density.begin(), density.end(), 0);
        //                rho0 = sum / particles.count;
    }

    void setNeighborSearchH(const T h)
    {
        neighbor_search_h = h;
    }

    double totalEnergy(const DisjointRanges& subrange) override
    {
        ZIRAN_ASSERT(false, "FDecoupledBasedMpmForceHelper::totalEnergy() not implemented");
    }

    void computeStressDifferential(const DisjointRanges& subrange, const StdVector<TM>& gradDv, StdVector<TM>& dstress, const TVStack& dvp, TVStack& dfp) override
    {
        ZIRAN_ASSERT(false, "FDecoupledBasedMpmForceHelper::computeStressDifferential() not implemented");
    }
};
} // namespace ZIRAN
#endif /* ifndef F_BASED_MPM_FORCE_HELPER */
