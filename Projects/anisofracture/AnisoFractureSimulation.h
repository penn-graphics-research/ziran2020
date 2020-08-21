#ifndef ANISO_FRACTURE_SIMULATION_H
#define ANISO_FRACTURE_SIMULATION_H

#include <MPM/MpmSimulationBase.h>
#include <Ziran/Math/MathTools.h>
#include <Ziran/Physics/ConstitutiveModel/NeoHookeanBorden.h>
#include <Partio.h>
#include "PhaseFieldSystem.h"
#include "ConjugateGradient.h"
#include <Ziran/Math/Linear/ImplicitQRSVD.h>
#include "AnisoVisualizer.h"

#include "AnisotropicPhaseField.h"

#undef B2

namespace ZIRAN {

template <class T, int dim>
class AnisoFractureSimulation : public MpmSimulationBase<T, dim> {
public:
    using Base = MpmSimulationBase<T, dim>;
    using Base::autorestart;
    using Base::begin_time_step_callbacks;
    using Base::block_offset;
    using Base::cfl;
    using Base::dt;
    using Base::dx;
    using Base::flip_pic_ratio;
    using Base::force;
    using Base::frame;
    using Base::grid;
    using Base::mls_mpm;
    using Base::num_nodes;
    using Base::particle_base_offset;
    using Base::particle_group;
    using Base::particle_order;
    using Base::particles;
    using Base::restart_callbacks;
    using Base::scratch_gradV;
    using Base::start_frame;
    using Base::step;
    using Base::symplectic;
    using Base::transfer_scheme;

    typedef Vector<T, dim> TV;
    typedef Vector<int, dim> IV;
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, Eigen::Dynamic> Vec;
    typedef Matrix<T, dim, Eigen::Dynamic> Mat;

    bool use_phase_field = false;
    bool useNaiveDamage = false; //turn on so we can use the naive damage and compare with PFF!
    T sigmaF = 0;
    T parabolic_M = 0;
    T delete_particle_threshold = 0;
    bool lumping = true;

    bool implicit = false; //default to false

    //Unique to Anisotropic Fracture

    using Simulation = AnisoFractureSimulation<T, dim>;
    using Objective = PhaseFieldSystem<Simulation>;

    Objective cg_objective;
    ConjugateGradient<T, Objective, Vec> cg;

    AnisoFractureSimulation(const bool implicitDamage)
        : implicit(implicitDamage), cg_objective(*this), cg(100)
    {
    }

    void writeState(std::ostream& out)
    {
        Base::writeState(out);
        std::string filename = SimulationBase::output_dir.absolutePath(SimulationBase::outputFileName("anisoFrax", ".bgeo"));

        // gather a_0 and F as flat arrays
        StdVector<StdVector<TV>> aData;
        StdVector<TM> FData;
        std::vector<T> dData, laplacians;
        for (auto iter = particles.iter(phase_field_range()); iter; ++iter) {
            aData.push_back(iter.template get<0>().a_0); //only grab first structural director
            dData.push_back(iter.template get<0>().d);
            laplacians.push_back(iter.template get<0>().laplacian);
        }
        for (auto iter = particles.iter(F_range()); iter; ++iter) {
            FData.push_back(iter.template get<0>());
        }

        //Compute updated directors as R*a_0
        Matrix<T, dim, dim> F, U, V, R;
        Vector<T, dim> a_0, sigma;
        std::vector<Vector<T, dim>> currAData;
        for (int i = 0; i < particles.count; i++) {
            a_0 = aData[i][0]; //grab first structural director only
            F = FData[i];
            singularValueDecomposition(F, U, sigma, V);
            R = U * V.transpose();

            currAData.push_back(R * a_0);
        }

        //currAData.resize(particles.count, TV::Zero());
        aniso_visualize_particles_vec(*this, currAData, dData, laplacians, filename);
    }

    inline static AttributeName<AnisotropicPhaseField<T, dim>> phase_field_range()
    {
        return AttributeName<AnisotropicPhaseField<T, dim>>("phase field");
    }

    inline static AttributeName<T> element_measure_range()
    {
        return AttributeName<T>("element measure");
    }

    inline static AttributeName<TM> F_range()
    {
        return AttributeName<TM>("F");
    }

    virtual void initialize()
    {
        Base::initialize();
    }

    virtual void particlesToGrid()
    {
        ZIRAN_TIMER();
        Base::particlesToGrid();

        if (!implicit) {
            computeLaplacians();
            updateDamage();
        }
        else {
            solveDamageSystem(); //this does damageP2G, grid damage solve, then damageG2P with updated model g value
        }
    }

    virtual void reinitialize()
    {
        //deleteParticles();

        Base::reinitialize();
    }

    void computeLaplacians()
    {

        ZIRAN_TIMER();

        auto& Xarray = particles.X.array;
        auto& Varray = particles.V.array;
        auto& marray = particles.mass.array;
        auto* pf_pointer = &particles.DataManager::get(phase_field_range());

        //Transfer damage to the grid
        for (uint64_t color = 0; color < (1 << dim); ++color) {
            tbb::parallel_for(0, (int)particle_group.size(), [&](int group_idx) {
                if ((block_offset[group_idx] & ((1 << dim) - 1)) != color)
                    return;
                for (int idx = particle_group[group_idx].first; idx <= particle_group[group_idx].second; ++idx) {
                    int i = particle_order[idx];
                    TV& Xp = Xarray[i];
                    BSplineWeights<T, dim> spline(Xp, dx);
                    grid.iterateKernel(spline, particle_base_offset[i],
                        [&](const IV& node, T w, const TV& dw, GridState<T, dim>& g) {
                            g.phase_field_multiplier += w;
                            g.phase_field += (*pf_pointer)[i].d * w;
                        });
                }
            });
        }
        grid.iterateGrid([&](IV node, GridState<T, dim>& g) {
            g.phase_field /= g.phase_field_multiplier;
        });

        //Compute and store damage laplacians
        tbb::parallel_for(0, (int)particle_group.size(), [&](int group_idx) {
            for (int idx = particle_group[group_idx].first; idx <= particle_group[group_idx].second; ++idx) {
                int i = particle_order[idx];

                TV& Xp = Xarray[i];
                T d_laplacian = (T)0;
                BSplineWeightsWithDDW<T, dim> spline(Xp, dx);
                grid.iterateKernelWithLaplacian(spline, particle_base_offset[i], [&](IV node, T w, TV dw, T laplacian, GridState<T, dim>& g) {
                    if (g.idx >= 0)
                        d_laplacian += g.phase_field * laplacian; //grab laplacians and use them to get the damage laplacian
                });

                // Assign this to be the laplacian of the particle
                (*pf_pointer)[i].laplacian = d_laplacian;
            }
        });

        //Update volume based on F
        tbb::parallel_for(particles.X.ranges, [&](DisjointRanges& subrange) {
            DisjointRanges subset(subrange, particles.commonRanges(phase_field_range(), element_measure_range(), F_range()));
            for (auto iter = particles.subsetIter(subset, phase_field_range(), element_measure_range(), F_range()); iter; ++iter) {
                auto& phase_field = iter.template get<0>();
                auto& vol0 = iter.template get<1>();
                auto& F = iter.template get<2>();
                phase_field.vol = vol0 * F.determinant();
            }
        });
    }
    template <class TName, class ScratchType>
    void updateDamageHelper(TName model_name)
    {
        auto ranges = particles.X.ranges;

        if (!particles.exist(model_name)) return;

        //Iterate over all particles and update their damages if necessary!
        tbb::parallel_for(ranges, [&](DisjointRanges& subrange) {
            DisjointRanges subset(subrange, particles.commonRanges(phase_field_range(), model_name, F_range()));
            for (auto iter = particles.subsetIter(subset, phase_field_range(), model_name, F_range()); iter; ++iter) {
                auto& pf = iter.template get<0>();
                auto& model = iter.template get<1>();
                auto& F = iter.template get<2>();

                //Compute geometric resistance using damage Laplacian
                T geometricResist = pf.d - (pf.l0 * pf.l0 * pf.laplacian);

                //Get undegraded Cauchy stress
                // T J = F.determinant();
                // TM tau;
                // ScratchType scratch_dummy;
                // scratch_dummy.F = F;
                // scratch_dummy.J = J;
                // model.g = 1; //set model.g to 1 so we compute based on the undegraded principal stress
                // model.kirchhoff(scratch_dummy, tau); //get tau (kirchoff stress)
                // TM cauchy = ((T)1 / J) * tau;

                //Get undegraded Cauchy stress
                auto model_dummy = model;
                TM cauchy;
                model_dummy.evaluateEffectiveCauchyStress(F, cauchy);

                //Now that we have cauchy stress take eigen value decomposition of it
                Eigen::EigenSolver<TM> es(cauchy);

                //Construct sigmaPlus, the tension portion of the cauchy stress
                TM sigmaPlus = TM::Zero(dim, dim);
                TV eigenVecs;
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++) {
                        eigenVecs(j) = es.eigenvectors().col(i)(j).real(); //get the real parts of each eigenvector
                    }
                    sigmaPlus += MATH_TOOLS::macaulay(es.eigenvalues()(i).real()) * (eigenVecs * eigenVecs.transpose());
                }

                //Construct structural tensor, A, using helper function
                TM A = constructStructuralTensor(pf.a_0, pf.alphas, F);

                //Compute phi
                T contraction = 0;
                TM Asig = A * sigmaPlus;
                TM sigA = sigmaPlus * A;
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++) {
                        contraction += (Asig(i, j) * sigA(i, j));
                    }
                }
                T phi = ((T)1 / (pf.sigma_crit * pf.sigma_crit)) * contraction;

                //Compute D_tilde
                T dTilde = pf.zeta * MATH_TOOLS::macaulay(phi - 1);

                //Update damage if necessary
                T diff = ((1 - pf.d) * dTilde) - geometricResist;
                T newD = pf.d + ((dt / pf.eta) * MATH_TOOLS::macaulay(diff)); //macaulay ensures that we only update when we should (if diff <= 0 expression returns 0)
                pf.d = std::min(newD, (T)1); //update damage
                T k = pf.residual_phase;

                //Update constitutive model g
                model.g = ((1 - pf.d) * (1 - pf.d) * (1 - k)) + k; //use monotonic deg function from cdmpm
            }
        });
    }

    template <class TName, class ScratchType>
    void updateDTildeHelper(TName model_name)
    {
        auto ranges = particles.X.ranges;

        if (!particles.exist(model_name)) return;

        //Iterate over all particles and update their damages if necessary!
        tbb::parallel_for(ranges, [&](DisjointRanges& subrange) {
            DisjointRanges subset(subrange, particles.commonRanges(phase_field_range(), model_name, F_range()));
            for (auto iter = particles.subsetIter(subset, phase_field_range(), model_name, F_range()); iter; ++iter) {
                auto& pf = iter.template get<0>();
                auto& model = iter.template get<1>();
                auto& F = iter.template get<2>();

                //Get undegraded Cauchy stress
                auto model_dummy = model;
                TM cauchy;
                model_dummy.evaluateEffectiveCauchyStress(F, cauchy);

                //Now that we have cauchy stress take eigen value decomposition of it
                Eigen::EigenSolver<TM> es(cauchy);

                //Construct sigmaPlus, the tension portion of the cauchy stress
                TM sigmaPlus = TM::Zero(dim, dim);
                TV eigenVecs;
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++) {
                        eigenVecs(j) = es.eigenvectors().col(i)(j).real(); //get the real parts of each eigenvector
                    }
                    sigmaPlus += MATH_TOOLS::macaulay(es.eigenvalues()(i).real()) * (eigenVecs * eigenVecs.transpose());
                }

                //Construct structural tensor, A, using helper function
                TM A = constructStructuralTensor(pf.a_0, pf.alphas, F);

                //Compute phi
                T contraction = 0;
                TM Asig = A * sigmaPlus;
                TM sigA = sigmaPlus * A;
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++) {
                        contraction += (Asig(i, j) * sigA(i, j));
                    }
                }
                T phi = ((T)1 / (pf.sigma_crit * pf.sigma_crit)) * contraction;

                //Compute D_tilde
                T dTilde = pf.zeta * MATH_TOOLS::macaulay(phi - 1);

                //Set particle dTilde to be the max of it's history and this new dtilde! (never decreasing)
                T dTilde_history = pf.maxDTilde;
                pf.maxDTilde = std::max(dTilde, dTilde_history);
            }
        });
    }

    //template <class TCONST>
    void updateDamage()
    {
        ZIRAN_TIMER();
        AttributeName<NeoHookeanBorden<T, dim>> name1(NeoHookeanBorden<T, dim>::name());
        updateDamageHelper<AttributeName<NeoHookeanBorden<T, dim>>, NeoHookeanBordenScratch<T, dim>>(name1);
        AttributeName<QRAnisotropic<T, dim>> name2(QRAnisotropic<T, dim>::name());
        updateDamageHelper<AttributeName<QRAnisotropic<T, dim>>, QRAnisotropicScratch<T, dim>>(name2);
        AttributeName<QRStableNeoHookean<T, dim>> name3(QRStableNeoHookean<T, dim>::name());
        updateDamageHelper<AttributeName<QRStableNeoHookean<T, dim>>, QRStableNeoHookeanScratch<T, dim>>(name3);
    }

    void updateDTilde()
    {
        ZIRAN_TIMER();
        AttributeName<NeoHookeanBorden<T, dim>> name1(NeoHookeanBorden<T, dim>::name());
        updateDTildeHelper<AttributeName<NeoHookeanBorden<T, dim>>, NeoHookeanBordenScratch<T, dim>>(name1);
        AttributeName<QRAnisotropic<T, dim>> name2(QRAnisotropic<T, dim>::name());
        updateDTildeHelper<AttributeName<QRAnisotropic<T, dim>>, QRAnisotropicScratch<T, dim>>(name2);
        AttributeName<QRStableNeoHookean<T, dim>> name3(QRStableNeoHookean<T, dim>::name());
        updateDTildeHelper<AttributeName<QRStableNeoHookean<T, dim>>, QRStableNeoHookeanScratch<T, dim>>(name3);
    }

    TM constructStructuralTensor(const StdVector<TV>& a_0, const StdVector<T>& alphas, const TM& F)
    {

        //Compute rotation R from decomposing F
        Matrix<T, dim, dim> U, V, R;
        Vector<T, dim> sigma;
        singularValueDecomposition(F, U, sigma, V);
        R = U * V.transpose();

        TM A = TM::Identity();

        ZIRAN_ASSERT(a_0.size() <= 2 && a_0.size() > 0, "ERROR: need to pass 1 or 2 structural directors, you passed: ", a_0.size());

        //Now compute structural tensor based on whether we have one or two structural directors
        if (a_0.size() == 1) {
            //one structural director means transvere isotropy
            TV a1 = R * a_0[0];
            A += alphas[0] * (a1 * a1.transpose());
            return A;
        }

        //two structural directors means orthotropy
        TV a1 = R * a_0[0];
        TV a2 = R * a_0[1];
        A += alphas[0] * (a1 * a1.transpose()) + alphas[1] * (a2 * a2.transpose());
        return A;
    }

    void damageP2G()
    {
        ZIRAN_TIMER();

        auto& Xarray = particles.X.array;
        auto& Varray = particles.V.array;
        auto& marray = particles.mass.array;
        auto* pf_pointer = &particles.DataManager::get(phase_field_range());

        //Transfer damage to the grid
        for (uint64_t color = 0; color < (1 << dim); ++color) {
            tbb::parallel_for(0, (int)particle_group.size(), [&](int group_idx) {
                if ((block_offset[group_idx] & ((1 << dim) - 1)) != color)
                    return;
                for (int idx = particle_group[group_idx].first; idx <= particle_group[group_idx].second; ++idx) {
                    int i = particle_order[idx];
                    TV& Xp = Xarray[i];
                    BSplineWeights<T, dim> spline(Xp, dx);
                    grid.iterateKernel(spline, particle_base_offset[i],
                        [&](const IV& node, T w, const TV& dw, GridState<T, dim>& g) {
                            g.phase_field_multiplier += w;
                            g.phase_field += (*pf_pointer)[i].d * w;
                        });
                }
            });
        }
        grid.iterateGrid([&](IV node, GridState<T, dim>& g) {
            g.phase_field /= g.phase_field_multiplier;
        });

        //Update volume based on F
        tbb::parallel_for(particles.X.ranges, [&](DisjointRanges& subrange) {
            DisjointRanges subset(subrange, particles.commonRanges(phase_field_range(), element_measure_range(), F_range()));
            for (auto iter = particles.subsetIter(subset, phase_field_range(), element_measure_range(), F_range()); iter; ++iter) {
                auto& phase_field = iter.template get<0>();
                auto& vol0 = iter.template get<1>();
                auto& F = iter.template get<2>();
                phase_field.vol = vol0 * F.determinant();
            }
        });
    }

    void solveDamageSystem()
    {
        ZIRAN_TIMER();

        auto& Xarray = particles.X.array;
        auto& Varray = particles.V.array;
        auto& marray = particles.mass.array;
        auto* pf_pointer = &particles.DataManager::get(phase_field_range());

        damageP2G(); //FLIP damage to the grid

        updateDTilde(); //update DTilde valus so we can construct our system to solve

        Vec x = Vec::Zero(num_nodes, 1);
        Vec rhs = Vec::Zero(num_nodes, 1);

        //build rhs
        for (uint64_t color = 0; color < (1 << dim); ++color) {
            tbb::parallel_for(0, (int)particle_group.size(), [&](int group_idx) {
                if ((block_offset[group_idx] & ((1 << dim) - 1)) != color)
                    return;
                for (int idx = particle_group[group_idx].first; idx <= particle_group[group_idx].second; ++idx) {
                    int i = particle_order[idx];
                    T vol = (*pf_pointer)[i].vol;
                    TV& Xp = particles.X[i];
                    T eta = (*pf_pointer)[i].eta;
                    T dTilde = (*pf_pointer)[i].maxDTilde;

                    BSplineWeights<T, dim> spline(Xp, dx);

                    grid.iterateKernel(spline, particle_base_offset[i], [&](IV node, T w, TV dw, GridState<T, dim>& g) {
                        int node_id = g.idx;
                        if (node_id < 0)
                            return;
                        rhs(node_id) += (((dt * dTilde) / eta) + g.phase_field) * vol * w; //different from PFF
                    });
                }
            });
        }

        cg_objective.setMultiplier([&](const Vec& x, Vec& b) {
            Vec c_scp = Vec::Zero(particles.count, 1);
            Mat gradc_scp = Mat::Zero(dim, particles.count);

            tbb::parallel_for(0, (int)particle_group.size(), [&](int group_idx) {
                for (int idx = particle_group[group_idx].first; idx <= particle_group[group_idx].second; ++idx) {
                    int i = particle_order[idx];
                    TV& Xp = particles.X[i];
                    T vol = (*pf_pointer)[i].vol;
                    T l0 = (*pf_pointer)[i].l0;
                    T eta = (*pf_pointer)[i].eta;

                    BSplineWeights<T, dim> spline(Xp, dx);

                    grid.iterateKernel(spline, particle_base_offset[i], [&](IV node, T w, TV dw, GridState<T, dim>& g) {
                        int node_id = g.idx;
                        if (node_id < 0)
                            return;
                        gradc_scp.col(i) += x(node_id) * dw; //same as PFF
                    });

                    gradc_scp.col(i) *= vol * (dt / eta) * l0 * l0; //different from PFF
                }
            });

            b.setZero();

            for (uint64_t color = 0; color < (1 << dim); ++color) {
                tbb::parallel_for(0, (int)particle_group.size(), [&](int group_idx) {
                    if ((block_offset[group_idx] & ((1 << dim) - 1)) != color)
                        return;
                    for (int idx = particle_group[group_idx].first; idx <= particle_group[group_idx].second; ++idx) {
                        int i = particle_order[idx];
                        T vol = (*pf_pointer)[i].vol;
                        TV& Xp = particles.X[i];
                        T eta = (*pf_pointer)[i].eta;
                        T dTilde = (*pf_pointer)[i].maxDTilde;

                        BSplineWeights<T, dim> spline(Xp, dx);

                        grid.iterateKernel(spline, particle_base_offset[i],
                            [&](IV node, T w, TV dw, GridState<T, dim>& g) {
                                int node_id = g.idx;
                                if (node_id < 0)
                                    return;
                                b(node_id) += vol * (1 + ((dt / eta) * (dTilde + 1))) * w * x(node_id); //diff from PFF
                                b(node_id) += gradc_scp.col(i).dot(dw); //diff from PFF
                            });
                    }
                });
            }
        });

        cg_objective.setPreconditioner([&](const Vec& in, Vec& out) {
            out.setZero();
            for (uint64_t color = 0; color < (1 << dim); ++color) {
                tbb::parallel_for(0, (int)particle_group.size(), [&](int group_idx) {
                    if ((block_offset[group_idx] & ((1 << dim) - 1)) != color)
                        return;
                    for (int idx = particle_group[group_idx].first; idx <= particle_group[group_idx].second; ++idx) {
                        int i = particle_order[idx];
                        T vol = (*pf_pointer)[i].vol;
                        TV& Xp = particles.X[i];
                        T eta = (*pf_pointer)[i].eta;
                        T dTilde = (*pf_pointer)[i].maxDTilde;

                        BSplineWeights<T, dim> spline(Xp, dx);

                        grid.iterateKernel(spline, particle_base_offset[i],
                            [&](IV node, T w, TV dw, GridState<T, dim>& g) {
                                int node_id = g.idx;
                                if (node_id < 0)
                                    return;
                                out(node_id) += vol * (1 + ((dt / eta) * (dTilde + 1))) * w; //diff from PFF
                            });
                    }
                });
            }
            for (int i = 0; i < num_nodes; ++i)
                out(i) = in(i) / out(i);
        });

        T tolerance = 1e-6;
        cg.setTolerance(tolerance);
        cg.solve(cg_objective, x, rhs, false);

        grid.iterateGrid([&](IV node, GridState<T, dim>& g) {
            g.phase_field = x(g.idx) - g.phase_field; //this encodes the DIFFERENCE between old and new!!
        });

        damageG2P(); //transfer to particle view to update particle phase
    }

    void damageG2P()
    {
        ZIRAN_TIMER();
        AttributeName<NeoHookeanBorden<T, dim>> name1(NeoHookeanBorden<T, dim>::name());
        damageG2PHelper<AttributeName<NeoHookeanBorden<T, dim>>, NeoHookeanBordenScratch<T, dim>>(name1);
        AttributeName<QRAnisotropic<T, dim>> name2(QRAnisotropic<T, dim>::name());
        damageG2PHelper<AttributeName<QRAnisotropic<T, dim>>, QRAnisotropicScratch<T, dim>>(name2);
        AttributeName<QRStableNeoHookean<T, dim>> name3(QRStableNeoHookean<T, dim>::name());
        damageG2PHelper<AttributeName<QRStableNeoHookean<T, dim>>, QRStableNeoHookeanScratch<T, dim>>(name3);
    }

    template <class TName, class ScratchType>
    void damageG2PHelper(TName model_name)
    {

        ZIRAN_TIMER();

        if (!particles.exist(model_name)) return; //check if this is the right model

        auto* pf_pointer = &particles.DataManager::get(phase_field_range());

        tbb::parallel_for(0, (int)particle_group.size(), [&](int group_idx) {
            for (int idx = particle_group[group_idx].first; idx <= particle_group[group_idx].second; ++idx) {
                int i = particle_order[idx];
                TV& Xp = particles.X[i];
                T pf = (T)0;

                BSplineWeights<T, dim> spline(Xp, dx);

                grid.iterateKernel(spline, particle_base_offset[i], [&](IV node, T w, TV dw, GridState<T, dim>& g) {
                    if (g.idx >= 0)
                        pf += w * g.phase_field;
                });

                // FLIP
                T& d = (*pf_pointer)[i].d;
                T new_d;

                new_d = d + pf; //pf encodes the transfered back difference in damage

                //want d to always increase, so take the max between old and new d, and then take the min between the new value and 1 (to enforce the ceiling)
                d = std::min(std::max(d, new_d), (T)1); //now we need to take the max
            }
        });

        auto ranges = particles.X.ranges;
        tbb::parallel_for(ranges,
            [&](DisjointRanges& subrange) {
                DisjointRanges subset(subrange,
                    particles.commonRanges(phase_field_range(),
                        model_name,
                        F_range()));
                for (auto iter = particles.subsetIter(subset, phase_field_range(), model_name, F_range()); iter; ++iter) {
                    auto& phase_field = iter.template get<0>();
                    auto& model = iter.template get<1>();
                    auto& F = iter.template get<2>();

                    //This section different from PFF, update model.g for degradation purposes
                    //Update constitutive model g
                    T d = phase_field.d;
                    T k = phase_field.residual_phase;
                    model.g = ((1 - d) * (1 - d) * (1 - k)) + k; //use monotonic deg function from cdmpm
                }
            });
    }
};

} // namespace ZIRAN

#endif
