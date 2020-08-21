#pragma once

#include "MpmParticleHandleBase.h"

#include <Ziran/CS/DataStructure/DataManager.h>
#include <Ziran/CS/DataStructure/KdTree.h>
#include <Ziran/CS/Util/DataDir.h>
#include <Ziran/CS/Util/AttributeNamesForward.h>
#include <Ziran/Math/Geometry/ObjIO.h>
#include <Ziran/Math/Geometry/PoissonDisk.h>
#include <Ziran/Math/Geometry/VoronoiNoise.h>
#include <Ziran/Math/Geometry/AnalyticLevelSet.h>
#include <Ziran/Math/Geometry/VdbLevelSet.h>
#include <Ziran/Math/Geometry/VtkIO.h>
#include <Ziran/Physics/LagrangianForce/LagrangianForce.h>
#include <Ziran/Physics/ConstitutiveModel/ConstitutiveModel.h>
#include <Ziran/Physics/PlasticityApplier.h>
#include <Ziran/Sim/Scene.h>

#include <MPM/Force/MpmPinningForceHelper.h>
#include <MPM/Force/MpmParallelPinningForceHelper.h>
#include <MPM/Force/CotangentBasedMpmForceHelper.h>
#include <MPM/Force/FBasedMpmForceHelper.h>
#include <MPM/Force/FDecoupledBasedMpmForceHelper.h>
#include <MPM/Force/FElasticNonequilibratedBasedMpmForceHelper.h>
#include <MPM/Force/FJMixedMpmForceHelper.h>
#include <MPM/Force/LinearCorotatedMpmForceHelper.h>
#include <MPM/Force/JBasedMpmForceHelper.h>
#include <MPM/Force/DensitySummationFluidMpmForceHelper.h>
#include <MPM/Force/MpmEtherDragForceHelper.h>
#include <MPM/Force/BulkViscosityMpmForceHelper.h>
#include <MPM/Force/ImplicitViscosityHelper.h>

#include "../../Projects/anisofracture/PhaseField.h"
#include "../../Projects/anisofracture/AnisotropicPhaseField.h"

namespace ZIRAN {

template <class T, int dim>
MpmParticleHandleBase<T, dim>::
    MpmParticleHandleBase(Particles<T, dim>& particles, Scene<T, dim>& scene, MpmForceBase<T, dim>* mpmforce,
        StdVector<std::unique_ptr<PlasticityApplierBase>>& plasticity_appliers,
        StdVector<TV>& scratch_xp, T& dt, Range particle_range, T total_volume, int cotangent_manifold_dim)
    : particles(particles)
    , scene(scene)
    , mpmforce(mpmforce)
    , plasticity_appliers(plasticity_appliers)
    , scratch_xp(scratch_xp)
    , dt(dt)
    , particle_range(particle_range)
    , total_volume(total_volume)
    , cotangent_manifold_dim(cotangent_manifold_dim)
{
}

// Creates a copy with new particles
template <class T, int dim>
MpmParticleHandleBase<T, dim>
MpmParticleHandleBase<T, dim>::copy()
{
    Range new_particle_range;
    new_particle_range.lower = particles.count;
    {
        auto ap = particles.appender();
        for (int i = particle_range.lower; i < particle_range.upper; i++)
            ap.append(particles.mass[i], particles.X[i], particles.V[i]);
    }
    new_particle_range.upper = particles.count;
    return MpmParticleHandleBase(particles, scene, mpmforce, plasticity_appliers, scratch_xp, dt, new_particle_range, total_volume);
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::transform(const std::function<void(int, Ref<T>, Vector<T, dim>&, Vector<T, dim>&)>& mapping)
{
    for (int i = particle_range.lower; i < particle_range.upper; ++i) {
        // lua does not support passing scalars by reference. This is a work around to actually change mass.
        mapping(i - particle_range.lower, particles.mass[i], particles.X[i], particles.V[i]);
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    addVolumeFraction(const T b)
{
    particles.add(volume_fraction_name<T>(), particle_range, b);
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    addBulkViscosity(const T b)
{
    using HelperT = BulkViscosityMpmForceHelper<T, dim>;

    BulkViscosity<T, dim> bulk(b);
    particles.add(HelperT::bulk_name(), particle_range, bulk);

    HelperT* hh = nullptr;
    for (auto& helper : mpmforce->helpers)
        if (HelperT* h = dynamic_cast<HelperT*>(helper.get()))
            hh = h;
    if (hh == nullptr) {
        mpmforce->helpers.emplace_back(std::make_unique<std::remove_const_t<HelperT>>(particles));
        hh = dynamic_cast<HelperT*>(mpmforce->helpers.back().get());
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    addImplicitViscosity(const T mu)
{
    using HelperT = ImplicitViscosityHelper<T, dim>;
    particles.add(HelperT::viscosity_name(), particle_range, mu);
    HelperT& h = mpmforce->getHelper(); // this will let mpmforce create a helper
    h.reinitialize();
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    addEtherDrag(const T drag)
{
    particles.add(ether_drag_name<T>(), particle_range, drag);
    MpmEtherDragForceHelper<T, dim>* helper = nullptr;
    for (auto& fh : mpmforce->helpers) {
        helper = dynamic_cast<MpmEtherDragForceHelper<T, dim>*>(fh.get());
        if (helper)
            break;
    }
    if (helper == nullptr)
        mpmforce->helpers.emplace_back(std::make_unique<MpmEtherDragForceHelper<T, dim>>(particles, scratch_xp, dt));
}

template <class T, int dim>
template <class TCONST>
void MpmParticleHandleBase<T, dim>::
    addFBasedMpmForce(const TCONST& model)
{
    if (cotangent_manifold_dim == 1)
        addCotangentBasedMpmForce<TCONST, 1>(model, particle_range);
    else if (cotangent_manifold_dim == 2)
        addCotangentBasedMpmForce<TCONST, 2>(model, particle_range);
    else if (cotangent_manifold_dim == 0)
        addFBasedMpmForceWithMeasure(model, particle_range, total_volume);
    else
        ZIRAN_ASSERT(false);
}

template <class T, int dim>
template <class TCONST>
void MpmParticleHandleBase<T, dim>::
    addFBasedMpmForceWithPhaseField(const T& percentage, const T& l0, const TCONST& model, bool allow_damage, const T random_fluctuation_percentage)
{
    ZIRAN_ASSERT(cotangent_manifold_dim == 0);
    FBasedMpmForceHelper<TCONST>& helper = mpmforce->getHelper(); // this will let mpmforce create a consitutive model helper
    particles.add(helper.constitutive_model_name(), particle_range, model);
    if (total_volume != 0) {
        particles.add(element_measure_name<T>(), particle_range, total_volume / particle_range.length());
        particles.add(helper.F_name(), particle_range, TM::Identity());
    }
    particles.add(helper.constitutive_model_scratch_name(), particle_range, typename FBasedMpmForceHelper<TCONST>::Scratch());
    PhaseField<T, dim> pf;
    pf.residual_phase = (T)0.001;
    pf.c = (T)1;
    pf.H = (T)0;
    pf.l0 = l0;
    pf.one_over_sigma_c = PhaseField<T, dim>::Get_Sigma_C_Based_On_Max_Deformation(percentage, model);
    pf.pf_Fp = (T)1;
    pf.H_max = std::numeric_limits<T>::max();
    pf.vol = total_volume / particle_range.length();
    pf.allow_damage = allow_damage;
    particles.add(phase_field_name<PhaseField<T, dim>>(), particle_range, pf);

    if (random_fluctuation_percentage) randomizePhaseFieldSigmaC(random_fluctuation_percentage);
}

template <class T, int dim>
template <class TCONST>
void MpmParticleHandleBase<T, dim>::addFBasedMpmForceWithAnisotropicPhaseField(const StdVector<Vector<T, dim>> a_0, const StdVector<T> alphas, const T& percentage, const T& l0, const TCONST& model, const T eta, const T zeta, bool allow_damage, const T residual)
{
    ZIRAN_ASSERT(cotangent_manifold_dim == 0);
    ZIRAN_ASSERT(a_0.size() == 1 || a_0.size() == 2, "Init Particle Handle Failure: Too many or too few structural directors!");
    for (size_t i = 0; i < a_0.size(); i++) {
        T epsilon = a_0[i].norm() - 1;
        ZIRAN_ASSERT(std::abs(epsilon) < 1e-5, "Structural directors must be normalized at initialization");
    }

    FBasedMpmForceHelper<TCONST>& helper = mpmforce->getHelper(); // this will let mpmforce create a consitutive model helper
    particles.add(helper.constitutive_model_name(), particle_range, model);
    if (total_volume != 0) {
        particles.add(element_measure_name<T>(), particle_range, total_volume / particle_range.length());
        particles.add(helper.F_name(), particle_range, TM::Identity());
    }
    particles.add(helper.constitutive_model_scratch_name(), particle_range, typename FBasedMpmForceHelper<TCONST>::Scratch());

    //Unique to Anisotropic Fracture
    AnisotropicPhaseField<T, dim> pf;
    pf.a_0.resize(a_0.size());
    for (size_t k = 0; k < a_0.size(); k++)
        pf.a_0[k] = TV::Unit(k); //this will always be just the world coordinate basis (e1, e2, e3)
    pf.alphas = alphas;
    pf.residual_phase = residual;
    pf.d = (T)0; //start with no damage (d starts at 0)
    pf.l0 = l0;
    pf.eta = eta;
    pf.zeta = zeta;
    pf.laplacian = (T)0;
    pf.vol = total_volume / particle_range.length();
    pf.allow_damage = allow_damage;

    pf.sigma_crit = model.getSigmaCrit(percentage); //added this to the model itself

    std::cout << "SigmaCrit = " << pf.sigma_crit << std::endl;

    // AnisotropicPhaseField<T, dim>::getSigmaCritFromPercentageStretch(percentage, model);

    particles.add(phase_field_name<AnisotropicPhaseField<T, dim>>(), particle_range, pf);

    //NOW, we initialize F to be the rotation defined by the structural directors we have defined!
    for (auto iter = particles.subsetIter(DisjointRanges{ particle_range }, F_name<T, dim>()); iter; ++iter) {
        auto& F = iter.template get<0>();
        F = initializeRotatedFHelper(a_0);
    }
}

template <class T, int dim>
Matrix<T, dim, dim> MpmParticleHandleBase<T, dim>::initializeRotatedFHelper(const StdVector<Vector<T, dim>> a_0)
{
    Matrix<T, dim, dim> F;
    if (a_0.size() == 1) {
        if constexpr (dim == 2) {
            TV first_column = a_0[0];
            Eigen::Rotation2D<T> r(M_PI / (T)2); // set from angle that is stored in q(0)
            TV second_column = r.toRotationMatrix() * first_column;
            F.col(0) = first_column;
            F.col(1) = second_column;
        }
        else {
            TV first_column = a_0[0];
            TV second_column;
            if (std::abs(a_0[0][0]) < (T)1e-6 && std::abs(a_0[0][1]) < (T)1e-6) {
                second_column = TV::Unit(0); //if we have 0,0,1 or 0,0,-1 then juse choose e_1 for second column
            }
            else {
                second_column(0) = -a_0[0][1];
                second_column(1) = a_0[0][0];
                second_column(2) = 0;
                second_column.normalize();
            }
            TV third_column = first_column.cross(second_column);
            F.col(0) = first_column;
            F.col(1) = second_column;
            F.col(2) = third_column;
        }
    }
    else {
        if constexpr (dim == 2) {
            ZIRAN_ASSERT(0, "Orthotropic is only possible in 3D!");
        }
        else {
            TV first_column = a_0[0];
            TV second_column = a_0[1];
            TV third_column = first_column.cross(second_column);
            F.col(0) = first_column;
            F.col(1) = second_column;
            F.col(2) = third_column;
        }
    }
    return F;
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::radialFibers(const TV center, const int zeroDimension)
{

    if (dim == 2) {
        int i = 0;
        StdVector<Vector<T, dim>> a_0;
        for (auto iter = particles.subsetIter(DisjointRanges{ particle_range }, F_name<T, dim>()); iter; ++iter) {
            auto& F = iter.template get<0>();

            TV position = particles.X[i];

            TV centeredPosition = position - center;

            a_0.clear();

            if (centeredPosition.norm() == 0) {
                a_0.push_back(TV::Unit(1)); //put up vector if norm is 0
                F = initializeRotatedFHelper(a_0);
                i++;
                continue;
            }
            else {
                a_0.push_back(centeredPosition.normalized());
                F = initializeRotatedFHelper(a_0);
                i++;
            }
        }
    }
    else {
        int i = 0;
        StdVector<Vector<T, dim>> a_0;
        for (auto iter = particles.subsetIter(DisjointRanges{ particle_range }, F_name<T, dim>()); iter; ++iter) {
            auto& F = iter.template get<0>();

            TV position = particles.X[i];

            TV centeredPosition = position - center;

            a_0.clear();

            if (centeredPosition.norm() == 0) {
                a_0.push_back(TV::Unit(1)); //put up vector if norm is 0
                F = initializeRotatedFHelper(a_0);
                i++;
                continue;
            }
            else {
                centeredPosition[zeroDimension] = 0; //zero this dimension for radial fibers
                TV zeroed = centeredPosition.normalized(); //THEN normalize
                a_0.push_back(zeroed);
                F = initializeRotatedFHelper(a_0);
                i++;
            }
        }
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::spiralFibers(const TV center, const int zeroDimension, const T theta)
{

    if (zeroDimension != 0) {
        ZIRAN_ASSERT(0, "for spiralFibers your tube must be aligned with x-axis");
    }

    if (dim == 2) {
        ZIRAN_ASSERT(0, "spiralFibers not implemented for 2D");
    }
    else {
        int i = 0;
        StdVector<Vector<T, dim>> a_0;
        for (auto iter = particles.subsetIter(DisjointRanges{ particle_range }, F_name<T, dim>()); iter; ++iter) {
            auto& F = iter.template get<0>();

            TV position = particles.X[i];

            TV centeredPosition = position - center;

            centeredPosition[zeroDimension] = 0; //zero this dimension

            TM rotate90; //THIS ONLY WORKS FOR ALIGNMENT WITH X DIRECTION
            rotate90 << 1, 0, 0, 0, std::cos(M_PI / 2), std::sin(M_PI / 2), 0, -1 * std::sin(M_PI / 2), std::cos(M_PI / 2);

            TV rotated = rotate90 * centeredPosition;

            rotated[0] = std::sin(theta * (M_PI / 180));

            a_0.clear();

            if (rotated.norm() == 0) {
                a_0.push_back(TV::Unit(1)); //put up vector if norm is 0
                F = initializeRotatedFHelper(a_0);
                i++;
                continue;
            }
            else {
                rotated.normalize();
                a_0.push_back(rotated);
                F = initializeRotatedFHelper(a_0);
                i++;
            }
        }
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::longitudinalFibers(const TV center, const T radius)
{
    //ZIRAN_ASSERT(dim == 2, "Longitudinal fibers not yet implemented for 3D!");

    if (dim == 2) {
        int i = 0;
        StdVector<Vector<T, dim>> a_0;
        for (auto iter = particles.subsetIter(DisjointRanges{ particle_range }, F_name<T, dim>()); iter; ++iter) {
            auto& F = iter.template get<0>();
            TV position = particles.X[i];

            TV centeredPosition = position - center;

            a_0.clear();

            if (centeredPosition.norm() == 0) {
                a_0.push_back(TV::Unit(1)); //put up vector if norm is 0
                F = initializeRotatedFHelper(a_0);
                i++;
                continue;
            }
            else {
                T x, y, a, x2, y2, r2, a2;
                x = centeredPosition[0];
                y = centeredPosition[1];
                x2 = x * x;
                y2 = y * y;
                r2 = radius * radius;
                a = std::sqrt(x2 / (1 - (y2 / r2)));
                a2 = a * a;

                TV a_1 = TV::Unit(1);
                T xDir, yDir;
                xDir = (2 * x) / a2;
                yDir = (2 * y) / r2;

                T theta = M_PI / 2; //90 deg in radians
                TM rotateTheta;
                rotateTheta << std::cos(theta), -1 * std::sin(theta), std::sin(theta), std::cos(theta);

                TV dirBefore;
                dirBefore[0] = xDir;
                dirBefore[1] = yDir;
                a_1 = rotateTheta * dirBefore;

                //Invert any values that need to be inverted to ensure correct directionality of vector field
                if (x <= 0) {
                    a_1[0] *= -1;
                    a_1[1] *= -1;
                }

                if (a_1.norm() == 0) {
                    a_0.push_back(TV::Unit(1));
                    F = initializeRotatedFHelper(a_0);
                    i++;
                    continue;
                }

                a_0.push_back(a_1.normalized());
                F = initializeRotatedFHelper(a_0);
                i++;
            }
        }
    }
    else {
        int i = 0;
        StdVector<Vector<T, dim>> a_0;
        for (auto iter = particles.subsetIter(DisjointRanges{ particle_range }, F_name<T, dim>()); iter; ++iter) {
            auto& F = iter.template get<0>();
            TV position = particles.X[i];

            TV centeredPosition = position - center;
            centeredPosition[2] = 0; //zero out the z dimension

            a_0.clear();

            if (centeredPosition.norm() == 0) {
                a_0.push_back(TV::Unit(1)); //put up vector if norm is 0
                F = initializeRotatedFHelper(a_0);
                i++;
                continue;
            }
            else {
                T x, y, a, x2, y2, r2, a2;
                x = centeredPosition[0];
                y = centeredPosition[1];
                x2 = x * x;
                y2 = y * y;
                r2 = radius * radius;
                a = std::sqrt(x2 / (1 - (y2 / r2)));
                a2 = a * a;

                TV a_1 = TV::Unit(1);
                T xDir, yDir;
                xDir = (2 * x) / a2;
                yDir = (2 * y) / r2;

                T theta = M_PI / 2; //90 deg in radians
                TM rotateTheta;
                rotateTheta << std::cos(theta), -1 * std::sin(theta), 0, std::sin(theta), std::cos(theta), 0, 0, 0, 1;

                TV dirBefore;
                dirBefore[0] = xDir;
                dirBefore[1] = yDir;
                dirBefore[2] = 0;
                a_1 = rotateTheta * dirBefore;

                //Invert any values that need to be inverted to ensure correct directionality of vector field
                if (x <= 0) {
                    a_1[0] *= -1;
                    a_1[1] *= -1;
                }

                if (a_1.norm() == 0) {
                    a_0.push_back(TV::Unit(1));
                    F = initializeRotatedFHelper(a_0);
                    i++;
                    continue;
                }

                a_0.push_back(a_1.normalized());
                F = initializeRotatedFHelper(a_0);
                i++;
            }
        }
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::fibersVaryWithY(const T yVal)
{
    ZIRAN_ASSERT(0, "Don't use this, it sucks lol");

    ZIRAN_ASSERT(dim == 2, "fibersVaryWithY not yet implemented for 3D!");

    int i = 0;
    for (auto iter = particles.subsetIter(DisjointRanges{ particle_range }, phase_field_name<AnisotropicPhaseField<T, dim>>()); iter; ++iter) {
        auto& pf = iter.template get<0>();
        TV position = particles.X[i];

        if (position[1] >= yVal) {
            pf.a_0.clear();
            //TV a_1(-1,1);
            TV a_1(2, 1);
            pf.a_0.push_back(a_1.normalized());
        }

        i++;
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::addOriginalPositionAsAttribute()
{
    // Create X0 as attribute
    particles.add(X0_name<T, dim>(), particle_range, TV::Zero());

    // Fill in X0 with current X
    for (auto iter = particles.iter(X0_name<T, dim>()); iter; ++iter) {
        Vector<T, dim>& X0 = iter.template get<0>();
        X0 = particles.X(iter.entryId());
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::randomizePhaseFieldSigmaC(const T random_fluctuation_percentage)
{
    RandomNumber<T> rn(123);
    for (auto iter = particles.subsetIter(DisjointRanges{ particle_range }, phase_field_name<PhaseField<T, dim>>()); iter; ++iter) {
        auto& pf = iter.template get<0>();
        T scale = 1 + rn.randReal(-random_fluctuation_percentage, random_fluctuation_percentage);
        pf.one_over_sigma_c *= scale;
    }
}

//Read in a file of voronoi points and use them to initialize sigmaC to be scaled based on point distance from voronoi surfaces
template <class T, int dim>
void MpmParticleHandleBase<T, dim>::voronoiSigmaC(std::string voronoiPointsFile, T radius)

//void MpmParticleHandleBase<T, dim>::voronoiSigmaC(std::string voronoiPointsFile, T radius, T minPercent, T maxPercent)
{
    //Read the file to grab our points from obj file
    StdVector<TV> vPoints;
    readPositionObj(voronoiPointsFile, vPoints);

    //Set up our KDTree and add each of our voronoi points to it!
    KdTree<dim> voronoiPoints;
    for (int i = 0; i < (int)vPoints.size(); i++) {
        voronoiPoints.addPoint(i, vPoints[i]);
    }
    voronoiPoints.optimize();

    //Now we can query this voronoi surface point KDTree for distances from our actual points!
    int i = 0;
    T zeta = -1; //set -1 until we calculate it for the first time
    for (auto iter = particles.subsetIter(DisjointRanges{ particle_range }, phase_field_name<PhaseField<T, dim>>()); iter; ++iter) {

        auto& pf = iter.template get<0>();

        int id;
        TV p;
        T dist;
        voronoiPoints.findNearest(particles.X[i], id, p, dist); //fill in "dist" with distance from nearest point

        //4th Way: use base to reverse engineer what value of zeta we need to fit the values to the curve y = e^(zeta * x) - 1
        if (zeta == -1) { //only compute once
            T baseSigmaC = (T)1 / pf.one_over_sigma_c;
            zeta = std::log(baseSigmaC + 1) / radius;
        }
        T newSigmaC = (T)1 / pf.one_over_sigma_c;
        if (dist < radius) {
            newSigmaC = std::exp(zeta * dist) - 1;
        }
        pf.one_over_sigma_c = (T)1 / newSigmaC;

        //3rd way: use 1 - log(dist/radius) and clamp all scales past (dist/radius)=1 to be 1
        //NOTE: this relies on scaling SPECIFICALLY one over sigma C!!!!!
        /*T x = dist / radius;
        T scale = 1;
        if (x < 1) {
            scale = (1 - std::log(x)) * magnitude; //scale by magnitude to allow for greater disparity between large and small sigmaC
        }
        pf.one_over_sigma_c *= scale; //NEED to scale one over sigma C here
        */

        //2nd Attempt: Exponential Way
        //T scale = std::exp(zeta * dist);

        //1st Way: linear scale based on dist, scale sigmaC!
        /*T proportion = (std::exp((dist / radius)) - 1) * (maxPercent - minPercent); //proportion through the interval min% to max% we are based on dist --> but using tanh instead of linear
        T scale = 1 + (minPercent + proportion);

        //Clamp all points outside the radius to be the max percent increase
        if (dist >= radius) {
            scale = 1 + maxPercent;
        }

        //clamp values to not be greater than 1 + maxPercent
        if(scale > (1+maxPercent)){
            scale = 1 + maxPercent;
        }

        //Clamp the scale to never be non-zero
        if (scale <= 0) {
            scale = 0.0000000001; //arbitrary small epsilon (to avoid divide by 0)
        }*/

        //want to scale the actual sigmaC, not its inverse!
        //T currSigmaC = (T)1 / pf.one_over_sigma_c;
        //T newSigmaC = currSigmaC * scale;
        //pf.one_over_sigma_c = (T)1 / newSigmaC;
        //pf.one_over_sigma_c *= scale; //for if we want to directly scale one over sigmaC

        i++; //increment particle index since our iterator is just for pf itself
    }
}

//Read in a file of voronoi points and use them to initialize sigmaC to be scaled based on point distance from voronoi surfaces
template <class T, int dim>
void MpmParticleHandleBase<T, dim>::yDimSigmaCInit(T yMin, T yMax, T maxScale)
{
    //Now we can query this voronoi surface point KDTree for distances from our actual points!
    int i = 0;
    for (auto iter = particles.subsetIter(DisjointRanges{ particle_range }, phase_field_name<PhaseField<T, dim>>()); iter; ++iter) {

        auto& pf = iter.template get<0>();
        T yVal = particles.X[i][1]; //grab y val

        //std::cout << "Y val: " << yVal << std::endl;

        if (yVal < yMin) {
            i++;
            continue;
        }
        else if (yVal > yMax) {
            i++;
            continue;
        }

        T dist = (yMax - yVal) / (yMax - yMin);

        T scale = dist * maxScale; //linear scale based on dist

        if (scale < 1) {
            scale = 1;
        }

        T newSigmaC = ((T)1 / pf.one_over_sigma_c) * scale;
        pf.one_over_sigma_c = (T)1 / newSigmaC;

        i++; //increment particle index since our iterator is just for pf itself
    }
}

template <class T, int dim>
template <class TCONST>
void MpmParticleHandleBase<T, dim>::
    addFDecoupledBasedMpmForce(const TCONST& model, T neighbor_search_h)
{
    FDecoupledBasedMpmForceHelper<TCONST>& helper = mpmforce->getHelper(); // this will let mpmforce create a constitutive model helper
    helper.setNeighborSearchH(neighbor_search_h);
    particles.add(helper.constitutive_model_name(), particle_range, model);
    if (total_volume != 0) {
        particles.add(element_measure_name<T>(), particle_range, total_volume / particle_range.length());
        particles.add(helper.F_name(), particle_range, TM::Identity());
        particles.add(F_Distortional_name<Eigen::Matrix<T, dim, dim>>(), particle_range, TM::Identity());
        particles.add(F_Dilational_name<T>(), particle_range, (T)1);
    }
    particles.add(helper.constitutive_model_scratch_name(), particle_range, typename FDecoupledBasedMpmForceHelper<TCONST>::Scratch());
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    scaleF(const T scale)
{
    for (auto iter = particles.subsetIter(DisjointRanges{ particle_range }, F_name<T, dim>()); iter; ++iter) {
        auto& F = iter.template get<0>();
        F *= scale;
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    scaleF(const TV scale)
{
    for (auto iter = particles.subsetIter(DisjointRanges{ particle_range }, F_name<T, dim>()); iter; ++iter) {
        auto& F = iter.template get<0>();
        if constexpr (dim >= 1)
            F.row(0) *= scale[0];
        if constexpr (dim >= 2)
            F.row(1) *= scale[1];
        if constexpr (dim >= 3)
            F.row(2) *= scale[2];
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    scaleJ(const T scale)
{
    for (auto iter = particles.subsetIter(DisjointRanges{ particle_range }, J_name<T>()); iter; ++iter) {
        auto& J = iter.template get<0>();
        J *= scale;
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    updatePinningTarget(const Vector<T, 4>& R_in, const Vector<T, 3>& omega_in, const TV& b, const TV& dbdt)
{
    Rotation<T, dim> R(R_in);
    AngularVelocity<T, dim> omega;
    omega.set(omega_in);

    DisjointRanges subset(DisjointRanges{ particle_range },
        particles.commonRanges(AttributeName<PinningTarget<T, dim>>("PinningTarget")));

    for (auto iter = particles.subsetIter(subset, AttributeName<PinningTarget<T, dim>>("PinningTarget")); iter; ++iter) {
        auto& target = iter.template get<0>();
        target.x = R.rotation * target.X + b;
        target.v = omega.cross(target.x - b) + dbdt;
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    scaleFCurve(int frame, const std::function<T(int)>& growCurve)
{
    DisjointRanges subset{ particle_range };
    for (auto iter = particles.subsetIter(subset, F_name<T, dim>()); iter; ++iter) {
        auto& F = iter.template get<0>();
        if (frame == 0)
            F /= growCurve(0);
        // else
        //      F *= (growCurve(frame - 1) / growCurve(frame));
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    manipulateF(const std::function<Matrix<T, dim, dim>(Matrix<T, dim, dim>, Vector<T, dim>)>& op)
{
    DisjointRanges subset{ particle_range };
    for (auto iter = particles.subsetIter(subset, F_name<T, dim>(), particles.X_name()); iter; ++iter) {
        auto& F = iter.template get<0>();
        auto& X = iter.template get<1>();
        TM new_F = op(F, X);
        F = new_F;
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    standingPoolInitialization(const EquationOfState<T, dim> model, const TV& gravity, const TV& seaLevel)
{
    auto cname = JBasedMpmForceHelper<EquationOfState<T, dim>>::constitutive_model_name();
    DisjointRanges ranges(DisjointRanges{ particle_range }, particles.commonRanges(particles.X_name(), particles.mass_name(), J_name<T>(), element_measure_name<T>(), cname));
    ZIRAN_ASSERT(ranges[0] == particle_range);
    ZIRAN_INFO("Setting J for particles ", ranges);
    for (auto iter = particles.subsetIter(ranges, particles.X_name(), particles.mass_name(), J_name<T>(), element_measure_name<T>(), cname); iter; ++iter) {
        TV& x = iter.template get<0>();
        T& mass = iter.template get<1>();
        auto& J = iter.template get<2>();
        T& Vp0 = iter.template get<3>();
        auto& cons_model = iter.template get<4>();
        J = std::pow(mass / Vp0 / cons_model.bulk * gravity.dot(x - seaLevel) + 1, -(T)1 / cons_model.gamma);
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    resetDeformation()
{
    for (auto iter = particles.subsetIter(DisjointRanges{ particle_range }, F_name<T, dim>()); iter; ++iter) {
        auto& F = iter.template get<0>();
        F = Matrix<T, dim, dim>::Identity();
    }
}

template <class T, int dim>
template <class TCONST>
void MpmParticleHandleBase<T, dim>::
    addJBasedMpmForce(const TCONST& model)
{
    JBasedMpmForceHelper<TCONST>& helper = mpmforce->getHelper(); // this will let mpmforce create a consitutive model helper
    particles.add(helper.constitutive_model_name(), particle_range, model);
    if (total_volume != 0) {
        particles.add(element_measure_name<T>(), particle_range, total_volume / particle_range.length());
        particles.add(J_name<T>(), particle_range, (T)1);
    }
    particles.add(helper.constitutive_model_scratch_name(), particle_range, typename JBasedMpmForceHelper<TCONST>::Scratch());
    //addJBasedMpmForceWithMeasure(model, particle_range, total_volume);
}

template <class T, int dim>
template <class FCONST, class JCONST>
void MpmParticleHandleBase<T, dim>::
    addFJMixedMpmForce(const FCONST& f_model_input, const JCONST& j_model, MATERIAL_PHASE_ENUM phase, MATERIAL_PROPERTY_ENUM property, bool linear_corotated)
{
    FJMixedMpmForceHelper<FCONST, JCONST>& helper = mpmforce->getHelper(); // this will let mpmforce create a consitutive model helper
    helper.linear_corotated = linear_corotated;
    FCONST f_model = f_model_input;
    if (phase == MATERIAL_PHASE_FLUID) {
        f_model.mu = 0;
        f_model.lambda = 0;
    }
    particles.add(helper.f_constitutive_model_name(), particle_range, f_model);
    particles.add(helper.j_constitutive_model_name(), particle_range, j_model);
    if (total_volume != 0) {
        particles.add(element_measure_name<T>(), particle_range, total_volume / particle_range.length());
        particles.add(J_name<T>(), particle_range, (T)1);
        particles.add(F_name<T, dim>(), particle_range, TM::Identity());
    }
    particles.add(helper.f_constitutive_model_scratch_name(), particle_range, typename FJMixedMpmForceHelper<FCONST, JCONST>::FScratch());
    particles.add(helper.j_constitutive_model_scratch_name(), particle_range, typename FJMixedMpmForceHelper<FCONST, JCONST>::JScratch());
    particles.add(material_phase_name(), particle_range, phase);
    particles.add(material_property_name(), particle_range, property);
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::addDensitySummationFluidMpmForce(const T h)
{
    DensitySummationFluidMpmForceHelper<T, dim>& helper = mpmforce->getHelper(); // this will let mpmforce create a consitutive model helper
    if (total_volume != 0) {
        particles.add(density_name<T>(), particle_range, 0);
        particles.add(initial_density_name<T>(), particle_range, 0);
        particles.add(F_Distortional_name<TM>(), particle_range, TM::Identity());
        helper.preheat(h);
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    addLinearCorotatedMpmForce(const LinearCorotated<T, dim>& model)
{
    LinearCorotatedMpmForceHelper<T, dim>& helper = mpmforce->getHelper(); // this will let mpmforce create a consitutive model helper
    particles.add(helper.constitutive_model_name(), particle_range, model);
    if (total_volume != 0) {
        particles.add(element_measure_name<T>(), particle_range, total_volume / particle_range.length());
        particles.add(helper.F_name(), particle_range, TM::Identity());
    }
    particles.add(helper.constitutive_model_scratch_name(), particle_range, typename LinearCorotatedMpmForceHelper<T, dim>::Scratch());
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    addElementMeasure()
{
    if (total_volume != 0) {
        particles.add(element_measure_name<T>(), particle_range, total_volume / particle_range.length());
    }
}

// this should only be called once for a handle
template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    pinParticlesInLevelSet(AnalyticLevelSet<T, dim>& levelset, T stiffness, T damping)
{
    DisjointRanges dr;
    int count = 0;
    for (int i = particle_range.lower; i < particle_range.upper; i++) {
        if (levelset.inside((particles.X[i]))) {
            dr.append(Range{ i, i + 1 });
            count++;
        }
    }
    ZIRAN_INFO("pinParticlesInLevelSet pinned ", count, " particles.");

    pinParticles(dr, stiffness, damping);
}

// this should only be called once for a handle
template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    pinParticlesInVdbLevelSet(std::string filename, T stiffness, T damping)
{
    VdbLevelSet<float, dim> levelset(filename);
    DisjointRanges dr;
    for (int i = particle_range.lower; i < particle_range.upper; i++)
        if (levelset.inside((particles.X[i]).template cast<float>()))
            dr.append(Range{ i, i + 1 });
    pinParticles(dr, stiffness, damping);
}

// this should only be called once for a handle
template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    parallelPinParticlesInLevelSet(AnalyticLevelSet<T, dim>& levelset, T stiffness, T damping, const bool is_collider_spring)
{
    DisjointRanges dr;
    int count = 0;
    for (int i = particle_range.lower; i < particle_range.upper; i++) {
        if (levelset.inside((particles.X[i]))) {
            dr.append(Range{ i, i + 1 });
            count++;
        }
    }
    ZIRAN_INFO("pinParticlesInLevelSet pinned ", count, " particles.");

    parallelPinParticles(dr, stiffness, damping, is_collider_spring);
}

// this should only be called once for a handle
template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    parallelPinParticlesInVdbLevelSet(std::string filename, T stiffness, T damping)
{
    VdbLevelSet<float, dim> levelset(filename);
    DisjointRanges dr;
    for (int i = particle_range.lower; i < particle_range.upper; i++)
        if (levelset.inside((particles.X[i]).template cast<float>()))
            dr.append(Range{ i, i + 1 });
    parallelPinParticles(dr, stiffness, damping, false);
}

template <class T, int dim>
template <class TConst, class TPlastic>
void MpmParticleHandleBase<T, dim>::
    addPlasticity(const TConst& cons, const TPlastic& plasticity, std::string strain_name)
{
    using TStrain = typename TConst::Strain;
    PlasticityApplier<TConst, TPlastic, TStrain>* plasticity_model = nullptr;
    for (auto& p : plasticity_appliers) {
        plasticity_model = dynamic_cast<PlasticityApplier<TConst, TPlastic, TStrain>*>(p.get());
        if (plasticity_model && plasticity_model->strain_name.name == strain_name)
            break;
        else
            plasticity_model = nullptr;
    }
    if (plasticity_model == nullptr)
        plasticity_appliers.push_back(std::make_unique<PlasticityApplier<TConst, TPlastic, TStrain>>(strain_name));
    particles.add(TPlastic::name(), particle_range, plasticity);
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    addDummyPlasticity()
{
    auto p = DummyPlasticity<T>();
    particles.add(DummyPlasticity<T>::name(), particle_range, p);
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    prescoreSnow(AnalyticLevelSet<T, dim>& levelset, T grain_size, T density_min, T Jp_min)
{
    ZIRAN_INFO("Prescoring Snow Particles");
    TV min_corner, max_corner;
    levelset.getBounds(min_corner, max_corner);
    PoissonDisk<T, dim> pd(/*random seed*/ 123, grain_size, min_corner, max_corner);
    StdVector<TV> grain_centroids;
    if (dim == 3)
        pd.sampleFromPeriodicData(grain_centroids, [&](const TV& x) { return levelset.inside(x); });
    else
        pd.sample(grain_centroids, [&](const TV& x) { return levelset.inside(x); });
    auto spn = AttributeName<SnowPlasticity<T>>(SnowPlasticity<T>::name());
    T phi = T(0.5) * (1 - std::sqrt((T)5));
    RandomNumber<T> rand;
    for (auto iter = particles.subsetIter({ particle_range }, Particles<T, dim>::X_name(), Particles<T, dim>::mass_name(), spn, element_measure_name<T>()); iter; ++iter) {
        TV& x = iter.template get<0>();
        for (const TV& c : grain_centroids) {
            TV v = c - x;
            T alpha = std::min(v.norm() / ((T)1.2 * grain_size), (T)1);
            T scale = phi + 1 / (alpha - phi);
            scale *= rand.randReal(0.75, 1.0);
            x = x + scale * v;
        }
        T min_distance2 = std::numeric_limits<T>::max();
        for (const TV& c : grain_centroids)
            min_distance2 = std::min(min_distance2, (x - c).squaredNorm());
        T& mass = iter.template get<1>();
        SnowPlasticity<T>& p = iter.template get<2>();
        T element_measure = iter.template get<3>();
        T scale = 1 - std::min(std::sqrt(min_distance2) * rand.randReal(0.75, 1.25) / (2 * grain_size), (T)1);
        T mass_min = density_min * element_measure;
        mass = scale * (mass - mass_min) + mass_min;
        p.theta_c *= scale;
        p.theta_s *= scale;
        p.Jp = scale * (p.max_Jp - Jp_min) + Jp_min;
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    prescorePhaseFieldSigmaC(AnalyticLevelSet<T, dim>& levelset, T grain_size, T scale_lower_clamp)
{
    ZIRAN_INFO("Prescoring phase field sigma_c");
    TV min_corner, max_corner;
    levelset.getBounds(min_corner, max_corner);
    PoissonDisk<T, dim> pd(/*random seed*/ 123, grain_size, min_corner, max_corner);
    StdVector<TV> grain_centroids;
    if (dim == 3)
        pd.sampleFromPeriodicData(grain_centroids, [&](const TV& x) { return levelset.inside(x); });
    else
        pd.sample(grain_centroids, [&](const TV& x) { return levelset.inside(x); });
    T phi = T(0.5) * (1 - std::sqrt((T)5));
    RandomNumber<T> rand;
    for (auto iter = particles.subsetIter({ particle_range }, Particles<T, dim>::X_name(), phase_field_name<PhaseField<T, dim>>()); iter; ++iter) {
        TV& x = iter.template get<0>();
        for (const TV& c : grain_centroids) {
            TV v = c - x;
            T alpha = std::min(v.norm() / ((T)1.2 * grain_size), (T)1);
            T scale = phi + 1 / (alpha - phi);
            scale *= rand.randReal(0.75, 1.0);
            //x = x + scale * v;
        }
        T min_distance2 = std::numeric_limits<T>::max();
        for (const TV& c : grain_centroids)
            min_distance2 = std::min(min_distance2, (x - c).squaredNorm());
        auto& pf = iter.template get<1>();
        T scale = 1 - std::min(std::sqrt(min_distance2) * rand.randReal(0.75, 1.25) / (2 * grain_size), (T)1);
        scale = scale * (1 - scale_lower_clamp) + scale_lower_clamp;
        T old_sigma_c = (T)1 / pf.one_over_sigma_c;
        T new_sigma_c = old_sigma_c * scale;
        pf.one_over_sigma_c = (T)1 / new_sigma_c;
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    prescoreModifiedCamClay(T grain_size, T M_max, T beta_max, T Jp_min)
{
    ZIRAN_INFO("Prescoring Modified Cam Clay Particles");
    auto mccn = AttributeName<ModifiedCamClay<T>>(ModifiedCamClay<T>::name());
    T scale = 1 / grain_size;
    T min_logJp = std::log(Jp_min);
    for (auto iter = particles.subsetIter({ particle_range }, Particles<T, dim>::X_name(), mccn); iter; ++iter) {
        Vector<float, dim> x = scale * iter.template get<0>().template cast<float>();
        ModifiedCamClay<T>& p = iter.template get<1>();
        T alpha = T(0.125) * voronoiDistance(x);
        p.M = alpha * p.M + (1 - alpha) * M_max;
        p.beta = alpha * p.beta + (1 - alpha) * beta_max;
        p.logJp = (1 - alpha) * p.logJp + alpha * min_logJp;
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    prescoreWetSand2D(AnalyticLevelSet<T, dim>& levelset, T grain_size, T density_min, T Jp_min, T scale_min)
{
    ZIRAN_INFO("Prescoring Wet Sand Particles");
    TV min_corner, max_corner;
    levelset.getBounds(min_corner, max_corner);
    PoissonDisk<T, dim> pd(/*random seed*/ 123, grain_size, min_corner, max_corner);
    StdVector<TV> grain_centroids;
    if (dim == 3)
        pd.sampleFromPeriodicData(grain_centroids, [&](const TV& x) { return levelset.inside(x); });
    else
        pd.sample(grain_centroids, [&](const TV& x) { return levelset.inside(x); });

    auto dpshn = AttributeName<DruckerPragerStvkHencky<T>>(DruckerPragerStvkHencky<T>::name());
    auto stvkhi = AttributeName<StvkWithHenckyIsotropic<T, dim>>(StvkWithHenckyIsotropic<T, dim>::name());
    T phi = T(0.5) * (1 - std::sqrt((T)5));
    RandomNumber<T> rand;

    for (auto iter = particles.subsetIter({ particle_range }, Particles<T, dim>::X_name(), Particles<T, dim>::mass_name(), dpshn, element_measure_name<T>(), stvkhi); iter; ++iter) {
        TV& x = iter.template get<0>();
        for (const TV& c : grain_centroids) {
            TV v = c - x;
            T alpha = std::min(v.norm() / ((T)1.2 * grain_size), (T)1);
            T scale = phi + 1 / (alpha - phi);
            scale *= rand.randReal(0.75, 1.0);
            x = x + scale * v;
        }
        // scale scale
        T scale_scale = (T)1;
        T scale_scale_min = (T)1;
        if (x(0) > 0.43 && x(1) > 1.05 && (x(2) > -0.2 && x(2) < -0.08) && rand.randReal(0, 1) < 0.5) {
            scale_scale = (T)0.1;
            scale_scale_min = 0.1;
        }
        else if (x(0) > 0.43 && (x(1) > 0.8 && x(1) < 1.05) && (x(2) > 0.1 && x(2) < 0.26) && rand.randReal(0, 1) < 0.5) {
            scale_scale = (T)0.14;
            scale_scale_min = 0.1;
        }
        else if (x(0) > 0.43 && (x(1) > 0.75 && x(1) < 0.905) && (x(2) > -0.26 && x(2) < -0.13) && rand.randReal(0, 1) < 0.7) {
            scale_scale = (T)0.24;
            scale_scale_min = 0.1;
        }

        T min_distance2 = std::numeric_limits<T>::max();
        for (const TV& c : grain_centroids)
            min_distance2 = std::min(min_distance2, (x - c).squaredNorm());
        T& mass = iter.template get<1>();
        DruckerPragerStvkHencky<T>& p = iter.template get<2>();
        T element_measure = iter.template get<3>();
        auto& cm_elastic = iter.template get<4>();
        T scale = 1 - std::min(std::sqrt(min_distance2) * rand.randReal(0.75, 1.25) / (2 * grain_size), (T)1);
        T mass_min = density_min * element_measure;
        mass = scale * (mass - mass_min) + mass_min;
        p.cohesion *= scale * scale_scale;
        cm_elastic.lambda *= std::max(scale * scale_scale, scale_min * scale_scale_min);
        cm_elastic.mu *= std::max(scale * scale_scale, scale_min * scale_scale_min);
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    pinParticles(const DisjointRanges& ranges, const T stiffness, const T damping)
{
    ZIRAN_ASSERT(!already_pinned_particles, "pinParticels should only be called once per handle.");
    already_pinned_particles = true;

    MpmPinningForceHelper<T, dim>* helper = nullptr;
    for (auto& fh : mpmforce->helpers) {
        helper = dynamic_cast<MpmPinningForceHelper<T, dim>*>(fh.get());
        if (helper)
            break;
    }
    if (helper == nullptr)
        mpmforce->helpers.emplace_back(std::make_unique<MpmPinningForceHelper<T, dim>>(particles, scratch_xp, dt));

    auto& target = particles.DataManager::get(helper->target_name());
    DisjointRanges dr = target.ranges;
    dr.merge(ranges);
    target.lazyResize(dr);

    for (auto iter = particles.subsetIter(ranges, helper->target_name(), Particles<T, dim>::X_name(), Particles<T, dim>::mass_name()); iter; ++iter) {
        PinningTarget<T, dim>& t = iter.template get<0>();
        const TV& x = iter.template get<1>();
        const T mass = iter.template get<2>();
        t.k = stiffness * mass;
        t.d = damping * mass;
        t.x = x;
        t.v.setZero();
        t.X = t.x;
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::
    parallelPinParticles(const DisjointRanges& ranges, const T stiffness, const T damping, const bool is_collider_spring)
{
    ZIRAN_ASSERT(!already_parallel_pinned_particles, "pinParticels should only be called once per handle.");
    already_parallel_pinned_particles = true;

    MpmParallelPinningForceHelper<T, dim>* helper = nullptr;
    for (auto& fh : mpmforce->helpers) {
        helper = dynamic_cast<MpmParallelPinningForceHelper<T, dim>*>(fh.get());
        if (helper)
            break;
    }
    if (helper == nullptr)
        mpmforce->helpers.emplace_back(std::make_unique<MpmParallelPinningForceHelper<T, dim>>(particles, scratch_xp, dt, is_collider_spring));

    auto& target = particles.DataManager::get(helper->target_name());
    DisjointRanges dr = target.ranges;
    dr.merge(ranges);
    target.lazyResize(dr);

    for (auto iter = particles.subsetIter(ranges, helper->target_name(), Particles<T, dim>::X_name(), Particles<T, dim>::mass_name()); iter; ++iter) {
        ParallelPinningTarget<T, dim>& t = iter.template get<0>();
        const TV& x = iter.template get<1>();
        const T mass = iter.template get<2>();
        t.k = stiffness * mass;
        t.d = damping * mass;
        t.x.clear();
        t.v.clear();
    }
}

template <class T, int dim>
template <class TCONST, int manifold_dim>
void MpmParticleHandleBase<T, dim>::
    addCotangentBasedMpmForce(const TCONST& model, const Range& range)
{
    using HelperT = CotangentBasedMpmForceHelper<TCONST, manifold_dim>;
    HelperT* hh = nullptr;
    for (auto& helper : mpmforce->helpers)
        if (HelperT* h = dynamic_cast<HelperT*>(helper.get()))
            hh = h;
    // We didn't find it
    if (hh == nullptr) {
        SimplexElements<T, manifold_dim, dim>& elements = scene.getElementManager();
        mpmforce->helpers.emplace_back(std::make_unique<std::remove_const_t<HelperT>>(particles, elements));
        hh = dynamic_cast<HelperT*>(mpmforce->helpers.back().get());
    }

    particles.add(hh->constitutive_model_scratch_name(), range, typename HelperT::Scratch());

    // These two are already added in the particle handle
    // particles.add(hh->VP_name(), range, typename HelperT::TM());
    // particles.add(hh->VdP_name(), range, typename HelperT::TM());

    particles.add(hh->constitutive_model_name(), range, model);
}

template <class T, int dim>
template <class TCONST>
void MpmParticleHandleBase<T, dim>::
    addFBasedMpmForceWithMeasure(const TCONST& model, const Range& range, T total_volume)
{
    FBasedMpmForceHelper<TCONST>& helper = mpmforce->getHelper(); // this will let mpmforce create a consitutive model helper
    particles.add(helper.constitutive_model_name(), range, model);
    if (total_volume != 0) {
        particles.add(element_measure_name<T>(), range, total_volume / range.length());
        particles.add(helper.F_name(), range, TM::Identity());
    }
    particles.add(helper.constitutive_model_scratch_name(), range, typename FBasedMpmForceHelper<TCONST>::Scratch());
}

template <class T, int dim>
template <class TCONST>
void MpmParticleHandleBase<T, dim>::
    addFElasticNonequilibratedBasedMpmForce(const TCONST& model, T viscosity_d_input, T viscosity_v_input)
{
    FElasticNonequilibratedBasedMpmForceHelper<TCONST>& helper = mpmforce->getHelper(); // this will let mpmforce create a consitutive model helper
    particles.add(helper.constitutive_model_name(), particle_range, model);
    particles.add(helper.F_name(), particle_range, TM::Identity());
    particles.add(helper.constitutive_model_scratch_name(), particle_range, typename FElasticNonequilibratedBasedMpmForceHelper<TCONST>::Scratch());
    helper.setParameters(viscosity_d_input, viscosity_v_input);
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::setMassFromDensity(const T density)
{
    ZIRAN_INFO("Setting mass from densiy: total_volume = ", total_volume, ", particle.count = ", particle_range.length());
    T mp = density * total_volume / particle_range.length();
    for (auto iter = particles.subsetIter(DisjointRanges{ particle_range }, particles.mass_name()); iter; ++iter) {
        iter.template get<0>() = mp;
    }
}

template <class T, int dim>
void MpmParticleHandleBase<T, dim>::addInextensibility(const TM& F)
{
    particles.add(inextensibility_name<bool>(), particle_range, true);
    for (auto iter = particles.subsetIter(DisjointRanges{ particle_range }, F_name<T, dim>()); iter; ++iter) {
        iter.template get<0>() = F;
    }
}
} // namespace ZIRAN
