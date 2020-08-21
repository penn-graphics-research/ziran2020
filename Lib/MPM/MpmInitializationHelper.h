#ifndef MPM_INITIALIZATION_HELPER_H
#define MPM_INITIALIZATION_HELPER_H

#include <Ziran/CS/Util/RandomNumber.h>
#include <Ziran/Math/Geometry/CollisionObject.h>
#include <Ziran/Math/Geometry/AnimatedCollisionObject.h>
#include <Ziran/Math/Geometry/SourceCollisionObject.h>
#include <Ziran/Math/Geometry/PoissonDisk.h>
#include <Ziran/Math/Geometry/RandomSampling.h>
#include <Ziran/Math/Geometry/CurveFinder.h>
#include <Ziran/Math/Geometry/Grid.h>
#include <Ziran/Physics/LagrangianForce/FemCodimensional.h>
#include <Ziran/Physics/PlasticityApplier.h>
#include <MPM/Force/FJMixedMpmForceHelper.h>
#include <MPM/Force/FBasedMpmForceHelper.h>
#include <MPM/Forward/MpmForward.h>
#include <MPM/MpmParticleHandleBase.h>
#include <MPM/MpmSimulationBase.h>
#include <MPM/Force/MpmPinningForceHelper.h>
#include <MPM/Force/MpmParallelPinningForceHelper.h>
#include <MPM/Force/MpmEtherDragForceHelper.h>
#include <float.h>
#include <Ziran/Math/Geometry/VtkIO.h>

namespace ZIRAN {

template <class T, int dim>
class SourceCollisionObject;

template <class T, int dim>
class Sphere;

template <class T, int dim>
class Grid;

template <class T, int dim>
class VdbLevelSet;

template <class T, int dim>
class PoissonDisk;

template <class T, int dim>
class MpmInitializationHelper {
public:
    static constexpr int interpolation_degree = ZIRAN_MPM_DEGREE;
    using TV = Vector<T, dim>;
    using IV = Vector<int, dim>;

    MpmSimulationBase<T, dim>& mpm;
    Scene<T, dim>& scene;

    MpmInitializationHelper(MpmSimulationBase<T, dim>& mpm)
        : mpm(mpm)
        , scene(mpm.scene)
    {
    }

    template <class TMesh>
    MpmParticleHandleBase<T, dim> makeParticleHandle(const DeformableObjectHandle<T, dim, TMesh>& deformable, const T volume_scale)
    {
        // volume_scale is useful when e.g. in 3D for a thin shell, you want to scale the volume by a thickness to let it be reasonable.
        return MpmParticleHandleBase<T, dim>(mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(), mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(), deformable.particle_range, deformable.totalVolume() * volume_scale);
    }

    template <class BaryStack>
    void sampleBarycentricWeights(const int count, BaryStack& barys, RandomNumber<typename BaryStack::Scalar>& rand)
    {
        constexpr static int manifold_dim = BaryStack::RowsAtCompileTime - 1;
        static const T bary_table[4][3] = { { (T)1 / 3, (T)1 / 3, (T)1 / 3 }, { (T)1 / 6, (T)1 / 6, (T)4 / 6 }, { (T)4 / 6, (T)1 / 6, (T)1 / 6 }, { (T)1 / 6, (T)4 / 6, (T)1 / 6 } };

        for (int p = 0; p < count; p++) {
            if (manifold_dim == 1) { // hair
                barys(0, p) = (1 + p) * (T)1 / (count + 1);
                barys(1, p) = 1 - barys(0, p);
            }
            else if (manifold_dim == 2) { // 2d cloth in 3d
                if (p < 4) {
                    barys(0, p) = bary_table[p][0];
                    barys(1, p) = bary_table[p][1];
                    barys(2, p) = bary_table[p][2];
                }
                else {
                    barys.col(p) = rand.template randomBarycentricWeights<manifold_dim + 1>();
                }
            }
            else
                ZIRAN_ASSERT(false);
        }
    }

    template <class TMesh>
    void initializeElementVertexF(const DeformableObjectHandle<T, dim, TMesh>& deformable)
    {
        using TM = Matrix<T, dim, dim>;
        auto& elements = deformable.elements;
        auto element_measure_name = elements.element_measure_name();
        int offset = deformable.particle_range.lower;

        StdVector<TM> F(deformable.particle_range.length());

        for (auto iter = elements.subsetIter(DisjointRanges{ deformable.element_range }, element_measure_name); iter; ++iter) {
            int element_global_index = iter.entryId();

            auto indices = elements.indices[element_global_index];
            auto Ds = elements.dS(elements.indices[element_global_index], deformable.particles.X.array);
            Matrix<T, dim, dim> Q;
            inplaceGivensQR(Ds, Q); // Ds is dim x manifold_dim  ,  Q is dim x dim
            for (int i = 0; i < indices.size(); i++)
                F[indices[i] - offset] = Q;
        }
        deformable.particles.template add<TM>("F", deformable.particle_range, std::move(F));
    }

    MpmParticleHandleBase<T, dim> sampleOneParticle(const TV& position, const TV& velocity, T density = 1, T total_volume = 1)
    {
        ZIRAN_INFO("Sampling one particle");
        size_t N = 1;
        Range particle_range = mpm.particles.getNextRange(N);
        mpm.particles.add(mpm.particles.X_name(), particle_range, std::move(position));
        mpm.particles.add(mpm.particles.V_name(), particle_range, std::move(velocity));
        mpm.particles.add(mpm.particles.mass_name(), particle_range, total_volume * density / N);
        return MpmParticleHandleBase<T, dim>(mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(), mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(), particle_range, total_volume);
    }

    MpmParticleHandleBase<T, dim> sampleFromVdbFile(std::string filename, T density, T particles_per_cell = (1 << dim))
    {
        VdbLevelSet<float, dim> vdbls(filename); // houdini's vdb files are stored with floats
        Vector<float, dim> min_corner, max_corner;
        vdbls.getBounds(min_corner, max_corner);

        PoissonDisk<T, dim> pd(/*random seed*/ 123, 0, min_corner.template cast<T>(), max_corner.template cast<T>());
#if 0
        pd.setDistanceByParticlesPerCell(mpm.dx, particles_per_cell);

        StdVector<TV> full_samples;
        AxisAlignedAnalyticBox<T, dim> box(min_corner.template cast<T>(), max_corner.template cast<T>());
        if (dim == 3)
            pd.sampleFromPeriodicData(full_samples, [&](const TV& x) { return box.inside(x); });
        else
            pd.sample(full_samples, [&](const TV& x) { return box.inside(x); });
#else
        pd.setDistanceByParticlesPerCell(mpm.dx, particles_per_cell * 2);
        AxisAlignedAnalyticBox<T, dim> box(min_corner.template cast<T>(), max_corner.template cast<T>());
        StdVector<TV> unscaled_samples;
        if (dim == 3)
            pd.sampleFromPeriodicData(unscaled_samples, [&](const TV& x) { return box.inside(x); });
        else
            pd.sample(unscaled_samples, [&](const TV& x) { return box.inside(x); });
        T real_ppc = (T)unscaled_samples.size() / ((max_corner - min_corner).template cast<T>() / mpm.dx).prod();
        ZIRAN_ASSERT(real_ppc > particles_per_cell, "Samples particles should be more dense.");
        T scale = std::pow(real_ppc / particles_per_cell, (T)1 / (T)dim);
        StdVector<TV> full_samples;
        for (auto X : unscaled_samples) {
            TV center = (min_corner + max_corner).template cast<T>() / (T)2;
            TV scaled_X = (X - center) * scale + center;
            if (box.inside(scaled_X))
                full_samples.push_back(scaled_X);
        }
#endif
        T per_particle_volume = box.volume() / (T)full_samples.size();

        StdVector<TV> samples;
        for (auto x : full_samples)
            if (vdbls.inside(x.template cast<float>()))
                samples.push_back(x);
        size_t N = samples.size();
        Range particle_range = mpm.particles.getNextRange(samples.size());
        mpm.particles.add(mpm.particles.X_name(), particle_range, std::move(samples));
        mpm.particles.add(mpm.particles.V_name(), particle_range, TV::Zero());
        mpm.particles.add(mpm.particles.mass_name(), particle_range, per_particle_volume * density);
        return MpmParticleHandleBase<T, dim>(mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(), mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(), particle_range, per_particle_volume * N);
    }

    MpmParticleHandleBase<T, dim> sampleFromVdbFileWithExistingPoints(StdVector<TV>& existing_samples, std::string filename, T density, T particles_per_cell = (1 << dim))
    {
        T per_particle_volume = std::pow(mpm.dx, dim) / particles_per_cell;
        VdbLevelSet<float, dim> vdbls(filename); // houdini's vdb files are stored with floats
        Vector<float, dim> min_corner, max_corner;
        vdbls.getBounds(min_corner, max_corner);

        PoissonDisk<T, dim> pd(/*random seed*/ 123, 0, min_corner.template cast<T>(), max_corner.template cast<T>());
#if 0
        pd.setDistanceByParticlesPerCell(mpm.dx, particles_per_cell);
        StdVector<TV> samples;
        if (dim == 3)
            pd.sampleFromPeriodicData(samples, [&](const TV& x) { return vdbls.inside(x.template cast<float>()); });
        else
            pd.sample(samples, [&](const TV& x) { return vdbls.inside(x.template cast<float>()); });
#else
        pd.setDistanceByParticlesPerCell(mpm.dx, particles_per_cell * 2);
        AxisAlignedAnalyticBox<T, dim> box(min_corner.template cast<T>(), max_corner.template cast<T>());
        StdVector<TV> unscaled_samples;
        if (dim == 3)
            pd.sampleFromPeriodicData(unscaled_samples, [&](const TV& x) { return box.inside(x); });
        else
            pd.sample(unscaled_samples, [&](const TV& x) { return box.inside(x); });
        T real_ppc = (T)unscaled_samples.size() / ((max_corner - min_corner).template cast<T>() / mpm.dx).prod();
        ZIRAN_ASSERT(real_ppc > particles_per_cell, "Samples particles should be more dense.");
        T scale = std::pow(real_ppc / particles_per_cell, (T)1 / (T)dim);
        StdVector<TV> samples;
        StdVector<TV> scaled_samples_in_box;
        for (auto X : unscaled_samples) {
            TV center = (min_corner + max_corner).template cast<T>() / (T)2;
            TV scaled_X = (X - center) * scale + center;
            if (vdbls.inside(scaled_X.template cast<float>()))
                samples.push_back(scaled_X);
            if (box.inside(scaled_X))
                scaled_samples_in_box.push_back(scaled_X);
        }
        per_particle_volume = box.volume() / (T)scaled_samples_in_box.size();
#endif
        existing_samples.insert(existing_samples.end(), samples.begin(), samples.end());
        samples = existing_samples;

        size_t N = samples.size();
        Range particle_range = mpm.particles.getNextRange(samples.size());
        mpm.particles.add(mpm.particles.X_name(), particle_range, std::move(samples));
        mpm.particles.add(mpm.particles.V_name(), particle_range, TV::Zero());
        mpm.particles.add(mpm.particles.mass_name(), particle_range, per_particle_volume * density);
        return MpmParticleHandleBase<T, dim>(mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(), mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(), particle_range, per_particle_volume * N);
    }

    // sample particles from vtk tetmesh
    MpmParticleHandleBase<T, dim> sampleFromVtkFile(std::string filename, const T density)
    {
        StdVector<TV> samples;
        StdVector<Vector<int, 4>> indices;

        // std::string absolute_path = DataDir().absolutePath(filename);
        // readPositionObj(absolute_path, samples);
        std::string absolute_path = DataDir().absolutePath(filename);

        readTetmeshVtk(absolute_path, samples, indices);

        T total_volume = 0;
        ///// TODO: compute tet mesh total volume
        for (size_t i = 0; i < indices.size(); i++) {
            Eigen::Matrix4d A;
            TV p0 = samples[indices[i](0)], p1 = samples[indices[i](1)],
               p2 = samples[indices[i](2)], p3 = samples[indices[i](3)];
            A << 1, p0(0), p0(1), p0(2), 1, p1(0), p1(1), p1(2), 1, p2(0), p2(1),
                p2(2), 1, p3(0), p3(1), p3(2);
            T temp = A.determinant() / (T)6;
            total_volume += (temp > 0 ? temp : (-temp));
        }
        ZIRAN_INFO("Total volume of the tet mesh: ", total_volume);

        T total_mass = total_volume * density;

        ZIRAN_INFO("Read in ", samples.size(), "Particles");
        size_t N = samples.size();

        T mass_per_particle = total_mass / (T)N;

        Range particle_range = mpm.particles.getNextRange(samples.size());
        mpm.particles.add(mpm.particles.X_name(), particle_range,
            std::move(samples));
        mpm.particles.add(mpm.particles.V_name(), particle_range, TV::Zero());
        mpm.particles.add(mpm.particles.mass_name(), particle_range,
            mass_per_particle);
        return MpmParticleHandleBase<T, dim>(
            mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(),
            mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(),
            particle_range, total_volume);
    }

    // sample particles from tetwild ".mesh" tetmesh
    MpmParticleHandleBase<T, dim> sampleFromTetWildFile(std::string filename, const T density)
    {
        StdVector<TV> samples;
        StdVector<Vector<int, 4>> indices;

        // std::string absolute_path = DataDir().absolutePath(filename);
        // readPositionObj(absolute_path, samples);
        std::string absolute_path = DataDir().absolutePath(filename);

        readTetMeshTetWild(absolute_path, samples, indices);

        T total_volume = 0;
        ///// TODO: compute tet mesh total volume
        for (size_t i = 0; i < indices.size(); i++) {
            Eigen::Matrix4d A;
            TV p0 = samples[indices[i](0)], p1 = samples[indices[i](1)],
               p2 = samples[indices[i](2)], p3 = samples[indices[i](3)];
            A << 1, p0(0), p0(1), p0(2), 1, p1(0), p1(1), p1(2), 1, p2(0), p2(1),
                p2(2), 1, p3(0), p3(1), p3(2);
            T temp = A.determinant() / (T)6;
            total_volume += (temp > 0 ? temp : (-temp));
        }
        ZIRAN_INFO("Total volume of the tet mesh: ", total_volume);

        T total_mass = total_volume * density;

        ZIRAN_INFO("Read in ", samples.size(), "Particles");
        size_t N = samples.size();

        T mass_per_particle = total_mass / (T)N;

        Range particle_range = mpm.particles.getNextRange(samples.size());
        mpm.particles.add(mpm.particles.X_name(), particle_range,
            std::move(samples));
        mpm.particles.add(mpm.particles.V_name(), particle_range, TV::Zero());
        mpm.particles.add(mpm.particles.mass_name(), particle_range,
            mass_per_particle);
        return MpmParticleHandleBase<T, dim>(
            mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(),
            mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(),
            particle_range, total_volume);
    }

    // sample particles from obj trimesh with z coordinate ignored
    MpmParticleHandleBase<T, dim> sampleFromObjFile2D(std::string filename,
        const T density)
    {
        StdVector<TV> samples;
        StdVector<Vector<int, 3>> indices;
        std::string absolute_path = DataDir().absolutePath(filename);

        readTrimeshObj(absolute_path, samples, indices);

        T total_volume = 0;
        ///// TODO: compute tri mesh total volume
        for (size_t i = 0; i < indices.size(); i++) {
            Eigen::Matrix3d A;
            TV p0 = samples[indices[i](0)], p1 = samples[indices[i](1)],
               p2 = samples[indices[i](2)];
            A << 1, p0(0), p0(1), 1, p1(0), p1(1), 1, p2(0), p2(1);
            T temp = A.determinant() / (T)2;
            total_volume += (temp > 0 ? temp : (-temp));
        }

        T total_mass = total_volume * density;

        ZIRAN_INFO("Read in ", samples.size(), "Particles");
        size_t N = samples.size();

        T mass_per_particle = total_mass / (T)N;

        Range particle_range = mpm.particles.getNextRange(samples.size());
        mpm.particles.add(mpm.particles.X_name(), particle_range,
            std::move(samples));
        mpm.particles.add(mpm.particles.V_name(), particle_range, TV::Zero());
        mpm.particles.add(mpm.particles.mass_name(), particle_range,
            mass_per_particle);
        return MpmParticleHandleBase<T, dim>(
            mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(),
            mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(),
            particle_range, total_volume);
    }

    void sampleSourceAtTheBeginning(int pos, T density, T particles_per_cell = (1 << dim))
    {
        auto source = dynamic_cast<SourceCollisionObject<T, dim>*>(mpm.collision_objects[pos].get());
        ZIRAN_ASSERT(source != nullptr, "mpm.collision_objects[pos] is not a source collision object!");
        source->poissonSample(mpm.dx, particles_per_cell, density);
    }

    size_t sourceSampleAndPrune(int pos, T density, T particles_per_cell = (1 << dim))
    {
        auto source = dynamic_cast<SourceCollisionObject<T, dim>*>(mpm.collision_objects[pos].get());
        ZIRAN_ASSERT(source != nullptr, "mpm.collision_objects[pos] is not a source collision object!");
        source->sampleAndPrune(mpm.dt, mpm.dx);
        return source->add_samples.size();
    }

    MpmParticleHandleBase<T, dim> getParticlesFromSource(int pos, T density, T particles_per_cell = (1 << dim))
    {
        auto source = dynamic_cast<SourceCollisionObject<T, dim>*>(mpm.collision_objects[pos].get());
        ZIRAN_ASSERT(source != nullptr, "mpm.collision_objects[pos] is not a source collision object!");
        T per_particle_volume = std::pow(mpm.dx, dim) / particles_per_cell;
        size_t N = source->add_samples.size();
        ZIRAN_ASSERT(N);
        Range particle_range = mpm.particles.getNextRange(N);
        mpm.particles.add(mpm.particles.X_name(), particle_range, std::move(source->add_samples));
        mpm.particles.add(mpm.particles.V_name(), particle_range, source->const_material_velocity);
        mpm.particles.add(mpm.particles.mass_name(), particle_range, per_particle_volume * density);
        return MpmParticleHandleBase<T, dim>(mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(), mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(), particle_range, per_particle_volume * N);
    }

    MpmParticleHandleBase<T, dim> sampleFromObjPointCloudFile(std::string filename, const T mass_per_particle, const T volume_per_particle)
    {
        StdVector<TV> samples;
        std::string absolute_path = DataDir().absolutePath(filename);
        readPositionObj(absolute_path, samples);
        size_t N = samples.size();
        Range particle_range = mpm.particles.getNextRange(samples.size());
        mpm.particles.add(mpm.particles.X_name(), particle_range, std::move(samples));
        mpm.particles.add(mpm.particles.V_name(), particle_range, TV::Zero());
        mpm.particles.add(mpm.particles.mass_name(), particle_range, mass_per_particle);
        return MpmParticleHandleBase<T, dim>(mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(), mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(), particle_range, N * volume_per_particle);
    }

    T sampleInAnalyticLevelSetHelper(AnalyticLevelSet<T, dim>& levelset, T particles_per_cell, StdVector<Vector<T, dim>>& samples)
    {
        using TV = Vector<T, dim>;
        ZIRAN_INFO("Sampling ", particles_per_cell, " particles per cell in the levelset");
        TV min_corner, max_corner;
        levelset.getBounds(min_corner, max_corner);

        PoissonDisk<T, dim> pd(/*random seed*/ 123, 0, min_corner, max_corner);
#if 0
        pd.setDistanceByParticlesPerCell(mpm.dx, particles_per_cell);

        if (dim == 3)
            pd.sampleFromPeriodicData(samples, [&](const TV& x) { return levelset.inside(x); });
        else
            pd.sample(samples, [&](const TV& x) { return levelset.inside(x); });

        StdVector<TV> full_samples;
        AxisAlignedAnalyticBox<T, dim> box(min_corner, max_corner);
        if (dim == 3)
            pd.sampleFromPeriodicData(full_samples, [&](const TV& x) { return box.inside(x); });
        else
            pd.sample(full_samples, [&](const TV& x) { return box.inside(x); });
        T per_particle_volume = box.volume() / (T)full_samples.size();
#else
        pd.setDistanceByParticlesPerCell(mpm.dx, particles_per_cell * 2);
        AxisAlignedAnalyticBox<T, dim> box(min_corner, max_corner);
        StdVector<TV> unscaled_samples;
        if (dim == 3)
            pd.sampleFromPeriodicData(unscaled_samples, [&](const TV& x) { return box.inside(x); });
        else
            pd.sample(unscaled_samples, [&](const TV& x) { return box.inside(x); });
        T real_ppc = (T)unscaled_samples.size() / ((max_corner - min_corner) / mpm.dx).prod();
        ZIRAN_ASSERT(real_ppc > particles_per_cell, "Samples particles should be more dense.");
        T scale = std::pow(real_ppc / particles_per_cell, (T)1 / (T)dim);
        StdVector<TV> scaled_samples_in_box;
        for (auto X : unscaled_samples) {
            TV center = (min_corner + max_corner) / (T)2;
            TV scaled_X = (X - center) * scale + center;
            if (levelset.inside(scaled_X))
                samples.push_back(scaled_X);
            if (box.inside(scaled_X))
                scaled_samples_in_box.push_back(scaled_X);
        }
        T per_particle_volume = box.volume() / (T)scaled_samples_in_box.size();
#endif

        T total_volume = per_particle_volume * (T)samples.size();

        return total_volume;
    }

    T sampleWaterInWaterLevelSetOutsideSandLevelSetHelper(AnalyticLevelSet<T, dim>& water_levelset, AnalyticLevelSet<T, dim>& sand_levelset, T particles_per_cell, StdVector<Vector<T, dim>>& samples)
    {
        using TV = Vector<T, dim>;
        ZIRAN_INFO("Sampling ", particles_per_cell, " particles per cell in the levelset");
        TV min_corner, max_corner;
        water_levelset.getBounds(min_corner, max_corner);
        T total_volume = water_levelset.volume();
        PoissonDisk<T, dim> pd(/*random seed*/ 123, 0, min_corner, max_corner);
        pd.setDistanceByParticlesPerCell(mpm.dx, particles_per_cell);

        if (dim == 3)
            pd.sampleFromPeriodicData(samples, [&](const TV& x) { return water_levelset.inside(x) && !sand_levelset.inside(x); });
        else
            pd.sample(samples, [&](const TV& x) { return water_levelset.inside(x) && !sand_levelset.inside(x); });
        return total_volume;
    }

    T sampleInAnalyticLevelSetHelperSpecial(AnalyticLevelSet<T, dim>& levelset, T particles_per_cell, StdVector<Vector<T, dim>>& samples, const Vector<T, dim>& min_bound, const Vector<T, dim>& max_bound)
    {
        using TV = Vector<T, dim>;
        ZIRAN_INFO("Sampling ", particles_per_cell, " particles per cell in the levelset");
        TV min_corner, max_corner;
        levelset.getBounds(min_corner, max_corner);
        T total_volume = levelset.volume();
        PoissonDisk<T, dim> pd(/*random seed*/ 123, 0, min_corner, max_corner);
        pd.setDistanceByParticlesPerCell(mpm.dx, particles_per_cell);
        if (dim == 3)
            pd.sampleFromPeriodicData(samples, [&](const TV& x) { return levelset.inside(x); });
        else
            pd.sample(samples, [&](const TV& x) { return (levelset.inside(x) && (x(0) >= min_bound(0) && x(0) <= max_bound(0)) && (x(1) >= min_bound(1) && x(1) <= max_bound(1))); });
        return total_volume;
    }

    T sampleInAnalyticLevelSetHelperSpecial(AnalyticLevelSet<T, dim>& levelset_in, AnalyticLevelSet<T, dim>& levelset_out, T particles_per_cell, StdVector<Vector<T, dim>>& samples, const Vector<T, dim>& min_bound, const Vector<T, dim>& max_bound)
    {
        using TV = Vector<T, dim>;
        ZIRAN_INFO("Sampling ", particles_per_cell, " particles per cell in the levelset");
        TV min_corner, max_corner;
        levelset_in.getBounds(min_corner, max_corner);
        T total_volume = levelset_in.volume();
        PoissonDisk<T, dim> pd(/*random seed*/ 123, 0, min_corner, max_corner);
        pd.setDistanceByParticlesPerCell(mpm.dx, particles_per_cell);
        if (dim == 3)
            pd.sampleFromPeriodicData(samples, [&](const TV& x) { return levelset_in.inside(x); });
        else
            pd.sample(samples, [&](const TV& x) { return levelset_in.inside(x) && !((levelset_out.inside(x) && (x(0) >= min_bound(0) && x(0) <= max_bound(0)) && (x(1) >= min_bound(1) && x(1) <= max_bound(1)))); });
        return total_volume;
    }

    MpmParticleHandleBase<T, dim> sampleWaterInWaterLevelSetOutsideSandLevelSet(AnalyticLevelSet<T, dim>& water_levelset, AnalyticLevelSet<T, dim>& sand_levelset, T density, T particles_per_cell)
    {
        StdVector<TV> samples;
        T total_volume = sampleWaterInWaterLevelSetOutsideSandLevelSetHelper(water_levelset, sand_levelset, particles_per_cell, samples);
        size_t N = samples.size();
        Range particle_range = mpm.particles.getNextRange(samples.size());
        mpm.particles.add(mpm.particles.X_name(), particle_range, std::move(samples));
        mpm.particles.add(mpm.particles.V_name(), particle_range, TV::Zero());
        mpm.particles.add(mpm.particles.mass_name(), particle_range, total_volume * density / N);
        return MpmParticleHandleBase<T, dim>(mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(), mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(), particle_range, total_volume);
    }

    MpmParticleHandleBase<T, dim> readFromFile(const char* filename, int number, T density)
    {
        // comment another element_measure_name() function first

        StdVector<TV> samples;
        StdVector<TV> vs;

        TV v = TV::Zero();
        T mass, total_mass = 0, vol;
        StdVector<T> vols;
        StdVector<T> masss;
        FILE* f = fopen(filename, "r");
        for (int i = 0; i < number; ++i) {
            TV pos;
            if constexpr (std::is_same<T, float>::value) {
                for (int d = 0; d < dim; ++d)
                    fscanf(f, "%f", &pos(d));
                samples.push_back(pos);
                for (int d = 0; d < dim; ++d)
                    fscanf(f, "%f", &v(d));
                vs.push_back(v);
                fscanf(f, "%f", &mass);
                masss.push_back(mass);
                total_mass += mass;
                fscanf(f, "%f", &vol);
                vols.push_back(vol);
            }
            else {
                for (int d = 0; d < dim; ++d)
                    fscanf(f, "%lf", &pos(d));
                samples.push_back(pos);
                for (int d = 0; d < dim; ++d)
                    fscanf(f, "%lf", &v(d));
                vs.push_back(v);
                fscanf(f, "%lf", &mass);
                masss.push_back(mass);
                total_mass += mass;
                fscanf(f, "%lf", &vol);
                vols.push_back(vol);
            }
        }
        fclose(f);

        size_t N = samples.size();
        Range particle_range = mpm.particles.getNextRange(samples.size());
        mpm.particles.add(mpm.particles.X_name(), particle_range, std::move(samples));
        mpm.particles.add(mpm.particles.V_name(), particle_range, vs);
        mpm.particles.add(mpm.particles.mass_name(), particle_range, masss);
        mpm.particles.add(element_measure_name<T>(), particle_range, vols);
        return MpmParticleHandleBase<T, dim>(mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(), mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(), particle_range, total_mass / density);
    }

    //
    // example usage: MpmParticleHandleBase<T, dim> p_handle = init_helper1.legoSampleInAnalyticLevelSet(box1, fluid_density, /* particles per dimension */ 3, /* sample grid offset */  0.5);
    //
    MpmParticleHandleBase<T, dim> legoSampleInAnalyticLevelSet(AnalyticLevelSet<T, dim>& levelset, T density, int particles_per_dimension, T grid_offset)
    {
        using IV = Vector<int, dim>;
        ZIRAN_ASSERT(particles_per_dimension == 2 || particles_per_dimension == 3);
        ZIRAN_ASSERT(grid_offset >= 0 && grid_offset < 1);
        int ppc = std::pow(particles_per_dimension, dim);
        ZIRAN_ASSERT(ppc == 4 || ppc == 9 || ppc == 8 || ppc == 27);
        ZIRAN_INFO("dd");
        T per_particle_volume = std::pow(mpm.dx, (T)dim) / (T)ppc;
        TV min_corner, max_corner;
        levelset.getBounds(min_corner, max_corner);
        ZIRAN_INFO("min_corner ", min_corner.transpose(), max_corner.transpose());
        min_corner -= 4 * mpm.dx * TV::Ones();
        max_corner += 4 * mpm.dx * TV::Ones();
        IV sample_grid_min_idx = IV::Zero();
        IV sample_grid_max_idx = IV::Zero();
        for (int d = 0; d < dim; d++) {
            sample_grid_min_idx(d) = (int)(min_corner(d) / mpm.dx);
            sample_grid_max_idx(d) = (int)(max_corner(d) / mpm.dx);
        }
        Box<int, dim> localBox(sample_grid_min_idx, sample_grid_max_idx);
        ZIRAN_INFO("sample grid min:", sample_grid_min_idx.transpose());
        ZIRAN_INFO("sample grid max:", sample_grid_max_idx.transpose());

        StdVector<TV> samples;
        for (MaxExclusiveBoxIterator<dim> it(localBox); it.valid(); ++it) {
            TV cell_corner = TV::Zero();
            for (int d = 0; d < dim; d++) cell_corner(d) = it.index(d) * mpm.dx + grid_offset * mpm.dx;
            TV cell_center = cell_corner + TV::Ones() * mpm.dx / 2;
            if (!levelset.inside(cell_center)) continue;

            if constexpr (dim == 2) {
                if (particles_per_dimension == 2) {
                    for (int i = 0; i < 2; i++)
                        for (int j = 0; j < 2; j++)
                            samples.emplace_back(cell_corner + TV(0.25 * mpm.dx + i * 0.5 * mpm.dx, 0.25 * mpm.dx + j * 0.5 * mpm.dx));
                }
                else if (particles_per_dimension == 3) {
                    for (int i = 0; i < 3; i++)
                        for (int j = 0; j < 3; j++)
                            samples.emplace_back(cell_corner + TV(mpm.dx / 6 + i * mpm.dx / 3, mpm.dx / 6 + j * mpm.dx / 3));
                }
            }
            else if constexpr (dim == 3) {
                if (particles_per_dimension == 2) {
                    for (int i = 0; i < 2; i++)
                        for (int j = 0; j < 2; j++)
                            for (int k = 0; k < 2; k++)
                                samples.emplace_back(cell_corner + TV(0.25 * mpm.dx + i * 0.5 * mpm.dx, 0.25 * mpm.dx + j * 0.5 * mpm.dx, 0.25 * mpm.dx + k * 0.5 * mpm.dx));
                }
                else if (particles_per_dimension == 3) {
                    for (int i = 0; i < 3; i++)
                        for (int j = 0; j < 3; j++)
                            for (int k = 0; k < 3; k++)
                                samples.emplace_back(cell_corner + TV(mpm.dx / 6 + i * mpm.dx / 3, mpm.dx / 6 + j * mpm.dx / 3, mpm.dx / 6 + k * mpm.dx / 3));
                }
            }
        }
        StdVector<TV> prune_samples;
        for (auto X : samples)
            if (levelset.inside(X)) prune_samples.push_back(X);
        samples = prune_samples;
        ZIRAN_INFO("sampled particle count: ", samples.size());

        T total_volume = per_particle_volume * (T)samples.size();
        size_t N = samples.size();
        ZIRAN_INFO("total volume: ", total_volume);

        Range particle_range = mpm.particles.getNextRange(samples.size());
        mpm.particles.add(mpm.particles.X_name(), particle_range, std::move(samples));
        mpm.particles.add(mpm.particles.V_name(), particle_range, TV::Zero());
        mpm.particles.add(mpm.particles.mass_name(), particle_range, total_volume * density / N);
        ZIRAN_INFO("dd");

        return MpmParticleHandleBase<T, dim>(mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(), mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(), particle_range, total_volume);
    }

    MpmParticleHandleBase<T, dim> sampleInAnalyticLevelSet(AnalyticLevelSet<T, dim>& levelset, T density, T particles_per_cell)
    {
        StdVector<TV> samples;
        T total_volume = sampleInAnalyticLevelSetHelper(levelset, particles_per_cell, samples);
        size_t N = samples.size();
        Range particle_range = mpm.particles.getNextRange(samples.size());
        mpm.particles.add(mpm.particles.X_name(), particle_range, std::move(samples));
        mpm.particles.add(mpm.particles.V_name(), particle_range, TV::Zero());
        mpm.particles.add(mpm.particles.mass_name(), particle_range, total_volume * density / N);
        return MpmParticleHandleBase<T, dim>(mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(), mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(), particle_range, total_volume);
    }

    MpmParticleHandleBase<T, dim> sampleFromParticlesList(StdVector<TV>& particles_list, T density, T particles_per_cell, T dx)
    {
        StdVector<TV> samples;
        for (auto particle : particles_list)
            samples.push_back(particle);
        T total_volume = std::pow(dx, dim) / T(particles_per_cell) * T(samples.size());
        size_t N = samples.size();
        Range particle_range = mpm.particles.getNextRange(samples.size());
        mpm.particles.add(mpm.particles.X_name(), particle_range, std::move(samples));
        mpm.particles.add(mpm.particles.V_name(), particle_range, TV::Zero());
        mpm.particles.add(mpm.particles.mass_name(), particle_range, total_volume * density / N);
        return MpmParticleHandleBase<T, dim>(mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(), mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(), particle_range, total_volume);
    }

    MpmParticleHandleBase<T, dim> sampleInAnalyticLevelSet(AnalyticLevelSet<T, dim>& levelset, T density, T particles_per_cell, const StdVector<TV>& samples_in)
    {
        StdVector<TV> samples;
        T total_volume = sampleInAnalyticLevelSetHelper(levelset, particles_per_cell, samples);
        total_volume *= (T)samples_in.size() / samples.size();
        samples = samples_in;
        size_t N = samples.size();
        Range particle_range = mpm.particles.getNextRange(samples.size());
        mpm.particles.add(mpm.particles.X_name(), particle_range, std::move(samples));
        mpm.particles.add(mpm.particles.V_name(), particle_range, TV::Zero());
        mpm.particles.add(mpm.particles.mass_name(), particle_range, total_volume * density / N);
        return MpmParticleHandleBase<T, dim>(mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(), mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(), particle_range, total_volume);
    }

    MpmParticleHandleBase<T, dim> sampleInAnalyticLevelSetSpecial(AnalyticLevelSet<T, dim>& levelset, T density, T particles_per_cell, const Vector<T, dim>& min_bound, const Vector<T, dim>& max_bound)
    {
        StdVector<TV> samples;
        T total_volume = sampleInAnalyticLevelSetHelperSpecial(levelset, particles_per_cell, samples, min_bound, max_bound);
        size_t N = samples.size();
        Range particle_range = mpm.particles.getNextRange(samples.size());
        mpm.particles.add(mpm.particles.X_name(), particle_range, std::move(samples));
        mpm.particles.add(mpm.particles.V_name(), particle_range, TV::Zero());
        mpm.particles.add(mpm.particles.mass_name(), particle_range, total_volume * density / N);
        return MpmParticleHandleBase<T, dim>(mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(), mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(), particle_range, total_volume);
    }

    MpmParticleHandleBase<T, dim> sampleInAnalyticLevelSetSpecial(AnalyticLevelSet<T, dim>& levelset_in, AnalyticLevelSet<T, dim>& levelset_out, T density, T particles_per_cell, const Vector<T, dim>& min_bound, const Vector<T, dim>& max_bound)
    {
        StdVector<TV> samples;
        T total_volume = sampleInAnalyticLevelSetHelperSpecial(levelset_in, levelset_out, particles_per_cell, samples, min_bound, max_bound);
        size_t N = samples.size();
        Range particle_range = mpm.particles.getNextRange(samples.size());
        mpm.particles.add(mpm.particles.X_name(), particle_range, std::move(samples));
        mpm.particles.add(mpm.particles.V_name(), particle_range, TV::Zero());
        mpm.particles.add(mpm.particles.mass_name(), particle_range, total_volume * density / N);
        return MpmParticleHandleBase<T, dim>(mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(), mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(), particle_range, total_volume);
    }

    MpmParticleHandleBase<T, dim> sampleInAnalyticLevelSetWithExistingPoints(StdVector<TV>& existing_samples, AnalyticLevelSet<T, dim>& levelset, T density, T particles_per_cell)
    {
        StdVector<TV> samples;
        T total_volume = sampleInAnalyticLevelSetHelper(levelset, particles_per_cell, samples);
        existing_samples.insert(existing_samples.end(), samples.begin(), samples.end());
        samples = existing_samples;
        size_t N = samples.size();
        Range particle_range = mpm.particles.getNextRange(samples.size());
        mpm.particles.add(mpm.particles.X_name(), particle_range, std::move(samples));
        mpm.particles.add(mpm.particles.V_name(), particle_range, TV::Zero());
        mpm.particles.add(mpm.particles.mass_name(), particle_range, total_volume * density / N);
        return MpmParticleHandleBase<T, dim>(mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(), mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(), particle_range, total_volume);
    }

    MpmParticleHandleBase<T, dim> sampleInAnalyticLevelSetPartial0(AnalyticLevelSet<T, dim>& levelset, T density, T particles_per_cell)
    {
        StdVector<TV> _samples;
        T total_volume = sampleInAnalyticLevelSetHelper(levelset, particles_per_cell, _samples);
        StdVector<TV> samples;
        for (const auto& pos : _samples) {
            T y = pos[1];
            if (y < 5)
                samples.push_back(pos);
        }
        size_t N = samples.size();
        Range particle_range = mpm.particles.getNextRange(samples.size());
        mpm.particles.add(mpm.particles.X_name(), particle_range, std::move(samples));
        mpm.particles.add(mpm.particles.V_name(), particle_range, TV::Zero());
        mpm.particles.add(mpm.particles.mass_name(), particle_range, total_volume * density / N);
        return MpmParticleHandleBase<T, dim>(mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(), mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(), particle_range, total_volume);
    }

    MpmParticleHandleBase<T, dim> sampleInAnalyticLevelSetPartial(AnalyticLevelSet<T, dim>& levelset, T density, T particles_per_cell)
    {
        StdVector<TV> _samples;
        T total_volume = sampleInAnalyticLevelSetHelper(levelset, particles_per_cell, _samples);
        StdVector<TV> samples;
        for (const auto& pos : _samples) {
            T sqrt2 = std::sqrt((T)2.0);
            T x = pos[0];
            T y = pos[1];
            if (y > -x + 2.5 + 0.2 * sqrt2 && y > x - 0.1 + 0.2 * sqrt2)
                samples.push_back(pos);
        }
        size_t N = samples.size();
        Range particle_range = mpm.particles.getNextRange(samples.size());
        mpm.particles.add(mpm.particles.X_name(), particle_range, std::move(samples));
        mpm.particles.add(mpm.particles.V_name(), particle_range, TV::Zero());
        mpm.particles.add(mpm.particles.mass_name(), particle_range, total_volume * density / N);
        return MpmParticleHandleBase<T, dim>(mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(), mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(), particle_range, total_volume);
    }

    MpmParticleHandleBase<T, dim> sampleInAnalyticLevelSetPartial2(AnalyticLevelSet<T, dim>& levelset, T density, T particles_per_cell)
    {
        StdVector<TV> _samples;
        T total_volume = sampleInAnalyticLevelSetHelper(levelset, particles_per_cell, _samples);
        StdVector<TV> samples;
        for (const auto& pos : _samples) {
            T y = pos[1];
            if (y < 3)
                samples.push_back(pos);
        }
        size_t N = samples.size();
        Range particle_range = mpm.particles.getNextRange(samples.size());
        mpm.particles.add(mpm.particles.X_name(), particle_range, std::move(samples));
        mpm.particles.add(mpm.particles.V_name(), particle_range, TV::Zero());
        mpm.particles.add(mpm.particles.mass_name(), particle_range, total_volume * density / N);
        return MpmParticleHandleBase<T, dim>(mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(), mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(), particle_range, total_volume);
    }

    MpmParticleHandleBase<T, dim> sampleInAnalyticLevelSetPartial3(AnalyticLevelSet<T, dim>& levelset, T density, T particles_per_cell, TV center, T radius)
    {
        ZIRAN_WARN("sample particles");
        StdVector<TV> _samples;
        T total_volume = sampleInAnalyticLevelSetHelper(levelset, particles_per_cell, _samples);
        StdVector<TV> samples;
        for (const auto& pos : _samples) {
            if ((pos - center).norm() > radius)
                samples.push_back(pos);
        }
        size_t N = samples.size();
        Range particle_range = mpm.particles.getNextRange(samples.size());
        mpm.particles.add(mpm.particles.X_name(), particle_range, std::move(samples));
        mpm.particles.add(mpm.particles.V_name(), particle_range, TV::Zero());
        mpm.particles.add(mpm.particles.mass_name(), particle_range, total_volume * density / N);
        ZIRAN_WARN("sample particles ends");
        return MpmParticleHandleBase<T, dim>(mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(), mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(), particle_range, total_volume);
    }

    MpmParticleHandleBase<T, dim> sampleOnLine(const TV& A, const TV& B, T density, T h, T thickness)
    {
        int N = int((A - B).norm() / h) + 1;
        TV d = (B - A) / (N - 1);
        StdVector<TV> samples;
        for (int i = 0; i < N; i++)
            samples.push_back(A + i * d);
        T total_volume = (N - 1) * h * thickness;
        Range particle_range = mpm.particles.getNextRange(samples.size());
        mpm.particles.add(mpm.particles.X_name(), particle_range, std::move(samples));
        mpm.particles.add(mpm.particles.V_name(), particle_range, TV::Zero());
        mpm.particles.add(mpm.particles.mass_name(), particle_range, total_volume * density / N);
        return MpmParticleHandleBase<T, dim>(mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(), mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(), particle_range, total_volume);
    }

    MpmParticleHandleBase<T, dim> samplePackedSpheresInAnalyticLevelSet(AnalyticLevelSet<T, dim>& levelset, T density, T particles_per_cell, T sphere_radius, T gap)
    {
        TV min_corner, max_corner;
        levelset.getBounds(min_corner, max_corner);
        T total_volume = levelset.volume();
        PoissonDisk<T, dim> pd(/*random seed*/ 123, sphere_radius * 2 + gap, min_corner, max_corner);
        StdVector<TV> center_samples;
        if (dim == 3)
            pd.sampleFromPeriodicData(center_samples, [&](const TV& x) { return levelset.inside(x); });
        else
            pd.sample(center_samples, [&](const TV& x) { return levelset.inside(x); });
        size_t N_spheres = center_samples.size();

        ZIRAN_INFO("N spheres:", N_spheres);

        StdVector<TV> samples;
        for (size_t k = 0; k < N_spheres; k++) {
            const TV center = center_samples[k];
            Sphere<T, dim> local_sphere(center, sphere_radius);
            PoissonDisk<T, dim> pd_local(/*random seed*/ 123, 0, center - sphere_radius * TV::Ones(), center + sphere_radius * TV::Ones());
            pd_local.setDistanceByParticlesPerCell(mpm.dx, particles_per_cell);

            StdVector<TV> local_samples;
            if (dim == 3)
                pd_local.sampleFromPeriodicData(local_samples, [&](const TV& x) { return local_sphere.inside(x); });
            else
                pd_local.sample(local_samples, [&](const TV& x) { return local_sphere.inside(x); });

            for (auto s : local_samples)
                samples.push_back(s);
        }
        size_t N = samples.size();
        ZIRAN_INFO("sampled particle: ", N);
        Range particle_range = mpm.particles.getNextRange(samples.size());
        mpm.particles.add(mpm.particles.X_name(), particle_range, std::move(samples));
        mpm.particles.add(mpm.particles.V_name(), particle_range, TV::Zero());
        mpm.particles.add(mpm.particles.mass_name(), particle_range, total_volume * density / N);
        return MpmParticleHandleBase<T, dim>(mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(), mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(), particle_range, total_volume);
    }

    MpmParticleHandleBase<T, dim> addAlexeyMagicSnowBall(T density, int number_of_particles, const Sphere<T, dim>& guide_sphere, TV velocity, Vector<T, 3> angular_velocity)
    {
        AngularVelocity<T, dim> angular;
        angular.set(angular_velocity);
        T ball_volume = guide_sphere.volume();
        Sphere<T, dim> padded_ball(guide_sphere.center, guide_sphere.radius * 1.1);

        RandomNumber<T> random(123);
#define PERTURB(x) random.randReal(1 - x, 1 + x)
#define UNIFORM(x, y) random.randReal(x, y)

        const T volume_per_particle = ball_volume / number_of_particles;
        StdVector<Sphere<T, dim>> spheres;
        TV zero = TV::Zero();
        for (int i = 0; i < 500; i++) {
            TV offset = random.randInBall(zero, padded_ball.radius);
            if (offset.norm() < padded_ball.radius * 0.5 || offset.norm() > padded_ball.radius * 0.9) {
                i--;
                continue;
            }
            T radius = random.randReal(0, padded_ball.radius - offset.norm());
            if (radius < padded_ball.radius * 0.1) {
                i--;
                continue;
            }
            spheres.emplace_back(offset + padded_ball.center, radius);
        }

        const T E = 100, nu = 0.25;

        const T h = 10;
        const T a = 1.025;
        const T ea = std::exp((a - 1) * h);

        StdVector<TV> Xs;
        StdVector<TV> Vs;
        StdVector<T> masses;
        StdVector<CorotatedElasticity<T, dim>> constitutive_models;
        StdVector<SnowPlasticity<T>> plasticity_models;

        T total_volume = 0;
        for (int i = 0; i < number_of_particles; i++) {
            TV X = random.randInBall(padded_ball.center, padded_ball.radius);

            bool inside = false;
            int count = 0;
            for (size_t j = 0; j < spheres.size(); j++) {
                inside |= spheres[j].inside(X);
                count += spheres[j].inside(X);
            }
            inside |= guide_sphere.inside(X);
            if (!inside) {
                i--;
                continue;
            }

            TV v = velocity;
            v += angular.cross(X - guide_sphere.center);
            T mass = density * volume_per_particle * PERTURB(0.5);
            total_volume += volume_per_particle;

            CorotatedElasticity<T, dim> constitutive_model(E * PERTURB(0.5), nu * PERTURB(0.1));
            SnowPlasticity<T> plasticity(h * PERTURB(0.01), .025 * PERTURB(0.01), .005 * PERTURB(0.01));

            const int n = 10;
            T array[n] = { .1, .2, .3, .4, .5, .6, .7, .8, .9, 1 };

            int c = 0;
            for (int i = 0; i < n; i++)
                if ((X - guide_sphere.center).norm() > array[i] * guide_sphere.radius) {
                    c++;
                    mass *= a;
                    constitutive_model.lambda *= ea;
                    constitutive_model.mu *= ea;
                }

            T mult = 1;
            for (int i = 0; i < n - c; i++)
                mult *= (1 / a);

            if (UNIFORM(0, 1) > mult)
                continue;
            Xs.emplace_back(X);
            Vs.emplace_back(v);
            masses.emplace_back(mass);
            constitutive_models.emplace_back(constitutive_model);
            plasticity_models.emplace_back(plasticity);
        }

        for (int k = 1; k <= 500; k++) {
            const TV offset = random.randInBall(zero, padded_ball.radius * 5);
            const T radius = random.randReal(padded_ball.radius, padded_ball.radius * 10);
            Sphere<T, dim> sphere_big(offset + padded_ball.center, radius + mpm.dx);
            Sphere<T, dim> sphere_small(offset + padded_ball.center, radius - mpm.dx);
            const TV offset_new = random.randInBall(zero, padded_ball.radius * 5);
            const T radius_new = random.randReal(padded_ball.radius, padded_ball.radius * 10);
            Sphere<T, dim> sphere_new(offset_new + padded_ball.center, radius_new);
            if (offset_new.norm() + guide_sphere.radius < radius_new)
                continue;
            if ((offset - offset_new).norm() + radius < radius_new)
                continue;
            for (size_t i = 0; i < Xs.size(); i++)
                if (sphere_big.inside(Xs[i]) && !sphere_small.inside(Xs[i]) && sphere_new.inside(Xs[i])) {
                    constitutive_models[i].lambda *= 0.5;
                    constitutive_models[i].mu *= 0.5;
                }
        }
#undef PERTURB
#undef UNIFORM

        Range particle_range = mpm.particles.getNextRange(Xs.size());

        mpm.particles.add(mpm.particles.X_name(), particle_range, std::move(Xs));
        mpm.particles.add(mpm.particles.V_name(), particle_range, std::move(Vs));
        mpm.particles.add(mpm.particles.mass_name(), particle_range, std::move(masses));

        using TConst = CorotatedElasticity<T, dim>;
        using TPlastic = SnowPlasticity<T>;
        using TM = Matrix<T, dim, dim>;
        using TStrain = TM;

        FBasedMpmForceHelper<TConst>& helper = mpm.getMpmForce()->getHelper();
        mpm.particles.add(TConst::name(), particle_range, std::move(constitutive_models));
        mpm.particles.add(element_measure_name<T>(), particle_range, total_volume / particle_range.length());
        mpm.particles.add(helper.F_name(), particle_range, TM::Identity());
        mpm.particles.add(helper.constitutive_model_scratch_name(), particle_range, typename FBasedMpmForceHelper<TConst>::Scratch());

        PlasticityApplier<TConst, TPlastic, TStrain>* plasticity_model = nullptr;
        for (auto& p : mpm.getPlasticityAppliers()) {
            plasticity_model = dynamic_cast<PlasticityApplier<TConst, TPlastic, TStrain>*>(p.get());
            if (plasticity_model && plasticity_model->strain_name.name == helper.F_name().name)
                break;
            else
                plasticity_model = nullptr;
        }
        if (plasticity_model == nullptr)
            mpm.plasticity_appliers.push_back(std::make_unique<PlasticityApplier<TConst, TPlastic, TStrain>>(helper.F_name().name));
        mpm.particles.add(TPlastic::name(), particle_range, std::move(plasticity_models));

        return MpmParticleHandleBase<T, dim>(mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(), mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(), particle_range, total_volume);
    }

    template <class TMesh>
    MpmParticleHandleBase<T, dim> sampleQuadratureParticles(const DeformableObjectHandle<T, dim, TMesh>& deformable, const T thickness, const T density, int runtime_qc)
    {
        if (runtime_qc == 1) {
            return sampleQuadratureParticlesHelper<TMesh, 1>(deformable, thickness, density, runtime_qc);
        }
        else if (runtime_qc == 2) {
            return sampleQuadratureParticlesHelper<TMesh, 2>(deformable, thickness, density, runtime_qc);
        }
        else if (runtime_qc == 3) {
            return sampleQuadratureParticlesHelper<TMesh, 3>(deformable, thickness, density, runtime_qc);
        }
        else if (runtime_qc == 4) {
            return sampleQuadratureParticlesHelper<TMesh, 4>(deformable, thickness, density, runtime_qc);
        }
        else if (runtime_qc > 4) {
            return sampleQuadratureParticlesHelper<TMesh, Eigen::Dynamic>(deformable, thickness, density, runtime_qc);
        }
        ZIRAN_ASSERT(false, "Quadrature Points must be a positive integer");
    }

    MeshHandle<T, 3, SimplexMesh<1>> addBendingSprings(const DeformableObjectHandle<T, 3, SimplexMesh<1>>& hair3d)
    {
        std::vector<int> path_vertex_counts;
        CurveFinder c(hair3d.mesh->indices);
        StdVector<int> pids;
        c.constructPaths(pids, path_vertex_counts, false);

        std::shared_ptr<SimplexMesh<1>> mesh(new SimplexMesh<1>());

        ZIRAN_INFO("number of paths :", path_vertex_counts.size());

        int start_id = 0;
        for (size_t i = 0; i < path_vertex_counts.size(); ++i) {
            int N = path_vertex_counts[i];
            ZIRAN_INFO("path has ", N, " points");
            for (int k = 0; k < N - 2; k++) {
                mesh->indices.emplace_back(pids[start_id + k], pids[start_id + k + 2]);
            }
            start_id += N;
        }

        std::vector<int> cycle_vertex_counts;
        pids.clear();
        c.constructCycles(pids, cycle_vertex_counts);

        ZIRAN_INFO("number of cyclies :", cycle_vertex_counts.size());

        start_id = 0;
        for (size_t i = 0; i < cycle_vertex_counts.size(); ++i) {
            int N = cycle_vertex_counts[i];

            ZIRAN_INFO("cycle has ", N, " points");

            if (N < 3)
                continue;
            for (int k = 0; k < N; k++) {
                int myself = k + start_id;
                int nextnext = (k + 2) % N + start_id;
                mesh->indices.emplace_back(pids[myself], pids[nextnext]);
            }
            start_id += N;
        }

        MeshHandle<T, 3, SimplexMesh<1>> handle(hair3d.particles, mesh, hair3d.particle_range);
        handle.particle_index_offset = 0;
        return handle;
    }

    MeshHandle<T, 2, SimplexMesh<1>> addBendingSprings2D(const DeformableObjectHandle<T, 2, SimplexMesh<1>>& hair2d)
    {
        int offset = hair2d.particle_index_offset;

        std::vector<int> path_vertex_counts;
        CurveFinder c(hair2d.mesh->indices);
        StdVector<int> pids;
        c.constructPaths(pids, path_vertex_counts, false);

        std::shared_ptr<SimplexMesh<1>> mesh(new SimplexMesh<1>());

        ZIRAN_INFO("number of paths :", path_vertex_counts.size());

        int start_id = 0;
        for (size_t i = 0; i < path_vertex_counts.size(); ++i) {
            int N = path_vertex_counts[i];
            ZIRAN_INFO("path has ", N, " points");
            for (int k = 0; k < N - 2; k++) {
                mesh->indices.emplace_back(pids[start_id + k] + offset, pids[start_id + k + 2] + offset);
            }
            start_id += N;
        }

        std::vector<int> cycle_vertex_counts;
        pids.clear();
        c.constructCycles(pids, cycle_vertex_counts);

        ZIRAN_INFO("number of cyclies :", cycle_vertex_counts.size());

        start_id = 0;
        for (size_t i = 0; i < cycle_vertex_counts.size(); ++i) {
            int N = cycle_vertex_counts[i];

            ZIRAN_INFO("cycle has ", N, " points");

            if (N < 3)
                continue;
            for (int k = 0; k < N; k++) {
                int myself = k + start_id;
                int nextnext = (k + 2) % N + start_id;
                mesh->indices.emplace_back(pids[myself] + offset, pids[nextnext] + offset);
            }
            start_id += N;
        }

        MeshHandle<T, 2, SimplexMesh<1>> handle(hair2d.particles, mesh, hair2d.particle_range);
        handle.particle_index_offset = 0;
        return handle;
    }

    MeshHandle<T, 3, SimplexMesh<1>> addBendingSpringsForTriangles(const DeformableObjectHandle<T, 3, SimplexMesh<2>>& cloth3d)
    {
        std::shared_ptr<SimplexMesh<1>> mesh(new SimplexMesh<1>());

        int offset = cloth3d.particle_index_offset;
        StdVector<Vector<int, 2>> touching_triangles;
        cloth3d.mesh->constructListOfFaceNeighboringElements(touching_triangles);
        ZIRAN_INFO("Adding bending springs for ", touching_triangles.size(), " edge touching triangles.");
        for (auto pair : touching_triangles) {
            int triA = pair(0), triB = pair(1);
            HashTable<int, int> count;
            for (int a = 0; a < 3; a++)
                count.insert(cloth3d.mesh->indices[triA](a), 0).value += 1;
            for (int b = 0; b < 3; b++)
                count.insert(cloth3d.mesh->indices[triB](b), 0).value += 1;
            int A = -1, B = -1;
            for (auto p : count) {
                if (p.value == 1) {
                    if (A == -1)
                        A = p.key;
                    else
                        B = p.key;
                }
            }
            ZIRAN_ASSERT(A != -1 && B != -1);
            mesh->indices.emplace_back(A + offset, B + offset);
        }

        MeshHandle<T, 3, SimplexMesh<1>> handle(cloth3d.particles, mesh, cloth3d.particle_range);
        handle.particle_index_offset = 0;
        return handle;
    }

    MeshHandle<T, 3, SimplexMesh<1>> addTorsionSprings(const DeformableObjectHandle<T, 3, SimplexMesh<1>>& hair3d)
    {
        std::vector<int> path_vertex_counts;
        CurveFinder c(hair3d.mesh->indices);
        StdVector<int> pids;
        c.constructPaths(pids, path_vertex_counts, false);

        std::shared_ptr<SimplexMesh<1>> mesh(new SimplexMesh<1>());

        ZIRAN_INFO("number of paths :", path_vertex_counts.size());

        int start_id = 0;
        for (size_t i = 0; i < path_vertex_counts.size(); ++i) {
            int N = path_vertex_counts[i];
            ZIRAN_INFO("path has ", N, " points");
            for (int k = 0; k < N - 3; k++) {
                mesh->indices.emplace_back(pids[start_id + k], pids[start_id + k + 3]);
            }
            start_id += N;
        }

        std::vector<int> cycle_vertex_counts;
        pids.clear();
        c.constructCycles(pids, cycle_vertex_counts);

        ZIRAN_INFO("number of cyclies :", cycle_vertex_counts.size());

        start_id = 0;
        for (size_t i = 0; i < cycle_vertex_counts.size(); ++i) {
            int N = cycle_vertex_counts[i];

            ZIRAN_INFO("cycle has ", N, " points");

            if (N < 3)
                continue;
            for (int k = 0; k < N; k++) {
                int myself = k + start_id;
                int nextnext = (k + 3) % N + start_id;
                mesh->indices.emplace_back(pids[myself], pids[nextnext]);
            }
            start_id += N;
        }

        MeshHandle<T, 3, SimplexMesh<1>> handle(hair3d.particles, mesh, hair3d.particle_range);
        handle.particle_index_offset = 0;
        return handle;
    }

    MeshHandle<T, 3, SimplexMesh<3>> addBendingTetrahedrons(const DeformableObjectHandle<T, 3, SimplexMesh<1>>& hair3d)
    {
        std::shared_ptr<SimplexMesh<3>> mesh(new SimplexMesh<3>());
        MeshHandle<T, 3, SimplexMesh<3>> handle(hair3d.particles, mesh, hair3d.particle_range);
        ZIRAN_ASSERT(false, "not implemented and wont be implemented");
        return handle;
    }

    template <class TMesh, int quadrature_count>
    MpmParticleHandleBase<T, dim> sampleQuadratureParticlesHelper(const DeformableObjectHandle<T, dim, TMesh>& deformable, const T thickness, const T density, int runtime_qc = quadrature_count)
    {
        RandomNumber<T> rand;

        static const int manifold_dim = TMesh::manifold_dim;

        using TM = Matrix<T, dim, dim>;
        using TM2 = Matrix<T, dim, manifold_dim>;
        using IV = Eigen::Matrix<int, manifold_dim + 1, 1>;
        using FemForce = FemCodimensional<T, manifold_dim, dim, quadrature_count>;
        auto& elements = deformable.elements;

        /*
            // ELement:  tangent_F (done)
                         tangent_dF (done)
                         quadrature_points (done)
                         barycentric_weights (done)
                         element_measure (done)
                         Dm_inv (done)

               Particl:  parent_element (done )
                         element_measure (done)
                         cotangent (doned)
                         mass  (done
                         position done(
                         velocity done
                         constitutive model (will be called later on the handle when the handle adds CotangentBasedMpmForceHelper)

               ElementV  mass (done)

               Other     Add FemCodimensional force (done)
            */

        ZIRAN_ASSERT(runtime_qc != Eigen::Dynamic);
        auto element_measure_name = elements.element_measure_name();

        // new element attributes

        StdVector<TM2> tangent_F;
        StdVector<TM2> tangent_dF;
        StdVector<Vector<int, quadrature_count>> quadrature_points;
        StdVector<Matrix<T, manifold_dim + 1, quadrature_count>> barycentric_weights;

        int element_count = deformable.element_range.length();
        tangent_F.resize(element_count);
        tangent_dF.resize(element_count);
        quadrature_points.resize(element_count);
        barycentric_weights.resize(element_count);

        // get exisiting VP starting index
        auto& VP = deformable.particles.add(FemForce::VP_name());
        int quad_pid = VP.size();

        auto& vertex_measure = deformable.particles.add(element_measure_name, deformable.particle_range, 0);
        int vertex_offset = deformable.particle_range.lower - vertex_measure.valueId(deformable.particle_range.lower);

        // get existing tan
        auto& tF = elements.add(FemForce::tangent_F_name());
        int tF_start = tF.size();
        Range quadrature_point_range = deformable.particles.getNextRange(element_count * runtime_qc);

        // particle appender scope
        {
            auto ap = deformable.particles.appender(FemForce::parent_element_name(), FemForce::element_measure_name(), FemForce::cotangent_name());

            int e_count = 0;
            for (auto iter = elements.subsetIter(DisjointRanges{ deformable.element_range }, element_measure_name); iter; ++iter, ++e_count) {
                int element_global_index = iter.entryId();
                Matrix<T, dim, manifold_dim + 1> element_vertices_X = elements.getVertices(elements.indices[element_global_index], deformable.particles.X.array);
                Matrix<T, dim, manifold_dim + 1> element_vertices_V = elements.getVertices(elements.indices[element_global_index], deformable.particles.V.array);

                const T& Vt = iter.template get<0>(); // tangent element volume
                T V = Vt * thickness;
                if (dim - manifold_dim == 2)
                    V *= M_PI * thickness;
                T Vp = V / (runtime_qc + manifold_dim + 1);
                T mp = Vp * density;

                quadrature_points[e_count].resize(runtime_qc);
                Matrix<T, manifold_dim + 1, quadrature_count>& barys = barycentric_weights[e_count];
                barys.resize(manifold_dim + 1, runtime_qc);

                for (int p = 0; p < runtime_qc; p++)
                    quadrature_points[e_count](p) = quad_pid++;

                sampleBarycentricWeights(runtime_qc, barys, rand);

                Matrix<T, dim, quadrature_count> quadrature_points_X = element_vertices_X * barys;
                Matrix<T, dim, quadrature_count> quadrature_points_V = element_vertices_V * barys;

                auto Ds = elements.dS(elements.indices[element_global_index], deformable.particles.X.array);
                Matrix<T, dim, dim> Q;
                inplaceGivensQR(Ds, Q); // Ds is dim x manifold_dim  ,  Q is dim x dim
                TM ct = Q;
                for (int p = 0; p < runtime_qc; p++) {
                    ap.append(mp, quadrature_points_X.col(p), quadrature_points_V.col(p), tF_start + e_count, Vp, ct);
                }

                IV element_node_global_indices = elements.indices[element_global_index];
                for (int i = 0; i < manifold_dim + 1; i++) {
                    int vertex_id = element_node_global_indices(i);
                    deformable.particles.mass[vertex_id] += mp;
                    vertex_measure[vertex_id - vertex_offset] += Vp;
                }
            }
            elements.add(FemForce::quadrature_points_name(), deformable.element_range, std::move(quadrature_points));
            elements.add(FemForce::barycentric_weights_name(), deformable.element_range, std::move(barycentric_weights));
            elements.add(FemForce::tangent_F_name(), deformable.element_range, std::move(tangent_F));
            elements.add(FemForce::tangent_dF_name(), deformable.element_range, std::move(tangent_dF));
        }

        deformable.particles.add(FemForce::VP_name(), quadrature_point_range, TM::Zero());
        deformable.particles.add(FemForce::VdP_name(), quadrature_point_range, TM::Zero());

        FemForce* f = nullptr;
        for (auto& force : scene.forces)
            if ((f = dynamic_cast<FemForce*>(force.get())))
                break;

        if (f == nullptr) {
            scene.forces.emplace_back(std::make_unique<FemForce>(elements, deformable.particles.X.array, deformable.particles));
            mpm.end_time_step_callbacks.emplace_back([&VP,
                                                         &elements,
                                                         &particles = mpm.particles](int, int) {
                auto& X = particles.X.array;
                auto& V = particles.V.array;
                elements.parallel_for([&](const auto& indices, const auto& barys, const auto& qp) {
                    Matrix<T, dim, manifold_dim + 1> element_vertices_X = elements.getVertices(indices, X);
                    Matrix<T, dim, manifold_dim + 1> element_vertices_V = elements.getVertices(indices, V);
                    Matrix<T, dim, quadrature_count> quadrature_points_X = element_vertices_X * barys;
                    Matrix<T, dim, quadrature_count> quadrature_points_V = element_vertices_V * barys;
                    for (int p = 0; p < qp.rows(); p++) {
                        VP[qp(p)].col(0) = quadrature_points_X.col(p);
                        VP[qp(p)].col(1) = quadrature_points_V.col(p);
                    }
                },
                    elements.indices_name(), FemForce::barycentric_weights_name(), FemForce::quadrature_points_name());
                particles.parallel_for([&](auto& X_local, auto& V_local, const auto& VP_local) {
                    X_local = VP_local.col(0);
                    V_local = VP_local.col(1);
                },
                    particles.X_name(), particles.V_name(), FemForce::VP_name());
            });
        }

        return MpmParticleHandleBase<T, dim>(mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(), mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(), quadrature_point_range, (T)0, manifold_dim);
    }

    MpmParticleHandleBase<T, dim> uniformSampleInBox(const AxisAlignedAnalyticBox<T, dim>& box, T density, T ppc)
    {
        T ppe = std::pow(ppc, (T)1 / dim);
        Vector<int, dim> particle_grid_cells;
        TV particle_grid_corner;
        for (int d = 0; d < dim; ++d) {
            particle_grid_cells[d] = (int)std::round((box.box.half_edges[d] * 2) / mpm.dx * ppe);
            particle_grid_corner[d] = box.box.b[d] - box.box.half_edges[d];
        }
        Grid<T, dim> particle_grid(particle_grid_cells, mpm.dx / ppe, particle_grid_corner);
        return uniformGridSample(particle_grid, density);
    }

    MpmParticleHandleBase<T, dim> uniformGridSample(const Grid<T, dim>& grid, T density)
    {
        ZIRAN_INFO("Sampling totally ", grid.numberNodes(), " particles on a Cartesian grid");
        StdVector<TV> samples;
        grid.getAllPositions(samples);
        int point_number = samples.size();
        ZIRAN_ASSERT(point_number == grid.numberNodes());
        T total_volume = grid.gridVolume();

        Range particle_range = mpm.particles.getNextRange(point_number);
        ZIRAN_ASSERT(particle_range.upper - particle_range.lower == point_number);
        mpm.particles.add(mpm.particles.X_name(), particle_range, std::move(samples));
        mpm.particles.add(mpm.particles.V_name(), particle_range, TV::Zero());
        mpm.particles.add(mpm.particles.mass_name(), particle_range, total_volume * density / point_number);
        return MpmParticleHandleBase<T, dim>(mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(), mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(), particle_range, total_volume);
    }

    //!!! point_number is total number of points sampled instead of particles per cell
    MpmParticleHandleBase<T, dim> uniformSampleInAnalyticLevelSet(AnalyticLevelSet<T, dim>& levelset, T density, int point_number)
    {
        ZIRAN_INFO("Sampling totally ", point_number, " particles in the levelset");
        TV min_corner, max_corner;
        levelset.getBounds(min_corner, max_corner);
        RandomSampling<T, dim> rs(123, min_corner, max_corner, point_number);
        StdVector<TV> samples;
        rs.sample(samples, [&](const TV& x) { return levelset.inside(x); });
        ZIRAN_ASSERT(samples.size() == (size_t)point_number);
        Range particle_range = mpm.particles.getNextRange(point_number);
        ZIRAN_ASSERT(particle_range.upper - particle_range.lower == point_number);
        mpm.particles.add(mpm.particles.X_name(), particle_range, std::move(samples));
        mpm.particles.add(mpm.particles.V_name(), particle_range, TV::Zero());
        T total_volume = levelset.volume();
        ZIRAN_ASSERT(total_volume == total_volume);
        mpm.particles.add(mpm.particles.mass_name(), particle_range, total_volume * density / point_number);
        return MpmParticleHandleBase<T, dim>(mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(), mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(), particle_range, total_volume);
    }

    MpmParticleHandleBase<T, dim> uniformSampleInAnalyticLevelSetSpecifyPPC(AnalyticLevelSet<T, dim>& levelset, T density, T ppc)
    {
        ZIRAN_INFO("Sampling on average ", ppc, " particles per cell randomly in the levelset");
        T volume = levelset.volume();
        T cells = volume / std::pow(mpm.dx, dim);
        int point_number = cells * ppc;
        TV min_corner, max_corner;
        levelset.getBounds(min_corner, max_corner);
        RandomSampling<T, dim> rs(123, min_corner, max_corner, point_number);
        StdVector<TV> samples;
        rs.sample(samples, [&](const TV& x) { return levelset.inside(x); });
        ZIRAN_ASSERT(samples.size() == (size_t)point_number);
        Range particle_range = mpm.particles.getNextRange(point_number);
        ZIRAN_ASSERT(particle_range.upper - particle_range.lower == point_number);
        mpm.particles.add(mpm.particles.X_name(), particle_range, std::move(samples));
        mpm.particles.add(mpm.particles.V_name(), particle_range, TV::Zero());
        T total_volume = levelset.volume();
        ZIRAN_ASSERT(total_volume == total_volume);
        mpm.particles.add(mpm.particles.mass_name(), particle_range, total_volume * density / point_number);
        return MpmParticleHandleBase<T, dim>(mpm.getParticles(), mpm.getScene(), mpm.getMpmForce(), mpm.getPlasticityAppliers(), mpm.getScratchXp(), mpm.getDt(), particle_range, total_volume);
    }

    void addExplicitVelocityField(const std::function<void(T&, const TV&, TV&)>& mapping)
    {
        mpm.explicit_velocity_field = mapping;
    }

    void addAnalyticCollisionObject(AnalyticCollisionObject<T, dim>& c)
    {
        mpm.addCollisionObject(std::move(c));
    }

    void addAnimatedCollisionObject(AnimatedCollisionObject<T, dim>& c)
    {
        mpm.collision_objects.emplace_back(std::make_unique<AnimatedCollisionObject<T, dim>>(std::move(c)));
    }

    int addSourceCollisionObject(SourceCollisionObject<T, dim>& c)
    {
        return mpm.addSourceCollisionObject(std::move(c));
    }

    void addExternalBodyForce(T scale, T frequency, const TV& direction)
    {
        auto ff = [=](const TV& x, const T t) -> TV {
            TV f = TV::Zero();
            f += scale * (std::sin(frequency * t) + 1) * direction;
            return f;
        };
        mpm.fext.emplace_back(std::move(ff));
    }

    int getNumNodes()
    {
        return mpm.num_nodes;
    }

    void scaleCotangent(Matrix<T, dim, dim> scale)
    {
        using TM = Matrix<T, dim, dim>;
        Range particle_range(0, mpm.particles.count);

        DisjointRanges subset(DisjointRanges{ particle_range },
            mpm.particles.commonRanges(AttributeName<TM>("cotangent")));

        for (auto iter = mpm.particles.subsetIter(subset, AttributeName<TM>("cotangent")); iter; ++iter) {
            auto& F = iter.template get<0>();
            F.array() = F.array() * scale.array(); // coefficient wise scale
        }
    }

    void applyPlasticity()
    {
        mpm.applyPlasticity();
    }

    T signedDistanceToCollisionObject(const int pid, const int object_id)
    {
        return mpm.collision_objects[object_id]->signedDistance(mpm.particles.X(pid));
    }

    template <int splat_size>
    void addFS(const MpmParticleHandleBase<T, dim>& particles_handle)
    {
        using TMFS = Matrix<T, splat_size - 1, dim>;
        Matrix<T, splat_size - 1, dim> FS = Matrix<T, splat_size - 1, dim>::Zero();
        particles_handle.particles.add(AttributeName<TMFS>("fullS"), particles_handle.particle_range, FS);
    }

    template <int splat_size>
    void addFSToAllParticles()
    {
        using TMFS = Matrix<T, splat_size - 1, dim>;
        Range range;
        range.lower = 0;
        range.upper = mpm.particles.count;
        Matrix<T, splat_size - 1, dim> FS = Matrix<T, splat_size - 1, dim>::Zero();
        mpm.particles.add(AttributeName<TMFS>("fullS"), range, FS);
    }

    //#########################################################################
    // A helper function to add all walls in the [0,max_domain_boundary] domain with offset.
    //#########################################################################
    template <class BoundaryType>
    void addAllWallsInDomain(const T max_domain_boundary, const T offset, BoundaryType boundary_type) // boundary_type = AnalyticCollisionObject<T, dim>::SLIP/STICKY/SEPARATE
    {
        for (int d = 0; d < dim; d++) {
            for (int s = -1; s <= 1; s += 2) {
                TV O;
                if (s == 1)
                    O = offset * TV::Unit(d);
                else
                    O = (TV::Unit(d) * (max_domain_boundary - offset));
                TV N = TV::Unit(d) * s;
                HalfSpace<T, dim> ls(O, N);
                AnalyticCollisionObject<T, dim> ground([&](T, AnalyticCollisionObject<T, dim>&) {}, ls, boundary_type);
                addAnalyticCollisionObject(ground);
            }
        }
    }
};
} // namespace ZIRAN

#endif
