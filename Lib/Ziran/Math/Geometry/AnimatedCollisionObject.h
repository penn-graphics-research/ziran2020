#pragma once

#include <mutex>
#include <Ziran/Math/Geometry/AnalyticLevelSet.h>
#include <Ziran/Math/Geometry/ObjIO.h>
#include <Ziran/Math/MathTools.h>
#include <Partio.h>

#include "Visualizer.h"

namespace ZIRAN {

template <class T, int dim>
class AnimatedLevelSet : public AnalyticLevelSet<T, dim> {
public:
    using TV = Vector<T, dim>;
    using IV = Vector<int, dim>;
    using TM = Matrix<T, dim, dim>;
    using TV3 = Vector<T, 3>;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    StdVector<TV> samples_l, samples_r;
    int old_frame_l = -1, old_frame_r = -1;

    std::string file_in;
    int frames;
    T duration;
    MpmSimulationBase<T, dim>& sim;
    bool mesh_invert;

    StdVector<TV> samples;
    StdVector<Vector<int, 3>> indices;
    std::vector<TV> velocities;
    std::vector<TV> normals;

    std::vector<TV> vs;
    std::vector<TV> ns;
    std::vector<T> ones;
    std::vector<bool> collisions;

    AnimatedLevelSet(const std::string file_in, int frames, T duration, MpmSimulationBase<T, dim>& sim, bool mesh_invert)
        : file_in(file_in), frames(frames), duration(duration), sim(sim), mesh_invert(mesh_invert) {}

    void loadSamples(int frame, StdVector<TV>& samples)
    {
        std::string absolute_path = DataDir().absolutePath(file_in + std::to_string(frame) + ".obj");
        readTrimeshObj(absolute_path, samples, indices);
    }

    void updateSamples(T time)
    {
        T interval = duration / frames;
        int frame_l = std::min((int)(time / interval), frames);
        int frame_r = std::min(frame_l + 1, frames);
        T percentage = (time - frame_l * interval) / interval;
        MATH_TOOLS::clamp(percentage, (T)0, (T)1);
        // update samples_l and samples_r
        if (frame_l != old_frame_l && frame_l == old_frame_r) {
            samples_l = samples_r;
            old_frame_l = frame_l;
        }
        if (frame_l != old_frame_l) {
            loadSamples(frame_l, samples_l);
            old_frame_l = frame_l;
        }
        if (frame_r != old_frame_r) {
            loadSamples(frame_r, samples_r);
            old_frame_r = frame_r;
        }
        // update samples and velocities
        samples.clear();
        velocities.clear();
        int num_samples = samples_l.size();
        for (int i = 0; i < num_samples; ++i) {
            samples.push_back(samples_l[i] * (1 - percentage) + samples_r[i] * percentage);
            velocities.push_back((samples_r[i] - samples_l[i]) / interval);
        }
        // update normals
        normals.clear();
        normals.resize(num_samples, TV::Zero());
        int num_triangles = indices.size();
        for (int tri = 0; tri < num_triangles; ++tri) {
            int i = indices[tri][0], j = indices[tri][1], k = indices[tri][2];
            TV x = samples[i], y = samples[j], z = samples[k];
            TV n = ((y - x).cross(z - x)).normalized();
            normals[i] += n;
            normals[j] += n;
            normals[k] += n;
        }
        for (auto& n : normals) {
            n.normalize();
            if (mesh_invert) n = -n;
        }
    }

    void updateNodes()
    {
        vs.clear();
        vs.resize(sim.num_nodes, TV::Zero());
        ns.clear();
        ns.resize(sim.num_nodes, TV::Zero());
        ones.clear();
        ones.resize(sim.num_nodes, (T)0);
        collisions.clear();
        collisions.resize(sim.num_nodes, false);
        int num_triangles = (int)indices.size();
        for (int tri = 0; tri < num_triangles; ++tri) {
            int i = indices[tri][0], j = indices[tri][1], k = indices[tri][2];
            TV x = samples[i], y = samples[j], z = samples[k];
            TV vx = velocities[i], vy = velocities[j], vz = velocities[k];
            TV nx = normals[i], ny = normals[j], nz = normals[k];
            int num_p = (int)((y - x).norm() / sim.dx) + 1;
            int num_q = (int)((z - x).norm() / sim.dx) + 1;
            num_p = std::max(num_p, num_q);
            num_q = num_p;
            for (int p = 0; p <= num_p; ++p)
                for (int q = 0; q <= num_q - p; ++q) {
                    T u = (T)p / (T)num_p;
                    T v = (T)q / (T)num_q;
                    TV position = (1 - u - v) * x + u * y + v * z;
                    TV velocity = (1 - u - v) * vx + u * vy + v * vz;
                    TV normal = (1 - u - v) * nx + u * ny + v * nz;
                    BSplineWeights<T, dim> spline(position, sim.dx);
                    uint64_t base_offset = MpmGrid<T, dim>::SparseMask::Linear_Offset(spline.base_node[0], spline.base_node[1], spline.base_node[2]);
                    sim.grid.iterateKernelWithValidation(spline, base_offset,
                        [&](const IV& node, T w, const TV& dw, GridState<T, dim>& g) {
                            if (g.idx < 0 || w == 0) return;
                            vs[g.idx] += velocity * w;
                            ns[g.idx] += normal * w;
                            ones[g.idx] += w;
                            collisions[g.idx] = true;
                        });
                }
        }
        sim.grid.iterateGrid([&](IV node, GridState<T, dim>& g) {
            if (ones[g.idx] > 0) {
                vs[g.idx] /= ones[g.idx];
                ns[g.idx].normalize();
            }
        });
    }

    void writeAnimiatedMesh(std::string filename)
    {
        Partio::ParticlesDataMutable* parts = Partio::create();
        // visualize particles info
        Partio::ParticleAttribute posH;
        posH = parts->addAttribute("position", Partio::VECTOR, 3);
        for (int k = 0; k < (int)samples.size(); k++) {
            int idx = parts->addParticle();
            float* posP = parts->dataWrite<float>(posH, idx);
            for (int d = 0; d < 3; ++d) posP[d] = 0;
            for (int d = 0; d < dim; ++d) posP[d] = samples[k](d);
        }
        Partio::write(filename.c_str(), *parts);
        parts->release();
    }

    int getNodeIdx(const TV& X) const
    {
        IV node;
        for (int d = 0; d < dim; ++d)
            node[d] = std::round(X[d] / sim.dx);
        auto offset = MpmGrid<T, dim>::SparseMask::Linear_Offset(node[0], node[1], node[2]);
        auto grid_array = sim.grid.grid->Get_Array();
        auto& g = reinterpret_cast<GridState<T, dim>&>(grid_array(offset));
        return g.idx;
    }

    T evalMaxSpeed(const TV& p_min_corner, const TV& p_max_corner) const
    {
        T max_speed = 0;
        int num_samples = samples.size();
        for (int i = 0; i < num_samples; ++i) {
            bool inside = true;
            for (int d = 0; d < dim; ++d)
                if (samples[i][d] < p_min_corner[d]) inside = false;
            for (int d = 0; d < dim; ++d)
                if (samples[i][d] > p_max_corner[d]) inside = false;
            if (inside)
                max_speed = std::max(max_speed, velocities[i].norm());
        }
        return max_speed;
    }

    T signedDistance(const TV& X) const override
    {
        IV node;
        for (int d = 0; d < dim; ++d)
            node[d] = std::round(X[d] / sim.dx);
        auto offset = MpmGrid<T, dim>::SparseMask::Linear_Offset(node[0], node[1], node[2]);
        if (!sim.grid.page_map->Test_Page(offset))
            return 1;
        int idx = getNodeIdx(X);
        return idx < 0 ? 1 : (collisions[idx] ? -1 : 1);
    }

    TV getMaterialVelocity(const TV& X) const
    {
        int idx = getNodeIdx(X);
        ZIRAN_ASSERT(collisions[idx], "this should be called only when there is collision");
        return vs[idx];
    }

    TV normal(const TV& X) const override
    {
        int idx = getNodeIdx(X);
        ZIRAN_ASSERT(collisions[idx], "this should be called only when there is collision");
        return ns[idx];
    }

    std::unique_ptr<AnalyticLevelSet<T, dim>> createPtr() const
    {
        return std::make_unique<AnimatedLevelSet<T, dim>>(*this);
    }
};

template <class T, int dim>
class AnimatedCollisionObject : public AnalyticCollisionObject<T, dim> {
    using Base = AnalyticCollisionObject<T, dim>;
    using TV = Vector<T, dim>;

public:
    AnimatedCollisionObject(const std::string file_in, int frames, T duration, MpmSimulationBase<T, dim>& sim, typename Base::COLLISION_OBJECT_TYPE type, bool mesh_invert = false)
        : Base(nullptr, type)
    {
        Base::ls = std::make_unique<AnimatedLevelSet<T, dim>>(file_in, frames, duration, sim, mesh_invert);
        Base::updateState = [](T time, AnalyticCollisionObject<T, dim>& object) {
            AnimatedLevelSet<T, dim>* animated_ls = dynamic_cast<AnimatedLevelSet<T, dim>*>(object.ls.get());
            animated_ls->updateSamples(time);
        };
        sim.before_euler_callbacks.emplace_back([& objects = sim.collision_objects](void) {
            for (size_t k = 0; k < objects.size(); ++k)
                if (dynamic_cast<AnimatedLevelSet<T, dim>*>(objects[k]->ls.get())) {
                    AnimatedLevelSet<T, dim>* animated_ls = dynamic_cast<AnimatedLevelSet<T, dim>*>(objects[k]->ls.get());
                    animated_ls->updateNodes();
                }
        });
    }

    AnimatedCollisionObject(AnimatedCollisionObject&& other) = default;
    AnimatedCollisionObject(const AnimatedCollisionObject& other) = delete;

    ~AnimatedCollisionObject() {}

    TV getMaterialVelocity(const TV& X) const override
    {
        AnimatedLevelSet<T, dim>* animated_ls = dynamic_cast<AnimatedLevelSet<T, dim>*>(Base::ls.get());
        return animated_ls->getMaterialVelocity(X);
    }

    virtual T evalMaxSpeed(const TV& p_min_corner, const TV& p_max_corner) const override
    {
        AnimatedLevelSet<T, dim>* animated_ls = dynamic_cast<AnimatedLevelSet<T, dim>*>(Base::ls.get());
        return animated_ls->evalMaxSpeed(p_min_corner, p_max_corner);
    }
};

} // namespace ZIRAN
