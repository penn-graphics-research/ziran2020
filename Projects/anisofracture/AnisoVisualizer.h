#pragma once

#include <MPM/MpmGrid.h>
#include <MPM/MpmSimulationBase.h>
#include <Ziran/CS/Util/Forward.h>
#include <Ziran/Math/Linear/Minres.h>

#include <Partio.h>
#include <mutex>

namespace ZIRAN {

template <class T, int dim>
void aniso_visualize_particles(MpmSimulationBase<T, dim>& sim, std::vector<T> info, std::string filename)
{
    Partio::ParticlesDataMutable* parts = Partio::create();

    // visualize particles info
    Partio::ParticleAttribute posH, infoH;
    posH = parts->addAttribute("position", Partio::VECTOR, 3);
    infoH = parts->addAttribute("info", Partio::VECTOR, 1);

    for (int k = 0; k < sim.particles.count; k++) {
        int idx = parts->addParticle();
        float* posP = parts->dataWrite<float>(posH, idx);
        float* infoP = parts->dataWrite<float>(infoH, idx);
        for (int d = 0; d < 3; ++d) posP[d] = 0;
        for (int d = 0; d < dim; ++d) posP[d] = sim.particles.X.array[k](d);
        infoP[0] = info[k];
    }

    Partio::write(filename.c_str(), *parts);
    parts->release();
}

//Using this in particular for anisoFracture.bgeo files!
template <class T, int dim>
void aniso_visualize_particles_vec(MpmSimulationBase<T, dim>& sim, std::vector<Vector<T, dim>> info, std::vector<T> info2, std::vector<T> info3, std::string filename)
{
    Partio::ParticlesDataMutable* parts = Partio::create();

    // visualize particles info
    Partio::ParticleAttribute posH, infoH, info2H, info3H;
    posH = parts->addAttribute("position", Partio::VECTOR, 3);
    infoH = parts->addAttribute("a_1", Partio::VECTOR, 3);
    info2H = parts->addAttribute("d", Partio::VECTOR, 1);
    info3H = parts->addAttribute("laplacian", Partio::VECTOR, 1);

    for (int k = 0; k < sim.particles.count; k++) {
        int idx = parts->addParticle();
        float* posP = parts->dataWrite<float>(posH, idx);
        float* infoP = parts->dataWrite<float>(infoH, idx);
        float* info2P = parts->dataWrite<float>(info2H, idx);
        float* info3P = parts->dataWrite<float>(info3H, idx);
        for (int d = 0; d < 3; ++d) posP[d] = 0;
        for (int d = 0; d < dim; ++d) posP[d] = sim.particles.X.array[k](d);
        for (int d = 0; d < 3; ++d) infoP[d] = 0;
        for (int d = 0; d < dim; ++d) infoP[d] = info[k](d);
        info2P[0] = info2[k];
        info3P[0] = info3[k];
    }

    Partio::write(filename.c_str(), *parts);
    parts->release();
}

template <class T, int dim>
void aniso_visualize_grid(MpmSimulationBase<T, dim>& sim, std::vector<T> info, std::string filename)
{
    Partio::ParticlesDataMutable* parts = Partio::create();

    // visualize grid info
    Partio::ParticleAttribute posH, infoH;
    posH = parts->addAttribute("position", Partio::VECTOR, 3);
    infoH = parts->addAttribute("info", Partio::VECTOR, 1);

    std::mutex mtx;
    sim.grid.iterateGrid([&](Vector<int, dim> node, GridState<T, dim>& g) {
        mtx.lock();
        int idx = parts->addParticle();
        float* posP = parts->dataWrite<float>(posH, idx);
        float* infoP = parts->dataWrite<float>(infoH, idx);
        for (int d = 0; d < 3; ++d) posP[d] = 0;
        for (int d = 0; d < dim; ++d) posP[d] = (float)node[d] * sim.dx;
        infoP[0] = info[g.idx];
        mtx.unlock();
    });

    Partio::write(filename.c_str(), *parts);
    parts->release();
}

template <class T, int dim>
void aniso_visualize_grid_vec(MpmSimulationBase<T, dim>& sim, std::vector<Vector<T, dim>>& info, std::string filename)
{
    Partio::ParticlesDataMutable* parts = Partio::create();

    // visualize grid info
    Partio::ParticleAttribute posH, infoH;
    posH = parts->addAttribute("position", Partio::VECTOR, 3);
    infoH = parts->addAttribute("info", Partio::VECTOR, 3);

    std::mutex mtx;
    sim.grid.iterateGrid([&](Vector<int, dim> node, GridState<T, dim>& g) {
        mtx.lock();
        int idx = parts->addParticle();
        float* posP = parts->dataWrite<float>(posH, idx);
        float* infoP = parts->dataWrite<float>(infoH, idx);
        for (int d = 0; d < 3; ++d) posP[d] = 0;
        for (int d = 0; d < dim; ++d) posP[d] = (float)node[d] * sim.dx;
        for (int d = 0; d < 3; ++d) infoP[d] = 0;
        for (int d = 0; d < dim; ++d) infoP[d] = (float)info[g.idx](d);
        mtx.unlock();
    });

    Partio::write(filename.c_str(), *parts);
    parts->release();
}

template <class T, int dim>
void aniso_visualize_g2p_vec(MpmSimulationBase<T, dim>& sim, std::vector<Vector<T, dim>>& info_g, std::string filename)
{
    using IV = Vector<int, dim>;
    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;
    std::vector<Vector<T, dim>> info(sim.particles.count, TV::Zero());
    auto& Xarray = sim.particles.X.array;
    tbb::parallel_for(0, (int)sim.particles.count, [&](int i) {
        TV& Xp = Xarray[i];
        BSplineWeights<T, dim> spline(Xp, sim.dx);
        sim.grid.iterateKernel(spline, sim.particle_base_offset[i], [&](IV node, T w, TV dw, GridState<T, dim>& g) {
            if (g.idx < 0)
                return;
            info[i] += w * info_g[g.idx];
        });
    });

    Partio::ParticlesDataMutable* parts = Partio::create();

    // visualize particles info
    Partio::ParticleAttribute posH, infoH;
    posH = parts->addAttribute("position", Partio::VECTOR, 3);
    infoH = parts->addAttribute("info", Partio::VECTOR, 3);

    for (int k = 0; k < sim.particles.count; k++) {
        int idx = parts->addParticle();
        float* posP = parts->dataWrite<float>(posH, idx);
        float* infoP = parts->dataWrite<float>(infoH, idx);
        for (int d = 0; d < 3; ++d) posP[d] = 0;
        for (int d = 0; d < dim; ++d) posP[d] = sim.particles.X.array[k](d);
        for (int d = 0; d < 3; ++d) infoP[d] = 0;
        for (int d = 0; d < dim; ++d) infoP[d] = info[k](d);
    }

    Partio::write(filename.c_str(), *parts);
    parts->release();
}

} // namespace ZIRAN
