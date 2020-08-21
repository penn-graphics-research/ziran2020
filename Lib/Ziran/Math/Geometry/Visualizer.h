#pragma once

#include <MPM/MpmGrid.h>
#include <MPM/MpmSimulationBase.h>
#include <Ziran/CS/Util/Forward.h>
#include <Ziran/Math/Linear/Minres.h>
#include <Ziran/Math/Geometry/PolyIO.h>

#include <Partio.h>
#include <mutex>

namespace ZIRAN {

template <class T, int dim>
void visualize_particles(MpmSimulationBase<T, dim>& sim, std::vector<T> info, std::string filename)
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

template <class T, int dim>
void visualize_particles_vec(MpmSimulationBase<T, dim>& sim, std::vector<Vector<T, dim>> info, std::string filename)
{
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

template <class T, int dim>
void visualize_grid(MpmSimulationBase<T, dim>& sim, std::vector<T> info, std::string filename)
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
void visualize_grid_vec(MpmSimulationBase<T, dim>& sim, std::vector<Vector<T, dim>>& info, std::string filename)
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
void visualize_g2p_vec(MpmSimulationBase<T, dim>& sim, std::vector<Vector<T, dim>>& info_g, std::string filename)
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

template <class T, int dim>
void visualize_g2p_vec(MpmSimulationBase<T, dim>& sim, Matrix<T, dim, Eigen::Dynamic>& info_stack, std::string filename)
{
    filename = sim.output_dir.absolutePath(sim.outputFileName(filename, ".bgeo"));
    if (info_stack.cols() != sim.num_nodes)
        info_stack = Matrix<T, dim, Eigen::Dynamic>::Zero(dim, sim.num_nodes);
    std::vector<Vector<T, dim>> info(info_stack.cols());
    for (int i = 0; i < info_stack.cols(); ++i)
        info[i] = info_stack.col(i);
    visualize_g2p_vec(sim, info, filename);
}

template <class T, int dim>
void write_linear_system(const Eigen::SparseMatrix<T>& output, const MpmGrid<T, dim>& grid2)
{
    T linear[256][256];
    std::map<int, int> mp;
    grid2.iterateGridSerial([&](Vector<int, dim> node, GridState<T, dim>& g) {
        int x = node[0] - 200;
        int y = node[1] - 200;
        mp[g.idx] += x * 16 + y;
    });
    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 256; ++j) linear[i][j] = 0;
    for (int k = 0; k < output.outerSize(); ++k)
        for (typename Eigen::SparseMatrix<T>::InnerIterator it(output, k); it; ++it) {
            int i = it.row();
            int j = it.col();
            T value = it.value();
            linear[mp[i]][mp[j]] += value;
        }
    FILE* f = fopen("/home/squarefk/Desktop/data.txt", "w");
    for (int i = 0; i < 256; ++i) {
        for (int j = 0; j < 256; ++j)
            fprintf(f, "%.20f ", linear[i][j]);
        fprintf(f, "\n");
    }
    fclose(f);
    getchar();
    exit(0);
}

template <class T, int dim>
void visualize_points(MpmSimulationBase<T, dim>& sim, StdVector<Vector<T, dim>>& points, std::string filename)
{
    filename = sim.output_dir.absolutePath(sim.outputFileName(filename, ".bgeo"));
    Partio::ParticlesDataMutable* parts = Partio::create();

    // visualize particles info
    Partio::ParticleAttribute posH, infoH;
    posH = parts->addAttribute("position", Partio::VECTOR, 3);

    for (int k = 0; k < points.size(); k++) {
        int idx = parts->addParticle();
        float* posP = parts->dataWrite<float>(posH, idx);
        for (int d = 0; d < 3; ++d) posP[d] = 0;
        for (int d = 0; d < dim; ++d) posP[d] = points[k](d);
    }

    Partio::write(filename.c_str(), *parts);
    parts->release();
}

template <class T, int dim>
void visualize_points(MpmSimulationBase<T, dim>& sim, StdVector<Vector<T, dim>>& points, StdVector<T>& info, std::string filename)
{
    filename = sim.output_dir.absolutePath(sim.outputFileName(filename, ".bgeo"));
    Partio::ParticlesDataMutable* parts = Partio::create();

    // visualize particles info
    Partio::ParticleAttribute posH, infoH;
    posH = parts->addAttribute("position", Partio::VECTOR, 3);
    infoH = parts->addAttribute("info", Partio::VECTOR, 1);

    for (int k = 0; k < points.size(); k++) {
        int idx = parts->addParticle();
        float* posP = parts->dataWrite<float>(posH, idx);
        float* infoP = parts->dataWrite<float>(infoH, idx);
        for (int d = 0; d < 3; ++d) posP[d] = 0;
        for (int d = 0; d < dim; ++d) posP[d] = points[k](d);
        infoP[0] = info[k];
    }

    Partio::write(filename.c_str(), *parts);
    parts->release();
}

template <class T, int dim>
void visualize_points(MpmSimulationBase<T, dim>& sim, StdVector<Vector<T, dim>>& points, StdVector<Vector<T, dim>>& info, std::string filename)
{
    filename = sim.output_dir.absolutePath(sim.outputFileName(filename, ".bgeo"));
    Partio::ParticlesDataMutable* parts = Partio::create();

    // visualize particles info
    Partio::ParticleAttribute posH, infoH;
    posH = parts->addAttribute("position", Partio::VECTOR, 3);
    infoH = parts->addAttribute("info", Partio::VECTOR, 3);

    for (int k = 0; k < points.size(); k++) {
        int idx = parts->addParticle();
        float* posP = parts->dataWrite<float>(posH, idx);
        float* infoP = parts->dataWrite<float>(infoH, idx);
        for (int d = 0; d < 3; ++d) posP[d] = 0;
        for (int d = 0; d < dim; ++d) posP[d] = points[k](d);
        for (int d = 0; d < 3; ++d) infoP[d] = 0;
        for (int d = 0; d < dim; ++d) infoP[d] = info[k](d);
    }

    Partio::write(filename.c_str(), *parts);
    parts->release();
}

template <class T, int dim>
void visualize_segments(MpmSimulationBase<T, dim>& sim, StdVector<Vector<T, dim>>& points, StdVector<Vector<int, 2>>& segments, std::string filename)
{
    filename = sim.output_dir.absolutePath(sim.outputFileName(filename, ".ply"));
    writeSegmeshPoly(filename, points, segments);
}

template <class T>
void visualize_sparse_matrix(const Eigen::SparseMatrix<T>& m, std::string filename)
{
    FILE* f = fopen(filename.c_str(), "w");
    for (int i = 0; i < m.rows(); ++i) {
        for (int j = 0; j < m.cols(); ++j)
            fprintf(f, "%.30f ", m.coeff(i, j));
        fprintf(f, "\n");
    }
    fclose(f);
}

} // namespace ZIRAN
