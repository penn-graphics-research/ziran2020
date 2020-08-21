#include "SplittingSimulation.h"

namespace ZIRAN {

template <class T, int dim>
template <int degree, int order>
void SplittingSimulation<T, dim>::solidKernel(MpmGrid<T, dim, degree>& grid3, MpmGrid<T, dim, order>& grid4, SM& M_inv, SM& G, SM& D, SM& S, TStack& a, TStack& b)
{
    ZIRAN_TIMER();
    auto& Xarray = particles.X.array;
    auto* J_pointer = particles.DataManager::getPointer(J_name<T>());
    auto* vol_pointer = particles.DataManager::getPointer(element_measure_name<T>());

    // build M
    num_v = grid3.getNumNodes();
    M_inv = Eigen::SparseMatrix<T>(num_v * dim, num_v * dim);
    std::vector<Eigen::Triplet<T>> M_inv_tri;
    grid3.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
        for (int d = 0; d < dim; ++d)
            M_inv_tri.emplace_back(g.idx * dim + d, g.idx * dim + d, Base::dt / g.m);
    });
    M_inv.setFromTriplets(M_inv_tri.begin(), M_inv_tri.end());
    // build G
    G = Eigen::SparseMatrix<T>(num_v * dim, num_p);
    std::vector<Eigen::Triplet<T>> G_tri(particles.count * grid3.kernel_size * grid4.kernel_size * dim);
    {
        ZIRAN_TIMER();
        tbb::parallel_for(0, particles.count, [&](int i) {
            TV Xp = Xarray[i];
            T J = (*J_pointer)[i];
            T vol0 = (*vol_pointer)[i];
            BSplineWeights<T, dim, degree> spline3(degree == 2 ? Xp : (Xp - TV::Ones() * Base::dx * 0.5), Base::dx);
            BSplineWeights<T, dim, order> spline4(Xp - TV::Ones() * Base::dx * 0.5, Base::dx);
            uint64_t offset3 = Base::SparseMask::Linear_Offset(to_std_array(spline3.base_node));
            uint64_t offset4 = Base::SparseMask::Linear_Offset(to_std_array(spline4.base_node));
            int cnt = i * grid3.kernel_size * grid4.kernel_size * dim;
            grid3.iterateKernel(spline3, offset3, [&](IV node3, T w3, TV dw3, GridState<T, dim>& g3) {
                int idx3 = g3.idx;
                if (idx3 < 0) return;
                grid4.iterateKernel(spline4, offset4, [&](IV node4, T w4, TV dw4, GridState<T, dim>& g4) {
                    int idx4 = g4.idx;
                    if (idx4 < 0) return;
                    for (int alpha = 0; alpha < dim; ++alpha)
                        G_tri[cnt++] = Eigen::Triplet<T>(idx3 * dim + alpha, idx4, -vol0 * J * w4 * dw3[alpha]);
                });
            });
        });
    }
    G.setFromTriplets(G_tri.begin(), G_tri.end());
    // build D
    D = Eigen::SparseMatrix<T>(-G.transpose());
    // build S
    S = Eigen::SparseMatrix<T>(num_p, num_p);
    S.setZero();
    grid4.iterateGridSerial([&](IV node2, GridState<T, dim>& g2) {
        int idx = g2.idx;
        S.coeffRef(idx, idx) += g2.v[0] / Base::dt;
    });

    // build a
    a = TStack::Zero(num_v * dim);
    grid3.iterateGrid([&](IV node, GridState<T, dim>& g) {
        for (int d = 0; d < dim; ++d)
            a(g.idx * dim + d) += g.m / Base::dt * (g.new_v[d] + Base::dt * Base::gravity[d]);
    });
    // build b
    b = TStack::Zero(num_p);
    grid4.iterateGrid([&](IV node, GridState<T, dim>& g) {
        int idx = g.idx;
        b(idx) += g.v[0] * g.v[1] / Base::dt;
    });
}

template <class T, int dim>
template <int order>
void SplittingSimulation<T, dim>::buildSolidSystem()
{
    ZIRAN_TIMER();
    auto& Xarray = particles.X.array;
    auto* J_pointer = particles.DataManager::getPointer(J_name<T>());
    auto* vol_pointer = particles.DataManager::getPointer(element_measure_name<T>());
    auto* j_model = particles.DataManager::getPointer(FJHelpher::j_constitutive_model_name());

    MpmGrid<T, dim, order> grid4;
    grid4.pollute(particles.X.array, Base::dx, 0, 1, -TV::Ones() * Base::dx * 0.5);
    Base::parallel_for_updating_grid([&](int i) {
        TV Xp = Xarray[i];
        T J = (*J_pointer)[i];
        T vol0 = (*vol_pointer)[i];
        BSplineWeights<T, dim, order> spline(Xp - TV::Ones() * Base::dx * 0.5, Base::dx);
        uint64_t offset = Base::SparseMask::Linear_Offset(to_std_array(spline.base_node));
        grid4.iterateKernel(spline, offset, [&](IV node4, T w4, TV dw4, GridState<T, dim>& g4) {
            T lambda = (*j_model)[i].lambda;
            T p = -lambda * (J - 1);
            g4.v[0] += vol0 / lambda * w4;
            g4.v[1] += vol0 * J * p * w4;
            g4.m += vol0 * J * w4;
        });
    });
    num_p = grid4.getNumNodes();
    grid4.iterateGrid([&](IV node, GridState<T, dim>& g) {
        g.v[1] /= g.m;
    });

    solidKernel<2, order>(grid, grid4, M_inv, G, D, S, a, b);
    buildWMatrix(grid);
    num_v = grid.getNumNodes();
    num_p = num_p;
}

template void SplittingSimulation<float, 2>::buildSolidSystem<0>();
template void SplittingSimulation<double, 2>::buildSolidSystem<0>();
template void SplittingSimulation<float, 3>::buildSolidSystem<0>();
template void SplittingSimulation<double, 3>::buildSolidSystem<0>();
template void SplittingSimulation<float, 2>::buildSolidSystem<1>();
template void SplittingSimulation<double, 2>::buildSolidSystem<1>();
template void SplittingSimulation<float, 3>::buildSolidSystem<1>();
template void SplittingSimulation<double, 3>::buildSolidSystem<1>();
} // namespace ZIRAN