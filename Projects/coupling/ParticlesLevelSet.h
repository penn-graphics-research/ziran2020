#pragma once

#include "../splitting/UMGrid.h"
#include <MPM/MpmGrid.h>
#include <MPM/MpmSimulationBase.h>
#include <algorithm>
#include <utility>
#include <queue>

namespace ZIRAN {

template <class T, int dim>
class ParticlesLevelSet {
public:
    using TV = Vector<T, dim>;
    using IV = Vector<int, dim>;
    using TM = Matrix<T, dim, dim>;

    std::priority_queue<std::pair<T, uint64_t>> pq;
    MpmGrid<T, dim, 1> phi_grid;
    T h;

    ParticlesLevelSet() = default;

    void insertHeap(IV node)
    {
        if (phi_grid[node].m > 0)
            return;
        T min_value[dim];
        for (int d = 0; d < dim; ++d) {
            min_value[d] = std::numeric_limits<T>::max();
            for (int delta = -1; delta <= 1; delta += 2) {
                IV new_node = node;
                new_node[d] += delta;
                if (phi_grid[new_node].m > 0)
                    min_value[d] = std::min(min_value[d], phi_grid[new_node].v[0]);
            }
        }
        std::sort(min_value, min_value + dim);
        if (min_value[0] > std::numeric_limits<T>::max() / (T)2)
            return;
        T result = std::numeric_limits<T>::max();
        if (dim == 2) {
            T a = min_value[0], b = min_value[1];
            result = std::min(result, a + h);
            if (a + h > b) {
                T ab = ((a + b) + std::sqrt(2.0 * h * h - (a - b) * (a - b))) / 2.0;
                result = std::min(result, ab);
            }
        }
        if (dim == 3) {
            T a = min_value[0], b = min_value[1], c = min_value[2];
            result = std::min(result, a + h);
            if (a + h > b) {
                T ab = ((a + b) + std::sqrt(2.0 * h * h - (a - b) * (a - b))) / 2.0;
                result = std::min(result, ab);
                if (ab > c) {
                    T abc = 1.0 / 6.0 * (std::sqrt((-2.0 * a - 2.0 * b - 2.0 * c) * (-2.0 * a - 2.0 * b - 2.0 * c) - 12.0 * (a * a + b * b + c * c - h * h)) + 2.0 * a + 2.0 * b + 2.0 * c);
                    result = std::min(result, abc);
                }
            }
        }
        uint64_t offset = MpmSimulationBase<T, dim>::SparseMask::Linear_Offset(to_std_array(node));
        pq.push(std::make_pair(-result, offset));
    }

    void build(SplittingSimulation<T, dim>& Ssim, T build_dx, T blob_radius, T erosion)
    {
        ZIRAN_TIMER();
        h = build_dx;
        int kernel_size = ((int)std::ceil(blob_radius / h) + 1) * 2;
        phi_grid.pollute(Ssim.particles.X.array, h, -1, 1);
        Ssim.parallel_for_updating_grid([&](int i) {
            auto& Xp = Ssim.particles.X.array[i];
            BSplineWeights<T, dim, 1> spline(Xp, h);
            for (int d = 0; d < dim; ++d) {
                ZIRAN_ASSERT(spline.base_node[d] + kernel_size / 2 < phi_grid.spgrid_size, "Exceed SPGrid range");
            }
            IV region = IV::Ones() * kernel_size;
            iterateRegion(region, [&](const IV& offset) {
                IV node = spline.base_node + offset - IV::Ones() * (kernel_size / 2 - 1);
                TV pos = node.template cast<T>() * h;
                T phi = (Xp - pos).norm() - blob_radius;
                if (phi_grid[node].m > 0) {
                    phi_grid[node].v[0] = std::max(-phi, phi_grid[node].v[0]);
                }
                else {
                    phi_grid[node].v[0] = -phi;
                    phi_grid[node].m = 1;
                }
            });
        });

        int phi_dof = phi_grid.getNumNodes();
        StdVector<IV> ready_to_insert;
        phi_grid.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
            if (g.v[0] > 0) {
                g.v[0] = 0;
                g.m = 0;
                ready_to_insert.push_back(node);
            }
        });
        for (auto& node : ready_to_insert) {
            insertHeap(node);
        }

        while (!pq.empty()) {
            auto item = pq.top();
            pq.pop();
            auto coord = MpmSimulationBase<T, dim>::SparseMask::LinearToCoord(item.second);
            IV node;
            for (int d = 0; d < dim; ++d)
                node[d] = coord[d];
            if (phi_grid[node].m == 0) {
                phi_grid[node].v[0] = -item.first;
                phi_grid[node].m = 1;
                for (int d = 0; d < dim; ++d) {
                    for (int delta = -1; delta <= 1; delta += 2) {
                        IV new_node = node;
                        new_node[d] += delta;
                        insertHeap(new_node);
                    }
                }
            }
        }
        phi_grid.iterateGrid([&](IV node, GridState<T, dim>& g) {
            g.v[0] = -g.v[0] + erosion;
        });
    }

    bool queryInside(TV& particleX, T& phi, TV& normal, T isocontour)
    {
        BSplineWeights<T, dim, 1> spline(particleX, h);
        uint64_t offset = MpmSimulationBase<T, dim>::SparseMask::Linear_Offset(to_std_array(spline.base_node));
        int cnt = 0;
        phi_grid.iterateKernelWithValidation(spline, offset, [&](const IV& node, T w, const TV& dw, GridState<T, dim>& g) {
            if (g.m > 0) ++cnt;
        });
        if (cnt != (1 << dim)) return false;
        phi = 0;
        normal = TV::Zero();
        phi_grid.iterateKernel(spline, offset, [&](const IV& node, T w, const TV& dw, GridState<T, dim>& g) {
            phi += w * g.v[0];
            normal += dw * g.v[0];
        });
        normal.normalize();
        return phi < isocontour;
    }

    void marchingSquare(StdVector<TV>& points, StdVector<Vector<int, 2>>& segments)
    {
        if constexpr (dim == 2) {
            points.clear();
            segments.clear();
            int count = 0;
            int phi_dof = phi_grid.getNumNodes();
            UMGrid<T, dim> cell_phi;
            cell_phi.clear();
            phi_grid.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
                cell_phi(node[0], node[1]) = g.v[0];
            });
            T dx = h;
            cell_phi.iterateGrid([&](int x, int y, T value) {
                int idx_a = -1;
                int idx_b = -1;
                int idx_c = -1;
                int idx_d = -1;
                if (cell_phi.find(x + 1, y) && cell_phi.find(x, y + 1) && cell_phi.find(x + 1, y + 1)) {
                    if (cell_phi(x, y) * cell_phi(x + 1, y) < 0) {
                        TV shift = TV(std::abs(cell_phi(x, y) / (cell_phi(x, y) - cell_phi(x + 1, y))) * dx, 0);
                        TV pos = TV(x, y) * dx + shift;
                        points.push_back(pos);
                        idx_a = count;
                        count += 1;
                    }
                    if (cell_phi(x + 1, y) * cell_phi(x + 1, y + 1) < 0) {
                        TV shift = TV(0, std::abs(cell_phi(x + 1, y) / (cell_phi(x + 1, y) - cell_phi(x + 1, y + 1))) * dx);
                        TV pos = TV(x + 1, y) * dx + shift;
                        points.push_back(pos);
                        idx_b = count;
                        count += 1;
                    }
                    if (cell_phi(x, y + 1) * cell_phi(x + 1, y + 1) < 0) {
                        TV shift = TV(std::abs(cell_phi(x, y + 1) / (cell_phi(x, y + 1) - cell_phi(x + 1, y + 1))) * dx, 0);
                        TV pos = TV(x, y + 1) * dx + shift;
                        points.push_back(pos);
                        idx_c = count;
                        count += 1;
                    }
                    if (cell_phi(x, y) * cell_phi(x, y + 1) < 0) {
                        TV shift = TV(0, std::abs(cell_phi(x, y) / (cell_phi(x, y) - cell_phi(x, y + 1))) * dx);
                        TV pos = TV(x, y) * dx + shift;
                        points.push_back(pos);
                        idx_d = count;
                        count += 1;
                    }
                }
                if (idx_a != -1 && idx_b != -1) segments.push_back(Vector<int, 2>(idx_a, idx_b));
                if (idx_b != -1 && idx_c != -1) segments.push_back(Vector<int, 2>(idx_b, idx_c));
                if (idx_c != -1 && idx_d != -1) segments.push_back(Vector<int, 2>(idx_c, idx_d));
                if (idx_d != -1 && idx_a != -1) segments.push_back(Vector<int, 2>(idx_d, idx_a));
                if (idx_a != -1 && idx_c != -1) segments.push_back(Vector<int, 2>(idx_a, idx_c));
                if (idx_b != -1 && idx_d != -1) segments.push_back(Vector<int, 2>(idx_b, idx_d));
            });
        }
    }
};

} // namespace ZIRAN