#pragma once

#include <MPM/MpmGrid.h>
#include <MPM/MpmSimulationBase.h>
#include <Ziran/CS/Util/Forward.h>

#include <vector>

namespace ZIRAN {

template <class T, int dim>
class HeatSolver {
    typedef Vector<int, dim> IV;
    typedef Vector<T, dim> TV;

    std::vector<T> q_conv;
    std::vector<T> q_rad;
    std::vector<T> q_laser;

    T surface_layer_thickness = 1e-5;
    T r0 = 5e-5;
    T theta_amb = 298;
    T c = 0.451 * 1e3;
    T kappa = 79.5;

    // convection with the air
    T h = 80;

    // radiation with the air
    T epsilon = 0.3;
    T sigma = 5.670367 * 1e-8;
    T alpha = 100;
    T P = 1;
    T T_melt = 1700;

public:
    std::vector<bool> onSurface;
    std::vector<bool> inLaser;

    void solve(MpmSimulationBase<T, dim>& sim, const T& center, std::vector<T>& q_theta, std::vector<std::pair<int, T>>& phase)
    {
        q_conv.assign(sim.particles.count, 0);
        q_rad.assign(sim.particles.count, 0);
        q_laser.assign(sim.particles.count, 0);
        onSurface.assign(sim.particles.count, false);
        inLaser.assign(sim.particles.count, false);
        tbb::parallel_for(0, (int)sim.particles.count, [&](int i) {
            TV Xp = sim.particles.X.array[i];
            // TODO: settings
            if (Xp(1) > 0.021 - surface_layer_thickness) {
                onSurface[i] = true;
                q_conv[i] = -h * (q_theta[i] - theta_amb);
                q_rad[i] = -epsilon * sigma * (q_theta[i] * q_theta[i] * q_theta[i] * q_theta[i] - theta_amb * theta_amb * theta_amb * theta_amb);
                TV cv;
                cv(0) = center;
                cv(1) = Xp(1);
                if (dim == 3) cv(2) = 0.02025;
                if ((Xp - cv).norm() < r0) {
                    inLaser[i] = true;
                    T r = (Xp - cv).norm();
                    q_laser[i] = (2 * alpha * P) / (M_PI * r0 * r0) * std::exp((-2 * r * r) / (r0 * r0));
                }
            }
        });

        std::vector<T> H(sim.num_nodes, 0);
        std::vector<T> theta(sim.num_nodes, 0);
        sim.parallel_for_updating_grid([&](int i) {
            TV& Xp = sim.particles.X.array[i];
            T mass = sim.particles.mass.array[i];
            BSplineWeights<T, dim> spline(Xp, sim.dx);
            sim.grid.iterateKernel(spline, sim.particle_base_offset[i], [&](const IV& node, T w, const TV& dw, GridState<T, dim>& g) {
                H[g.idx] += mass * c * w;
                theta[g.idx] += mass * c * q_theta[i] * w;
            });
        });
        sim.grid.iterateGrid([&](IV node, GridState<T, dim>& g) {
            theta[g.idx] /= H[g.idx];
        });

        std::vector<T> delta(sim.num_nodes, 0);
        auto* vol_pointer = sim.particles.getPointer(element_measure_name<T>());
        sim.parallel_for_updating_grid([&](int i) {
            TV& Xp = sim.particles.X.array[i];
            T mass = sim.particles.mass.array[i];
            T vol = (*vol_pointer)[i];
            BSplineWeights<T, dim> spline(Xp, sim.dx);
            sim.grid.iterateKernel(spline, sim.particle_base_offset[i], [&](const IV& node1, T w1, const TV& dw1, GridState<T, dim>& g1) {
                sim.grid.iterateKernel(spline, sim.particle_base_offset[i], [&](const IV& node2, T w2, const TV& dw2, GridState<T, dim>& g2) {
                    delta[g1.idx] -= kappa * vol * dw1.dot(dw2) * theta[g2.idx];
                });
            });
            T R, A;
            if constexpr (dim == 2) {
                R = std::sqrt(vol / M_PI);
                A = (T)2 * R;
            }
            else {
                R = std::pow(vol * 0.75 / M_PI, (T)1 / 3);
                A = M_PI * R * R;
            }
            sim.grid.iterateKernel(spline, sim.particle_base_offset[i], [&](const IV& node, T w, const TV& dw, GridState<T, dim>& g) {
                delta[g.idx] += A * w * (q_conv[i] + q_rad[i]);
                delta[g.idx] += A * w * (q_laser[i]);
            });
        });

        std::vector<T> new_theta(sim.num_nodes, 0);
        sim.grid.iterateGrid([&](IV node, GridState<T, dim>& g) {
            new_theta[g.idx] = theta[g.idx] + delta[g.idx] / H[g.idx] * sim.dt;
        });

        tbb::parallel_for(0, (int)sim.particles.count, [&](int i) {
            TV& Xp = sim.particles.X.array[i];
            BSplineWeights<T, dim> spline(Xp, sim.dx);
            sim.grid.iterateKernel(spline, sim.particle_base_offset[i], [&](const IV& node, T w, const TV& dw, GridState<T, dim>& g) {
                q_theta[i] += (new_theta[g.idx] - theta[g.idx]) * w;
            });
        });

        // phase change
        tbb::parallel_for(0, (int)sim.particles.count, [&](int i) {
            if (q_theta[i] > T_melt && phase[i].first == 0) {
                phase[i].second += q_theta[i] - T_melt;
                q_theta[i] = T_melt;
                if (phase[i].second > 100) {
                    q_theta[i] += phase[i].second - 100;
                    phase[i].first = 1;
                    phase[i].second = 100;
                }
            }
            if (q_theta[i] < T_melt && phase[i].first == 1) {
                phase[i].second -= T_melt - q_theta[i];
                q_theta[i] = T_melt;
                if (phase[i].second < 0) {
                    q_theta[i] += phase[i].second;
                    phase[i].first = 0;
                    phase[i].second = 0;
                }
            }
        });
    }
};

} // namespace ZIRAN