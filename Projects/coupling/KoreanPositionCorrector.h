#pragma once

#include "../splitting/SplittingSimulation.h"

namespace ZIRAN {

namespace KOREAN_POSITION_CORRECTOR {

template <class T, int dim>
void solve(SplittingSimulation<T, dim>& sim)
{
    using TV = Vector<T, dim>;
    using IV = Vector<int, dim>;

    MpmGrid<T, dim> subgrid;
    MpmGrid<T, dim, 1> subgrid_linear;
    StdVector<uint64_t> suboffsets;
    subgrid.pollute(sim.particles.X.array, suboffsets, sim.dx / (T)2);

    T grid_vol = (dim == 2) ? (sim.dx * sim.dx) : (sim.dx * sim.dx * sim.dx);

    sim.parallel_for_updating_grid([&](int i) {
        TV Xp = sim.particles.X.array[i];
        BSplineWeights<T, dim> spline(Xp, sim.dx / (T)2);
        subgrid.iterateKernel(spline, suboffsets[i], [&](IV node, T w, TV dw, GridState<T, dim>& g) {
            g.m += 1 * w;
        });
    });

    sim.parallel_for_updating_grid([&](int i) {
        TV Xp = sim.particles.X.array[i];
        T rho = 0;

        // BSplineWeights<T, dim, 1> spline_linear(Xp, sim.dx / (T)2);
        // uint64_t offset_linear = MpmSimulationBase<T, dim>::SparseMask::Linear_Offset(to_std_array(spline_linear.base_node));
        // subgrid_linear.iterateKernel(spline_linear, offset_linear, [&](IV node, T w, TV dw, GridState<T, dim>& g) {
        //     rho += subgrid[node].m / grid_vol * w;
        // });
        {
            BSplineWeights<T, dim> spline(Xp, sim.dx / (T)2);
            subgrid.iterateKernel(spline, suboffsets[i], [&](IV node, T w, TV dw, GridState<T, dim>& g) {
                rho += g.m / grid_vol * w;
            });
        }

        BSplineWeights<T, dim> spline(Xp, sim.dx / (T)2);
        T p = sim.dx / (T)2 * (rho * grid_vol - 1);
        subgrid.iterateKernel(spline, suboffsets[i], [&](IV node, T w, TV dw, GridState<T, dim>& g) {
            g.v += p * dw * grid_vol;
        });
    });

    sim.parallel_for_updating_grid([&](int i) {
        TV Xp = sim.particles.X.array[i];
        TV Vp = TV::Zero();
        BSplineWeights<T, dim> spline(Xp, sim.dx / (T)2);
        subgrid.iterateKernel(spline, suboffsets[i], [&](IV node, T w, TV dw, GridState<T, dim>& g) {
            if (g.m == 0) return;
            Vp += g.v / g.m * w;
        });
        subgrid.iterateKernel(spline, suboffsets[i], [&](IV node, T w, TV dw, GridState<T, dim>& g) {
            if (g.m == 0) return;
            g.new_v += Vp / g.m * w;
        });
    });

    sim.parallel_for_updating_grid([&](int i) {
        TV& Xp = sim.particles.X.array[i];
        TV delta_X = TV::Zero();
        BSplineWeights<T, dim> spline(Xp, sim.dx / (T)2);
        subgrid.iterateKernel(spline, suboffsets[i], [&](IV node, T w, TV dw, GridState<T, dim>& g) {
            Xp += g.new_v * w;
            delta_X += g.new_v * w;
        });
        TV& Vp = sim.particles.V.array[i];
        Vp += delta_X / sim.dt;
    });
}
}

} // namespace ZIRAN::KOREAN_POSITION_CORRECTOR