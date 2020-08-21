#include "SplittingSimulation.h"
#include "SmallGrid.h"

namespace ZIRAN {

template <class T, int dim>
void SplittingSimulation<T, dim>::rebuildInterfaceQuad()
{
    ZIRAN_TIMER();

    StdVector<T> SVQ_mass = particles.mass.array;
    const StdVector<TV>& SVQ_v = particles.V.array;
    // const StdVector<TM>& SVQ_C = particles.DataManager::get(C_name<TM>()).array;

    StdVector<T> AVQ_mass(solid_positions.size(), 0);
    StdVector<TV> AVQ_v(solid_positions.size(), TV::Zero());
    // StdVector<TM> AVQ_C(solid_positions.size(), TM::Zero());

    T radius = Base::dx * 0.5;
    spatial_hash.rebuild(radius, solid_positions);
    Base::parallel_for_updating_grid([&](int i) {
        TV Xp = particles.X.array[i];
        StdVector<int> unfiltered_neighbors;
        StdVector<int> neighbors;
        spatial_hash.oneLayerNeighbors(Xp, unfiltered_neighbors);
        for (auto& idx : unfiltered_neighbors)
            if (!particleInSolid(Xp) && !particleInSolid(solid_positions[idx]))
                if ((Xp - solid_positions[idx]).norm() < radius)
                    neighbors.push_back(idx);
        T unit_mass = SVQ_mass[i] / (T)(neighbors.size() + 1);
        SVQ_mass[i] = unit_mass;
        for (auto idx : neighbors) {
            AVQ_mass[idx] += unit_mass;
            AVQ_v[idx] += unit_mass * SVQ_v[i];
            // AVQ_C[idx] += unit_mass * SVQ_C[i];
        }
    });
    for (int i = 0; i < solid_positions.size(); ++i)
        if (AVQ_mass[i] > 0) {
            AVQ_v[i] /= AVQ_mass[i];
            // AVQ_C[i] /= AVQ_mass[i];
        }

    lego.volume_GQ.clear();
    lego.volume_mass.clear();
    lego.volume_v.clear();
    lego.volume_C.clear();
    lego.face_GQ.clear();
    lego.face_N.clear();
    interface_GQ.clear();
    interface_N.clear();
    interface_L.clear();
    ZIRAN_ASSERT(Base::interpolation_degree != 1, "do not support linear kernel");
    for (int i = 0; i < particles.count; ++i) {
        lego.volume_GQ.push_back(particles.X.array[i]);
        lego.volume_mass.push_back(SVQ_mass[i]);
        lego.volume_v.push_back(SVQ_v[i]);
        if (fluid_Q1Q0)
            lego.volume_C.push_back(((Base::apic_rpic_ratio + 1) * (T)0.5) * Base::scratch_gradV[i] + ((Base::apic_rpic_ratio - 1) * (T)0.5) * Base::scratch_gradV[i].transpose());
        else {
            auto* Carray_pointer = particles.DataManager::getPointer(C_name<TM>());
            lego.volume_C.push_back((*Carray_pointer)[i]);
        }
    }
    for (int i = 0; i < solid_positions.size(); ++i)
        if (AVQ_mass[i] > 0) {
            lego.volume_GQ.push_back(solid_positions[i]);
            lego.volume_mass.push_back(AVQ_mass[i]);
            lego.volume_v.push_back(AVQ_v[i]);
            lego.volume_C.push_back(TM::Zero());
            interface_GQ.push_back(solid_positions[i]);
            interface_N.push_back(solid_normals[i]);
            interface_L.push_back(solid_lens[i]);
        }
}

// build M M_inv G D S a b
template <class T, int dim>
template <int degree, int order>
void SplittingSimulation<T, dim>::fluidKernel(MpmGrid<T, dim, degree>& grid3, MpmGrid<T, dim, order>& grid4, MpmGrid<T, dim, order>& grid5, SM& M_inv, SM& G, SM& D, SM& S, TStack& a, TStack& b)
{
    ZIRAN_TIMER();
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
    G = Eigen::SparseMatrix<T>(num_v * dim, num_p + lego.total_id);
    std::vector<Eigen::Triplet<T>> G_tri(lego.volume_GQ.size() * grid3.kernel_size * grid4.kernel_size * dim);
    {
        ZIRAN_TIMER();
        for (int i = 0; i < (int)lego.volume_GQ.size(); ++i) {
            TV Xp = lego.volume_GQ[i];
            T rho = particles.mass.array[0] / (*vol_pointer)[0];
            T vol = lego.volume_mass[i] / rho;
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
                        G_tri[cnt++] = Eigen::Triplet<T>(idx3 * dim + alpha, idx4, -vol * w4 * dw3[alpha]);
                });
            });
        }
    }
    {
        ZIRAN_TIMER();
        T len = (dim == 2 ? Base::dx : (Base::dx * Base::dx));
        for (int i = 0; i < (int)lego.face_GQ.size(); ++i) {
            TV Xp = lego.face_GQ[i];
            TV normal = lego.face_N[i];
            BSplineWeights<T, dim, degree> spline3(degree == 2 ? Xp : (Xp - TV::Ones() * Base::dx * 0.5), Base::dx);
            uint64_t offset3 = Base::SparseMask::Linear_Offset(to_std_array(spline3.base_node));
            grid3.iterateKernel(spline3, offset3, [&](IV node3, T w3, TV dw3, GridState<T, dim>& g3) {
                int idx3 = g3.idx;
                if (idx3 < 0) return;
                for (int delta = -1; delta <= 1; delta += 2) {
                    TV tmp = lego.face_GQ[i] + lego.face_N[i] * (T)0.5 * Base::dx * (T)delta;
                    BSplineWeights<T, dim, order> spline5(tmp - TV::Ones() * Base::dx * 0.5, Base::dx);
                    int idx5 = grid5[spline5.base_node].idx;
                    ZIRAN_ASSERT(idx5 >= 0, "dof not matching");
                    T w5 = 1;
                    for (int alpha = 0; alpha < dim; ++alpha)
                        G_tri.emplace_back(idx3 * dim + alpha, num_p + idx5, len * w3 * w5 * normal[alpha]);
                }
            });
        }
        G.setFromTriplets(G_tri.begin(), G_tri.end());
    }

    // build D
    D = Eigen::SparseMatrix<T>(-G.transpose());
    // build S
    S = Eigen::SparseMatrix<T>(num_p + lego.total_id, num_p + lego.total_id);
    S.setZero();
    // Fanfu hacking condition
    if (0) {
        T rho = particles.mass.array[0] / (*vol_pointer)[0];
        std::vector<Eigen::Triplet<T>> S_tri;
        for (int i = 0; i < num_p + lego.total_id; i++)
            S_tri.emplace_back(i, i, (std::pow(Base::dx, (T)dim) / Base::dt / rho) / (T)100000);
        S.setFromTriplets(S_tri.begin(), S_tri.end());
    }

    // build a
    a = TStack::Zero(num_v * dim);
    grid3.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
        for (int d = 0; d < dim; ++d)
            a(g.idx * dim + d) += g.m / Base::dt * g.new_v[d];
    });
    // build b
    b = TStack::Zero(num_p + lego.total_id);
}

template <class T, int dim>
template <int degree, int order>
void SplittingSimulation<T, dim>::buildFluidSystem()
{
    ZIRAN_TIMER();
    lego.work(*this);
    rebuildInterfaceQuad();

    MpmGrid<T, dim, degree>& grid3 = grid_comm;
    if (degree == 1)
        grid3.pollute(lego.volume_GQ, Base::dx, 0, 1, -TV::Ones() * Base::dx * 0.5);
    else
        grid3.pollute(lego.volume_GQ, Base::dx);
    for (int i = 0; i < lego.volume_GQ.size(); ++i) {
        TV Xp = lego.volume_GQ[i];
        if (degree == 1) Xp -= TV::Ones() * Base::dx * 0.5;
        T mass = lego.volume_mass[i];
        TV momentum = mass * lego.volume_v[i];
        TM C = mass * lego.volume_C[i];
        TM4 velocity_density = TM4::Zero();
        velocity_density.template topLeftCorner<dim, dim>() = C;
        velocity_density.template topRightCorner<dim, 1>() = momentum;
        velocity_density(3, 3) = mass;
        BSplineWeights<T, dim, degree> spline(Xp, Base::dx);
        uint64_t offset = MpmSimulationBase<T, dim>::SparseMask::Linear_Offset(to_std_array(spline.base_node));
        grid3.iterateKernel(spline, offset, [&](const IV& node, T w, const TV& dw, GridState<T, dim>& g) {
            TV4 xi_minus_xp = TV4::Zero();
            xi_minus_xp.template topLeftCorner<dim, 1>() = node.template cast<T>() * Base::dx - Xp; // top dim entries non-zero
            xi_minus_xp(3) = 1;
            TV4 velocity_delta = velocity_density * xi_minus_xp * w;
            g.m += velocity_delta(3);
            g.new_v += velocity_delta.template topLeftCorner<dim, 1>();
        });
    }
    num_v = grid3.getNumNodes();
    grid3.iterateGrid([&](IV node, GridState<T, dim>& g) {
        g.new_v = g.new_v / g.m + Base::dt * Base::gravity;
        g.v = g.new_v;
    });

    bool do_analytical_integration = true;
    if (do_analytical_integration && fluid_Q1Q0) {
        ZIRAN_TIMER();
        ZIRAN_ASSERT(degree == 1, "only work for FQ1Q0");
        auto* vol_pointer = particles.DataManager::getPointer(element_measure_name<T>());
        MpmGrid<T, dim, 1> cell_kind;
        cell_kind.pollute(lego.volume_GQ, Base::dx, -1, 1, -TV::Ones() * Base::dx * 0.5);
        // prune VQ outside boundary
        StdVector<TV> prune_volume_GQ;
        StdVector<T> prune_volume_mass;
        for (int i = 0; i < lego.volume_GQ.size(); ++i)
            if (!particleInSolid(lego.volume_GQ[i])) {
                prune_volume_GQ.push_back(lego.volume_GQ[i]);
                prune_volume_mass.push_back(lego.volume_mass[i]);
            }
        lego.volume_GQ = prune_volume_GQ;
        lego.volume_mass = prune_volume_mass;
        // idx :
        //      0 inner fluid
        //      1 near free surface or near solid
        // m :
        for (auto& Xp : lego.volume_GQ) {
            BSplineWeights<T, dim, 1> spline(Xp - TV::Ones() * Base::dx * 0.5, Base::dx);
            cell_kind[spline.base_node].idx = 0;
        }
        cell_kind.iterateGrid([&](IV node, GridState<T, dim>& g) {
            IV region = IV::Ones() * 3;
            iterateRegion(region, [&](const IV& offset) {
                IV neighbor = node + offset - IV::Ones();
                if (!gridInSolid(neighbor) && cell_kind[neighbor].idx == -1)
                    cell_kind[node].idx = 1;
            });
        });
        for (auto& Xp : interface_GQ) {
            BSplineWeights<T, dim, 1> spline(Xp - TV::Ones() * Base::dx * 0.5, Base::dx);
            cell_kind[spline.base_node].idx = 1;
        }

        StdVector<TV> tmp_volume_GQ = lego.volume_GQ;
        StdVector<T> tmp_volume_mass = lego.volume_mass;
        lego.volume_GQ.clear();
        lego.volume_mass.clear();
        for (int i = 0; i < tmp_volume_GQ.size(); ++i) {
            TV Xp = tmp_volume_GQ[i];
            BSplineWeights<T, dim, 1> spline(Xp - TV::Ones() * Base::dx * 0.5, Base::dx);
            if (cell_kind[spline.base_node].m == 0) {
                for (int d = 0; d < dim; ++d)
                    for (int delta = -1; delta <= 1; delta += 2) {
                        IV node = spline.base_node;
                        node[d] += delta;
                        if (gridInSolid(node)) {
                            TV tmp = (node + IV::Ones()).template cast<T>() * Base::dx;
                            tmp[d] -= (T)delta * (T)0.5 * Base::dx;
                            lego.face_GQ.push_back(tmp);
                            lego.face_N.push_back(TV::Unit(d) * (T)delta);
                        }
                    }
                cell_kind[spline.base_node].m = 1;
            }
            if (cell_kind[spline.base_node].idx == 1) {
                lego.volume_GQ.push_back(tmp_volume_GQ[i]);
                lego.volume_mass.push_back(tmp_volume_mass[i]);
            }
            if (cell_kind[spline.base_node].idx == 0) {
                T vol = (dim == 2 ? (Base::dx * Base::dx) : (Base::dx * Base::dx * Base::dx));
                T rho = particles.mass.array[0] / (*vol_pointer)[0];
                lego.volume_GQ.push_back(spline.base_node.template cast<T>() * Base::dx + TV::Ones() * Base::dx);
                lego.volume_mass.push_back(vol * rho);
                cell_kind[spline.base_node].idx = 2;
            }
        }

        auto clamp_position = [&](StdVector<TV>& Xarray) {
            for (auto& Xp : Xarray) {
                BSplineWeights<T, dim, 1> spline(Xp - TV::Ones() * Base::dx * 0.5, Base::dx);
                for (int d = 0; d < dim; ++d) {
                    T lower = ((T)spline.base_node(d) + (T)0.6) * Base::dx;
                    T upper = ((T)spline.base_node(d) + (T)1.4) * Base::dx;
                    MATH_TOOLS::clamp(Xp(d), lower, upper);
                }
            }
        };
        clamp_position(lego.volume_GQ);
        // clamp_position(lego.face_GQ);
        clamp_position(interface_GQ);

        grid3.iterateGrid([&](IV node, GridState<T, dim>& g) {
            g.m = 0;
            g.idx = -1;
        });
        for (int i = 0; i < lego.volume_GQ.size(); ++i) {
            TV Xp = lego.volume_GQ[i];
            T mass = lego.volume_mass[i];
            BSplineWeights<T, dim, degree> spline(Xp - TV::Ones() * Base::dx * 0.5, Base::dx);
            uint64_t offset = Base::SparseMask::Linear_Offset(to_std_array(spline.base_node));
            grid3.iterateKernel(spline, offset, [&](const IV& node, T w, const TV& dw, GridState<T, dim>& g) {
                g.m += mass * w;
            });
        }
        num_v = grid3.getNumNodes();
    }

    MpmGrid<T, dim, order> grid4;
    grid4.pollute(lego.volume_GQ, Base::dx, 0, 1, -TV::Ones() * Base::dx * 0.5);
    for (auto& Xp : lego.volume_GQ) {
        BSplineWeights<T, dim, order> spline4(Xp - TV::Ones() * Base::dx * 0.5, Base::dx);
        uint64_t offset4 = Base::SparseMask::Linear_Offset(to_std_array(spline4.base_node));
        grid4.iterateKernel(spline4, offset4, [&](const IV& node4, T w4, const TV& dw4, GridState<T, dim>& g4) {
            g4.m += w4;
        });
    }
    num_p = grid4.getNumNodes();

    MpmGrid<T, dim, order> grid5;
    grid5.pollute(lego.face_GQ, Base::dx, -1, 1, -TV::Ones() * Base::dx * 0.5);
    for (int i = 0; i < lego.face_GQ.size(); ++i)
        for (int delta = -1; delta <= 1; delta += 2) {
            TV Xp = lego.face_GQ[i] + lego.face_N[i] * (T)0.5 * Base::dx * (T)delta;
            BSplineWeights<T, dim, order> spline5(Xp - TV::Ones() * Base::dx * 0.5, Base::dx);
            grid5[spline5.base_node].m += (T)1;
        }
    lego.total_id = grid5.getNumNodes();

    fluidKernel(grid3, grid4, grid5, M_inv, G, D, S, a, b);
    buildWMatrix(grid3);
    num_v = grid.getNumNodes();
    num_p = num_p + lego.total_id;
}

template void SplittingSimulation<float, 2>::buildFluidSystem<1, 0>();
template void SplittingSimulation<double, 2>::buildFluidSystem<1, 0>();
template void SplittingSimulation<float, 3>::buildFluidSystem<1, 0>();
template void SplittingSimulation<double, 3>::buildFluidSystem<1, 0>();

} // namespace ZIRAN
