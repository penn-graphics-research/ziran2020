#ifndef PPIC_SIMULATION_H
#define PPIC_SIMULATION_H

#include <MPM/MpmSimulationBase.h>
#include <MPM/MpmSimulationDataAnalysis.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Dense.h>

namespace ZIRAN {

//linear interpolation, dim = 2
template <class T>
void gridToParticlesFullHelper(const VDBBSplineWeights<T, 2, 1>& weights, Matrix<T, 4, 4>& BTMBinvBTM, const Matrix<T, 2, 4>& node_positions, const Vector<T, 2>& Xp, const T& dx)
{
    for (int i = 0; i <= 1; i++) {
        T wi = weights.w[0](i);
        T dwi = weights.dw[0](i);
        for (int j = 0; j <= 1; j++) {
            T wj = weights.w[1](j);
            T dwj = weights.dw[1](j);
            int flat_index = i * 2 + j;
            BTMBinvBTM(0, flat_index) = wi * wj;
            BTMBinvBTM(1, flat_index) = dwi * wj;
            BTMBinvBTM(2, flat_index) = wi * dwj;
            BTMBinvBTM(3, flat_index) = dwi * dwj;
        }
    }
}

//quaratic interpolation, dim = 2
template <class T>
void gridToParticlesFullHelper(const VDBBSplineWeights<T, 2, 2>& weights, Matrix<T, 9, 9>& BTMBinvBTM, const Matrix<T, 2, 9>& node_positions, const Vector<T, 2>& Xp, const T& dx)
{
    using TV = Vector<T, 2>;
    for (int i = 0; i <= 2; i++) {
        T wi = weights.w[0](i);
        for (int j = 0; j <= 2; j++) {
            T wj = weights.w[1](j);
            int flat_index = i * 3 + j;

            TV xi_minus_xp_over_h = (node_positions.col(flat_index) - Xp) / dx;
            T rix = xi_minus_xp_over_h(0);
            T rjy = xi_minus_xp_over_h(1);
            T minus_two_to_i_mod_2 = (i == 1) ? (-2) : 1;
            T minus_two_to_j_mod_2 = (j == 1) ? (-2) : 1;
            T weight = wi * wj;
            BTMBinvBTM(0, flat_index) = weight;
            BTMBinvBTM(1, flat_index) = 4 * weight * rix;
            BTMBinvBTM(2, flat_index) = 4 * weight * rjy;
            BTMBinvBTM(3, flat_index) = 16 * weight * rix * rjy;

            BTMBinvBTM(4, flat_index) = 4 * minus_two_to_i_mod_2 * wj;
            BTMBinvBTM(5, flat_index) = 4 * minus_two_to_j_mod_2 * wi;
            BTMBinvBTM(6, flat_index) = 16 * minus_two_to_i_mod_2 * wj * rjy;
            BTMBinvBTM(7, flat_index) = 16 * minus_two_to_j_mod_2 * wi * rix;

            BTMBinvBTM(8, flat_index) = 16 * minus_two_to_i_mod_2 * minus_two_to_j_mod_2;
        }
    }
}

//linear interpolation, dim = 3
template <class T, int dim = 3>
void gridToParticlesFullHelper(const VDBBSplineWeights<T, 3, 1>& weights, Matrix<T, 8, 8>& BTMBinvBTM, const Matrix<T, 3, 8>& node_positions, const Vector<T, 3>& Xp, const T& dx)
{
    using TV = Vector<T, dim>;
    for (int i = 0; i <= 1; i++) {
        T wi = weights.w[0](i);
        for (int j = 0; j <= 1; j++) {
            T wj = weights.w[1](j);
            for (int k = 0; k <= 1; k++) {
                T wk = weights.w[2](k);
                int flat_index = i * 4 + j * 2 + k;

                TV xi_minus_xp_over_h = (node_positions.col(flat_index) - Xp) / dx;
                T rix = xi_minus_xp_over_h(0);
                T rjy = xi_minus_xp_over_h(1);
                T rkz = xi_minus_xp_over_h(2);

                T weight = wi * wj * wk;
                BTMBinvBTM(0, flat_index) = weight;
                BTMBinvBTM(1, flat_index) = 4 * weight * rix;
                BTMBinvBTM(2, flat_index) = 4 * weight * rjy;
                BTMBinvBTM(3, flat_index) = 4 * weight * rkz;
                BTMBinvBTM(4, flat_index) = 16 * weight * rix * rjy;
                BTMBinvBTM(5, flat_index) = 16 * weight * rix * rkz;
                BTMBinvBTM(6, flat_index) = 16 * weight * rjy * rkz;
                BTMBinvBTM(7, flat_index) = 64 * weight * rix * rjy * rkz;
            }
        }
    }
}

//quadratic interpolation, dim = 3
template <class T, int dim = 3>
void gridToParticlesFullHelper(const VDBBSplineWeights<T, 3, 2>& weights, Matrix<T, 27, 27>& BTMBinvBTM, const Matrix<T, 3, 27>& node_positions, const Vector<T, 3>& Xp, const T& dx)
{
    using TV = Vector<T, dim>;
    for (int i = 0; i <= 2; i++) {
        T wi = weights.w[0](i);
        for (int j = 0; j <= 2; j++) {
            T wj = weights.w[1](j);
            for (int k = 0; k <= 2; k++) {
                T wk = weights.w[2](k);
                int flat_index = i * 9 + j * 3 + k;

                TV xi_minus_xp_over_h = (node_positions.col(flat_index) - Xp) / dx;
                T rix = xi_minus_xp_over_h(0);
                T rjy = xi_minus_xp_over_h(1);
                T rkz = xi_minus_xp_over_h(2);
                T minus_two_to_i_mod_2 = (i == 1) ? (-2) : 1;
                T minus_two_to_j_mod_2 = (j == 1) ? (-2) : 1;
                T minus_two_to_k_mod_2 = (k == 1) ? (-2) : 1;

                T weight = wi * wj * wk;
                BTMBinvBTM(0, flat_index) = weight;
                BTMBinvBTM(1, flat_index) = 4 * weight * rix;
                BTMBinvBTM(2, flat_index) = 4 * weight * rjy;
                BTMBinvBTM(3, flat_index) = 4 * weight * rkz;
                BTMBinvBTM(4, flat_index) = 16 * weight * rix * rjy;
                BTMBinvBTM(5, flat_index) = 16 * weight * rix * rkz;
                BTMBinvBTM(6, flat_index) = 16 * weight * rjy * rkz;
                BTMBinvBTM(7, flat_index) = 64 * weight * rix * rjy * rkz;

                BTMBinvBTM(8, flat_index) = 4 * minus_two_to_i_mod_2 * wj * wk;
                BTMBinvBTM(9, flat_index) = 4 * minus_two_to_j_mod_2 * wi * wk;
                BTMBinvBTM(10, flat_index) = 4 * minus_two_to_k_mod_2 * wi * wj;
                BTMBinvBTM(11, flat_index) = 16 * minus_two_to_i_mod_2 * minus_two_to_j_mod_2 * wk;
                BTMBinvBTM(12, flat_index) = 16 * minus_two_to_i_mod_2 * minus_two_to_k_mod_2 * wj;
                BTMBinvBTM(13, flat_index) = 16 * minus_two_to_j_mod_2 * minus_two_to_k_mod_2 * wi;
                BTMBinvBTM(14, flat_index) = 64 * minus_two_to_i_mod_2 * minus_two_to_j_mod_2 * minus_two_to_k_mod_2;

                BTMBinvBTM(15, flat_index) = 16 * minus_two_to_i_mod_2 * rjy * wj * wk;
                BTMBinvBTM(16, flat_index) = 16 * minus_two_to_i_mod_2 * rkz * wj * wk;
                BTMBinvBTM(17, flat_index) = 64 * minus_two_to_i_mod_2 * rjy * rkz * wj * wk;

                BTMBinvBTM(18, flat_index) = 16 * minus_two_to_j_mod_2 * rix * wi * wk;
                BTMBinvBTM(19, flat_index) = 16 * minus_two_to_j_mod_2 * rkz * wi * wk;
                BTMBinvBTM(20, flat_index) = 64 * minus_two_to_j_mod_2 * rix * rkz * wi * wk;

                BTMBinvBTM(21, flat_index) = 16 * minus_two_to_k_mod_2 * rix * wi * wj;
                BTMBinvBTM(22, flat_index) = 16 * minus_two_to_k_mod_2 * rjy * wi * wj;
                BTMBinvBTM(23, flat_index) = 64 * minus_two_to_k_mod_2 * rix * rjy * wi * wj;

                BTMBinvBTM(24, flat_index) = 64 * minus_two_to_i_mod_2 * minus_two_to_j_mod_2 * rkz * wk;
                BTMBinvBTM(25, flat_index) = 64 * minus_two_to_i_mod_2 * minus_two_to_k_mod_2 * rjy * wj;
                BTMBinvBTM(26, flat_index) = 64 * minus_two_to_j_mod_2 * minus_two_to_k_mod_2 * rix * wi;
            }
        }
    }
}

//linear interpolation, dim = 2
template <class T, int dim = 2>
void splatToPadsFullHelper(VDBBSplineWeights<T, 2, 1>& weights, Matrix<T, 4, 4>& MB, Vector<T, 4>& local_mass, const T& particle_mass, const Matrix<T, 2, 4>& node_positions, const Vector<T, 2>& Xp, const T& dx)
{
    //MB(flat_index, py*2 + px) = mp*wx_i*wy_i*rhx_i^px*rhy_j^py;
    //splat mass
    using TV = Vector<T, 2>;
    auto constructMB = [&](int flat_index, const T& weight) {
        TV xi_minus_xp_over_h = (node_positions.col(flat_index) - Xp) / dx;
        T r2_term = particle_mass * weight;
        for (int py = 0; py < 2; py++) {
            T r1_term = 1;
            for (int px = 0; px < 2; px++) {
                int col_id = py * 2 + px;
                MB(flat_index, col_id) = r1_term * r2_term;
                r1_term *= xi_minus_xp_over_h(0);
            }
            r2_term *= xi_minus_xp_over_h(1);
        }
        local_mass(flat_index) += particle_mass * weight;
    };
    weights.flatMap(constructMB);
}

//quaratic interpolation, dim = 2
template <class T, int dim = 2>
void splatToPadsFullHelper(const VDBBSplineWeights<T, 2, 2>& weights, Matrix<T, 9, 9>& MB, Vector<T, 9>& local_mass, const T& particle_mass, const Matrix<T, 2, 9>& node_positions, const Vector<T, 2>& Xp, const T& dx)
{
    using TV = Vector<T, 2>;
    T weight_prod_x = (T)1;
    T weight_prod_y = (T)1;
    for (int i = 0; i <= 2; i++) {
        weight_prod_x *= weights.w[0](i);
        weight_prod_y *= weights.w[1](i);
    }

    for (int i = 0; i <= 2; i++) {
        T wi = weights.w[0](i);
        for (int j = 0; j <= 2; j++) {
            T wj = weights.w[1](j);
            int flat_index = i * 3 + j;
            TV xi_minus_xp_over_h = (node_positions.col(flat_index) - Xp) / dx;
            T rix = xi_minus_xp_over_h(0);
            T rjy = xi_minus_xp_over_h(1);
            T minus_two_to_i_mod_2 = (i == 1) ? (-2) : 1;
            T minus_two_to_j_mod_2 = (j == 1) ? (-2) : 1;

            T weight = wi * wj;
            MB(flat_index, 0) = weight;
            MB(flat_index, 1) = weight * rix;
            MB(flat_index, 2) = weight * rjy;
            MB(flat_index, 3) = weight * rix * rjy;

            MB(flat_index, 4) = wj * weight_prod_x * minus_two_to_i_mod_2;
            MB(flat_index, 5) = wi * weight_prod_y * minus_two_to_j_mod_2;

            MB(flat_index, 6) = rjy * MB(flat_index, 4);
            MB(flat_index, 7) = rix * MB(flat_index, 5);

            MB(flat_index, 8) = weight_prod_x * weight_prod_y * minus_two_to_i_mod_2 * minus_two_to_j_mod_2;
            local_mass(flat_index) += particle_mass * weight;
        }
    }
    MB *= particle_mass;
}
//linear interpolation, dim = 3
template <class T, int dim = 3>
void splatToPadsFullHelper(VDBBSplineWeights<T, 3, 1>& weights, Matrix<T, 8, 8>& MB, Vector<T, 8>& local_mass, const T& particle_mass, const Matrix<T, 3, 8>& node_positions, const Vector<T, 3>& Xp, const T& dx)
{

    using TV = Vector<T, dim>;
    auto constructMB = [&](int flat_index, const T& weight) {
        TV xi_minus_xp_over_h = (node_positions.col(flat_index) - Xp) / dx;
        T rix = xi_minus_xp_over_h(0);
        T rjy = xi_minus_xp_over_h(1);
        T rkz = xi_minus_xp_over_h(2);
        MB(flat_index, 0) = weight * 1;
        MB(flat_index, 1) = weight * rix;
        MB(flat_index, 2) = weight * rjy;
        MB(flat_index, 3) = weight * rkz;
        MB(flat_index, 4) = weight * rix * rjy;
        MB(flat_index, 5) = weight * rix * rkz;
        MB(flat_index, 6) = weight * rjy * rkz;
        MB(flat_index, 7) = weight * rix * rjy * rkz;
        local_mass(flat_index) += particle_mass * weight;
    };
    weights.flatMap(constructMB);
    MB *= particle_mass;
}

//quadratic interpolation, dim = 3
template <class T, int dim = 3>
void splatToPadsFullHelper(const VDBBSplineWeights<T, 3, 2>& weights, Matrix<T, 27, 27>& MB, Vector<T, 27>& local_mass, const T& particle_mass, const Matrix<T, 3, 27>& node_positions, const Vector<T, 3>& Xp, const T& dx)
{
    using TV = Vector<T, dim>;
    T weight_prod_x = (T)1;
    T weight_prod_y = (T)1;
    T weight_prod_z = (T)1;
    for (int i = 0; i <= 2; i++) {
        weight_prod_x *= weights.w[0](i);
        weight_prod_y *= weights.w[1](i);
        weight_prod_z *= weights.w[2](i);
    }

    for (int i = 0; i <= 2; i++) {
        T wi = weights.w[0](i);
        for (int j = 0; j <= 2; j++) {
            T wj = weights.w[1](j);
            for (int k = 0; k <= 2; k++) {
                T wk = weights.w[2](k);

                int flat_index = i * 9 + j * 3 + k;
                TV xi_minus_xp_over_h = (node_positions.col(flat_index) - Xp) / dx;
                T rix = xi_minus_xp_over_h(0);
                T rjy = xi_minus_xp_over_h(1);
                T rkz = xi_minus_xp_over_h(2);

                T prod_x_term = (i == 1) ? (-2 * weight_prod_x) : weight_prod_x;
                T prod_y_term = (j == 1) ? (-2 * weight_prod_y) : weight_prod_y;
                T prod_z_term = (k == 1) ? (-2 * weight_prod_z) : weight_prod_z;

                T weight = wi * wj * wk;
                MB(flat_index, 0) = weight;
                MB(flat_index, 1) = rix * weight;
                MB(flat_index, 2) = rjy * weight;
                MB(flat_index, 3) = rkz * weight;
                MB(flat_index, 4) = rix * rjy * weight;
                MB(flat_index, 5) = rix * rkz * weight;
                MB(flat_index, 6) = rjy * rkz * weight;
                MB(flat_index, 7) = rix * rjy * rkz * weight;

                MB(flat_index, 8) = wj * wk * prod_x_term;
                MB(flat_index, 9) = wi * wk * prod_y_term;
                MB(flat_index, 10) = wi * wj * prod_z_term;
                MB(flat_index, 11) = wk * prod_x_term * prod_y_term;
                MB(flat_index, 12) = wj * prod_x_term * prod_z_term;
                MB(flat_index, 13) = wi * prod_y_term * prod_z_term;
                MB(flat_index, 14) = prod_x_term * prod_y_term * prod_z_term;

                MB(flat_index, 15) = MB(flat_index, 8) * rjy;
                MB(flat_index, 16) = MB(flat_index, 8) * rkz;
                MB(flat_index, 17) = MB(flat_index, 8) * rjy * rkz;

                MB(flat_index, 18) = MB(flat_index, 9) * rix;
                MB(flat_index, 19) = MB(flat_index, 9) * rkz;
                MB(flat_index, 20) = MB(flat_index, 9) * rix * rkz;

                MB(flat_index, 21) = MB(flat_index, 10) * rix;
                MB(flat_index, 22) = MB(flat_index, 10) * rjy;
                MB(flat_index, 23) = MB(flat_index, 10) * rix * rjy;

                MB(flat_index, 24) = MB(flat_index, 11) * rkz;
                MB(flat_index, 25) = MB(flat_index, 12) * rjy;
                MB(flat_index, 26) = MB(flat_index, 13) * rix;

                local_mass(flat_index) += particle_mass * weight;
            }
        }
    }
    MB *= particle_mass;
}

template <class T, int _dim>
class PpicSimulation : public MpmSimulationBase<T, _dim> {
public:
    using Base = MpmSimulationBase<T, _dim>;
    using Base::applyPlasticity;
    using Base::autorestart;
    using Base::dim;
    using Base::dx;
    using Base::force;
    using Base::frame;
    using Base::interpolation_degree;
    using Base::leaf_to_bin;
    using Base::new_velocity;
    using Base::output_dir;
    using Base::outputFileName;
    using Base::particles;
    using Base::point_partitioner;
    using Base::restart_callbacks;
    using Base::scratch_gradV;
    using Base::splat_size;
    using Base::start_frame;
    using Base::step;
    using Base::tpads;
    using Base::velocity;
    using Base::vpads;
    using typename Base::PointPartitioner;
    using typename Base::Scalar;
    using typename Base::TM;
    using typename Base::TM4;
    using typename Base::TV;
    using typename Base::TV4;
    using typename Base::VdbTV;

    typedef Matrix<T, splat_size - 1, dim> TMFS;

    bool use_fpic;
    int fpic_num_basis;

    inline static AttributeName<TMFS> FS_name()
    {
        return AttributeName<TMFS>("fullS");
    }

    PpicSimulation()
        : Base()
        , use_fpic(false)
        , fpic_num_basis(splat_size)
    {
    }

    template <TICK_REQUIRES(dim == 2), class VelocityGridType>
    void printAngularMomentum(std::string& filename, VelocityGridType& v)
    {
        MpmSimulationDataAnalysis<T, dim> ppic_data(*this);
        AngularVelocity<T, dim> LG;
        LG = ppic_data.evalTotalAngularMomentumGrid(v);
        std::ofstream fs;
        fs.open(filename, std::ios_base::app);
        T ang_vel = LG.angular_velocity;
        fs << ang_vel << "\n";
        fs.close();
    }

    template <TICK_REQUIRES(dim == 3), class VelocityGridType>
    void printAngularMomentum(std::string& filename, VelocityGridType& v)
    {
        MpmSimulationDataAnalysis<T, dim> ppic_data(*this);
        AngularVelocity<T, dim> LG;
        LG = ppic_data.evalTotalAngularMomentumGrid(v);
        std::ofstream fs;
        fs.open(filename, std::ios_base::app);
        TV ang_vel = LG.angular_velocity;
        for (int i = 0; i < dim; i++)
            fs << " " << ang_vel(i);
        fs << "\n";
        fs.close();
    }

    void printLinearMomentum(std::string& filename)
    {
        MpmSimulationDataAnalysis<T, dim> ppic_data(*this);
        TV lin_mom = ppic_data.evalTotalLinearMomentumParticles();
        std::ofstream fs;
        fs.open(filename, std::ios_base::app);
        for (int i = 0; i < dim; i++)
            fs << " " << lin_mom(i);
        if (dim == 2)
            fs << " 0";
        fs << "\n";
    }

    void gridToParticles(double dt) override
    {
        // angular momentum after P2G
        std::string filename_ang = output_dir.absolutePath("am_afterp2g.txt");
        printAngularMomentum(filename_ang, velocity);
        // end computing angular momentum
        // angular momentum before G2P
        std::string filename = output_dir.absolutePath("am_beforeg2p.txt");
        printAngularMomentum(filename, new_velocity);
        // end computing angular momentum
        if (!use_fpic) {
            Base::gridToParticles(dt);
            return;
        }

        ZIRAN_INFO("Using FPIC G2P");

        sim();
        size_t num_buckets = point_partitioner.size();
        std::atomic<bool> faster_than_grid_cell(false);
        std::atomic<bool> faster_than_half_grid_cell(false);

        tbb::parallel_for(tbb::blocked_range<size_t>(0, num_buckets),
            [&](const tbb::blocked_range<size_t>& range) {
                bool local_faster_than_grid_cell = false;
                bool local_faster_than_half_grid_cell = false;

                auto& FS = particles.DataManager::get(PpicSimulation<T, dim>::FS_name());

                for (size_t b = range.begin(), b_end = range.end(); b < b_end; ++b) { // b is bucket index
                    openvdb::CoordBBox big_box = point_partitioner.getBBox(b);
                    big_box.expand(big_box.max() + openvdb::Coord(interpolation_degree));
                    openvdb::tools::Dense<VdbTV> dense_velocity(big_box);
                    openvdb::tools::copyToDense(*new_velocity, dense_velocity, true);

                    typename PointPartitioner::IndexIterator iter = point_partitioner.indices(b);

                    openvdb::Coord old_base_node(INT_MAX);

                    using TInfluencedV = Matrix<T, dim, splat_size>;
                    using TBTM = Matrix<T, splat_size, splat_size>;

                    TInfluencedV local_velocity = TInfluencedV::Zero();
                    TInfluencedV node_positions;
                    TBTM BTMBinvBTM = TBTM::Zero();

                    // go through all particles
                    for (; iter; ++iter) {
                        size_t i = *iter;
                        TV Xp = particles.X[i];
                        VDBBSplineWeights<T, dim, interpolation_degree> weights(Xp, dx);
                        if (old_base_node != weights.base_node) {
                            auto gather_local_velocity = [&](const openvdb::Coord& ijk, int flat_index) {
                                assert(big_box.isInside(ijk));
                                TV new_local_velocity = eigenMap<T, dim>(dense_velocity.getValue(ijk));
                                local_velocity.col(flat_index) = new_local_velocity;
                                node_positions.col(flat_index) = nodePosition<T, dim>(dx, ijk);
                            };
                            weights.loop(gather_local_velocity);
                            old_base_node = weights.base_node;
                        }

                        if (interpolation_degree <= 2)
                            gridToParticlesFullHelper<T>(weights, BTMBinvBTM, node_positions, Xp, dx);
                        else
                            ZIRAN_ASSERT(false, "only interpolation degree <= 2 are implemented!");

                        particles.V[i] = BTMBinvBTM.row(0) * local_velocity.transpose();
                        FS[i] = BTMBinvBTM.block(1, 0, splat_size - 1, splat_size) * local_velocity.transpose();
                        //FS[i].block(use_full_splat_num_basis, 0, splat_size - 1 - use_full_splat_num_basis, dim).setZero();
                        // Workaround for eigen assertion
                        for (int d = 0; d < dim; d++)
                            for (int r = fpic_num_basis - 1; r < splat_size - 1; r++)
                                FS[i](r, d) = T(0);
                        // ZIRAN_DBUG(i, " : \n", FS[i]);
                        TV increment = dt * particles.V[i];
                        particles.X[i] += increment;
                        local_faster_than_grid_cell = local_faster_than_grid_cell + (increment.squaredNorm() > dx * dx);
                        TM& gradVp = scratch_gradV[i];
                        if (gradVp != gradVp) //check for NAN/unitialized data
                            continue;
                        TM4 fat_gradVp = TM4::Zero();
                        auto add_gradV = [&](int flat_index, const TV4& dweight) {
                            TV4 new_local_velocity = TV4::Zero();
                            new_local_velocity.template head<dim>() = local_velocity.col(flat_index);
                            fat_gradVp += new_local_velocity * dweight.transpose();
                        };
                        weights.flatMapGradients(add_gradV);
                        gradVp = fat_gradVp.template topLeftCorner<dim, dim>();
                    }
                }
                if (local_faster_than_half_grid_cell)
                    faster_than_half_grid_cell.store(true, std::memory_order_relaxed);

                if (local_faster_than_grid_cell)
                    faster_than_grid_cell.store(true, std::memory_order_relaxed);
            });

        static int last_frame_restarted = 0;
        if (faster_than_grid_cell && autorestart) {
            ZIRAN_WARN("Particle traveling more than a grid cell detected");
            int frame_to_restart = frame - 1;
            if (step.max_dt <= step.min_dt) {
                ZIRAN_WARN("Unable to shrink dt further");
                frame_to_restart--;
                ZIRAN_ASSERT(frame_to_restart >= start_frame, "Unstable intial conditions detected, shrink min_dt or change other parameters");
            }
            else {
                double new_max_dt = std::max(step.max_dt / 2, step.min_dt);
                ZIRAN_WARN("Shrinking max dt to ", new_max_dt);

                restart_callbacks.emplace_back([new_max_dt, frame_to_restart, this](int frame) {
                    step.max_dt = new_max_dt;
                });
            }
            last_frame_restarted = frame_to_restart;
            throw RestartException(frame_to_restart);
        }

        if (!faster_than_half_grid_cell && frame > last_frame_restarted + 1) {
            double new_max_dt = std::min(step.max_dt_original, step.max_dt * (double)1);
            if (new_max_dt != step.max_dt) {
                ZIRAN_WARN("All particles traveled less than half a grid cell");
                step.max_dt = new_max_dt;
                ZIRAN_WARN("Increasing max dt to ", step.max_dt);
            }
        }

        force->evolveStrain(dt);

        applyPlasticity();
    }

    void splatToPads() override
    {
        if (!use_fpic) {
            Base::splatToPads();
            // particle linear momentum
            std::string filename_lin = output_dir.absolutePath("lin_momentum.txt");
            printLinearMomentum(filename_lin);
            // end computing linear momentum
            return;
        }

        ZIRAN_INFO("Using FPIC P2G");

        ZIRAN_QUIET_TIMER();
        size_t num_buckets = point_partitioner.size();
        auto& Xarray = particles.X.array;
        auto& Varray = particles.V.array;
        auto& marray = particles.mass.array;

        const StdVector<Matrix<T, splat_size - 1, dim>>* FSarray_pointer;
        FSarray_pointer = &(particles.DataManager::get(PpicSimulation<T, dim>::FS_name()).array);

        // build hashtables and splat mass to density arrays
        tbb::parallel_for(tbb::blocked_range<size_t>(0, num_buckets),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t b = range.begin(), b_end = range.end(); b < b_end; ++b) { // b is bucket index
                    // --------------- splat -----------------------------------
                    auto& dense_mass = tpads[b];
                    auto& dense_momentum = vpads[b];
                    dense_mass.fill((T)0);
                    dense_momentum.fill(VdbTV(0));
                    typename PointPartitioner::IndexIterator iter = point_partitioner.indices(b);
                    openvdb::Coord origin = point_partitioner.origin(b); // origin (min) node index of this leaf
                    openvdb::Coord max_corner = point_partitioner.origin(b); //  max influnced grid node index of the local point cloud

                    using TInfluencedV = Matrix<T, dim, splat_size>;
                    using TInfluencedm = Vector<T, splat_size>;
                    using TMB = Matrix<T, splat_size, splat_size>;

                    TInfluencedV local_momentum = TInfluencedV::Zero();
                    TInfluencedm local_mass = TInfluencedm::Zero();
                    TInfluencedV node_positions = TInfluencedV::Zero();
                    TMB MB = TMB::Zero();

                    size_t i = *iter;
                    TV Xp = Xarray[i];

                    VDBBSplineWeights<T, dim, interpolation_degree> weights(Xp, dx);
                    auto gather_node_position = [&](const openvdb::Coord& ijk, int flat_index) {
                        // update max corner
                        max_corner.maxComponent(ijk);
                        node_positions.col(flat_index) = nodePosition<T, dim>(dx, ijk);
                    };
                    bool need_new_data = true;

                    while (iter) {
                        weights.compute(Xp);
                        if (need_new_data) {
                            local_momentum = TInfluencedV::Zero();
                            local_mass = TInfluencedm::Zero();
                            MB = TMB::Zero();
                            weights.loop(gather_node_position);
                            need_new_data = false;
                        }
                        T particle_mass = marray[i];

                        if (interpolation_degree <= 2)
                            splatToPadsFullHelper<T>(weights, MB, local_mass, particle_mass, node_positions, Xp, dx);
                        else
                            ZIRAN_ASSERT(false, "only interpolation degree <= 2 are implemented!");
                        TInfluencedV momentum_incre = (MB.col(0) * Varray[i].transpose() + MB.block(0, 1, splat_size, splat_size - 1) * (*FSarray_pointer)[i]).transpose();
                        local_momentum += momentum_incre;

                        ++iter;

                        need_new_data = !iter;
                        if (iter) {
                            i = *iter;
                            Xp = Xarray[i];
                            TV grid_space_x = Xp * weights.one_over_dx;
                            openvdb::Coord new_base_node;
                            for (int d = 0; d < dim; d++)
                                new_base_node[d] = baseNode<interpolation_degree>(grid_space_x[d]);
                            need_new_data = (new_base_node != weights.base_node);
                        }
                        if (need_new_data) {
                            auto splat_to_dense = [&](const openvdb::Coord& ijk, int flat_index) {
                                openvdb::Coord dense_index = ijk - origin;
                                T mass = local_mass(flat_index);
                                TV momentum = local_momentum.col(flat_index);
                                // splat to mass and velocity dense array
                                dense_mass(dense_index) += mass;
                                dense_momentum(dense_index) += vdbVec(momentum);
                            };
                            weights.loop(splat_to_dense);
                        }
                    }
                    // --------------- splat END--------------------------------

                    // --------------- hashtable construction-------------------
                    for (int table_id = 0; table_id < (1 << dim); ++table_id) {
                        bool need_to_add = true;
                        openvdb::Coord leaf_base_index(0);
                        for (int d = 0; d < dim; d++) {
                            bool flag = table_id & (1 << d);
                            leaf_base_index[d] = origin[d] + (flag ? (1 << ZIRAN_VDB_SIZE) : 0);
                            if (flag && (max_corner[d] - origin[d] < (1 << ZIRAN_VDB_SIZE)))
                                need_to_add = false;
                        }
                        if (need_to_add)
                            leaf_to_bin[table_id][leaf_base_index] = b;
                    }
                    // ---------------  hashtable construction END--------------
                }
            });
        // particle linear momentum
        std::string filename_lin = output_dir.absolutePath("lin_momentum.txt");
        printLinearMomentum(filename_lin);
        // end computing linear momentum
    }
    const char* name() override { return "ppic"; }
};
} // namespace ZIRAN

#endif
