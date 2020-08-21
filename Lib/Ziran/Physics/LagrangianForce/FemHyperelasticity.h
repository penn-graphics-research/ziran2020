#ifndef FEM_HYPERELASTICITY_H
#define FEM_HYPERELASTICITY_H
#include <Ziran/Math/Geometry/Elements.h>
#include <Ziran/Math/Linear/ImplicitQRSVD.h>
#include <Ziran/Math/Nonlinear/NewtonsMethod.h>
#include <Ziran/Physics/ConstitutiveModel/ConstitutiveModel.h>
#include <tbb/tbb.h>
#include <tick/requires.h>

namespace ZIRAN {

template <class T, int dim>
class LagrangianForce;

/**
This is the class for computing the FEM discretization of hyperelasticity over simplex meshes with linear interpolation.
**/
template <class TCONST, int dim>
class FemHyperelasticity : public LagrangianForce<typename TCONST::Scalar, dim> {
public:
    using Base = LagrangianForce<typename TCONST::Scalar, dim>;
    using T = typename TCONST::Scalar;
    using Scratch = typename TCONST::Scratch;
    using TM = typename TCONST::TM;
    using THessian = typename TCONST::Hessian;
    static const int manifold_dim = TM::RowsAtCompileTime;
    static const int TM_size = dim * dim;
    using TV = Vector<T, dim>;
    using IV = Eigen::Matrix<int, manifold_dim + 1, 1>;
    using Sparse = Eigen::SparseMatrix<T, Eigen::RowMajor>;
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;
    using VecBlock = typename Base::VecBlock;
    using Vec = typename Base::Vec;

    const StdVector<TV>& particles_X;
    SimplexElements<T, manifold_dim, dim>& elements;
    DataArray<TCONST>& constitutive_models;
    DataArray<Scratch>& scratch;

    inline static AttributeName<TCONST> constitutive_model_name()
    {
        return AttributeName<TCONST>(TCONST::name());
    }

    inline static AttributeName<Scratch> constitutive_model_scratch_name()
    {
        return AttributeName<Scratch>(TCONST::scratch_name());
    }

    FemHyperelasticity(SimplexElements<T, manifold_dim, dim>& elements, const StdVector<TV>& particles_X)
        : particles_X(particles_X)
        , elements(elements)
        , constitutive_models(elements.add(constitutive_model_name()))
        , scratch(elements.add(constitutive_model_scratch_name()))
    {
    }

    virtual ~FemHyperelasticity() {}

    bool isThisMyElementManager(ElementManager<T, dim>& em) override
    {
        SimplexElements<T, manifold_dim, dim>* se = dynamic_cast<SimplexElements<T, manifold_dim, dim>*>(&em);
        return se == &elements;
    }

    // Update position based state with the referred particle X array
    void updatePositionBasedState() override
    {
        updatePositionBasedState(particles_X);
    }

    // Update position based state with an input position array
    void updatePositionBasedState(const StdVector<TV>& x) override
    {
        // update scratch
        scratch.lazyResize(constitutive_models.ranges);
        elements.updateF(x, elements.F_name());
        auto master = elements.iter(constitutive_model_name(), constitutive_model_scratch_name(), elements.F_name());
        tbb::blocked_range<int> range(0, master.common_ranges->length(), 16);
        tbb::parallel_for(range, [&](const tbb::blocked_range<int>& subrange) {
            auto end = master + subrange.end();
            for (auto iter = master + subrange.begin(); iter != end; ++iter)
                iter.template get<0>().updateScratch(iter.template get<2>(), iter.template get<1>());
        });
    }

    T totalEnergy() const override
    {
        T pe = (T)0;
        for (auto iter = elements.iter(elements.element_measure_name(), constitutive_model_name(), constitutive_model_scratch_name()); iter; ++iter)
            pe += iter.template get<0>() * iter.template get<1>().psi(iter.template get<2>());
        return pe;
    }

    // add to pad, no zero out (for parallel)
    void splatToPad(const T scale, const int pad_id) const override
    {
        TVStack& pad = elements.pads[pad_id];
        Range r{ elements.partition_offsets[pad_id], elements.partition_offsets[pad_id + 1] };
        addScaledForcesHelper(scale, pad, r, elements.local_indices);
    }

    // add to pad, no zero out (for parallel)
    void splatDifferentialToPad(const T scale, const int pad_id, const TVStack& pad_dx) const override
    {
        TVStack& pad = elements.pads[pad_id];
        Range r{ elements.partition_offsets[pad_id], elements.partition_offsets[pad_id + 1] };
        addScaledForceDifferentialHelper(scale, pad_dx, pad, r, elements.local_indices);
    }

    TICK_MEMBER_REQUIRES(dim == manifold_dim)
    void addScaledForcesHelper(const T scale, TVStack& forces, const Range& subrange, const StdVector<IV>& element_indices) const
    {
        //set up interpolating functions derivatives over unit simplex
        Eigen::Matrix<T, manifold_dim, manifold_dim + 1> grad_N_hat;
        grad_N_hat << Vector<T, manifold_dim>::Constant(-1), Matrix<T, manifold_dim, manifold_dim>::Identity();

        DisjointRanges subset(DisjointRanges({ subrange }),
            elements.commonRanges(elements.element_measure_name(),
                elements.Dm_inv_name(),
                constitutive_model_name(),
                constitutive_model_scratch_name()));
        TM P;
        for (auto iter = elements.subsetIter(subset, elements.element_measure_name(), elements.Dm_inv_name(), constitutive_model_name(), constitutive_model_scratch_name()); iter; ++iter) {
            const T& element_measure = iter.template get<0>();
            const TM& Dm_inverse = iter.template get<1>();
            const TCONST& cons_model = iter.template get<2>();
            const typename TCONST::Scratch& s = iter.template get<3>();
            const int id = iter.entryId();
            const IV& indices = element_indices[id];
            Eigen::Matrix<T, manifold_dim, manifold_dim + 1> grad_N = Dm_inverse.transpose() * grad_N_hat;
            cons_model.firstPiola(s, P);
            //the matrix G contains the forces on nodes 1 through dim, the force on node 0 is the negative sum of those on the other nodes
            Eigen::Matrix<T, dim, manifold_dim + 1> G;
            G = scale * element_measure * P * grad_N;
            for (int ln = 0; ln < manifold_dim + 1; ln++) {
                int node_index = indices(ln);
                forces.col(node_index) -= G.col(ln);
            }
        }
    }

    TICK_MEMBER_REQUIRES(dim != manifold_dim)
    void addScaledForcesHelper(const T scale, TVStack& forces, const Range& subrange, const StdVector<IV>& element_indices) const
    {
        //set up interpolating functions derivatives over unit simplex
        Eigen::Matrix<T, manifold_dim, manifold_dim + 1> grad_N_hat;
        grad_N_hat << Vector<T, manifold_dim>::Constant(-1), Matrix<T, manifold_dim, manifold_dim>::Identity();

        DisjointRanges subset(DisjointRanges({ subrange }),
            elements.commonRanges(elements.element_measure_name(),
                elements.Dm_inv_name(),
                constitutive_model_name(),
                constitutive_model_scratch_name(),
                elements.Q_name()));
        TM P;
        for (auto iter = elements.subsetIter(subset, elements.element_measure_name(), elements.Dm_inv_name(), constitutive_model_name(), constitutive_model_scratch_name(), elements.Q_name()); iter; ++iter) {
            const T& element_measure = iter.template get<0>();
            const TM& Dm_inverse = iter.template get<1>();
            const TCONST& cons_model = iter.template get<2>();
            const typename TCONST::Scratch& s = iter.template get<3>();
            const auto& Q = iter.template get<4>();
            const int id = iter.entryId();
            const IV& indices = element_indices[id];
            Eigen::Matrix<T, manifold_dim, manifold_dim + 1> grad_N = Dm_inverse.transpose() * grad_N_hat;
            cons_model.firstPiola(s, P);
            //the matrix G contains the forces on nodes 1 through dim, the force on node 0 is the negative sum of those on the other nodes
            Eigen::Matrix<T, dim, manifold_dim + 1> G;
            G = scale * element_measure * Q * P * grad_N;
            for (int ln = 0; ln < manifold_dim + 1; ln++) {
                int node_index = indices(ln);
                forces.col(node_index) -= G.col(ln);
            }
        }
    }

    // serial
    void addScaledForces(const T scale, TVStack& forces) const override
    {
        Range r{ 0, elements.count };
        addScaledForcesHelper(scale, forces, r, elements.indices.array);
    }

    TICK_MEMBER_REQUIRES(dim == manifold_dim)
    void addScaledForceDifferentialHelper(const T scale, const TVStack& dx, TVStack& df, const Range& subrange, const StdVector<IV>& element_indices) const
    {
        //set up interpolating functions derivatives over unit simplex
        Eigen::Matrix<T, manifold_dim, manifold_dim + 1> grad_N_hat;
        grad_N_hat << Vector<T, manifold_dim>::Constant(-1), Matrix<T, manifold_dim, manifold_dim>::Identity();

        DisjointRanges subset(DisjointRanges({ subrange }),
            elements.commonRanges(elements.element_measure_name(),
                elements.Dm_inv_name(),
                constitutive_model_name(),
                constitutive_model_scratch_name()));
        TM dP;
        for (auto iter = elements.subsetIter(subset, elements.element_measure_name(), elements.Dm_inv_name(), constitutive_model_name(), constitutive_model_scratch_name()); iter; ++iter) {
            const T& element_measure = iter.template get<0>();
            const TM& Dm_inverse = iter.template get<1>();
            const TCONST& cons_model = iter.template get<2>();
            const typename TCONST::Scratch& s = iter.template get<3>();
            const int id = iter.entryId();
            const IV& indices = element_indices[id];
            Eigen::Matrix<T, manifold_dim, manifold_dim + 1> grad_N = Dm_inverse.transpose() * grad_N_hat;
            TM dF = elements.dS(indices, dx) * Dm_inverse;
            cons_model.firstPiolaDifferential(s, dF, dP);
            Eigen::Matrix<T, dim, manifold_dim + 1> G;
            //the matrix G contains the force differentials on nodes 1 through dim, the force differential on node 0 is the negative sum of those on the other nodes
            G = scale * element_measure * dP * grad_N;
            for (int ln = 0; ln < manifold_dim + 1; ln++) {
                int node_index = indices(ln);
                df.col(node_index) -= G.col(ln);
            }
        }
    }

    TICK_MEMBER_REQUIRES(dim != manifold_dim)
    void addScaledForceDifferentialHelper(const T scale, const TVStack& dx, TVStack& df, const Range& subrange, const StdVector<IV>& element_indices) const
    {
        //set up interpolating functions derivatives over unit simplex
        Eigen::Matrix<T, manifold_dim, manifold_dim + 1> grad_N_hat;
        grad_N_hat << Vector<T, manifold_dim>::Constant(-1), Matrix<T, manifold_dim, manifold_dim>::Identity();

        //the matrix G contains the force differentials on nodes 1 through dim, the force differential on node 0 is the negative sum of those on the other nodes
        DisjointRanges subset(DisjointRanges({ subrange }),
            elements.commonRanges(elements.element_measure_name(),
                elements.Dm_inv_name(),
                constitutive_model_name(),
                constitutive_model_scratch_name(),
                elements.Q_name()));
        TM dP;
        for (auto iter = elements.subsetIter(subset, elements.element_measure_name(), elements.Dm_inv_name(), constitutive_model_name(), constitutive_model_scratch_name(), elements.Q_name()); iter; ++iter) {
            const T& element_measure = iter.template get<0>();
            const TM& Dm_inverse = iter.template get<1>();
            const TCONST& cons_model = iter.template get<2>();
            const typename TCONST::Scratch& s = iter.template get<3>();
            const auto& Q = iter.template get<4>();
            const int id = iter.entryId();
            const IV& indices = element_indices[id];

            Eigen::Matrix<T, manifold_dim, manifold_dim + 1> grad_N = Dm_inverse.transpose() * grad_N_hat;
            TM dF = Q.transpose() * elements.dS(indices, dx) * Dm_inverse;
            cons_model.firstPiolaDifferential(s, dF, dP);

            Eigen::Matrix<T, dim, manifold_dim + 1> G;
            G = scale * element_measure * Q * dP * grad_N;
            for (int ln = 0; ln < manifold_dim + 1; ln++) {
                int node_index = indices(ln);
                df.col(node_index) -= G.col(ln);
            }
        }
    }

    // serial
    void addScaledForceDifferential(const T scale, const TVStack& dx, TVStack& df) const override
    {
        Range r{ 0, elements.count };
        addScaledForceDifferentialHelper(scale, dx, df, r, elements.indices.array);
    }

    struct SparseMatBlocks {
        using Index = std::remove_reference_t<decltype(*Sparse().outerIndexPtr())>;
        using BlockMat = Eigen::Map<Eigen::Matrix<T, dim, dim, Eigen::RowMajor>, 0, Eigen::Stride<Eigen::Dynamic, 1>>;

        Index offset[manifold_dim + 1][manifold_dim + 1];
        Index stride[manifold_dim + 1][manifold_dim + 1];

        SparseMatBlocks() {}

        SparseMatBlocks(Sparse& matrix, const IV& indices)
        {
            // The sparse matrix can't be const because the coeffRef function will insert entries
            // if they are not found. To check whether this occured we check that the matrix
            // is compressed before and after going through
            ZIRAN_ASSERT(matrix.isCompressed());
            const Index* outerIndex = matrix.outerIndexPtr();
            const T* valuePtr = matrix.valuePtr();
            for (int i = 0; i < manifold_dim + 1; i++) {
                int row_node = indices(i);
                for (int j = 0; j < manifold_dim + 1; j++) {
                    int col_node = indices(j);
                    Index row_index = dim * row_node;
                    Index col_index = dim * col_node;
                    const T* entry = &matrix.coeffRef(row_index, col_index);
                    offset[i][j] = entry - valuePtr;
                    stride[i][j] = outerIndex[row_index + 1] - outerIndex[row_index];
                    assert((dim <= 1) || (stride[i][j] == (Index)(outerIndex[row_index + 2] - outerIndex[row_index + 1])));
                    assert((dim <= 2) || (stride[i][j] == (Index)(outerIndex[row_index + 3] - outerIndex[row_index + 2])));
                }
            }
            ZIRAN_ASSERT(matrix.isCompressed());
        }

        BlockMat getBlock(Sparse& matrix, int i, int j) const
        {
            T* valuePtr = matrix.valuePtr();
            return BlockMat(valuePtr + offset[i][j], Eigen::Stride<Eigen::Dynamic, 1>(stride[i][j], 1));
        }
    };

    TICK_MEMBER_REQUIRES(dim == manifold_dim)
    void addScaledStiffnessEntriesHelper(const T scale, Sparse& newton_matrix) const
    {
        //set up interpolating functions derivatives over unit simplex
        Eigen::Matrix<T, manifold_dim, manifold_dim + 1> grad_N_hat;
        grad_N_hat << Vector<T, manifold_dim>::Constant(-1), Matrix<T, manifold_dim, manifold_dim>::Identity();

        //the element stiffness matrix is Ke_i_alpha_j_beta = dP_{alpha,gamma}dF{beta,tau} dN_idX_gamma dN_jdX_tau
        // Ke_i_alpha_j_beta computed, scaled and added in to the appropriate entries

        TM dPdF_contract_grad_Ni_grad_Nj = TM::Zero();
        THessian dPdF;
        for (auto iter = elements.iter(elements.element_measure_name(), elements.Dm_inv_name(), constitutive_model_name(), constitutive_model_scratch_name(), block_name()); iter; ++iter) {
            const T& element_measure = iter.template get<0>();
            const TM& Dm_inverse = iter.template get<1>();
            const TCONST& cons_model = iter.template get<2>();
            const typename TCONST::Scratch& s = iter.template get<3>();
            const SparseMatBlocks& blocks = iter.template get<4>();

            cons_model.firstPiolaDerivative(s, dPdF);
            Eigen::Matrix<T, manifold_dim, manifold_dim + 1> gNe = Dm_inverse.transpose() * grad_N_hat;
            T a = scale * element_measure;
            for (int i = 0; i < manifold_dim + 1; i++) {
                TV scaled_gNei = a * gNe.col(i);
                for (int j = 0; j < manifold_dim + 1; j++) {
                    dPdFContractTwoVectors(dPdF_contract_grad_Ni_grad_Nj, dPdF, scaled_gNei, gNe.col(j));
                    blocks.getBlock(newton_matrix, i, j) += dPdF_contract_grad_Ni_grad_Nj;
                }
            }
        }
    }

    TICK_MEMBER_REQUIRES(dim != manifold_dim)
    void addScaledStiffnessEntriesHelper(const T scale, Sparse& newton_matrix) const
    {
        //set up interpolating functions derivatives over unit simplex
        using TVE = Vector<T, manifold_dim>;
        Eigen::Matrix<T, manifold_dim, manifold_dim + 1> grad_N_hat;
        grad_N_hat << TVE::Constant(-1), Matrix<T, manifold_dim, manifold_dim>::Identity();

        //the element stiffness matrix is Ke_i_alpha_j_beta = dP_{alpha,gamma}dF{beta,tau} dN_idX_gamma dN_jdX_tau
        // Ke_i_alpha_j_beta computed, scaled and added in to the appropriate entries

        TM dPdF_contract_grad_Ni_grad_Nj = TM::Zero();
        THessian dPdF;
        for (auto iter = elements.iter(elements.element_measure_name(), elements.Dm_inv_name(), constitutive_model_name(), constitutive_model_scratch_name(), block_name(), elements.Q_name()); iter; ++iter) {
            const T& element_measure = iter.template get<0>();
            const TM& Dm_inverse = iter.template get<1>();
            const TCONST& cons_model = iter.template get<2>();
            const typename TCONST::Scratch& s = iter.template get<3>();
            const SparseMatBlocks& blocks = iter.template get<4>();
            const auto& Q = iter.template get<5>();

            cons_model.firstPiolaDerivative(s, dPdF);
            Eigen::Matrix<T, manifold_dim, manifold_dim + 1> gNe = Dm_inverse.transpose() * grad_N_hat;
            T a = scale * element_measure;
            for (int i = 0; i < manifold_dim + 1; i++) {
                TVE scaled_gNei = a * gNe.col(i);
                for (int j = 0; j < manifold_dim + 1; j++) {
                    dPdFContractTwoVectors(dPdF_contract_grad_Ni_grad_Nj, dPdF, scaled_gNei, gNe.col(j));
                    blocks.getBlock(newton_matrix, i, j) += Q * dPdF_contract_grad_Ni_grad_Nj * Q.transpose();
                }
            }
        }
    }

    void addScaledStiffnessEntries(const T scale, Eigen::SparseMatrix<T, Eigen::RowMajor>& newton_matrix) const override
    {
        addScaledStiffnessEntriesHelper(scale, newton_matrix);
    }

    void initializeStiffnessSparsityPattern(StdVector<Eigen::Triplet<T>>& tripletList) const override
    {
        tripletList.reserve(tripletList.size() + elements.get(constitutive_model_name()).ranges.length() * (dim + 1));
        for (auto iter = elements.iter(constitutive_model_name(), elements.indices_name()); iter; ++iter) {
            const IV& indices = iter.template get<1>();
            for (int ln1 = 0; ln1 < manifold_dim + 1; ln1++) {
                for (int d1 = 0; d1 < dim; d1++) {
                    int row_index = dim * indices[ln1] + d1;
                    for (int ln2 = 0; ln2 < manifold_dim + 1; ln2++) {
                        for (int d2 = 0; d2 < dim; d2++) {
                            int column_index = dim * indices[ln2] + d2;
                            tripletList.emplace_back(row_index, column_index, (T)0);
                        }
                    }
                }
            }
        }
    }

    void updateStiffnessSparsityPatternBasedState(Eigen::SparseMatrix<T, Eigen::RowMajor>& newton_matrix) override
    {
        elements.add(block_name()).lazyResize(constitutive_models.ranges);

        for (auto iter = elements.iter(constitutive_model_name(), elements.indices_name(), block_name()); iter; ++iter) {
            const IV& indices = iter.template get<1>();
            iter.template get<2>() = SparseMatBlocks(newton_matrix, indices);
        }
    }

    static AttributeName<SparseMatBlocks> block_name()
    {
        std::string n = TCONST::name();
        n += "Block";
        return AttributeName<SparseMatBlocks>(n.c_str());
    }
};
} // namespace ZIRAN

#endif
