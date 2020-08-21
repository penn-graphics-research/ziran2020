#include <Ziran/Physics/LagrangianForce/FemCodimensional.h>
#include <Ziran/Math/Geometry/Elements.h>
#include <Ziran/Math/Geometry/Particles.h>
#include <Ziran/Math/Linear/ImplicitQRSVD.h>
#include <Ziran/Physics/ConstitutiveModel/ConstitutiveModel.h>

namespace ZIRAN {

template <class T, int manifold_dim, int dim, int quadrature_count>
FemCodimensional<T, manifold_dim, dim, quadrature_count>::FemCodimensional(SimplexElements<T, manifold_dim, dim>& elements, const StdVector<TV>& particles_X, Particles<T, dim>& particles)
    : particles_X(particles_X)
    , particles(particles)
    , elements(elements)
    , VP(particles.add(VP_name()))
    , VdP(particles.add(VdP_name()))
    , quadrature_points(elements.get(quadrature_points_name()))
    , tangent_F(elements.add(tangent_F_name()))
    , tangent_dF(elements.add(tangent_dF_name()))
{
    ZIRAN_ASSERT(dim > manifold_dim);

    VP.registerReorderingCallback(
        [&](const DataArrayBase::ValueIDToNewValueID& old_to_new) {
            for (size_t i = 0; i < quadrature_points.size(); i++) {
                QArray& qp = quadrature_points[i];
                for (int j = 0; j < qp.rows(); j++)
                    qp[j] = old_to_new(qp[j]);
            }
        });
}

template <class T, int manifold_dim, int dim, int quadrature_count>
FemCodimensional<T, manifold_dim, dim, quadrature_count>::~FemCodimensional()
{
}

template <class T, int manifold_dim, int dim, int quadrature_count>
bool FemCodimensional<T, manifold_dim, dim, quadrature_count>::isThisMyElementManager(ElementManager<T, dim>& em)
{
    SimplexElements<T, manifold_dim, dim>* se = dynamic_cast<SimplexElements<T, manifold_dim, dim>*>(&em);
    return se == &elements;
}

// Update position based state with the referred particle X array
template <class T, int manifold_dim, int dim, int quadrature_count>
void FemCodimensional<T, manifold_dim, dim, quadrature_count>::updatePositionBasedState()
{
    updatePositionBasedState(particles_X);
}

template <class T, int manifold_dim, int dim, int quadrature_count>
// Update position based state with an input position array
void FemCodimensional<T, manifold_dim, dim, quadrature_count>::updatePositionBasedState(const StdVector<TV>& x)
{
    elements.updateF(x, tangent_F_name());
}

template <class T, int manifold_dim, int dim, int quadrature_count>
void FemCodimensional<T, manifold_dim, dim, quadrature_count>::updatePositionDifferentialBasedState(const TVStack& dx)
{
    elements.updateF(dx, tangent_dF_name());
}

template <class T, int manifold_dim, int dim, int quadrature_count>
T FemCodimensional<T, manifold_dim, dim, quadrature_count>::totalEnergy() const
{
    return 0; // handled by MpmCoCo
}

// add to pad, no zero out (for parallel)
template <class T, int manifold_dim, int dim, int quadrature_count>
void FemCodimensional<T, manifold_dim, dim, quadrature_count>::splatToPad(const T scale, const int pad_id) const
{
    TVStack& pad = elements.pads[pad_id];
    Range r{ elements.partition_offsets[pad_id], elements.partition_offsets[pad_id + 1] };
    addScaledForcesHelper(scale, pad, r, elements.local_indices);
}

// add to pad, no zero out (for parallel)
template <class T, int manifold_dim, int dim, int quadrature_count>
void FemCodimensional<T, manifold_dim, dim, quadrature_count>::splatDifferentialToPad(const T scale, const int pad_id, const TVStack& pad_dx) const
{
    TVStack& pad = elements.pads[pad_id];
    Range r{ elements.partition_offsets[pad_id], elements.partition_offsets[pad_id + 1] };
    addScaledForceDifferentialHelper(scale, pad_dx, pad, r, elements.local_indices);
}

template <class T, int manifold_dim, int dim, int quadrature_count>
void FemCodimensional<T, manifold_dim, dim, quadrature_count>::addScaledForcesHelper(const T scale, TVStack& forces, const Range& subrange, const StdVector<IV>& element_indices) const
{
    //set up interpolating functions derivatives over unit simplex
    Eigen::Matrix<T, manifold_dim, manifold_dim + 1> grad_N_hat;
    grad_N_hat << Vector<T, manifold_dim>::Constant(-1), Matrix<T, manifold_dim, manifold_dim>::Identity();

    DisjointRanges subset(DisjointRanges({ subrange }),
        elements.commonRanges(elements.Dm_inv_name(),
            quadrature_points_name()));

    for (auto iter = elements.subsetIter(subset, elements.Dm_inv_name(), quadrature_points_name()); iter; ++iter) {
        const auto& Dm_inverse = iter.template get<0>();
        const QArray& points = iter.template get<1>();
        const int id = iter.entryId();
        const IV& indices = element_indices[id];
        Eigen::Matrix<T, manifold_dim, manifold_dim + 1> grad_N = Dm_inverse.transpose() * grad_N_hat;

        TM P;
        P.setZero();
        for (int r = 0; r < points.rows(); r++) {
            int p = points(r);
            P += VP(p);
        }
        //the matrix G contains the forces on nodes 1 through dim, the force on node 0 is the negative sum of those on the other nodes
        Eigen::Matrix<T, dim, manifold_dim + 1> G;
        G = scale * P.template topLeftCorner<dim, manifold_dim>() * grad_N;
        for (int ln = 0; ln < manifold_dim + 1; ln++) {
            int node_index = indices(ln);
            forces.col(node_index) -= G.col(ln);
        }
    }
}

// serial
template <class T, int manifold_dim, int dim, int quadrature_count>
void FemCodimensional<T, manifold_dim, dim, quadrature_count>::addScaledForces(const T scale, TVStack& forces) const
{
    Range r{ 0, elements.count };
    addScaledForcesHelper(scale, forces, r, elements.indices.array);
}

template <class T, int manifold_dim, int dim, int quadrature_count>
void FemCodimensional<T, manifold_dim, dim, quadrature_count>::addScaledForceDifferentialHelper(const T scale, const TVStack& dx, TVStack& df, const Range& subrange, const StdVector<IV>& element_indices) const
{
    //set up interpolating functions derivatives over unit simplex
    Eigen::Matrix<T, manifold_dim, manifold_dim + 1> grad_N_hat;
    grad_N_hat << Vector<T, manifold_dim>::Constant(-1), Matrix<T, manifold_dim, manifold_dim>::Identity();

    DisjointRanges subset(DisjointRanges({ subrange }),
        elements.commonRanges(
            elements.Dm_inv_name(),
            quadrature_points_name()));

    for (auto iter = elements.subsetIter(subset, elements.Dm_inv_name(), quadrature_points_name()); iter; ++iter) {
        const auto& Dm_inverse = iter.template get<0>();
        const QArray& points = iter.template get<1>();
        const int id = iter.entryId();
        const IV& indices = element_indices[id];
        Eigen::Matrix<T, manifold_dim, manifold_dim + 1> grad_N = Dm_inverse.transpose() * grad_N_hat;

        TM dP;
        dP.setZero();
        for (int r = 0; r < points.rows(); r++) {
            int p = points(r);
            dP += VdP(p);
        }
        //the matrix G contains the dforces on nodes 1 through dim, the dforce on node 0 is the negative sum of those on the other nodes
        Eigen::Matrix<T, dim, manifold_dim + 1> G;
        G = scale * dP.template topLeftCorner<dim, manifold_dim>() * grad_N;
        for (int ln = 0; ln < manifold_dim + 1; ln++) {
            int node_index = indices(ln);
            df.col(node_index) -= G.col(ln);
        }
    }
}

// serial
template <class T, int manifold_dim, int dim, int quadrature_count>
void FemCodimensional<T, manifold_dim, dim, quadrature_count>::addScaledForceDifferential(const T scale, const TVStack& dx, TVStack& df) const
{
    Range r{ 0, elements.count };
    addScaledForceDifferentialHelper(scale, dx, df, r, elements.indices.array);
}

template <class T, int manifold_dim, int dim, int quadrature_count>
void FemCodimensional<T, manifold_dim, dim, quadrature_count>::addScaledStiffnessEntries(const T scale, Eigen::SparseMatrix<T, Eigen::RowMajor>& newton_matrix) const
{
    ZIRAN_ASSERT(false, "FemCodimensional only supports matrixfree mpm");
}

template <class T, int manifold_dim, int dim, int quadrature_count>
void FemCodimensional<T, manifold_dim, dim, quadrature_count>::initializeStiffnessSparsityPattern(StdVector<Eigen::Triplet<T>>& tripletList) const
{
    ZIRAN_ASSERT(false, "not implemented");
}

template <class T, int manifold_dim, int dim, int quadrature_count>
void FemCodimensional<T, manifold_dim, dim, quadrature_count>::updateStiffnessSparsityPatternBasedState(Eigen::SparseMatrix<T, Eigen::RowMajor>& newton_matrix)
{
    ZIRAN_ASSERT(false, "not implemented");
}

template <class T, int manifold_dim, int dim, int quadrature_count>
AttributeName<typename FemCodimensional<T, manifold_dim, dim, quadrature_count>::TTM> FemCodimensional<T, manifold_dim, dim, quadrature_count>::tangent_F_name()
{
    return AttributeName<TTM>("tangent F");
}

template <class T, int manifold_dim, int dim, int quadrature_count>
AttributeName<typename FemCodimensional<T, manifold_dim, dim, quadrature_count>::TTM> FemCodimensional<T, manifold_dim, dim, quadrature_count>::tangent_dF_name()
{
    return AttributeName<TTM>("tangent dF");
}

template <class T, int manifold_dim, int dim, int quadrature_count>
AttributeName<typename FemCodimensional<T, manifold_dim, dim, quadrature_count>::TM> FemCodimensional<T, manifold_dim, dim, quadrature_count>::VP_name() // volume scaled P
{
    return AttributeName<TM>("VP");
}

template <class T, int manifold_dim, int dim, int quadrature_count>
AttributeName<typename FemCodimensional<T, manifold_dim, dim, quadrature_count>::TM> FemCodimensional<T, manifold_dim, dim, quadrature_count>::VdP_name() // volume scaled dP
{
    return AttributeName<TM>("VdP");
}

template <class T, int manifold_dim, int dim, int quadrature_count>
AttributeName<typename FemCodimensional<T, manifold_dim, dim, quadrature_count>::QArray> FemCodimensional<T, manifold_dim, dim, quadrature_count>::quadrature_points_name()
{
    return AttributeName<QArray>("quadrature points");
}

template <class T, int manifold_dim, int dim, int quadrature_count>
AttributeName<typename FemCodimensional<T, manifold_dim, dim, quadrature_count>::BaryStack> FemCodimensional<T, manifold_dim, dim, quadrature_count>::barycentric_weights_name()
{
    return AttributeName<BaryStack>("barycentric weights");
}

template <class T, int manifold_dim, int dim, int quadrature_count>
AttributeName<T> FemCodimensional<T, manifold_dim, dim, quadrature_count>::element_measure_name()
{
    return AttributeName<T>("element measure");
}

template <class T, int manifold_dim, int dim, int quadrature_count>
AttributeName<typename FemCodimensional<T, manifold_dim, dim, quadrature_count>::TM> FemCodimensional<T, manifold_dim, dim, quadrature_count>::cotangent_name()
{
    return AttributeName<TM>("cotangent");
}

template <class T, int manifold_dim, int dim, int quadrature_count>
AttributeName<int> FemCodimensional<T, manifold_dim, dim, quadrature_count>::parent_element_name()
{
    return AttributeName<int>("parent element" + std::to_string(manifold_dim));
}
template class FemCodimensional<double, 1, 2, -1>;
template class FemCodimensional<double, 1, 2, 1>;
template class FemCodimensional<double, 1, 2, 2>;
template class FemCodimensional<double, 1, 2, 3>;
template class FemCodimensional<double, 1, 2, 4>;
template class FemCodimensional<double, 1, 3, -1>;
template class FemCodimensional<double, 1, 3, 1>;
template class FemCodimensional<double, 1, 3, 2>;
template class FemCodimensional<double, 1, 3, 3>;
template class FemCodimensional<double, 1, 3, 4>;
template class FemCodimensional<double, 2, 2, -1>;
template class FemCodimensional<double, 2, 2, 1>;
template class FemCodimensional<double, 2, 2, 2>;
template class FemCodimensional<double, 2, 2, 3>;
template class FemCodimensional<double, 2, 2, 4>;
template class FemCodimensional<double, 2, 3, -1>;
template class FemCodimensional<double, 2, 3, 1>;
template class FemCodimensional<double, 2, 3, 2>;
template class FemCodimensional<double, 2, 3, 3>;
template class FemCodimensional<double, 2, 3, 4>;
template class FemCodimensional<float, 1, 2, -1>;
template class FemCodimensional<float, 1, 2, 1>;
template class FemCodimensional<float, 1, 2, 2>;
template class FemCodimensional<float, 1, 2, 3>;
template class FemCodimensional<float, 1, 2, 4>;
template class FemCodimensional<float, 1, 3, -1>;
template class FemCodimensional<float, 1, 3, 1>;
template class FemCodimensional<float, 1, 3, 2>;
template class FemCodimensional<float, 1, 3, 3>;
template class FemCodimensional<float, 1, 3, 4>;
template class FemCodimensional<float, 2, 2, -1>;
template class FemCodimensional<float, 2, 2, 1>;
template class FemCodimensional<float, 2, 2, 2>;
template class FemCodimensional<float, 2, 2, 3>;
template class FemCodimensional<float, 2, 2, 4>;
template class FemCodimensional<float, 2, 3, -1>;
template class FemCodimensional<float, 2, 3, 1>;
template class FemCodimensional<float, 2, 3, 2>;
template class FemCodimensional<float, 2, 3, 3>;
template class FemCodimensional<float, 2, 3, 4>;
} // namespace ZIRAN
