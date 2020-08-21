#ifndef FEM_CODIMENSIONAL_H
#define FEM_CODIMENSIONAL_H
#include <Ziran/Physics/LagrangianForce/LagrangianForce.h>
#include <Ziran/CS/Util/Forward.h>
#include <Ziran/Math/Geometry/SimplexElements.h>
#include <Ziran/Math/Geometry/ElementManager.h>
#include <Ziran/CS/DataStructure/DataManager.h>
#include <Eigen/Sparse>

namespace ZIRAN {

// TODO (create tangent_F , tangent_dF, quadrature_points, parent_element, Dm_inverse, element_measure, cotangent

// quadrature_count is allowed to be Eigen::Dynamic. It encodes # of quadrature points per element, should be at least 1
// This class is supposed to work together with CotangentBasedMpmForceHelper in a MPM sim.
template <class T, int manifold_dim, int dim, int quadrature_count>
class FemCodimensional : public LagrangianForce<T, dim> {
public:
    using Base = LagrangianForce<T, dim>;
    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;
    using TTM = Matrix<T, dim, manifold_dim>;
    typedef Matrix<T, dim, dim - manifold_dim> CTM;
    using IV = Matrix<int, manifold_dim + 1, 1>;
    using QArray = Vector<int, quadrature_count>;
    using BaryStack = Matrix<T, manifold_dim + 1, quadrature_count>;

    using Sparse = Eigen::SparseMatrix<T, Eigen::RowMajor>;
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;
    using VecBlock = typename Base::VecBlock;

    const StdVector<TV>& particles_X; // this is for implicit
    Particles<T, dim>& particles;
    SimplexElements<T, manifold_dim, dim>& elements;

    DataArray<TM>& VP; //  belongs to particles, volume scaled first Piola Kichhoff stress, computed by the cotangent force
    const DataArray<TM>& VdP; // belongs to particles, volume scaled first Piola Kichhoff stress differential, computed by the cotangent force
    DataArray<QArray>& quadrature_points; // initialization should take care of the size of this.

    // !!!!!!!!!!!!!!!!!!! this index into the local data array of VP, not the global element index!

    DataArray<TTM>& tangent_F; // resized and initialzied when adding the
    DataArray<TTM>& tangent_dF;

    FemCodimensional(SimplexElements<T, manifold_dim, dim>& elements, const StdVector<TV>& particles_X, Particles<T, dim>& particles);

    virtual ~FemCodimensional();

    bool isThisMyElementManager(ElementManager<T, dim>& em) override;

    // Update position based state with the referred particle X array
    void updatePositionBasedState() override;

    // Update position based state with an input position array
    void updatePositionBasedState(const StdVector<TV>& x) override;

    T totalEnergy() const override;

    void updatePositionDifferentialBasedState(const TVStack& dx) override;

    // add to pad, no zero out (for parallel)
    void splatToPad(const T scale, const int pad_id) const override;

    // add to pad, no zero out (for parallel)
    void splatDifferentialToPad(const T scale, const int pad_id, const TVStack& pad_dx) const override;

    void addScaledForcesHelper(const T scale, TVStack& forces, const Range& subrange, const StdVector<IV>& element_indices) const;

    // serial
    void addScaledForces(const T scale, TVStack& forces) const override;

    void addScaledForceDifferentialHelper(const T scale, const TVStack& dx, TVStack& df, const Range& subrange, const StdVector<IV>& element_indices) const;

    // serial
    void addScaledForceDifferential(const T scale, const TVStack& dx, TVStack& df) const override;

    void addScaledStiffnessEntries(const T scale, Eigen::SparseMatrix<T, Eigen::RowMajor>& newton_matrix) const override;

    void initializeStiffnessSparsityPattern(StdVector<Eigen::Triplet<T>>& tripletList) const override;

    void updateStiffnessSparsityPatternBasedState(Eigen::SparseMatrix<T, Eigen::RowMajor>& newton_matrix) override;

    static AttributeName<TTM> tangent_F_name();

    static AttributeName<TTM> tangent_dF_name();

    static AttributeName<TM> VP_name(); // volume scaled P

    static AttributeName<TM> VdP_name(); // volume scaled dP

    static AttributeName<QArray> quadrature_points_name();

    static AttributeName<BaryStack> barycentric_weights_name();

    static AttributeName<T> element_measure_name();

    static AttributeName<TM> cotangent_name();

    static AttributeName<int> parent_element_name();
};
} // namespace ZIRAN

#endif
