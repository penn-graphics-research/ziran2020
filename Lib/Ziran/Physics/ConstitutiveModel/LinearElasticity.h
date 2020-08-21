#ifndef LINEAR_ELASTICITY_H
#define LINEAR_ELASTICITY_H
#include <Ziran/CS/Util/Forward.h>
#include <Ziran/CS/Util/BinaryIO.h>
#include <Ziran/Physics/ConstitutiveModel/HyperelasticConstitutiveModel.h>

namespace ZIRAN {

template <class T, int dim>
struct LinearScratch {
    using TM = Matrix<T, dim, dim>;
    TM F, epsilon;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    LinearScratch()
        : F(TM::Identity())
        , epsilon(TM::Zero())
    {
    }

    static const char* name()
    {
        return "LinearScratch";
    }
};

template <class T, int _dim>
class LinearElasticity : public HyperelasticConstitutiveModel<LinearElasticity<T, _dim>> {
public:
    static const int dim = _dim;
    using Base = HyperelasticConstitutiveModel<LinearElasticity<T, dim>>;
    using TV = typename Base::TV;
    using TM = typename Base::TM;
    using Strain = TM;
    using Hessian = typename Base::Hessian;
    using Scalar = typename Base::Scalar;
    using Scratch = typename HyperelasticTraits<LinearElasticity<T, dim>>::ScratchType;

    T mu, lambda;

    LinearElasticity(const T E = (T)1, const T nu = (T)0.3);

    void setLameParameters(const T E, const T nu);

    void updateScratch(const TM& new_F, Scratch& scratch);

    static constexpr bool diagonalDifferentiable() { return false; }

    T psi(const Scratch& s) const;

    void firstPiola(const Scratch& s, TM& P) const;

    // void firstPiolaDerivative(const Scratch& s, Hessian& dPdF) const;

    void firstPiolaDifferential(const Scratch& s, const TM& dF, TM& dP) const;

    bool isC2(const Scratch& s, T tolerance) const;

    void write(std::ostream& out) const;

    static LinearElasticity<T, _dim> read(std::istream& in);

    static const char* name() { return "Linear"; }

    static const char* scratch_name() { return Scratch::name(); }
};

template <class T, int dim>
struct HyperelasticTraits<LinearElasticity<T, dim>> {
    using ScratchType = LinearScratch<T, dim>;
};

template <class T, int dim>
struct RW<LinearScratch<T, dim>> {
    using Tag = NoWriteTag<LinearScratch<T, dim>>;
};
} // namespace ZIRAN

#endif
