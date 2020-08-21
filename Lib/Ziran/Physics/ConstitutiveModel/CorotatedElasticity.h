#ifndef COROTATED_ELASTICITY_H
#define COROTATED_ELASTICITY_H
#include <Ziran/CS/Util/Forward.h>
#include <Ziran/CS/Util/BinaryIO.h>
#include <Ziran/Physics/ConstitutiveModel/HyperelasticConstitutiveModel.h>

namespace ZIRAN {

template <class T, int dim>
struct CorotatedScratch {
    using TM = Matrix<T, dim, dim>;
    T J;
    TM F, R, S, JFinvT;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CorotatedScratch()
        : F(TM::Identity())
    {
    }

    static const char* name()
    {
        return "CorotatedScratch";
    }
};

template <class T, int _dim>
class CorotatedElasticity : public HyperelasticConstitutiveModel<CorotatedElasticity<T, _dim>> {
public:
    static const int dim = _dim;
    using Base = HyperelasticConstitutiveModel<CorotatedElasticity<T, dim>>;
    using TV = typename Base::TV;
    using TM = typename Base::TM;
    using Strain = TM;
    using Hessian = typename Base::Hessian;
    using Scalar = typename Base::Scalar;
    using Scratch = typename HyperelasticTraits<CorotatedElasticity<T, dim>>::ScratchType;

    T mu, lambda;

    CorotatedElasticity(const T E = (T)1, const T nu = (T)0.3);

    void setLameParameters(const T E, const T nu);

    Matrix<T, 2, 2> Bij(const TV& sigma, int i, int j, T clamp_value) const;

    void updateScratch(const TM& new_F, Scratch& scratch);

    static constexpr bool diagonalDifferentiable() { return true; }

    T psi(const Scratch& s) const;

    void firstPiola(const Scratch& s, TM& P) const;

    void firstPiolaDerivative(const Scratch& s, Hessian& dPdF) const;

    T psiDiagonal(const TV& sigma) const;

    TV firstPiolaDiagonal(const TV& sigma) const;

    TM firstPiolaDerivativeDiagonal(const TV& sigma) const;

    void firstPiolaDifferential(const Scratch& s, const TM& dF, TM& dP) const;

    bool isC2(const Scratch& s, T tolerance) const;

    void write(std::ostream& out) const;

    static CorotatedElasticity<T, _dim> read(std::istream& in);

    static const char* name() { return "Corotated"; }

    static const char* scratch_name() { return Scratch::name(); }
};

template <class T, int dim>
struct HyperelasticTraits<CorotatedElasticity<T, dim>> {
    using ScratchType = CorotatedScratch<T, dim>;
};

template <class T, int dim>
struct RW<CorotatedScratch<T, dim>> {
    using Tag = NoWriteTag<CorotatedScratch<T, dim>>;
};
} // namespace ZIRAN

#endif
