/**
 This code implements psi(F) = gamma * J(F)
 **/
#ifndef SURFACE_TENSION_H
#define SURFACE_TENSION_H
#include <Ziran/CS/Util/Forward.h>

namespace ZIRAN {

template <class Derived>
class HyperelasticConstitutiveModel;

template <typename Derived>
struct HyperelasticTraits;

template <class T, int dim>
struct SurfaceTensionScratch {
    using TM = Matrix<T, dim, dim>;
    T J;
    TM F, JFinvT;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SurfaceTensionScratch()
        : F(TM::Identity())
    {
    }

    static const char* name()
    {
        return "SurfaceTensionScratch";
    }
};

template <class T, int _dim>
class SurfaceTension : public HyperelasticConstitutiveModel<SurfaceTension<T, _dim>> {
public:
    static const int dim = _dim;
    using Base = HyperelasticConstitutiveModel<SurfaceTension<T, dim>>;
    using TV = typename Base::TV;
    using TM = typename Base::TM;
    using Strain = TM;
    using Hessian = typename Base::Hessian;
    using Scalar = typename Base::Scalar;
    using Scratch = typename HyperelasticTraits<SurfaceTension<T, dim>>::ScratchType;
    using Vec = Vector<T, Eigen::Dynamic>;
    using VecBlock = Eigen::VectorBlock<Vec>;

    T gamma;

    SurfaceTension(const T gamma_in = (T)0.05)
    {
        setSurfaceTensionCoefficient(gamma_in);
    }

    void setSurfaceTensionCoefficient(const T gamma_in)
    {
        gamma = gamma_in;
    }

    void updateScratch(const TM& new_F, Scratch& s)
    {
        using namespace EIGEN_EXT;
        s.F = new_F;
        s.J = s.F.determinant();
        cofactorMatrix(s.F, s.JFinvT);
    }

    T psi(const Scratch& s) const
    {
        return gamma * s.J;
    }

    void firstPiola(const Scratch& s, TM& P) const
    {
        P = gamma * s.JFinvT;
    }

    void firstPiolaDerivative(const Scratch& s, Hessian& dPdF) const
    {
        using namespace EIGEN_EXT;
        dPdF.setZero();
        addScaledCofactorMatrixDerivative(s.F, gamma, dPdF);
    }

    void firstPiolaDifferential(const Scratch& s, const TM& dF, TM& dP) const
    {
        using namespace EIGEN_EXT;
        dP.setZero();
        addScaledCofactorMatrixDifferential(s.F, dF, gamma, dP);
    }

    bool isC2(const Scratch& s, T tolerance) const
    {
        return s.J > tolerance;
    }

    constexpr static bool diagonalDifferentiable()
    {
        return false;
        // TODO change to true and implement diagonal version
    }

    void write(std::ostream& out) const
    {
        writeEntry(out, gamma);
    }

    static SurfaceTension<T, dim> read(std::istream& in)
    {
        SurfaceTension<T, dim> model;
        model.gamma = readEntry<T>(in);
        return model;
    }

    static const char* name()
    {
        return "SurfaceTension";
    }

    static const char* scratch_name()
    {
        return Scratch::name();
    }
};

template <class T, int dim>
struct HyperelasticTraits<SurfaceTension<T, dim>> {
    using ScratchType = SurfaceTensionScratch<T, dim>;
};

template <class T, int dim>
struct RW<SurfaceTensionScratch<T, dim>> {
    using Tag = NoWriteTag<SurfaceTensionScratch<T, dim>>;
};
} // namespace ZIRAN
#endif
