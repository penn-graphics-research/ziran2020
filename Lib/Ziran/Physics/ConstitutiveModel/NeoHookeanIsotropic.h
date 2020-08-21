#ifndef NEO_HOOKEAN_ISOTROPIC_H
#define NEO_HOOKEAN_ISOTROPIC_H
#include <iostream>
#include <Ziran/Physics/ConstitutiveModel/SvdBasedIsotropicHelper.h>
#include <Ziran/CS/Util/Forward.h>
#include <Ziran/CS/Util/BinaryIO.h>
#include <Ziran/Physics/ConstitutiveModel/HyperelasticConstitutiveModel.h>

namespace ZIRAN {

template <class T, int dim>
struct NeoHookeanIsotropicScratch {
    using TM = Matrix<T, dim, dim>;
    using TV = Vector<T, dim>;
    T J, logJ;
    TM F, U, V, FinvT;
    TV sigma;
    SvdBasedIsotropicHelper<T, dim> isotropic;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    NeoHookeanIsotropicScratch()
        : F(TM::Identity()), isotropic(0)
    {
    }

    static const char* name()
    {
        return "NeoHookeanIsotropicScratch";
    }
};

template <class T, int _dim>
class NeoHookeanIsotropic : public HyperelasticConstitutiveModel<NeoHookeanIsotropic<T, _dim>> {
public:
    static const int dim = _dim;
    static constexpr T eps = (T)1e-6;
    using Base = HyperelasticConstitutiveModel<NeoHookeanIsotropic<T, dim>>;
    using TV = typename Base::TV;
    using TM = typename Base::TM;
    using Strain = TM;
    using Hessian = typename Base::Hessian;
    using Scalar = typename Base::Scalar;
    using Scratch = typename HyperelasticTraits<NeoHookeanIsotropic<T, dim>>::ScratchType;

    bool project{ true };
    T mu, lambda;

    NeoHookeanIsotropic(const T E = (T)1, const T nu = (T)0.3);

    void setLameParameters(const T E, const T nu);

    void updateScratch(const TM& new_F, Scratch& scratch);

    void updateScratchSVD(const TM& new_F, Scratch& scratch) const;

    static constexpr bool diagonalDifferentiable() { return true; }

    T psi(const Scratch& s) const;

    void kirchhoff(const Scratch& s, TM& tau) const;

    void firstPiolaDerivative(const Scratch& s, Hessian& dPdF) const;

    void firstPiolaDifferential(const Scratch& s, const TM& dF, TM& dP) const;

    bool isC2(const Scratch& s, T tolerance) const;

    void write(std::ostream& out) const;

    static NeoHookeanIsotropic<T, _dim> read(std::istream& in);

    static const char* name() { return "NeoHookeanIsotropic"; }

    static const char* scratch_name() { return Scratch::name(); }
};

template <class T, int dim>
struct HyperelasticTraits<NeoHookeanIsotropic<T, dim>> {
    using ScratchType = NeoHookeanIsotropicScratch<T, dim>;
};

template <class T, int dim>
struct RW<NeoHookeanIsotropicScratch<T, dim>> {
    using Tag = NoWriteTag<NeoHookeanIsotropicScratch<T, dim>>;
};
} // namespace ZIRAN

#endif
