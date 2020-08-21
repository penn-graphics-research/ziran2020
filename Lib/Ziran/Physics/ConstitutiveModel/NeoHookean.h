#ifndef NEO_HOOKEAN_H
#define NEO_HOOKEAN_H
#include <iostream>
#include <Ziran/CS/Util/Forward.h>
#include <Ziran/CS/Util/BinaryIO.h>
#include <Ziran/Physics/ConstitutiveModel/HyperelasticConstitutiveModel.h>

namespace ZIRAN {

template <class T, int dim>
struct NeoHookeanScratch {
    using TM = Matrix<T, dim, dim>;

    T J, logJ;
    TM F, FinvT;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    NeoHookeanScratch()
        : F(TM::Identity())
    {
    }

    static const char* name()
    {
        return "NeoHookeanScratch";
    }
};

template <class T, int _dim>
class NeoHookean : public HyperelasticConstitutiveModel<NeoHookean<T, _dim>> {
public:
    static const int dim = _dim;
    using Base = HyperelasticConstitutiveModel<NeoHookean<T, dim>>;
    using TV = typename Base::TV;
    using TM = typename Base::TM;
    using Strain = TM;
    using Hessian = typename Base::Hessian;
    using Scalar = typename Base::Scalar;
    using Scratch = typename HyperelasticTraits<NeoHookean<T, dim>>::ScratchType;

    T mu, lambda;

    NeoHookean(const T E = (T)1, const T nu = (T)0.3);

    void setLameParameters(const T E, const T nu);

    void updateScratch(const TM& new_F, Scratch& scratch);

    static constexpr bool diagonalDifferentiable() { return true; }

    T psi(const Scratch& s) const;

    void kirchhoff(const Scratch& s, TM& tau) const;

    void firstPiolaDerivative(const Scratch& s, Hessian& dPdF) const;

    void firstPiolaDifferential(const Scratch& s, const TM& dF, TM& dP) const;

    bool isC2(const Scratch& s, T tolerance) const;

    void write(std::ostream& out) const;

    static NeoHookean<T, _dim> read(std::istream& in);

    static const char* name() { return "NeoHookean"; }

    static const char* scratch_name() { return Scratch::name(); }
};

template <class T, int dim>
struct HyperelasticTraits<NeoHookean<T, dim>> {
    using ScratchType = NeoHookeanScratch<T, dim>;
};

template <class T, int dim>
struct RW<NeoHookeanScratch<T, dim>> {
    using Tag = NoWriteTag<NeoHookeanScratch<T, dim>>;
};
} // namespace ZIRAN

#endif
