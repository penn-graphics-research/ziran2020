#ifndef COROTATED_DISTORTIONAL_DILATIONAL_H
#define COROTATED_DISTORTIONAL_DILATIONAL_H
#include <Ziran/CS/Util/BinaryIO.h>
#include <Ziran/CS/Util/Forward.h>
#include <Ziran/Physics/ConstitutiveModel/HyperelasticConstitutiveModel.h>
#include <Ziran/Physics/ConstitutiveModel/CorotatedElasticity.h>

namespace ZIRAN {

template <class T, int dim>
struct CorotatedDistortionalDilationalScratch {
    using TM = Matrix<T, dim, dim>;
    T J;
    TM F, F_distortional, JFinvT;
    CorotatedScratch<T, dim> corotated_scratch;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CorotatedDistortionalDilationalScratch()
        : F(TM::Identity()) {}

    static const char* name() { return "CorotatedDistortionalDilationalScratch"; }
};

template <class T, int _dim>
class CorotatedDistortionalDilational
    : public HyperelasticConstitutiveModel<
          CorotatedDistortionalDilational<T, _dim>> {
public:
    static const int dim = _dim;
    using Base = HyperelasticConstitutiveModel<CorotatedDistortionalDilational<T, dim>>;
    using TV = typename Base::TV;
    using TM = typename Base::TM;
    using Strain = TM;
    using Hessian = typename Base::Hessian;
    using Scalar = typename Base::Scalar;
    using Scratch = typename HyperelasticTraits<
        CorotatedDistortionalDilational<T, dim>>::ScratchType;

    T mu, lambda;

    CorotatedDistortionalDilational(const T E = (T)1, const T nu = (T)0.3);

    void setLameParameters(const T E, const T nu);

    void updateScratch(const TM& new_F, Scratch& scratch);

    bool hessianImplemented() const
    {
        return true;
    } // derivative test skips dPdF test if this is false

    static constexpr bool diagonalDifferentiable() { return true; }

    T psi(const Scratch& s) const;

    void firstPiola(const Scratch& s, TM& P) const;

    void firstPiolaDifferential(const Scratch& s, const TM& dF, TM& dP) const;

    bool isC2(const Scratch& s, T tolerance) const;

    void write(std::ostream& out) const;

    static CorotatedDistortionalDilational<T, _dim> read(std::istream& in);

    static const char* name() { return "CorotatedDistortionalDilational"; }

    static const char* scratch_name() { return Scratch::name(); }
};

template <class T, int dim>
struct HyperelasticTraits<CorotatedDistortionalDilational<T, dim>> {
    using ScratchType = CorotatedDistortionalDilationalScratch<T, dim>;
};

template <class T, int dim>
struct RW<CorotatedDistortionalDilationalScratch<T, dim>> {
    using Tag = NoWriteTag<CorotatedDistortionalDilationalScratch<T, dim>>;
};
} // namespace ZIRAN

#endif
