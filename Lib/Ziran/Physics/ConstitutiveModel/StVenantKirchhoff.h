#ifndef ST_VENANT_KIRCHHOFF_H
#define ST_VENANT_KIRCHHOFF_H

#include <iostream>
#include <Ziran/CS/Util/Forward.h>
#include <Ziran/CS/Util/BinaryIO.h>
#include <Ziran/Physics/ConstitutiveModel/HyperelasticConstitutiveModel.h>

namespace ZIRAN {

template <typename Derived>
struct HyperelasticTraits;

template <class T, int dim>
struct StVenantKirchhoffScratch {
    using TM = Matrix<T, dim, dim>;
    using TV = Vector<T, dim>;
    TM F, U, V;
    TV sigma, sigma_square_m1;
    T trE;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    StVenantKirchhoffScratch()
        : F(TM::Identity())
    {
    }

    static const char* name()
    {
        return "StVenantKirchhoffScratch";
    }
};

template <class T, int _dim>
class StVenantKirchhoff : public HyperelasticConstitutiveModel<StVenantKirchhoff<T, _dim>> {
public:
    static const int dim = _dim;
    using Base = HyperelasticConstitutiveModel<StVenantKirchhoff<T, dim>>;
    using TV = typename Base::TV;
    using TM = typename Base::TM;
    using Strain = TM;
    using Hessian = typename Base::Hessian;
    using Scalar = typename Base::Scalar;
    using Scratch = typename HyperelasticTraits<StVenantKirchhoff<T, dim>>::ScratchType;

    T mu, lambda;

    StVenantKirchhoff(const T E = (T)1, const T nu = (T)0.3);

    void setLameParameters(const T E, const T nu);

    void updateScratch(const TM& new_F, Scratch& scratch);

    static constexpr bool diagonalDifferentiable() { return true; }

    T psi(const Scratch& s) const;

    void firstPiola(const Scratch& s, TM& P) const;

    void firstPiolaDerivative(const Scratch& s, Hessian& dPdF) const;

    bool isC2(const Scratch& s, T tolerance) const;

    void write(std::ostream& out) const;

    static StVenantKirchhoff<T, _dim> read(std::istream& in);

    static const char* name() { return "StVenantKirchhoff"; }

    static const char* scratch_name() { return Scratch::name(); }
};

template <class T, int dim>
struct HyperelasticTraits<StVenantKirchhoff<T, dim>> {
    using ScratchType = StVenantKirchhoffScratch<T, dim>;
};

template <class T, int dim>
struct RW<StVenantKirchhoffScratch<T, dim>> {
    using Tag = NoWriteTag<StVenantKirchhoffScratch<T, dim>>;
};
} // namespace ZIRAN

#endif
