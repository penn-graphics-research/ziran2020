#ifndef LINEAR_COROTATED_H
#define LINEAR_COROTATED_H
#include <Ziran/CS/Util/Forward.h>

namespace ZIRAN {

template <class T, int dim>
struct LinearCorotatedScratch {
    using TM = Matrix<T, dim, dim>;
    TM F;
    TM e_hat;
    T trace_e_hat;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    LinearCorotatedScratch()
        : F(TM::Identity())
    {
    }

    static const char* name()
    {
        return "LinearCorotatedScratch";
    }
};

template <class T, int _dim>
class LinearCorotated : public HyperelasticConstitutiveModel<LinearCorotated<T, _dim>> {
public:
    static const int dim = _dim;
    using Base = HyperelasticConstitutiveModel<LinearCorotated<T, dim>>;
    using TV = typename Base::TV;
    using TM = typename Base::TM;
    using Strain = TM;
    using Hessian = typename Base::Hessian;
    using Scalar = typename Base::Scalar;
    using Scratch = typename HyperelasticTraits<LinearCorotated<T, dim>>::ScratchType;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    T mu, lambda;
    //    Eigen::Matrix<T, dim, dim, Eigen::DontAlign> R;
    TM R;

    LinearCorotated(const T E = (T)1, const T nu = (T)0.3);

    void setLameParameters(const T E, const T nu);

    void rebuildR(const TM& new_F);

    void updateScratch(const TM& new_F, Scratch& scratch);

    static constexpr bool diagonalDifferentiable() { return false; }

    T psi(const Scratch& s) const;

    void firstPiola(const Scratch& s, TM& P) const;

    void firstPiolaDifferential(const Scratch& s, const TM& dF, TM& dP) const;

    bool isC2(const Scratch& s, T tolerance) const;

    void write(std::ostream& out) const;

    static LinearCorotated<T, _dim> read(std::istream& in);

    static const char* name() { return "LinearCorotated"; }

    static const char* scratch_name() { return Scratch::name(); }
};

template <class T, int dim>
struct HyperelasticTraits<LinearCorotated<T, dim>> {
    using ScratchType = LinearCorotatedScratch<T, dim>;
};

template <class T, int dim>
struct RW<LinearCorotated<T, dim>> {
    using Tag = CustomTypeTag<LinearCorotated<T, dim>>;
};

template <class T, int dim>
struct RW<LinearCorotatedScratch<T, dim>> {
    using Tag = NoWriteTag<LinearCorotatedScratch<T, dim>>;
};
} // namespace ZIRAN

#endif
