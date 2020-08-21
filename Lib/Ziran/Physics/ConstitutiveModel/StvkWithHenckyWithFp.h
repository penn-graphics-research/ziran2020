#ifndef STVK_WITH_HENCKY_WITH_FP_H
#define STVK_WITH_HENCKY_WITH_FP_H
#include <iostream>
#include <Ziran/CS/Util/Forward.h>
#include <Ziran/Physics/ConstitutiveModel/SvdBasedIsotropicHelper.h>
#include <Ziran/CS/Util/BinaryIO.h>
#include <Ziran/Physics/ConstitutiveModel/HyperelasticConstitutiveModel.h>

namespace ZIRAN {
template <class T, int dim>
struct StvkWithHenckyWithFpScratch {
    using TM = Matrix<T, dim, dim>;
    using TV = Vector<T, dim>;
    TM F, U, V;
    TV sigma;
    TV log_sigma;
    SvdBasedIsotropicHelper<T, dim> isotropic;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    StvkWithHenckyWithFpScratch()
        : isotropic(0)
    {
    }

    static const char* name()
    {
        return "StvkWithHenckyWithFpScratch";
    }
};

template <class T, int _dim>
class StvkWithHenckyWithFp : public HyperelasticConstitutiveModel<StvkWithHenckyWithFp<T, _dim>> {
public:
    static const int dim = _dim;
    static constexpr T eps = (T)1e-6;
    using Base = HyperelasticConstitutiveModel<StvkWithHenckyWithFp<T, _dim>>;
    using TM = typename Base::TM;
    using TV = typename Base::TV;

    using Strain = TM;
    using Hessian = typename Base::Hessian;
    using Scalar = typename Base::Scalar;
    using Scratch = typename HyperelasticTraits<StvkWithHenckyWithFp<T, dim>>::ScratchType;
    using Base::firstPiolaDerivative;
    using Vec = Vector<T, Eigen::Dynamic>;
    using VecBlock = Eigen::VectorBlock<Vec>;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Matrix<T, dim, dim, Eigen::DontAlign> Fp_inv; // this is different from StvkWithHencky
    T mu, lambda;
    T damage_scale = 1;

    StvkWithHenckyWithFp(const T E = (T)1, const T nu = (T)0.3);

    void setLameParameters(const T E, const T nu);

    void updateScratchSVD(const TM& new_F, Scratch& s) const; // private

    void updateScratch(const TM& new_F, Scratch& s) const;
    static constexpr bool diagonalDifferentiable()
    {
        return false; // this is different from StvkWithHencky
    }

    /**
       psi = mu tr((log S)^2) + 1/2 lambda (tr(log S))^2
     */
    T psi(const Scratch& s) const;

    /**
       P = U (2 mu S^{-1} (log S) + lambda tr(log S) S^{-1}) V^T
     */
    void firstPiola(const Scratch& s, TM& P) const;

    void firstPiolaDifferential(const Scratch& s, const TM& dF, TM& dP) const;

    bool isC2(const Scratch& s, T tolerance) const;

    /**
      Returns whether dP (or dPdF) is implemented
      */
    bool hessianImplemented() const;

    void write(std::ostream& out) const;

    static StvkWithHenckyWithFp<T, _dim> read(std::istream& in);

    static const char* name();

    static const char* scratch_name();
};

template <class T, int dim>
struct HyperelasticTraits<StvkWithHenckyWithFp<T, dim>> {
    using ScratchType = StvkWithHenckyWithFpScratch<T, dim>;
};

template <class T, int dim>
struct RW<StvkWithHenckyWithFp<T, dim>> {
    using Tag = CustomTypeTag<StvkWithHenckyWithFp<T, dim>>;
};

template <class T, int dim>
struct RW<StvkWithHenckyWithFpScratch<T, dim>> {
    using Tag = NoWriteTag<StvkWithHenckyWithFpScratch<T, dim>>;
};

extern template class StvkWithHenckyWithFp<float, 2>;
extern template class StvkWithHenckyWithFp<float, 3>;
extern template class StvkWithHenckyWithFp<double, 2>;
extern template class StvkWithHenckyWithFp<double, 3>;
} // namespace ZIRAN

#endif
