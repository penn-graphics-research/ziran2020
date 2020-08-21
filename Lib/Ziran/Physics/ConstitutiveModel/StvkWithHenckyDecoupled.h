#ifndef STVK_WITH_HENCKY_DECOUPLED_H
#define STVK_WITH_HENCKY_DECOUPLED_H
#include <Ziran/Physics/ConstitutiveModel/SvdBasedIsotropicHelper.h>
#include <Ziran/CS/Util/Forward.h>
#include <Ziran/CS/Util/BinaryIO.h>
#include <Ziran/Math/MathTools.h>
#include <tick/requires.h>
#include <limits>

namespace ZIRAN {

template <class Derived>
class HyperelasticConstitutiveModel;

template <typename Derived>
struct ScratchTrait;

template <class T, int _dim>
class StvkWithHenckyDecoupled;

// scratch (non-state) variables for the consitutive model
template <class T, int dim>
struct StvkWithHenckyDecoupledScratch {
    using TM = Matrix<T, dim, dim>;
    using TV = Vector<T, dim>;
    TM F, U, V;
    TV sigma;
    TV logS;
    SvdBasedIsotropicHelper<T, dim> isotropic;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    StvkWithHenckyDecoupledScratch()
        : isotropic(0)
    {
    }

    static const char* name()
    {
        return "StvkWithHenckyDecoupledScratch";
    }
};

template <class T, int _dim>
class StvkWithHenckyDecoupled : public HyperelasticConstitutiveModel<StvkWithHenckyDecoupled<T, _dim>> {
public:
    static const int dim = _dim;
    static constexpr T eps = (T)1e-6;
    using Base = HyperelasticConstitutiveModel<StvkWithHenckyDecoupled<T, dim>>;
    using TM = typename Base::TM;
    using TV = typename Base::TV;
    using Strain = TM;
    using Hessian = typename Base::Hessian;
    using Scalar = typename Base::Scalar;
    using Scratch = typename HyperelasticTraits<StvkWithHenckyDecoupled<T, dim>>::ScratchType;
    using Base::firstPiolaDerivative; // TODO: a more efficient version
    using Vec = Vector<T, Eigen::Dynamic>;
    using VecBlock = Eigen::VectorBlock<Vec>;

    T mu, lambda;

    StvkWithHenckyDecoupled(const T E = (T)1, const T nu = (T)0.3)
    {
        setLameParameters(E, nu);
    }

    void setLameParameters(const T E, const T nu)
    {
        lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
        mu = E / ((T)2 * ((T)1 + nu));
    }

    void updateScratchSVD(const TM& new_F, Scratch& s) const // private
    {
        s.F = new_F;
        singularValueDecomposition(s.F, s.U, s.sigma, s.V);
        s.logS = s.sigma.array().abs().log();
    }

    TICK_MEMBER_REQUIRES(dim == 1)
    void updateScratch(const TM& new_F, Scratch& s) const
    {
        using namespace MATH_TOOLS;
        updateScratchSVD(new_F, s);
        T g = 2 * mu + lambda;
        T one_over_F = 1 / new_F(0, 0);
        s.isotropic.psi0 = g * one_over_F;
        s.isotropic.psi00 = g * sqr(one_over_F);
    }

    TICK_MEMBER_REQUIRES(dim == 2)
    void updateScratch(const TM& new_F, Scratch& s) const
    {
        using namespace MATH_TOOLS;
        updateScratchSVD(new_F, s);
        //T g = 2 * mu + lambda;
        T prod = s.sigma(0) * s.sigma(1);
        s.isotropic.psi0 = (lambda * (s.logS(0) + s.logS(1)) + mu * (s.logS(0) - s.logS(1))) / (s.sigma(0));
        s.isotropic.psi1 = (lambda * (s.logS(0) + s.logS(1)) + mu * (s.logS(1) - s.logS(0))) / (s.sigma(1));
        s.isotropic.psi00 = (2 * (mu + lambda) - 2 * mu * (s.logS(0) - s.logS(1)) - 2 * lambda * (s.logS(0) + s.logS(1))) / (2 * sqr(s.sigma(0)));
        s.isotropic.psi11 = (2 * (mu + lambda) + 2 * mu * (s.logS(0) - s.logS(1)) - 2 * lambda * (s.logS(0) + s.logS(1))) / (2 * sqr(s.sigma(1)));
        s.isotropic.psi01 = (lambda - mu) / prod;

        // (psi0-psi1)/(sigma0-sigma1)
        T local_eps = 10 * std::numeric_limits<T>::epsilon();
        if (std::abs(s.sigma(0) - s.sigma(1)) > local_eps)
            s.isotropic.m01 = (2 * (s.sigma(1) - s.sigma(0)) * lambda * s.logS(0) + (s.sigma(0) + s.sigma(1)) * mu * (s.logS(0) - s.logS(1)) + 2 * (s.sigma(1) - s.sigma(0)) * lambda * s.logS(1) - (s.sigma(0) + s.sigma(1)) * mu * (s.logS(1) - s.logS(0))) / (2 * prod * (s.sigma(0) - s.sigma(1)));
        else
            s.isotropic.m01 = -((2 * lambda * s.logS(0)) / sqr(s.sigma(0))) + (2 * mu) / sqr(s.sigma(0));

        // (psi0+psi1)/(sigma0+sigma1)
        s.isotropic.p01 = (s.isotropic.psi0 + s.isotropic.psi1) / clamp_small_magnitude(s.sigma(0) + s.sigma(1), eps);
    }

    TICK_MEMBER_REQUIRES(dim == 3)
    void updateScratch(const TM& new_F, Scratch& s) const
    {
        using namespace MATH_TOOLS;
        updateScratchSVD(new_F, s);
        T g = 2 * mu + lambda;
        T sum_log = s.logS(0) + s.logS(1) + s.logS(2);
        T prod01 = s.sigma(0) * s.sigma(1);
        T prod02 = s.sigma(0) * s.sigma(2);
        T prod12 = s.sigma(1) * s.sigma(2);
        s.isotropic.psi0 = (2 * mu * s.logS(0) + lambda * sum_log) / s.sigma(0);
        s.isotropic.psi1 = (2 * mu * s.logS(1) + lambda * sum_log) / s.sigma(1);
        s.isotropic.psi2 = (2 * mu * s.logS(2) + lambda * sum_log) / s.sigma(2);
        s.isotropic.psi00 = (g * (1 - s.logS(0)) - lambda * (s.logS(1) + s.logS(2))) / sqr(s.sigma(0));
        s.isotropic.psi11 = (g * (1 - s.logS(1)) - lambda * (s.logS(0) + s.logS(2))) / sqr(s.sigma(1));
        s.isotropic.psi22 = (g * (1 - s.logS(2)) - lambda * (s.logS(0) + s.logS(1))) / sqr(s.sigma(2));
        s.isotropic.psi01 = lambda / (s.sigma(0) * s.sigma(1));
        s.isotropic.psi02 = lambda / (s.sigma(0) * s.sigma(2));
        s.isotropic.psi12 = lambda / (s.sigma(1) * s.sigma(2));

        // (psiA-psiB)/(sigmaA-sigmaB)
        s.isotropic.m01 = -(lambda * sum_log + 2 * mu * diff_interlock_log_over_diff(s.sigma(0), s.sigma(1), s.logS(1), eps)) / prod01;
        s.isotropic.m02 = -(lambda * sum_log + 2 * mu * diff_interlock_log_over_diff(s.sigma(0), s.sigma(2), s.logS(2), eps)) / prod02;
        s.isotropic.m12 = -(lambda * sum_log + 2 * mu * diff_interlock_log_over_diff(s.sigma(1), s.sigma(2), s.logS(2), eps)) / prod12;

        // (psiA+psiB)/(sigmaA+sigmaB)
        s.isotropic.p01 = (s.isotropic.psi0 + s.isotropic.psi1) / clamp_small_magnitude(s.sigma(0) + s.sigma(1), eps);
        s.isotropic.p02 = (s.isotropic.psi0 + s.isotropic.psi2) / clamp_small_magnitude(s.sigma(0) + s.sigma(2), eps);
        s.isotropic.p12 = (s.isotropic.psi1 + s.isotropic.psi2) / clamp_small_magnitude(s.sigma(1) + s.sigma(2), eps);
    }

    static constexpr bool diagonalDifferentiable()
    {
        return false; //TODO implement diagonal functions
    }

    /**
       psi
     */
    T psi(const Scratch& s) const
    {
        using namespace MATH_TOOLS;
        return (mu / 2) * (sqr(s.logS(0) - s.logS(1))) + (lambda / 2) * sqr(s.logS(0) + s.logS(1));
    }

    void firstPiola(const Scratch& s, TM& P) const
    {
        TV P_hat;
        s.isotropic.computePHat(P_hat);
        P = s.U * P_hat.asDiagonal() * s.V.transpose();
    }

    void firstPiolaDifferential(const Scratch& s, const TM& dF, TM& dP) const
    {
        TM D = s.U.transpose() * dF * s.V;
        TM K;
        s.isotropic.dPdFOfSigmaContract(D, K);
        dP = s.U * K * s.V.transpose();
    }

    bool isC2(const Scratch& s, T tolerance) const
    {
        return s.sigma.prod() > tolerance; // due to the log sigma term
    }

    /**
       Returns whether dP (or dPdF) is implemented
    */
    bool hessianImplemented() const
    {
        return true;
    }

    void write(std::ostream& out) const
    {
        writeEntry(out, mu);
        writeEntry(out, lambda);
    }

    static StvkWithHenckyDecoupled<T, dim> read(std::istream& in)
    {
        StvkWithHenckyDecoupled<T, dim> model;
        model.mu = readEntry<T>(in);
        model.lambda = readEntry<T>(in);
        return model;
    }

    static const char* name()
    {
        return "StvkWithHenckyDecoupled";
    }

    inline static AttributeName<StvkWithHenckyDecoupled<T, dim>> attributeName()
    {
        return AttributeName<StvkWithHenckyDecoupled<T, dim>>("StvkWithHenckyDecoupled");
    }

    static const char* scratch_name()
    {
        return Scratch::name();
    }
};

template <class T, int dim>
struct HyperelasticTraits<StvkWithHenckyDecoupled<T, dim>> {
    using ScratchType = StvkWithHenckyDecoupledScratch<T, dim>;
};

template <class T, int dim>
struct RW<StvkWithHenckyDecoupledScratch<T, dim>> {
    using Tag = NoWriteTag<StvkWithHenckyDecoupledScratch<T, dim>>;
};
} // namespace ZIRAN

#endif
