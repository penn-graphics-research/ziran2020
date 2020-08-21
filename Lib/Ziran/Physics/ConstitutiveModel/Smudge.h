#ifndef SMUDGE_H
#define SMUDGE_H
#include <Ziran/Physics/ConstitutiveModel/SvdBasedIsotropicHelper.h>
#include <Ziran/CS/Util/Forward.h>
#include <Ziran/CS/Util/BinaryIO.h>
#include <Ziran/CS/DataStructure/DataManager.h>
#include <Ziran/Math/MathTools.h>
#include <tick/requires.h>

namespace ZIRAN {

template <class Derived>
class HyperelasticConstitutiveModel;

template <typename Derived>
struct ScratchTrait;

template <class T, int _dim>
class Smudge;

// scratch (non-state) variables for the consitutive model
template <class T, int dim>
struct SmudgeScratch {
    using TM = Matrix<T, dim, dim>;
    using TV = Vector<T, dim>;
    TM F, U, V;
    TV sigma;
    TV logS;
    SvdBasedIsotropicHelper<T, dim> isotropic;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SmudgeScratch()
        : isotropic(0)
    {
    }

    static const char* name()
    {
        return "SmudgeScratch";
    }
};

template <class T, int _dim>
class Smudge : public HyperelasticConstitutiveModel<Smudge<T, _dim>> {
public:
    static const int dim = _dim;
    static constexpr T eps = (T)1e-6;
    static constexpr T quartic_scale = 6.254421582537118;

    using Base = HyperelasticConstitutiveModel<Smudge<T, dim>>;
    using TM = typename Base::TM;
    using TV = typename Base::TV;
    using Strain = TM;
    using Hessian = typename Base::Hessian;
    using Scalar = typename Base::Scalar;
    using Scratch = typename HyperelasticTraits<Smudge<T, dim>>::ScratchType;
    using Base::firstPiolaDerivative; // TODO: a more efficient version
    using Vec = Vector<T, Eigen::Dynamic>;
    using VecBlock = Eigen::VectorBlock<Vec>;

    T mu, lambda;
    bool unilateral;

    Smudge(const T E = (T)1, const T nu = (T)0.3, const bool unilateral = true)
        : unilateral(unilateral)
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

    T divDifHelper(const T& fi, const T& fj, const T& log_fi, const T& log_fj) const
    {
        using namespace MATH_TOOLS;

        T mij = 0;
        T threshold = (T)(10 * std::numeric_limits<float>::epsilon());
        if (unilateral) {
            // We only deviate from 0 if (log_fi >= 0 && log_fj < 0) or the log's have different sign and they are separated by a certain threshold
            if (log_fi * log_fj <= 0 && (std::abs(fi - fj) >= threshold)) {
                // regular formula here
                mij = -4 * mu * quartic_scale * cube(log_fj) / (fj * (fi - fj));
            }
            else if (log_fi < 0 && log_fj < 0) {
                // magic formula here
                if (std::abs(fi - fj) >= threshold) {
                    mij = -4 * mu * quartic_scale * (fi * cube(log_fj) - fj * cube(log_fi)) / (fi * fj * (fi - fj));
                }
                else {
                    mij = -4 * mu * quartic_scale * (-3 + log_fj) * sqr(log_fj) / sqr(fj);
                }
            }
        }
        else {
            if (std::abs(fi - fj) >= threshold) {
                mij = -4 * mu * quartic_scale * (fi * cube(log_fj) - fj * cube(log_fi)) / (fi * fj * (fi - fj));
            }
            else {
                mij = -4 * mu * quartic_scale * (-3 + log_fj) * sqr(log_fj) / sqr(fj);
            }
        }

        return mij;
    }

    TICK_MEMBER_REQUIRES(dim == 1)
    void updateScratch(const TM& new_F, Scratch& s) const
    {
        ZIRAN_ASSERT(0, "1D smudge model is not supported.");
    }

    TICK_MEMBER_REQUIRES(dim == 2)
    void updateScratch(const TM& new_F, Scratch& s) const
    {
        using namespace MATH_TOOLS;
        updateScratchSVD(new_F, s);
        //T g = 2 * mu + lambda;
        T prod = s.sigma(0) * s.sigma(1);
        s.isotropic.psi0 = 0;
        s.isotropic.psi1 = 0;
        s.isotropic.psi00 = 0;
        s.isotropic.psi11 = 0;
        s.isotropic.psi01 = 0;
        s.isotropic.m01 = 0;

        // lambda term
        T alambda = quartic_scale * lambda;
        T logsum = s.logS(0) + s.logS(1);
        if (unilateral) {
            if (prod < 1) {
                s.isotropic.psi0 += 2 * alambda * cube(logsum) / s.sigma(0);
                s.isotropic.psi1 += 2 * alambda * cube(logsum) / s.sigma(1);
                s.isotropic.psi00 -= 2 * alambda * (-3 + logsum) * sqr(logsum) / sqr(s.sigma(0));
                s.isotropic.psi11 -= 2 * alambda * (-3 + logsum) * sqr(logsum) / sqr(s.sigma(1));
                s.isotropic.psi01 += 6 * alambda * sqr(logsum) / prod;
                s.isotropic.m01 -= 2 * alambda * cube(logsum) / prod; // (psi0-psi1)/(sigma0-sigma1)
            }
            // mu term
            if (s.sigma(0) < 1) {
                s.isotropic.psi0 += 4 * mu * quartic_scale * cube(s.logS(0)) / s.sigma(0);
                s.isotropic.psi00 -= (4 * mu * quartic_scale * (-3 + s.logS(0)) * sqr(s.logS(0))) / sqr(s.sigma(0));
            }
            if (s.sigma(1) < 1) {
                s.isotropic.psi1 += 4 * mu * quartic_scale * cube(s.logS(1)) / s.sigma(1);
                s.isotropic.psi11 -= (4 * mu * quartic_scale * (-3 + s.logS(1)) * sqr(s.logS(1))) / sqr(s.sigma(1));
            }
        }
        else {
            // lambda term
            s.isotropic.psi0 += 2 * alambda * cube(logsum) / s.sigma(0);
            s.isotropic.psi1 += 2 * alambda * cube(logsum) / s.sigma(1);
            s.isotropic.psi00 -= 2 * alambda * (-3 + logsum) * sqr(logsum) / sqr(s.sigma(0));
            s.isotropic.psi11 -= 2 * alambda * (-3 + logsum) * sqr(logsum) / sqr(s.sigma(1));
            s.isotropic.psi01 += 6 * alambda * sqr(logsum) / prod;
            s.isotropic.m01 -= 2 * alambda * cube(logsum) / prod; // (psi0-psi1)/(sigma0-sigma1)
            // mu term
            s.isotropic.psi0 += 4 * mu * quartic_scale * cube(s.logS(0)) / s.sigma(0);
            s.isotropic.psi00 -= (4 * mu * quartic_scale * (-3 + s.logS(0)) * sqr(s.logS(0))) / sqr(s.sigma(0));
            s.isotropic.psi1 += 4 * mu * quartic_scale * cube(s.logS(1)) / s.sigma(1);
            s.isotropic.psi11 -= (4 * mu * quartic_scale * (-3 + s.logS(1)) * sqr(s.logS(1))) / sqr(s.sigma(1));
        }

        // (psi0-psi1)/(sigma0-sigma1)
        s.isotropic.m01 += divDifHelper(s.sigma(0), s.sigma(1), s.logS(0), s.logS(1));

        // (psi0+psi1)/(sigma0+sigma1)
        s.isotropic.p01 = (s.isotropic.psi0 + s.isotropic.psi1) / clamp_small_magnitude(s.sigma(0) + s.sigma(1), eps);
    }

    TICK_MEMBER_REQUIRES(dim == 3)
    void updateScratch(const TM& new_F, Scratch& s) const
    {
        using namespace MATH_TOOLS;
        updateScratchSVD(new_F, s);
        //T g = 2 * mu + lambda;
        T logsum = s.logS(0) + s.logS(1) + s.logS(2);
        T prod01 = s.sigma(0) * s.sigma(1);
        T prod02 = s.sigma(0) * s.sigma(2);
        T prod12 = s.sigma(1) * s.sigma(2);
        T prod = s.sigma(0) * s.sigma(1) * s.sigma(2);

        // zero out all components
        s.isotropic.psi0 = 0;
        s.isotropic.psi1 = 0;
        s.isotropic.psi2 = 0;
        s.isotropic.psi00 = 0;
        s.isotropic.psi11 = 0;
        s.isotropic.psi22 = 0;
        s.isotropic.psi01 = 0;
        s.isotropic.psi02 = 0;
        s.isotropic.psi12 = 0;
        s.isotropic.m01 = 0;
        s.isotropic.m02 = 0;
        s.isotropic.m12 = 0;

        T alambda = quartic_scale * lambda;

        if (unilateral) {
            // lambda term
            if (prod < 1) {
                s.isotropic.psi0 += 2 * alambda * cube(logsum) / s.sigma(0);
                s.isotropic.psi1 += 2 * alambda * cube(logsum) / s.sigma(1);
                s.isotropic.psi2 += 2 * alambda * cube(logsum) / s.sigma(2);
                s.isotropic.psi00 -= 2 * alambda * (-3 + logsum) * sqr(logsum) / sqr(s.sigma(0));
                s.isotropic.psi11 -= 2 * alambda * (-3 + logsum) * sqr(logsum) / sqr(s.sigma(1));
                s.isotropic.psi22 -= 2 * alambda * (-3 + logsum) * sqr(logsum) / sqr(s.sigma(2));

                s.isotropic.psi01 += 6 * alambda * sqr(logsum) / prod01;
                s.isotropic.psi02 += 6 * alambda * sqr(logsum) / prod02;
                s.isotropic.psi12 += 6 * alambda * sqr(logsum) / prod12;

                // (psiA-psiB)/(sigmaA-sigmaB)
                s.isotropic.m01 += -2 * alambda * cube(logsum) / prod01;
                s.isotropic.m02 += -2 * alambda * cube(logsum) / prod02;
                s.isotropic.m12 += -2 * alambda * cube(logsum) / prod12;
            }

            // mu term
            if (s.sigma(0) < 1) {
                s.isotropic.psi0 += 4 * mu * quartic_scale * cube(s.logS(0)) / s.sigma(0);
                s.isotropic.psi00 -= (4 * mu * quartic_scale * (-3 + s.logS(0)) * sqr(s.logS(0))) / sqr(s.sigma(0));
            }
            if (s.sigma(1) < 1) {
                s.isotropic.psi1 += 4 * mu * quartic_scale * cube(s.logS(1)) / s.sigma(1);
                s.isotropic.psi11 -= (4 * mu * quartic_scale * (-3 + s.logS(1)) * sqr(s.logS(1))) / sqr(s.sigma(1));
            }
            if (s.sigma(2) < 1) {
                s.isotropic.psi2 += 4 * mu * quartic_scale * cube(s.logS(2)) / s.sigma(2);
                s.isotropic.psi22 -= (4 * mu * quartic_scale * (-3 + s.logS(2)) * sqr(s.logS(2))) / sqr(s.sigma(2));
            }
        }
        else {
            // lambda term
            s.isotropic.psi0 += 2 * alambda * cube(logsum) / s.sigma(0);
            s.isotropic.psi1 += 2 * alambda * cube(logsum) / s.sigma(1);
            s.isotropic.psi2 += 2 * alambda * cube(logsum) / s.sigma(2);
            s.isotropic.psi00 -= 2 * alambda * (-3 + logsum) * sqr(logsum) / sqr(s.sigma(0));
            s.isotropic.psi11 -= 2 * alambda * (-3 + logsum) * sqr(logsum) / sqr(s.sigma(1));
            s.isotropic.psi22 -= 2 * alambda * (-3 + logsum) * sqr(logsum) / sqr(s.sigma(2));
            s.isotropic.psi01 += 6 * alambda * sqr(logsum) / prod01;
            s.isotropic.psi02 += 6 * alambda * sqr(logsum) / prod02;
            s.isotropic.psi12 += 6 * alambda * sqr(logsum) / prod12;
            // (psiA-psiB)/(sigmaA-sigmaB)
            s.isotropic.m01 += -2 * alambda * cube(logsum) / prod01;
            s.isotropic.m02 += -2 * alambda * cube(logsum) / prod02;
            s.isotropic.m12 += -2 * alambda * cube(logsum) / prod12;

            // mu term
            s.isotropic.psi0 += 4 * mu * quartic_scale * cube(s.logS(0)) / s.sigma(0);
            s.isotropic.psi00 -= (4 * mu * quartic_scale * (-3 + s.logS(0)) * sqr(s.logS(0))) / sqr(s.sigma(0));
            s.isotropic.psi1 += 4 * mu * quartic_scale * cube(s.logS(1)) / s.sigma(1);
            s.isotropic.psi11 -= (4 * mu * quartic_scale * (-3 + s.logS(1)) * sqr(s.logS(1))) / sqr(s.sigma(1));
            s.isotropic.psi2 += 4 * mu * quartic_scale * cube(s.logS(2)) / s.sigma(2);
            s.isotropic.psi22 -= (4 * mu * quartic_scale * (-3 + s.logS(2)) * sqr(s.logS(2))) / sqr(s.sigma(2));
        }

        // (psiA-psiB)/(sigmaA-sigmaB)
        s.isotropic.m01 += divDifHelper(s.sigma(0), s.sigma(1), s.logS(0), s.logS(1));
        s.isotropic.m02 += divDifHelper(s.sigma(0), s.sigma(2), s.logS(0), s.logS(2));
        s.isotropic.m12 += divDifHelper(s.sigma(1), s.sigma(2), s.logS(1), s.logS(2));

        // (psiA+psiB)/(sigmaA+sigmaB)
        s.isotropic.p01 = (s.isotropic.psi0 + s.isotropic.psi1) / clamp_small_magnitude(s.sigma(0) + s.sigma(1), eps);
        s.isotropic.p02 = (s.isotropic.psi0 + s.isotropic.psi2) / clamp_small_magnitude(s.sigma(0) + s.sigma(2), eps);
        s.isotropic.p12 = (s.isotropic.psi1 + s.isotropic.psi2) / clamp_small_magnitude(s.sigma(1) + s.sigma(2), eps);
    }

    static constexpr bool diagonalDifferentiable() // TODO: remove this function from taiziran
    {
        return false;
    }

    /**
       psi = mu tr((log S)^2) + 1/2 lambda (tr(log S))^2
     */
    T psi(const Scratch& s) const
    {
        using namespace MATH_TOOLS;
        T mu_term = 0;
        T lambda_term = 0;
        for (int i = 0; i < dim; ++i) {
            if (unilateral) {
                if (s.logS(i) < 0)
                    mu_term += quartic_scale * cube(s.logS(i)) * s.logS(i);
            }
            else {
                mu_term += quartic_scale * cube(s.logS(i)) * s.logS(i);
            }
        }
        T trace_logS = s.logS.array().sum();
        if (unilateral) {
            lambda_term = trace_logS < 0 ? (T).5 * lambda * quartic_scale * cube(trace_logS) * trace_logS : 0;
        }
        else {
            lambda_term = (T).5 * lambda * quartic_scale * cube(trace_logS) * trace_logS;
        }
        return mu * mu_term + lambda_term;
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

    static Smudge<T, dim> read(std::istream& in)
    {
        Smudge<T, dim> model;
        model.mu = readEntry<T>(in);
        model.lambda = readEntry<T>(in);
        return model;
    }

    static const char* name()
    {
        return "Smudge";
    }

    inline static AttributeName<Smudge<T, dim>> attributeName()
    {
        return AttributeName<Smudge<T, dim>>("Smudge");
    }

    static const char* scratch_name()
    {
        return Scratch::name();
    }
};

template <class T, int dim>
struct HyperelasticTraits<Smudge<T, dim>> {
    using ScratchType = SmudgeScratch<T, dim>;
};

template <class T, int dim>
struct RW<SmudgeScratch<T, dim>> {
    using Tag = NoWriteTag<SmudgeScratch<T, dim>>;
};
} // namespace ZIRAN

#endif
