#ifndef COROTATED_UNILATERAL_H
#define COROTATED_UNILATERAL_H
#include <Ziran/Physics/ConstitutiveModel/SvdBasedIsotropicHelper.h>

namespace ZIRAN {

template <class Derived>
class HyperelasticConstitutiveModel;

template <typename Derived>
struct ScratchTrait;

template <class T, int _dim>
class CorotatedUnilateral;

// scratch (non-state) variables for the consitutive model
template <class T, int dim>
struct CorotatedUnilateralScratch {
    using TM = Matrix<T, dim, dim>;
    using TV = Vector<T, dim>;
    TM F, U, V;
    TV sigma;
    SvdBasedIsotropicHelper<T, dim> isotropic;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CorotatedUnilateralScratch()
    {
    }

    static const char* name()
    {
        return "CorotatedUnilateralScratch";
    }
};

template <class T, int _dim>
class CorotatedUnilateral : public HyperelasticConstitutiveModel<CorotatedUnilateral<T, _dim>> {
public:
    static const int dim = _dim;

    static constexpr T eps_collapse_sigma = (T)1e-6;
    static constexpr T eps_c2_extension = (T)1e-15;

    using Base = HyperelasticConstitutiveModel<CorotatedUnilateral<T, dim>>;
    using TM = typename Base::TM;
    using TV = typename Base::TV;
    using Strain = TM;
    using Hessian = typename Base::Hessian;
    using Scalar = typename Base::Scalar;
    using Scratch = typename HyperelasticTraits<CorotatedUnilateral<T, dim>>::ScratchType;
    using Base::firstPiolaDerivative; // TODO: a more efficient version
    using Vec = Vector<T, Eigen::Dynamic>;
    using VecBlock = Eigen::VectorBlock<Vec>;

    T mu, lambda, compress_level;

    CorotatedUnilateral(const T E = (T)1, const T nu = (T)0.3, const T compress_level_in = (T)0)
    {
        setLameParameters(E, nu);
        compress_level = compress_level_in;
    }

    void setLameParameters(const T E, const T nu)
    {
        lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
        mu = E / ((T)2 * ((T)1 + nu));

        // // Turn off lambda term!
        // lambda = 0;

        // Turn off mu term!
        // mu = 0;
    }

    void updateScratchSVD(const TM& new_F, Scratch& s) // private
    {
        s.F = new_F;
        singularValueDecomposition(s.F, s.U, s.sigma, s.V);
    }

    // h(x) = g(x+compress_level)
    static void hEval(const T x, const T s, T& g, T& gp, T& gpp)
    {
        gEval(x + s, g, gp, gpp);
    }

    // g(x) = x^2 sqrt(x^2+eps)   ... x<=0
    //      = 0                   ... x>0
    static void gEval(const T x, T& g, T& gp, T& gpp)
    {
        using MATH_TOOLS::sqr;
        using std::sqrt;
        T x2 = sqr(x);
        T x3 = x2 * x;
        if (x < 0) {
            T zz = sqrt(x2 + eps_c2_extension);
            g = x2 * zz;
            gp = (3 * x3 + 2 * x * eps_c2_extension) / zz;
            gpp = (6 * x2 + 3 * eps_c2_extension) / zz;
        }
        else {
            g = 0;
            gp = 0;
            gpp = 0;
        }
        if (x == 0) {
            gpp = 2 * sqrt(eps_c2_extension);
        }
    }

    TICK_MEMBER_REQUIRES(dim == 2)
    void updateScratch(const TM& new_F, Scratch& s)
    {
        using namespace MATH_TOOLS;
        updateScratchSVD(new_F, s);

        T g0, gp0, gpp0;
        hEval(s.sigma(0) - 1, compress_level, g0, gp0, gpp0);
        T g1, gp1, gpp1;
        hEval(s.sigma(1) - 1, compress_level, g1, gp1, gpp1);
        T g01, gp01, gpp01;
        hEval(s.sigma(0) * s.sigma(1) - 1, compress_level, g01, gp01, gpp01);

        s.isotropic.psi0 = mu * gp0 + lambda / 2 * gp01 * s.sigma(1);
        s.isotropic.psi1 = mu * gp1 + lambda / 2 * gp01 * s.sigma(0);
        s.isotropic.psi00 = mu * gpp0 + lambda / 2 * gpp01 * sqr(s.sigma(1));
        s.isotropic.psi11 = mu * gpp1 + lambda / 2 * gpp01 * sqr(s.sigma(0));
        s.isotropic.psi01 = lambda / 2 * (gpp01 * s.sigma(0) * s.sigma(1) + gp01);

        // (psi0-psi1)/(sigma0-sigma1)
        if (s.sigma(0) == s.sigma(1))
            s.isotropic.m01 = mu * (gpp0 + gpp1) / 2 - lambda / 2 * gp01;
        else
            s.isotropic.m01 = (s.isotropic.psi0 - s.isotropic.psi1) / (s.sigma(0) - s.sigma(1));

        // (psi0+psi1)/(sigma0+sigma1)
        s.isotropic.p01 = (s.isotropic.psi0 + s.isotropic.psi1) / clamp_small_magnitude(s.sigma(0) + s.sigma(1), eps_collapse_sigma);
    }

    TICK_MEMBER_REQUIRES(dim == 3)
    void updateScratch(const TM& new_F, Scratch& s)
    {
        ZIRAN_ASSERT(false, "not implemented");
    }

    static constexpr bool diagonalDifferentiable()
    {
        return false; //TODO implement diagonal functions
    }

    /**
       psi = mu
     */
    T psi(const Scratch& s) const
    {
        T result = 0;

        T g01, gp01, gpp01;
        hEval(s.sigma.prod() - 1, compress_level, g01, gp01, gpp01);

        for (int i = 0; i < dim; i++) {
            T g0, gp0, gpp0;
            hEval(s.sigma(i) - 1, compress_level, g0, gp0, gpp0);
            result += mu * g0;
        }

        result += lambda / 2 * g01;

        return result;
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
        TM S = s.sigma.asDiagonal();
        return !EIGEN_EXT::nearKink(S, tolerance);
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

    static CorotatedUnilateral<T, dim> read(std::istream& in)
    {
        CorotatedUnilateral<T, dim> model;
        model.mu = readEntry<T>(in);
        model.lambda = readEntry<T>(in);
        return model;
    }

    static const char* name()
    {
        return "CorotatedUnilateral";
    }

    static const char* scratch_name()
    {
        return Scratch::name();
    }
};

template <class T, int dim>
struct HyperelasticTraits<CorotatedUnilateral<T, dim>> {
    using ScratchType = CorotatedUnilateralScratch<T, dim>;
};

template <class T, int dim>
struct RW<CorotatedUnilateralScratch<T, dim>> {
    using Tag = NoWriteTag<CorotatedUnilateralScratch<T, dim>>;
};
} // namespace ZIRAN
#endif
