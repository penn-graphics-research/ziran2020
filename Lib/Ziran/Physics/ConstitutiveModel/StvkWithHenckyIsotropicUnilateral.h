#ifndef STVK_WITH_HENCKY_ISOTROPIC_UNILATERAL_H
#define STVK_WITH_HENCKY_ISOTROPIC_UNILATERAL_H
#include <Ziran/Physics/ConstitutiveModel/SvdBasedIsotropicHelper.h>
#include <Ziran/Physics/ConstitutiveModel/HyperelasticConstitutiveModel.h>
#include <Ziran/Math/MathTools.h>
#include <Ziran/Math/Linear/ImplicitQRSVD.h>
#include <Ziran/CS/Util/BinaryIO.h>
#include <iostream>
#include <iomanip>

namespace ZIRAN {

template <class Derived>
class HyperelasticConstitutiveModel;

template <typename Derived>
struct ScratchTrait;

template <class T, int _dim>
class StvkWithHenckyIsotropicUnilateral;

// scratch (non-state) variables for the consitutive model
template <class T, int dim>
struct StvkWithHenckyIsotropicUnilateralScratch {
    using TM = Matrix<T, dim, dim>;
    using TV = Vector<T, dim>;
    TM F, U, V;
    TV sigma;
    TV logS;
    T h;
    int region;
    SvdBasedIsotropicHelper<T, dim> isotropic;
    SvdBasedIsotropicHelper<T, dim> im; // isotropic_multiplier

    StvkWithHenckyIsotropicUnilateralScratch()
        : isotropic(0)
        , im(0)
    {
    }

    static const char* name()
    {
        return "StvkWithHenckyIsotropicUnilateralScratch";
    }
};

template <class T, int _dim>
class StvkWithHenckyIsotropicUnilateral : public HyperelasticConstitutiveModel<StvkWithHenckyIsotropicUnilateral<T, _dim>> {
public:
    static const int dim = _dim;
    static constexpr T eps = (T)1e-6;
    static constexpr T one_over_sqrt_2 = 0.70710678118;
    static constexpr T one_over_sqrt_3 = 0.57735026919;
    static constexpr T one_over_sqrt_6 = 0.40824829046;
    using Base = HyperelasticConstitutiveModel<StvkWithHenckyIsotropicUnilateral<T, dim>>;
    using Base::firstPiola;
    using Base::firstPiolaDifferential;
    using Base::psi;
    using TM = typename Base::TM;
    using TV = typename Base::TV;
    using Strain = TM;
    using Hessian = typename Base::Hessian;
    using Scalar = typename Base::Scalar;
    using Scratch = typename HyperelasticTraits<StvkWithHenckyIsotropicUnilateral<T, dim>>::ScratchType;
    using Base::firstPiolaDerivative; // TODO: a more efficient version
    using Vec = Vector<T, Eigen::Dynamic>;
    using VecBlock = Eigen::VectorBlock<Vec>;

    T mu, lambda, cohesion, a_const, b_const, k_const;

    StvkWithHenckyIsotropicUnilateral(const T E = (T)1, const T nu = (T)0.3, const T cohesion = (T)0, const T a_const = -1, const T b_const = 0, const T k_const = 1)
    {
        setLameParameters(E, nu);
        this->cohesion = cohesion;
        this->a_const = a_const;
        this->b_const = b_const;
        this->k_const = k_const;
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
        s.logS = s.sigma.array().abs().max(1e-7).log();
    }

    // 2D
    //
    void compute_H_shift(const T& z_input, T& H, T& gH, T& HH, int& region) const
    {
        if (z_input <= a_const + cohesion) {
            region = 0;
            H = 1;
            gH = 0;
            HH = 0;
        }
        else if (z_input >= b_const + cohesion) {
            region = 2;
            H = 0;
            gH = 0;
            HH = 0;
        }
        else {
            region = 1;
            T ibma = (T)1 / (b_const - a_const);
            T z = (z_input - a_const - cohesion) * ibma;
            T z2 = z * z;
            T z3 = z2 * z;
            T z4 = z3 * z;
            T z5 = z4 * z;
            H = 1 - 10 * z3 + 15 * z4 - 6 * z5;
            gH = 30 * ibma * z2 * (-1 + 2 * z - z2);
            HH = 60 * ibma * ibma * z * (-1 + 3 * z - 2 * z2);
        }
    }

    TICK_MEMBER_REQUIRES(dim == 2)
    void updateMultiplier(Scratch& s) const
    {
        using namespace ZIRAN::MATH_TOOLS;
        const TV xi_0(one_over_sqrt_2, one_over_sqrt_2);
        const TV xi_1(-one_over_sqrt_2, one_over_sqrt_2);
        TV x = s.logS;
        T u = x.dot(xi_0);
        T v = x.dot(xi_1);

        T h, ghs, Hhs;
        TV gh;
        TM Hh;

        T v3 = cube(v);
        T av3 = std::abs(v3);
        T i1pav3 = 1 / (1 + av3);
        T z = u + k_const * (v * v * v * v) * i1pav3;
        compute_H_shift(z, h, ghs, Hhs, s.region);
        if (s.region == 0 || s.region == 2) {
            gh = TV::Zero();
            Hh = TM::Zero();
        }
        else {
            // This is to compute dz / dx and dz
            T gzv = k_const * (v3 * (4 + av3)) * sqr(i1pav3);
            T Hzv = -6 * k_const * v * v * (-2 + av3) * cube(i1pav3);
            TV gz;
            TM Hz;
            gz(0) = one_over_sqrt_2 * (1 - gzv);
            gz(1) = one_over_sqrt_2 * (1 + gzv);
            Hz(0, 0) = Hzv / 2;
            Hz(1, 1) = Hz(0, 0);
            Hz(0, 1) = -Hzv / 2;
            Hz(1, 0) = Hz(0, 1);

            TV z_over_s = gz.array() / s.sigma.array();
            gh = ghs * z_over_s;
            TV one_over_s;
            for (size_t d = 0; d < dim; ++d)
                one_over_s(d) = (T)1 / s.sigma(d);
            TM one_over_s_outer = one_over_s * one_over_s.transpose();
            TM gz_over_s_squared = TM::Zero();
            for (size_t d = 0; d < dim; ++d)
                gz_over_s_squared(d, d) = gz(d) / (s.sigma(d) * s.sigma(d));
            TM hz_over_s_squared = Hz.array() * one_over_s_outer.array();
            Hh = Hhs * z_over_s * z_over_s.transpose() - ghs * gz_over_s_squared + ghs * hz_over_s_squared;
        }
        s.h = h;
        const T& s0 = s.sigma(0);
        const T& s1 = s.sigma(1);
        bool m_check = std::abs(s0 - s1) > 1e-11;
        bool p_check = std::abs(s0 + s1) > 1e-11;

        s.h = h;
        s.im.psi0 = gh(0);
        s.im.psi1 = gh(1);
        s.im.psi00 = Hh(0, 0);
        s.im.psi01 = Hh(0, 1);
        s.im.psi11 = Hh(1, 1);
        s.im.m01 = m_check ? (s.im.psi0 - s.im.psi1) / (s.sigma(0) - s.sigma(1)) : 0;
        s.im.p01 = p_check ? (s.im.psi0 + s.im.psi1) / (s.sigma(0) + s.sigma(1)) : 0;
    }

    TICK_MEMBER_REQUIRES(dim == 2)
    void updateScratch(const TM& new_F, Scratch& s) const
    {
        using namespace MATH_TOOLS;
        updateScratchSVD(new_F, s);
        updateMultiplier(s);
        if (s.region == 2) {
            s.isotropic.psi0 = (T)0;
            s.isotropic.psi1 = (T)0;
            s.isotropic.psi00 = (T)0;
            s.isotropic.psi11 = (T)0;
            s.isotropic.psi01 = (T)0;
            s.isotropic.m01 = (T)0;
            s.isotropic.p01 = (T)0;
        }
        else {
            T g = 2 * mu + lambda;
            T prod = s.sigma(0) * s.sigma(1);
            TV logS_squared = s.logS.array().square();
            T trace_logS = s.logS.array().sum();
            T psi = mu * logS_squared.array().sum() + (T).5 * lambda * trace_logS * trace_logS;
            T psi0 = (g * s.logS(0) + lambda * s.logS(1)) / s.sigma(0);
            T psi1 = (g * s.logS(1) + lambda * s.logS(0)) / s.sigma(1);
            T psi00 = (g * (1 - s.logS(0)) - lambda * s.logS(1)) / (sqr(s.sigma(0)));
            T psi11 = (g * (1 - s.logS(1)) - lambda * s.logS(0)) / (sqr(s.sigma(1)));
            T psi01 = lambda / prod;

            // (psi0-psi1)/(sigma0-sigma1)
            T q = s.sigma(0) / s.sigma(1) - 1;
            // for 2D dam breach example, use:
            //T q = std::max(s.sigma(0) / s.sigma(1) - 1, 1 + eps);
            T h = (std::abs(q) < eps) ? 1 : (std::log1p(std::abs(q)) / q);
            T t = h / s.sigma(1);
            T z = s.logS(1) - t * s.sigma(1);
            T m01 = -(lambda * (s.logS(0) + s.logS(1)) + 2 * mu * z) / prod;
            // (psi0+psi1)/(sigma0+sigma1)
            T p01 = (psi0 + psi1) / clamp_small_magnitude(s.sigma(0) + s.sigma(1), eps);

            if (s.region == 0) {
                s.isotropic.psi0 = psi0;
                s.isotropic.psi1 = psi1;
                s.isotropic.psi00 = psi00;
                s.isotropic.psi11 = psi11;
                s.isotropic.psi01 = psi01;
                s.isotropic.m01 = m01;
                s.isotropic.p01 = p01;
            }
            else {
                s.isotropic.psi0 = psi0 * s.h + s.im.psi0 * psi;
                s.isotropic.psi1 = psi1 * s.h + s.im.psi1 * psi;
                s.isotropic.psi00 = psi00 * s.h + 2 * psi0 * s.im.psi0 + psi * s.im.psi00;
                s.isotropic.psi11 = psi11 * s.h + 2 * psi1 * s.im.psi1 + psi * s.im.psi11;
                s.isotropic.psi01 = psi01 * s.h + psi0 * s.im.psi1 + psi1 * s.im.psi0 + psi * s.im.psi01;
                s.isotropic.m01 = m01 * s.h + s.im.m01 * psi;
                s.isotropic.p01 = p01 * s.h + s.im.p01 * psi;
            }
        }
    }

    TICK_MEMBER_REQUIRES(dim == 3)
    void updateMultiplier(Scratch& s) const
    {
        using namespace ZIRAN::MATH_TOOLS;
        const TV xi_0(one_over_sqrt_3, one_over_sqrt_3, one_over_sqrt_3);
        const TV xi_1(-one_over_sqrt_2, one_over_sqrt_2, 0);
        const TV xi_2(-one_over_sqrt_6, -one_over_sqrt_6, 2 * one_over_sqrt_6);
        TV x = s.logS;
        T u = x.dot(xi_0);
        T v = x.dot(xi_1);
        T w = x.dot(xi_2);

        T h, ghs, Hhs;
        TV gh;
        TM Hh;

        T vpws = sqr(v) + sqr(w);
        T nvpw = std::sqrt(vpws);
        T nvpw3 = cube(nvpw);
        T i1pnvpw3 = 1 / (1 + nvpw3);
        T z = u + (nvpw * nvpw * nvpw * nvpw) * i1pnvpw3;
        compute_H_shift(z, h, ghs, Hhs, s.region);
        if (s.region == 0 || s.region == 2) {
            gh = TV::Zero();
            Hh = TM::Zero();
        }
        else {
            // This is to compute dz / dx and dz
            TV gz;
            TM Hz;

            T v2 = sqr(v);
            T v4 = sqr(v2);
            T v6 = v4 * v2;
            T v8 = sqr(v4);
            T w2 = sqr(w);
            T w4 = sqr(w2);
            T w6 = w4 * w2;
            T w8 = sqr(w4);

            T vpws = sqr(v) + sqr(w);
            T nvpw = std::sqrt(vpws);
            T nvpw3 = cube(nvpw);
            T i1pnvpw3 = 1 / (1 + nvpw3);
            T squarei1pnvpw3 = sqr(i1pnvpw3);
            T cubei1pnvpw3 = cube(i1pnvpw3);
            z = u + k_const * (nvpw * nvpw * nvpw * nvpw) * i1pnvpw3;

            T zv = k_const * v * vpws * (4 + nvpw3) * squarei1pnvpw3;
            T zw = k_const * w * vpws * (4 + nvpw3) * squarei1pnvpw3;
            gz = xi_0 + zv * xi_1 + zw * xi_2;

            T zvv = k_const * (4 * w2 + v6 * w2 + 5 * nvpw * w4 + 3 * v4 * (-2 * nvpw + w4) + v2 * (12 - nvpw * w2 + 3 * w6) + w8) * cubei1pnvpw3;
            T zww = k_const * (v8 + 12 * w2 + 3 * v6 * w2 - 6 * nvpw * w4 + v4 * (5 * nvpw + 3 * w4) + v2 * (4 - nvpw * w2 + w6)) * cubei1pnvpw3;
            T zwv = -k_const * (v * w * (-8 + (v2 + w2) * (11 * nvpw + sqr(vpws)))) * cubei1pnvpw3;
            Hz = zvv * xi_1 * xi_1.transpose() + zwv * (xi_1 * xi_2.transpose() + xi_2 * xi_1.transpose()) + zww * xi_2 * xi_2.transpose();

            TV z_over_s = gz.array() / s.sigma.array();
            gh = ghs * z_over_s;
            TV one_over_s;
            for (size_t d = 0; d < dim; ++d)
                one_over_s(d) = (T)1 / s.sigma(d);
            TM one_over_s_outer = one_over_s * one_over_s.transpose();
            TM gz_over_s_squared = TM::Zero();
            for (size_t d = 0; d < dim; ++d)
                gz_over_s_squared(d, d) = gz(d) / (s.sigma(d) * s.sigma(d));
            TM hz_over_s_squared = Hz.array() * one_over_s_outer.array();
            Hh = Hhs * z_over_s * z_over_s.transpose() - ghs * gz_over_s_squared + ghs * hz_over_s_squared;
        }
        const T& s0 = s.sigma(0);
        const T& s1 = s.sigma(1);
        const T& s2 = s.sigma(2);
        bool m_check01 = std::abs(s0 - s1) > 1e-11;
        bool p_check01 = std::abs(s0 + s1) > 1e-11;
        bool m_check02 = std::abs(s0 - s2) > 1e-11;
        bool p_check02 = std::abs(s0 + s2) > 1e-11;
        bool m_check12 = std::abs(s1 - s2) > 1e-11;
        bool p_check12 = std::abs(s1 + s2) > 1e-11;
        s.h = h;
        s.im.psi0 = gh(0);
        s.im.psi1 = gh(1);
        s.im.psi2 = gh(2);
        s.im.psi00 = Hh(0, 0);
        s.im.psi11 = Hh(1, 1);
        s.im.psi22 = Hh(2, 2);
        s.im.psi01 = Hh(0, 1);
        s.im.psi02 = Hh(0, 2);
        s.im.psi12 = Hh(1, 2);
        s.im.m01 = m_check01 ? (s.im.psi0 - s.im.psi1) / (s.sigma(0) - s.sigma(1)) : (T)0;
        s.im.m02 = m_check02 ? (s.im.psi0 - s.im.psi2) / (s.sigma(0) - s.sigma(2)) : (T)0;
        s.im.m12 = m_check12 ? (s.im.psi1 - s.im.psi2) / (s.sigma(1) - s.sigma(2)) : (T)0;
        s.im.p01 = p_check01 ? (s.im.psi0 + s.im.psi1) / (s.sigma(0) + s.sigma(1)) : (T)0;
        s.im.p02 = p_check02 ? (s.im.psi0 + s.im.psi2) / (s.sigma(0) + s.sigma(2)) : (T)0;
        s.im.p12 = p_check12 ? (s.im.psi1 + s.im.psi2) / (s.sigma(1) + s.sigma(2)) : (T)0;
    }

    TICK_MEMBER_REQUIRES(dim == 3)
    void updateScratch(const TM& new_F, Scratch& s) const
    {
        using namespace MATH_TOOLS;
        updateScratchSVD(new_F, s);
        updateMultiplier(s);

        if (s.region == 2) {
            s.isotropic.psi0 = (T)0;
            s.isotropic.psi1 = (T)0;
            s.isotropic.psi2 = (T)0;
            s.isotropic.psi00 = (T)0;
            s.isotropic.psi11 = (T)0;
            s.isotropic.psi22 = (T)0;
            s.isotropic.psi01 = (T)0;
            s.isotropic.psi02 = (T)0;
            s.isotropic.psi12 = (T)0;
            s.isotropic.m01 = (T)0;
            s.isotropic.m02 = (T)0;
            s.isotropic.m12 = (T)0;
            s.isotropic.p01 = (T)0;
            s.isotropic.p02 = (T)0;
            s.isotropic.p12 = (T)0;
        }
        else {
            T g = 2 * mu + lambda;
            T sum_log = s.logS(0) + s.logS(1) + s.logS(2);
            T prod01 = s.sigma(0) * s.sigma(1);
            T prod02 = s.sigma(0) * s.sigma(2);
            T prod12 = s.sigma(1) * s.sigma(2);
            TV logS_squared = s.logS.array().square();
            T trace_logS = s.logS.array().sum();
            T psi = mu * logS_squared.array().sum() + (T).5 * lambda * trace_logS * trace_logS;
            T psi0 = (2 * mu * s.logS(0) + lambda * sum_log) / s.sigma(0);
            T psi1 = (2 * mu * s.logS(1) + lambda * sum_log) / s.sigma(1);
            T psi2 = (2 * mu * s.logS(2) + lambda * sum_log) / s.sigma(2);
            T psi00 = (g * (1 - s.logS(0)) - lambda * (s.logS(1) + s.logS(2))) / sqr(s.sigma(0));
            T psi11 = (g * (1 - s.logS(1)) - lambda * (s.logS(0) + s.logS(2))) / sqr(s.sigma(1));
            T psi22 = (g * (1 - s.logS(2)) - lambda * (s.logS(0) + s.logS(1))) / sqr(s.sigma(2));
            T psi01 = lambda / (s.sigma(0) * s.sigma(1));
            T psi02 = lambda / (s.sigma(0) * s.sigma(2));
            T psi12 = lambda / (s.sigma(1) * s.sigma(2));

            // (psiA-psiB)/(sigmaA-sigmaB)
            T m01 = -(lambda * sum_log + 2 * mu * diff_interlock_log_over_diff(s.sigma(0), s.sigma(1), s.logS(1), eps)) / prod01;
            T m02 = -(lambda * sum_log + 2 * mu * diff_interlock_log_over_diff(s.sigma(0), std::abs(s.sigma(2)), s.logS(2), eps)) / prod02;
            T m12 = -(lambda * sum_log + 2 * mu * diff_interlock_log_over_diff(s.sigma(1), std::abs(s.sigma(2)), s.logS(2), eps)) / prod12;

            // (psiA+psiB)/(sigmaA+sigmaB)
            T p01 = (psi0 + psi1) / clamp_small_magnitude(s.sigma(0) + s.sigma(1), eps);
            T p02 = (psi0 + psi2) / clamp_small_magnitude(s.sigma(0) + s.sigma(2), eps);
            T p12 = (psi1 + psi2) / clamp_small_magnitude(s.sigma(1) + s.sigma(2), eps);

            if (s.region == 0) {
                s.isotropic.psi0 = psi0;
                s.isotropic.psi1 = psi1;
                s.isotropic.psi2 = psi2;
                s.isotropic.psi00 = psi00;
                s.isotropic.psi11 = psi11;
                s.isotropic.psi22 = psi22;
                s.isotropic.psi01 = psi01;
                s.isotropic.psi02 = psi02;
                s.isotropic.psi12 = psi12;
                s.isotropic.m01 = m01;
                s.isotropic.m02 = m02;
                s.isotropic.m12 = m12;
                s.isotropic.p01 = p01;
                s.isotropic.p02 = p02;
                s.isotropic.p12 = p12;
            }
            else {
                s.isotropic.psi0 = psi0 * s.h + s.im.psi0 * psi;
                s.isotropic.psi1 = psi1 * s.h + s.im.psi1 * psi;
                s.isotropic.psi2 = psi2 * s.h + s.im.psi2 * psi;
                s.isotropic.psi00 = psi00 * s.h + 2 * psi0 * s.im.psi0 + psi * s.im.psi00;
                s.isotropic.psi11 = psi11 * s.h + 2 * psi1 * s.im.psi1 + psi * s.im.psi11;
                s.isotropic.psi22 = psi22 * s.h + 2 * psi2 * s.im.psi2 + psi * s.im.psi22;
                s.isotropic.psi01 = psi01 * s.h + psi0 * s.im.psi1 + psi1 * s.im.psi0 + psi * s.im.psi01;
                s.isotropic.psi02 = psi02 * s.h + psi0 * s.im.psi2 + psi2 * s.im.psi0 + psi * s.im.psi02;
                s.isotropic.psi12 = psi12 * s.h + psi1 * s.im.psi2 + psi2 * s.im.psi1 + psi * s.im.psi12;
                s.isotropic.m01 = m01 * s.h + s.im.m01 * psi;
                s.isotropic.m02 = m02 * s.h + s.im.m02 * psi;
                s.isotropic.m12 = m12 * s.h + s.im.m12 * psi;
                s.isotropic.p01 = p01 * s.h + s.im.p01 * psi;
                s.isotropic.p02 = p02 * s.h + s.im.p02 * psi;
                s.isotropic.p12 = p12 * s.h + s.im.p12 * psi;
            }
        }
    }

    static constexpr bool diagonalDifferentiable()
    {
        return false; //TODO implement diagonal functions
    }

    /**
       psi = mu tr((log S)^2) + 1/2 lambda (tr(log S))^2
     */
    T psi(const Scratch& s) const
    {
        T ret;
        if (s.region == 2) {
            ret = (T)0;
        }
        else {
            TV logS_squared = s.logS.array().square();
            T trace_logS = s.logS.array().sum();
            ret = mu * logS_squared.array().sum() + (T).5 * lambda * trace_logS * trace_logS;
            if (s.region == 1) {
                ret *= s.h;
            }
        }
        return ret;
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
        T sqrt_tol = std::sqrt(tolerance);
        bool check = true;
        for (int d = 0; d < dim; ++d) {
            check &= (s.sigma(d) > sqrt_tol);
        }
        return s.sigma.prod() > sqrt_tol && check; // due to the log sigma term
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

    static StvkWithHenckyIsotropicUnilateral<T, dim> read(std::istream& in)
    {
        StvkWithHenckyIsotropicUnilateral<T, dim> model;
        model.mu = readEntry<T>(in);
        model.lambda = readEntry<T>(in);
        return model;
    }

    static const char* name()
    {
        return "StvkWithHenckyIsotropicUnilateral";
    }

    inline static AttributeName<StvkWithHenckyIsotropicUnilateral<T, dim>> attributeName()
    {
        return AttributeName<StvkWithHenckyIsotropicUnilateral<T, dim>>("StvkWithHenckyIsotropicUnilateral");
    }

    static const char* scratch_name()
    {
        return Scratch::name();
    }
};

template <class T, int dim>
struct HyperelasticTraits<StvkWithHenckyIsotropicUnilateral<T, dim>> {
    using ScratchType = StvkWithHenckyIsotropicUnilateralScratch<T, dim>;
};

template <class T, int dim>
struct RW<StvkWithHenckyIsotropicUnilateralScratch<T, dim>> {
    using Tag = NoWriteTag<StvkWithHenckyIsotropicUnilateralScratch<T, dim>>;
};
} // namespace ZIRAN

#endif
