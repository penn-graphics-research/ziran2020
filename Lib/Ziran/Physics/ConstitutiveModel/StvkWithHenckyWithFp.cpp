#include "StvkWithHenckyWithFp.h"
#include <Ziran/Math/MathTools.h>
#include <Ziran/CS/Util/StaticIf.h>
#include <Ziran/CS/Util/ErrorContext.h>
#include <Ziran/Math/Linear/ImplicitQRSVD.h>

namespace ZIRAN {
template <class T, int _dim>
StvkWithHenckyWithFp<T, _dim>::StvkWithHenckyWithFp(const T E, const T nu)
    : Fp_inv(TM::Identity())
{
    setLameParameters(E, nu);
}

template <class T, int _dim>
void StvkWithHenckyWithFp<T, _dim>::setLameParameters(const T E, const T nu)
{
    lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
    mu = E / ((T)2 * ((T)1 + nu));
}

template <class T, int _dim>
void StvkWithHenckyWithFp<T, _dim>::updateScratchSVD(const TM& new_F, Scratch& s) const // private
{
    using namespace EIGEN_EXT;
    s.F = new_F * Fp_inv; //different from StvkWithHenckyIsotropic
    singularValueDecomposition(s.F, s.U, s.sigma, s.V);
    s.sigma = s.sigma.array().abs().max(1e-16);
    s.log_sigma = s.sigma.array().log();
}

template <class T, int _dim>
void StvkWithHenckyWithFp<T, _dim>::updateScratch(const TM& new_F, Scratch& s) const
{
    if constexpr (dim == 1) {
        using namespace MATH_TOOLS;
        this->updateScratchSVD(new_F, s);
        T g = 2 * mu + lambda;
        g *= damage_scale;
        T one_over_F = 1 / new_F(0, 0);
        s.isotropic.psi0 = g * one_over_F;
        s.isotropic.psi00 = g * sqr(one_over_F);
    }
    else if constexpr (dim == 2) {
        using namespace MATH_TOOLS;
        this->updateScratchSVD(new_F, s);
        T g = 2 * mu + lambda;
        T prod = s.sigma(0) * s.sigma(1);
        s.isotropic.psi0 = (g * s.log_sigma(0) + lambda * s.log_sigma(1)) / s.sigma(0);
        s.isotropic.psi1 = (g * s.log_sigma(1) + lambda * s.log_sigma(0)) / s.sigma(1);
        s.isotropic.psi00 = (g * (1 - s.log_sigma(0)) - lambda * s.log_sigma(1)) / sqr(s.sigma(0));
        s.isotropic.psi11 = (g * (1 - s.log_sigma(1)) - lambda * s.log_sigma(0)) / sqr(s.sigma(1));
        s.isotropic.psi01 = lambda / prod;

        // (psi0-psi1)/(sigma0-sigma1)
        T q = std::max(s.sigma(0) / s.sigma(1) - 1, -1 + eps);
        T h = (std::fabs(q) < eps) ? 1 : (std::log1p(q) / q);
        T t = h / s.sigma(1);
        T z = s.log_sigma(1) - t * s.sigma(1);
        s.isotropic.m01 = -(lambda * (s.log_sigma(0) + s.log_sigma(1)) + 2 * mu * z) / prod;

        // (psi0+psi1)/(sigma0+sigma1)
        s.isotropic.p01 = (s.isotropic.psi0 + s.isotropic.psi1) / clamp_small_magnitude(s.sigma(0) + s.sigma(1), eps);
    }
    else if constexpr (dim == 3) {
        using namespace MATH_TOOLS;
        this->updateScratchSVD(new_F, s);
        T g = 2 * mu + lambda;
        T sum_log = s.log_sigma(0) + s.log_sigma(1) + s.log_sigma(2);
        T prod01 = s.sigma(0) * s.sigma(1);
        T prod02 = s.sigma(0) * s.sigma(2);
        T prod12 = s.sigma(1) * s.sigma(2);
        s.isotropic.psi0 = (2 * mu * s.log_sigma(0) + lambda * sum_log) / s.sigma(0);
        s.isotropic.psi1 = (2 * mu * s.log_sigma(1) + lambda * sum_log) / s.sigma(1);
        s.isotropic.psi2 = (2 * mu * s.log_sigma(2) + lambda * sum_log) / s.sigma(2);
        s.isotropic.psi00 = (g * (1 - s.log_sigma(0)) - lambda * (s.log_sigma(1) + s.log_sigma(2))) / sqr(s.sigma(0));
        s.isotropic.psi11 = (g * (1 - s.log_sigma(1)) - lambda * (s.log_sigma(0) + s.log_sigma(2))) / sqr(s.sigma(1));
        s.isotropic.psi22 = (g * (1 - s.log_sigma(2)) - lambda * (s.log_sigma(0) + s.log_sigma(1))) / sqr(s.sigma(2));
        s.isotropic.psi01 = lambda / (s.sigma(0) * s.sigma(1));
        s.isotropic.psi02 = lambda / (s.sigma(0) * s.sigma(2));
        s.isotropic.psi12 = lambda / (s.sigma(1) * s.sigma(2));

        // (psiA-psiB)/(sigmaA-sigmaB)
        s.isotropic.m01 = -(lambda * sum_log + 2 * mu * diff_interlock_log_over_diff(s.sigma(0), s.sigma(1), s.log_sigma(1), eps)) / prod01;
        s.isotropic.m02 = -(lambda * sum_log + 2 * mu * diff_interlock_log_over_diff(s.sigma(0), s.sigma(2), s.log_sigma(2), eps)) / prod02;
        s.isotropic.m12 = -(lambda * sum_log + 2 * mu * diff_interlock_log_over_diff(s.sigma(1), s.sigma(2), s.log_sigma(2), eps)) / prod12;

        // (psiA+psiB)/(sigmaA+sigmaB)
        s.isotropic.p01 = (s.isotropic.psi0 + s.isotropic.psi1) / clamp_small_magnitude(s.sigma(0) + s.sigma(1), eps);
        s.isotropic.p02 = (s.isotropic.psi0 + s.isotropic.psi2) / clamp_small_magnitude(s.sigma(0) + s.sigma(2), eps);
        s.isotropic.p12 = (s.isotropic.psi1 + s.isotropic.psi2) / clamp_small_magnitude(s.sigma(1) + s.sigma(2), eps);
    }
}

/**
       psi = mu tr((log S)^2) + 1/2 lambda (tr(log S))^2
     */
template <class T, int _dim>
T StvkWithHenckyWithFp<T, _dim>::psi(const Scratch& s) const
{
    TV log_sigma_squared = s.log_sigma.array().square();
    T trace_log_sigma = s.log_sigma.array().sum();
    return damage_scale * (mu * log_sigma_squared.array().sum() + (T).5 * lambda * trace_log_sigma * trace_log_sigma);
}

/**
       P = U (2 mu S^{-1} (log S) + lambda tr(log S) S^{-1}) V^T
     */
template <class T, int _dim>
void StvkWithHenckyWithFp<T, _dim>::firstPiola(const Scratch& s, TM& P) const
{
    TV P_hat;
    s.isotropic.computePHat(P_hat);
    P = s.U * P_hat.asDiagonal() * s.V.transpose();
    P *= Fp_inv.transpose(); // this is different from StvkWithHencky
    P *= damage_scale;
}

template <class T, int _dim>
void StvkWithHenckyWithFp<T, _dim>::firstPiolaDifferential(const Scratch& s, const TM& dF, TM& dP) const
{
    //ZIRAN_ASSERT(false, "not implemented");
    TM D = s.U.transpose() * dF * Fp_inv * s.V;
    TM K;
    s.isotropic.dPdFOfSigmaContract(D, K);
    dP = s.U * K * s.V.transpose();
    dP *= Fp_inv.transpose();
    dP *= damage_scale;
}

template <class T, int _dim>
bool StvkWithHenckyWithFp<T, _dim>::isC2(const Scratch& s, T tolerance) const
{
    return s.sigma.prod() > tolerance; // due to the log sigma term
}

/**
      Returns whether dP (or dPdF) is implemented
      */
template <class T, int _dim>
bool StvkWithHenckyWithFp<T, _dim>::hessianImplemented() const
{
    return true;
}

template <class T, int _dim>
void StvkWithHenckyWithFp<T, _dim>::write(std::ostream& out) const
{
    writeEntry(out, Fp_inv);
    writeEntry(out, mu);
    writeEntry(out, lambda);
}

template <class T, int _dim>
StvkWithHenckyWithFp<T, _dim> StvkWithHenckyWithFp<T, _dim>::read(std::istream& in)
{
    StvkWithHenckyWithFp<T, _dim> model;
    model.Fp_inv = readEntry<TM>(in);
    model.mu = readEntry<T>(in);
    model.lambda = readEntry<T>(in);
    return model;
}

template <class T, int _dim>
const char* StvkWithHenckyWithFp<T, _dim>::name()
{
    return "StvkWithHenckyWithFp";
}

template <class T, int _dim>
const char* StvkWithHenckyWithFp<T, _dim>::scratch_name()
{
    return Scratch::name();
}

template class StvkWithHenckyWithFp<float, 1>;
template class StvkWithHenckyWithFp<float, 2>;
template class StvkWithHenckyWithFp<float, 3>;
template class StvkWithHenckyWithFp<double, 1>;
template class StvkWithHenckyWithFp<double, 2>;
template class StvkWithHenckyWithFp<double, 3>;
} // namespace ZIRAN
