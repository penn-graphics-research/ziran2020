#include <Ziran/Physics/ConstitutiveModel/NeoHookeanIsotropic.h>
#include <Ziran/Physics/ConstitutiveModel/HyperelasticConstitutiveModel.h>
#include <Ziran/Math/Linear/DenseExt.h>
#include <Ziran/CS/Util/BinaryIO.h>
#include <Ziran/Math/Linear/ImplicitQRSVD.h>
#include <cmath>
#include <tick/requires.h>
namespace ZIRAN {

template <class T, int _dim>
NeoHookeanIsotropic<T, _dim>::NeoHookeanIsotropic(const T E, const T nu)
{
    setLameParameters(E, nu);
}

template <class T, int _dim>
void NeoHookeanIsotropic<T, _dim>::setLameParameters(const T E, const T nu)
{
    lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
    mu = E / ((T)2 * ((T)1 + nu));
}

template <class T, int _dim>
void NeoHookeanIsotropic<T, _dim>::updateScratchSVD(const TM& new_F, Scratch& s) const // private
{
    s.F = new_F;
    singularValueDecomposition(s.F, s.U, s.sigma, s.V);
}

// origin
//template <class T, int _dim>
//void NeoHookeanIsotropic<T, _dim>::updateScratch(const TM& new_F, Scratch& scratch)
//{
//    using std::log;
//    using namespace EIGEN_EXT;
//    scratch.F = new_F;
//    scratch.J = scratch.F.determinant();
//
//    TM JFinvT;
//    EIGEN_EXT::cofactorMatrix(scratch.F, JFinvT);
//    scratch.FinvT = ((T)1 / scratch.J) * JFinvT;
//    scratch.logJ = log(scratch.J);
//}

template <class T, int _dim>
void NeoHookeanIsotropic<T, _dim>::updateScratch(const TM& new_F, Scratch& s)
{
    using std::log;
    using namespace MATH_TOOLS;
    updateScratchSVD(new_F, s);

    if constexpr (dim == 2)
        s.J = s.sigma(0) * s.sigma(1);
    else if constexpr (dim == 3)
        s.J = s.sigma(0) * s.sigma(1) * s.sigma(2);

    TM JFinvT;
    EIGEN_EXT::cofactorMatrix(s.F, JFinvT);
    s.FinvT = ((T)1 / s.J) * JFinvT;
    s.logJ = log(s.J);

    if constexpr (dim == 2) {
        T sigma_prod = s.sigma.prod();
        T log_sigmaProd = std::log(sigma_prod);
        T inv0 = T(1) / s.sigma(0);
        T inv1 = T(1) / s.sigma(1);

        s.isotropic.psi0 = mu * (s.sigma(0) - inv0) + lambda * inv0 * log_sigmaProd;
        s.isotropic.psi1 = mu * (s.sigma(1) - inv1) + lambda * inv1 * log_sigmaProd;

        T inv2_0 = T(1) / s.sigma(0) / s.sigma(0);
        T inv2_1 = T(1) / s.sigma(1) / s.sigma(1);

        s.isotropic.psi00 = mu * (T(1) + inv2_0) - lambda * inv2_0 * (log_sigmaProd - T(1));
        s.isotropic.psi11 = mu * (T(1) + inv2_1) - lambda * inv2_1 * (log_sigmaProd - T(1));
        s.isotropic.psi01 = lambda / s.sigma(0) / s.sigma(1);

        // (psi0-psi1)/(sigma0-sigma1)

        s.isotropic.m01 = mu + (mu - lambda * log_sigmaProd) / sigma_prod;

        // (psi0+psi1)/(sigma0+sigma1)
        s.isotropic.p01 = (s.isotropic.psi0 + s.isotropic.psi1) / clamp_small_magnitude(s.sigma(0) + s.sigma(1), eps);

        if (project) {
            s.isotropic.buildMatrixBlock();
            s.isotropic.projectABBlock();
        }
    }
    else if constexpr (dim == 3) {
        T log_sigmaProd = std::log(s.sigma.prod());
        T inv0 = T(1) / s.sigma(0);
        T inv1 = T(1) / s.sigma(1);
        T inv2 = T(1) / s.sigma(2);

        s.isotropic.psi0 = mu * (s.sigma(0) - inv0) + lambda * inv0 * log_sigmaProd;
        s.isotropic.psi1 = mu * (s.sigma(1) - inv1) + lambda * inv1 * log_sigmaProd;
        s.isotropic.psi2 = mu * (s.sigma(2) - inv2) + lambda * inv2 * log_sigmaProd;

        T inv2_0 = T(1) / s.sigma(0) / s.sigma(0);
        T inv2_1 = T(1) / s.sigma(1) / s.sigma(1);
        T inv2_2 = T(1) / s.sigma(2) / s.sigma(2);

        s.isotropic.psi00 = mu * (T(1) + inv2_0) - lambda * inv2_0 * (log_sigmaProd - T(1));
        s.isotropic.psi11 = mu * (T(1) + inv2_1) - lambda * inv2_1 * (log_sigmaProd - T(1));
        s.isotropic.psi22 = mu * (T(1) + inv2_2) - lambda * inv2_2 * (log_sigmaProd - T(1));
        s.isotropic.psi01 = lambda / s.sigma(0) / s.sigma(1);
        s.isotropic.psi12 = lambda / s.sigma(1) / s.sigma(2);
        s.isotropic.psi02 = lambda / s.sigma(0) / s.sigma(2);

        // (psiA-psiB)/(sigmaA-sigmaB)
        T common = mu - lambda * log_sigmaProd;
        s.isotropic.m01 = mu + common / s.sigma(0) / s.sigma(1);
        s.isotropic.m02 = mu + common / s.sigma(0) / s.sigma(2);
        s.isotropic.m12 = mu + common / s.sigma(1) / s.sigma(2);

        // (psiA+psiB)/(sigmaA+sigmaB)
        s.isotropic.p01 = (s.isotropic.psi0 + s.isotropic.psi1) / clamp_small_magnitude(s.sigma(0) + s.sigma(1), eps);
        s.isotropic.p02 = (s.isotropic.psi0 + s.isotropic.psi2) / clamp_small_magnitude(s.sigma(0) + s.sigma(2), eps);
        s.isotropic.p12 = (s.isotropic.psi1 + s.isotropic.psi2) / clamp_small_magnitude(s.sigma(1) + s.sigma(2), eps);

        if (project) {
            s.isotropic.buildMatrixBlock();
            s.isotropic.projectABBlock();
        }
    }
}

template <class T, int _dim>
T NeoHookeanIsotropic<T, _dim>::psi(const Scratch& s) const
{
    T I1 = EIGEN_EXT::firstInvariant(s.F);
    return (T)0.5 * mu * (I1 - _dim) - mu * s.logJ + (T)0.5 * lambda * s.logJ * s.logJ;
}

template <class T, int _dim>
void NeoHookeanIsotropic<T, _dim>::kirchhoff(const Scratch& s, TM& tau) const
{
    T scale = lambda * s.logJ - mu;
    tau = (mu * s.F * s.F.transpose() + scale * TM::Identity());
}

//origin
//template <class T, int _dim>
//void NeoHookeanIsotropic<T, _dim>::firstPiolaDerivative(const Scratch& s, Hessian& dPdF) const
//{
//    using namespace EIGEN_EXT;
//    T scale = lambda * s.logJ - mu;
//
//    dPdF = ((lambda - scale)) * vec(s.FinvT) * vec(s.FinvT).transpose();
//    dPdF.diagonal().array() += mu;
//
//    addScaledCofactorMatrixDerivative(s.F, (scale / s.J), dPdF);
//}

template <class T, int _dim>
void NeoHookeanIsotropic<T, _dim>::firstPiolaDerivative(const Scratch& ss, Hessian& dPdF) const
{
    if constexpr (dim == 2) {
        for (int ij = 0; ij < 4; ++ij) {
            int j = ij / 2;
            int i = ij & 1;
            for (int rs = 0; rs <= ij; ++rs) {
                int s = rs / 2;
                int r = rs & 1;
                if (project) {
                    dPdF(ij, rs) = dPdF(rs, ij) = ss.isotropic.Aij(0, 0) * ss.U(i, 0) * ss.V(j, 0) * ss.U(r, 0) * ss.V(s, 0)
                        + ss.isotropic.Aij(0, 1) * ss.U(i, 0) * ss.V(j, 0) * ss.U(r, 1) * ss.V(s, 1)
                        + ss.isotropic.B01(0, 0) * ss.U(i, 0) * ss.V(j, 1) * ss.U(r, 0) * ss.V(s, 1)
                        + ss.isotropic.B01(0, 1) * ss.U(i, 0) * ss.V(j, 1) * ss.U(r, 1) * ss.V(s, 0)
                        + ss.isotropic.B01(1, 0) * ss.U(i, 1) * ss.V(j, 0) * ss.U(r, 0) * ss.V(s, 1)
                        + ss.isotropic.B01(1, 1) * ss.U(i, 1) * ss.V(j, 0) * ss.U(r, 1) * ss.V(s, 0)
                        + ss.isotropic.Aij(1, 0) * ss.U(i, 1) * ss.V(j, 1) * ss.U(r, 0) * ss.V(s, 0)
                        + ss.isotropic.Aij(1, 1) * ss.U(i, 1) * ss.V(j, 1) * ss.U(r, 1) * ss.V(s, 1);
                }
                else
                    dPdF(ij, rs) = dPdF(rs, ij) = ss.isotropic.psi00 * ss.U(i, 0) * ss.V(j, 0) * ss.U(r, 0) * ss.V(s, 0)
                        + ss.isotropic.psi01 * ss.U(i, 0) * ss.V(j, 0) * ss.U(r, 1) * ss.V(s, 1)
                        + ss.isotropic.b01(0, 0) * ss.U(i, 0) * ss.V(j, 1) * ss.U(r, 0) * ss.V(s, 1)
                        + ss.isotropic.b01(0, 1) * ss.U(i, 0) * ss.V(j, 1) * ss.U(r, 1) * ss.V(s, 0)
                        + ss.isotropic.b01(1, 0) * ss.U(i, 1) * ss.V(j, 0) * ss.U(r, 0) * ss.V(s, 1)
                        + ss.isotropic.b01(1, 1) * ss.U(i, 1) * ss.V(j, 0) * ss.U(r, 1) * ss.V(s, 0)
                        + ss.isotropic.psi01 * ss.U(i, 1) * ss.V(j, 1) * ss.U(r, 0) * ss.V(s, 0)
                        + ss.isotropic.psi11 * ss.U(i, 1) * ss.V(j, 1) * ss.U(r, 1) * ss.V(s, 1);
            }
        }
    }
    else if constexpr (dim == 3) {
        for (int ij = 0; ij < 9; ++ij) {
            int j = ij / 3;
            int i = ij - j * 3;
            for (int rs = 0; rs <= ij; ++rs) {
                int s = rs / 3;
                int r = rs - s * 3;
                if (project) {
                    dPdF(ij, rs) = dPdF(rs, ij) = ss.isotropic.Aij(0, 0) * ss.U(i, 0) * ss.V(j, 0) * ss.U(r, 0) * ss.V(s, 0) + ss.isotropic.Aij(0, 1) * ss.U(i, 0) * ss.V(j, 0) * ss.U(r, 1) * ss.V(s, 1) + ss.isotropic.Aij(0, 2) * ss.U(i, 0) * ss.V(j, 0) * ss.U(r, 2) * ss.V(s, 2) + ss.isotropic.Aij(0, 1) * ss.U(i, 1) * ss.V(j, 1) * ss.U(r, 0) * ss.V(s, 0) + ss.isotropic.Aij(1, 1) * ss.U(i, 1) * ss.V(j, 1) * ss.U(r, 1) * ss.V(s, 1) + ss.isotropic.Aij(1, 2) * ss.U(i, 1) * ss.V(j, 1) * ss.U(r, 2) * ss.V(s, 2) + ss.isotropic.Aij(0, 2) * ss.U(i, 2) * ss.V(j, 2) * ss.U(r, 0) * ss.V(s, 0) + ss.isotropic.Aij(1, 2) * ss.U(i, 2) * ss.V(j, 2) * ss.U(r, 1) * ss.V(s, 1) + ss.isotropic.Aij(2, 2) * ss.U(i, 2) * ss.V(j, 2) * ss.U(r, 2) * ss.V(s, 2) +

                        ss.isotropic.B01(0, 0) * ss.U(i, 0) * ss.V(j, 1) * ss.U(r, 0) * ss.V(s, 1) + ss.isotropic.B01(0, 1) * ss.U(i, 0) * ss.V(j, 1) * ss.U(r, 1) * ss.V(s, 0) + ss.isotropic.B01(1, 0) * ss.U(i, 1) * ss.V(j, 0) * ss.U(r, 0) * ss.V(s, 1) + ss.isotropic.B01(1, 1) * ss.U(i, 1) * ss.V(j, 0) * ss.U(r, 1) * ss.V(s, 0) +

                        ss.isotropic.B12(0, 0) * ss.U(i, 1) * ss.V(j, 2) * ss.U(r, 1) * ss.V(s, 2) + ss.isotropic.B12(0, 1) * ss.U(i, 1) * ss.V(j, 2) * ss.U(r, 2) * ss.V(s, 1) + ss.isotropic.B12(1, 0) * ss.U(i, 2) * ss.V(j, 1) * ss.U(r, 1) * ss.V(s, 2) + ss.isotropic.B12(1, 1) * ss.U(i, 2) * ss.V(j, 1) * ss.U(r, 2) * ss.V(s, 1) +

                        ss.isotropic.B20(1, 1) * ss.U(i, 0) * ss.V(j, 2) * ss.U(r, 0) * ss.V(s, 2) + ss.isotropic.B20(1, 0) * ss.U(i, 0) * ss.V(j, 2) * ss.U(r, 2) * ss.V(s, 0) + ss.isotropic.B20(0, 1) * ss.U(i, 2) * ss.V(j, 0) * ss.U(r, 0) * ss.V(s, 2) + ss.isotropic.B20(0, 0) * ss.U(i, 2) * ss.V(j, 0) * ss.U(r, 2) * ss.V(s, 0);
                }
                else
                    dPdF(ij, rs) = dPdF(rs, ij) = ss.isotropic.psi00 * ss.U(i, 0) * ss.V(j, 0) * ss.U(r, 0) * ss.V(s, 0) + ss.isotropic.psi01 * ss.U(i, 0) * ss.V(j, 0) * ss.U(r, 1) * ss.V(s, 1) + ss.isotropic.psi02 * ss.U(i, 0) * ss.V(j, 0) * ss.U(r, 2) * ss.V(s, 2) + ss.isotropic.psi01 * ss.U(i, 1) * ss.V(j, 1) * ss.U(r, 0) * ss.V(s, 0) + ss.isotropic.psi11 * ss.U(i, 1) * ss.V(j, 1) * ss.U(r, 1) * ss.V(s, 1) + ss.isotropic.psi12 * ss.U(i, 1) * ss.V(j, 1) * ss.U(r, 2) * ss.V(s, 2) + ss.isotropic.psi02 * ss.U(i, 2) * ss.V(j, 2) * ss.U(r, 0) * ss.V(s, 0) + ss.isotropic.psi12 * ss.U(i, 2) * ss.V(j, 2) * ss.U(r, 1) * ss.V(s, 1) + ss.isotropic.psi22 * ss.U(i, 2) * ss.V(j, 2) * ss.U(r, 2) * ss.V(s, 2) +

                        ss.isotropic.b01(0, 0) * ss.U(i, 0) * ss.V(j, 1) * ss.U(r, 0) * ss.V(s, 1) + ss.isotropic.b01(0, 1) * ss.U(i, 0) * ss.V(j, 1) * ss.U(r, 1) * ss.V(s, 0) + ss.isotropic.b01(1, 0) * ss.U(i, 1) * ss.V(j, 0) * ss.U(r, 0) * ss.V(s, 1) + ss.isotropic.b01(1, 1) * ss.U(i, 1) * ss.V(j, 0) * ss.U(r, 1) * ss.V(s, 0) +

                        ss.isotropic.b12(0, 0) * ss.U(i, 1) * ss.V(j, 2) * ss.U(r, 1) * ss.V(s, 2) + ss.isotropic.b12(0, 1) * ss.U(i, 1) * ss.V(j, 2) * ss.U(r, 2) * ss.V(s, 1) + ss.isotropic.b12(1, 0) * ss.U(i, 2) * ss.V(j, 1) * ss.U(r, 1) * ss.V(s, 2) + ss.isotropic.b12(1, 1) * ss.U(i, 2) * ss.V(j, 1) * ss.U(r, 2) * ss.V(s, 1) +

                        ss.isotropic.b20(1, 1) * ss.U(i, 0) * ss.V(j, 2) * ss.U(r, 0) * ss.V(s, 2) + ss.isotropic.b20(1, 0) * ss.U(i, 0) * ss.V(j, 2) * ss.U(r, 2) * ss.V(s, 0) + ss.isotropic.b20(0, 1) * ss.U(i, 2) * ss.V(j, 0) * ss.U(r, 0) * ss.V(s, 2) + ss.isotropic.b20(0, 0) * ss.U(i, 2) * ss.V(j, 0) * ss.U(r, 2) * ss.V(s, 0);
            }
        }
    }
}

template <class T, int _dim>
void NeoHookeanIsotropic<T, _dim>::firstPiolaDifferential(const Scratch& s, const TM& dF, TM& dP) const
{
    using namespace EIGEN_EXT;

    cofactorMatrixDifferential(s.F, dF, dP);
    T scale = lambda * s.logJ - mu;
    dP = mu * dF + (lambda - scale) * s.FinvT.cwiseProduct(dF).sum() * s.FinvT + scale / s.J * dP;
}

template <class T, int _dim>
bool NeoHookeanIsotropic<T, _dim>::isC2(const Scratch& s, T tolerance) const
{
    return s.J > tolerance;
}

template <class T, int _dim>
void NeoHookeanIsotropic<T, _dim>::write(std::ostream& out) const
{
    writeEntry(out, mu);
    writeEntry(out, lambda);
}

template <class T, int _dim>
NeoHookeanIsotropic<T, _dim> NeoHookeanIsotropic<T, _dim>::read(std::istream& in)
{
    NeoHookeanIsotropic<T, _dim> model;
    model.mu = readEntry<T>(in);
    model.lambda = readEntry<T>(in);
    return model;
}

template class NeoHookeanIsotropic<double, 1>;
template class NeoHookeanIsotropic<double, 2>;
template class NeoHookeanIsotropic<double, 3>;
template class NeoHookeanIsotropic<float, 1>;
template class NeoHookeanIsotropic<float, 2>;
template class NeoHookeanIsotropic<float, 3>;
} // namespace ZIRAN
