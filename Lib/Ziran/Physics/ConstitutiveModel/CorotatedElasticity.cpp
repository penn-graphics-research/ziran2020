#include <Ziran/CS/Util/BinaryIO.h>
#include <Ziran/Physics/ConstitutiveModel/CorotatedElasticity.h>
#include <Ziran/Physics/ConstitutiveModel/HyperelasticConstitutiveModel.h>
#include <Ziran/Math/Linear/DenseExt.h>
#include <Ziran/Math/Linear/ImplicitQRSVD.h>

namespace ZIRAN {

template <class T, int _dim>
CorotatedElasticity<T, _dim>::CorotatedElasticity(const T E, const T nu)
{
    setLameParameters(E, nu);
}

template <class T, int _dim>
void CorotatedElasticity<T, _dim>::setLameParameters(const T E, const T nu)
{
    lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
    mu = E / ((T)2 * ((T)1 + nu));
}

template <class T, int _dim>
Matrix<T, 2, 2> CorotatedElasticity<T, _dim>::Bij(const TV& sigma, int i, int j, T clamp_value) const
{
    auto mingClampMagnitude = [&](const T input) {
        T magnitude = input > 0 ? input : -input;
        T sign = input > 0 ? 1.f : -1.f;
        T output = magnitude > clamp_value ? magnitude : clamp_value;
        return output * sign;
    };
    if constexpr (dim == 2) {
        TV dE = firstPiolaDiagonal(sigma);
        T B_Pij = 0.5 * (dE[i] + dE[j]) / mingClampMagnitude(sigma[i] + sigma[j]);
        T B_Mij = 0.5f * (2.f * mu - lambda * (sigma.prod() - 1));

        Matrix<T, 2, 2> B_P_Const;
        B_P_Const << 1, 1, 1, 1;
        Matrix<T, 2, 2> B_M_Const;
        B_M_Const << 1, -1, -1, 1;
        return B_M_Const * B_Pij + B_P_Const * B_Mij;
    }
    else {
        TV dE = firstPiolaDiagonal(sigma);
        T B_Pij = 0.5 * (dE[i] + dE[j]) / mingClampMagnitude(sigma[i] + sigma[j]);
        T J = sigma(0) * sigma(1) * sigma(2);
        T B_Mij = mu - lambda * (J - 1) * 0.5 * J / sigma(i) / sigma(j);
        Matrix<T, 2, 2> B_P_Const;
        B_P_Const << 1, 1, 1, 1;
        Matrix<T, 2, 2> B_M_Const;
        B_M_Const << 1, -1, -1, 1;
        return B_M_Const * B_Pij + B_P_Const * B_Mij;
    }
}

template <class T, int _dim>
void CorotatedElasticity<T, _dim>::updateScratch(const TM& new_F, Scratch& scratch)
{
    using namespace EIGEN_EXT;
    scratch.F = new_F;
    scratch.J = scratch.F.determinant();
    cofactorMatrix(scratch.F, scratch.JFinvT);
    polarDecomposition(scratch.F, scratch.R, scratch.S);
}

template <class T, int _dim>
T CorotatedElasticity<T, _dim>::psi(const Scratch& s) const
{
    T Jm1 = s.J - 1;
    return mu * ZIRAN::EIGEN_EXT::squaredNorm(s.F - s.R) + (T).5 * lambda * Jm1 * Jm1;
}

template <class T, int _dim>
void CorotatedElasticity<T, _dim>::firstPiola(const Scratch& s, TM& P) const
{
    P.noalias() = (T)2 * mu * (s.F - s.R) + lambda * (s.J - 1) * s.JFinvT;
}

template <class T, int _dim>
void CorotatedElasticity<T, _dim>::firstPiolaDerivative(const Scratch& s, Hessian& dPdF) const
{
    using namespace EIGEN_EXT;
    dPdF.noalias() = lambda * vec(s.JFinvT) * vec(s.JFinvT).transpose();
    dPdF.diagonal().array() += 2 * mu;
    addScaledRotationalDerivative(s.R, s.S, -2 * mu, dPdF);
    addScaledCofactorMatrixDerivative(s.F, lambda * (s.J - (T)1), dPdF);
}

template <class T, int _dim>
T CorotatedElasticity<T, _dim>::psiDiagonal(const TV& sigma) const
{
    T sigmam12Sum = (sigma - TV::Ones()).squaredNorm();
    T sigmaProdm1 = sigma.prod() - 1.0;
    return mu * sigmam12Sum + lambda / 2.0 * sigmaProdm1 * sigmaProdm1;
}

template <class T, int _dim>
typename CorotatedElasticity<T, _dim>::TV CorotatedElasticity<T, _dim>::firstPiolaDiagonal(const TV& sigma) const
{
    T sigmaProdm1lambda = lambda * (sigma.prod() - 1.0);
    TV sigmaProd_noI;
    if constexpr (dim == 2) {
        sigmaProd_noI[0] = sigma[1];
        sigmaProd_noI[1] = sigma[0];
    }
    else {
        sigmaProd_noI[0] = sigma[1] * sigma[2];
        sigmaProd_noI[1] = sigma[2] * sigma[0];
        sigmaProd_noI[2] = sigma[0] * sigma[1];
    }

    TV dE;
    dE[0] = (2 * mu * (sigma[0] - 1.0) + sigmaProd_noI[0] * sigmaProdm1lambda);
    dE[1] = (2 * mu * (sigma[1] - 1.0) + sigmaProd_noI[1] * sigmaProdm1lambda);
    if constexpr (dim == 3) {
        dE[2] = (2 * mu * (sigma[2] - 1.0) + sigmaProd_noI[2] * sigmaProdm1lambda);
    }
    return dE;
}

template <class T, int _dim>
typename CorotatedElasticity<T, _dim>::TM CorotatedElasticity<T, _dim>::firstPiolaDerivativeDiagonal(const TV& sigma) const
{
    T sigmaProd = sigma.prod();
    TV sigmaProd_noI;
    if (dim == 2) {
        sigmaProd_noI[0] = sigma[1];
        sigmaProd_noI[1] = sigma[0];
    }
    else {
        sigmaProd_noI[0] = sigma[1] * sigma[2];
        sigmaProd_noI[1] = sigma[2] * sigma[0];
        sigmaProd_noI[2] = sigma[0] * sigma[1];
    }

    TM ddE;
    ddE(0, 0) = 2 * mu + lambda * sigmaProd_noI[0] * sigmaProd_noI[0];
    ddE(1, 1) = 2 * mu + lambda * sigmaProd_noI[1] * sigmaProd_noI[1];
    if constexpr (dim == 3) {
        ddE(2, 2) = 2 * mu + lambda * sigmaProd_noI[2] * sigmaProd_noI[2];
    }

    if (dim == 2) {
        ddE(0, 1) = ddE(1, 0) = lambda * ((sigmaProd - 1.0) + sigmaProd_noI[0] * sigmaProd_noI[1]);
    }
    else {
        ddE(0, 1) = ddE(1, 0) = lambda * (sigma[2] * (sigmaProd - 1.0) + sigmaProd_noI[0] * sigmaProd_noI[1]);
        ddE(0, 2) = ddE(2, 0) = lambda * (sigma[1] * (sigmaProd - 1.0) + sigmaProd_noI[0] * sigmaProd_noI[2]);
        ddE(2, 1) = ddE(1, 2) = lambda * (sigma[0] * (sigmaProd - 1.0) + sigmaProd_noI[2] * sigmaProd_noI[1]);
    }
    return ddE;
}

template <class T, int _dim>
void CorotatedElasticity<T, _dim>::firstPiolaDifferential(const Scratch& s, const TM& dF, TM& dP) const
{
    using namespace EIGEN_EXT;
    dP.noalias() = lambda * s.JFinvT.cwiseProduct(dF).sum() * s.JFinvT;
    dP += 2 * mu * dF;
    addScaledRotationalDifferential(s.R, s.S, dF, -2 * mu, dP);
    addScaledCofactorMatrixDifferential(s.F, dF, lambda * (s.J - (T)1), dP);
}

template <class T, int _dim>
bool CorotatedElasticity<T, _dim>::isC2(const Scratch& s, T tolerance) const
{
    return !EIGEN_EXT::nearKink(s.S, tolerance);
}

template <class T, int _dim>
void CorotatedElasticity<T, _dim>::write(std::ostream& out) const
{
    writeEntry(out, mu);
    writeEntry(out, lambda);
}

template <class T, int _dim>
CorotatedElasticity<T, _dim> CorotatedElasticity<T, _dim>::read(std::istream& in)
{
    CorotatedElasticity<T, _dim> model;
    model.mu = readEntry<T>(in);
    model.lambda = readEntry<T>(in);
    return model;
}

template class CorotatedElasticity<double, 1>;
template class CorotatedElasticity<double, 2>;
template class CorotatedElasticity<double, 3>;
template class CorotatedElasticity<float, 1>;
template class CorotatedElasticity<float, 2>;
template class CorotatedElasticity<float, 3>;
} // namespace ZIRAN
