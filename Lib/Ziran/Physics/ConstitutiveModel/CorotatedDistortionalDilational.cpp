#include <Ziran/CS/Util/BinaryIO.h>
#include <Ziran/Math/Linear/DenseExt.h>
#include <Ziran/Math/Linear/ImplicitQRSVD.h>
#include <Ziran/Physics/ConstitutiveModel/CorotatedDistortionalDilational.h>
#include <Ziran/Physics/ConstitutiveModel/HyperelasticConstitutiveModel.h>

namespace ZIRAN {

template <class T, int _dim>
CorotatedDistortionalDilational<T, _dim>::CorotatedDistortionalDilational(
    const T E, const T nu)
{
    setLameParameters(E, nu);
}

template <class T, int _dim>
void CorotatedDistortionalDilational<T, _dim>::setLameParameters(const T E,
    const T nu)
{
    lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
    mu = E / ((T)2 * ((T)1 + nu));
}

template <class T, int _dim>
void CorotatedDistortionalDilational<T, _dim>::updateScratch(const TM& new_F, Scratch& scratch)
{
    using namespace EIGEN_EXT;
    scratch.F = new_F;
    scratch.J = scratch.F.determinant();
    cofactorMatrix(scratch.F, scratch.JFinvT);
    scratch.F_distortional = scratch.F * std::pow(scratch.J, -(T)1 / (T)dim);

    CorotatedElasticity<T, dim> corotated;
    corotated.mu = mu;
    corotated.lambda = 0;
    corotated.updateScratch(scratch.F_distortional, scratch.corotated_scratch);
}

template <class T, int _dim>
T CorotatedDistortionalDilational<T, _dim>::psi(const Scratch& s) const
{
    T result = 0;

    // lambda term
    T Jm1 = s.J - 1;
    result += (T).5 * lambda * Jm1 * Jm1;

    // mu term
    CorotatedElasticity<T, dim> corotated;
    corotated.mu = mu;
    corotated.lambda = 0;
    result += corotated.psi(s.corotated_scratch);

    return result;
}

template <class T, int _dim>
void CorotatedDistortionalDilational<T, _dim>::firstPiola(const Scratch& s, TM& P) const
{
    // mu term: using notations from the techdoc of [stomakhin2014augmented]
    static const T a = -(T)1 / (T)_dim;
    CorotatedElasticity<T, dim> corotated;
    corotated.mu = mu;
    corotated.lambda = 0;
    TM A;
    corotated.firstPiola(s.corotated_scratch, A);
    TM H = s.JFinvT / s.J;
    T F_contract_A = (s.F.array() * A.array()).sum();
    TM Ahat = std::pow(s.J, a) * (A + a * F_contract_A * H);

    P.noalias() = Ahat + lambda * (s.J - 1) * s.JFinvT; // lambda term
}

template <class T, int _dim>
void CorotatedDistortionalDilational<T, _dim>::firstPiolaDifferential(const Scratch& s, const TM& dF, TM& dP) const
{
    using namespace EIGEN_EXT;

    // lambda term
    dP.noalias() = lambda * s.JFinvT.cwiseProduct(dF).sum() * s.JFinvT;
    addScaledCofactorMatrixDifferential(s.F, dF, lambda * (s.J - (T)1), dP);

    // mu term: using notations from the techdoc of [stomakhin2014augmented]
    static const T a = -(T)1 / (T)_dim;
    CorotatedElasticity<T, dim> corotated;
    corotated.mu = mu;
    corotated.lambda = 0;

    TM A;
    corotated.firstPiola(s.corotated_scratch, A);
    TM Z = dF;
    TM H = s.JFinvT / s.J;
    T H_contract_Z = (H.array() * Z.array()).sum();
    TM B_contract_Z = std::pow(s.J, a) * (Z + a * H_contract_Z * s.F);
    TM C_contract_B_contract_Z = TM::Zero();
    corotated.firstPiolaDifferential(s.corotated_scratch, B_contract_Z, C_contract_B_contract_Z);
    T F_contract_C_contract_B_contract_Z = (s.F.array() * C_contract_B_contract_Z.array()).sum();
    dP += std::pow(s.J, a) * (C_contract_B_contract_Z + a * F_contract_C_contract_B_contract_Z * H); // term 1
    T F_contract_A = (s.F.array() * A.array()).sum();
    TM A_contract_B = std::pow(s.J, a) * (A + a * F_contract_A * H);
    dP += a * H_contract_Z * A_contract_B; // term 2
    T A_contract_Z = (A.array() * Z.array()).sum();
    dP += a * std::pow(s.J, a) * A_contract_Z * H; // term 3
    dP += -a * std::pow(s.J, a) * F_contract_A * H * (Z.transpose()) * H; // term 4
}

template <class T, int _dim>
bool CorotatedDistortionalDilational<T, _dim>::isC2(const Scratch& s, T tolerance) const
{
    CorotatedElasticity<T, dim> corotated;
    corotated.mu = mu;
    corotated.lambda = 0;
    return corotated.isC2(s.corotated_scratch, tolerance) && s.J > tolerance;
}

template <class T, int _dim>
void CorotatedDistortionalDilational<T, _dim>::write(std::ostream& out) const
{
    writeEntry(out, mu);
    writeEntry(out, lambda);
}

template <class T, int _dim>
CorotatedDistortionalDilational<T, _dim>
CorotatedDistortionalDilational<T, _dim>::read(std::istream& in)
{
    CorotatedDistortionalDilational<T, _dim> model;
    model.mu = readEntry<T>(in);
    model.lambda = readEntry<T>(in);
    return model;
}

template class CorotatedDistortionalDilational<double, 1>;
template class CorotatedDistortionalDilational<double, 2>;
template class CorotatedDistortionalDilational<double, 3>;
template class CorotatedDistortionalDilational<float, 1>;
template class CorotatedDistortionalDilational<float, 2>;
template class CorotatedDistortionalDilational<float, 3>;
} // namespace ZIRAN
