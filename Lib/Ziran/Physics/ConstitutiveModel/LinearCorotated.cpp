#include <Ziran/CS/Util/BinaryIO.h>
#include <Ziran/Physics/ConstitutiveModel/HyperelasticConstitutiveModel.h>
#include <Ziran/Math/Linear/DenseExt.h>
#include <Ziran/Math/Linear/ImplicitQRSVD.h>
#include "LinearCorotated.h"

namespace ZIRAN {

template <class T, int _dim>
LinearCorotated<T, _dim>::LinearCorotated(const T E, const T nu)
    : R(TM::Identity())
{
    setLameParameters(E, nu);
}

template <class T, int _dim>
void LinearCorotated<T, _dim>::setLameParameters(const T E, const T nu)
{
    lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
    mu = E / ((T)2 * ((T)1 + nu));
}

template <class T, int _dim>
void LinearCorotated<T, _dim>::rebuildR(const TM& F)
{
    TM S;
    polarDecomposition(F, R, S); // TODO: replace this with fast rotation extration (no need for SVD)
}

template <class T, int _dim>
void LinearCorotated<T, _dim>::updateScratch(const TM& new_F, Scratch& scratch)
{
    using namespace EIGEN_EXT;
    scratch.F = new_F;
    TM RtF = R.transpose() * new_F;
    scratch.e_hat = 0.5 * (RtF + RtF.transpose()) - TM::Identity();
    scratch.trace_e_hat = scratch.e_hat.diagonal().sum();
}

template <class T, int _dim>
T LinearCorotated<T, _dim>::psi(const Scratch& s) const
{
    return mu * s.e_hat.squaredNorm() + lambda * 0.5 * s.trace_e_hat * s.trace_e_hat;
}

template <class T, int _dim>
void LinearCorotated<T, _dim>::firstPiola(const Scratch& s, TM& P) const
{
    P.noalias() = (T)2 * mu * R * s.e_hat + lambda * s.trace_e_hat * R;
}

template <class T, int _dim>
void LinearCorotated<T, _dim>::firstPiolaDifferential(const Scratch& s, const TM& dF, TM& dP) const
{
    dP.noalias() = mu * dF + mu * R * dF.transpose() * R + lambda * (R.array() * dF.array()).sum() * R;
}

template <class T, int _dim>
bool LinearCorotated<T, _dim>::isC2(const Scratch& s, T tolerance) const
{
    return true;
}

template <class T, int _dim>
void LinearCorotated<T, _dim>::write(std::ostream& out) const
{
    writeEntry(out, mu);
    writeEntry(out, lambda);
    writeEntry(out, R);
}

template <class T, int _dim>
LinearCorotated<T, _dim> LinearCorotated<T, _dim>::read(std::istream& in)
{
    LinearCorotated<T, _dim> model;
    model.mu = readEntry<T>(in);
    model.lambda = readEntry<T>(in);
    model.R = readEntry<TM>(in);

    return model;
}

template class LinearCorotated<double, 1>;
template class LinearCorotated<double, 2>;
template class LinearCorotated<double, 3>;
template class LinearCorotated<float, 1>;
template class LinearCorotated<float, 2>;
template class LinearCorotated<float, 3>;
} // namespace ZIRAN
