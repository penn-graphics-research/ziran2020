#include <Ziran/Physics/ConstitutiveModel/NeoHookean.h>
#include <Ziran/Physics/ConstitutiveModel/HyperelasticConstitutiveModel.h>
#include <Ziran/Math/Linear/DenseExt.h>
#include <Ziran/CS/Util/BinaryIO.h>
#include <cmath>

namespace ZIRAN {

template <class T, int _dim>
NeoHookean<T, _dim>::NeoHookean(const T E, const T nu)
{
    setLameParameters(E, nu);
}

template <class T, int _dim>
void NeoHookean<T, _dim>::setLameParameters(const T E, const T nu)
{
    lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
    mu = E / ((T)2 * ((T)1 + nu));
}

template <class T, int _dim>
void NeoHookean<T, _dim>::updateScratch(const TM& new_F, Scratch& scratch)
{
    using std::log;
    using namespace EIGEN_EXT;
    scratch.F = new_F;
    scratch.J = scratch.F.determinant();

    TM JFinvT;
    EIGEN_EXT::cofactorMatrix(scratch.F, JFinvT);
    scratch.FinvT = ((T)1 / scratch.J) * JFinvT;
    scratch.logJ = log(scratch.J);
}

template <class T, int _dim>
T NeoHookean<T, _dim>::psi(const Scratch& s) const
{
    T I1 = EIGEN_EXT::firstInvariant(s.F);
    return (T)0.5 * mu * (I1 - _dim) - mu * s.logJ + (T)0.5 * lambda * s.logJ * s.logJ;
}

template <class T, int _dim>
void NeoHookean<T, _dim>::kirchhoff(const Scratch& s, TM& tau) const
{
    T scale = lambda * s.logJ - mu;
    tau = (mu * s.F * s.F.transpose() + scale * TM::Identity());
}

template <class T, int _dim>
void NeoHookean<T, _dim>::firstPiolaDerivative(const Scratch& s, Hessian& dPdF) const
{
    using namespace EIGEN_EXT;
    T scale = lambda * s.logJ - mu;

    dPdF = ((lambda - scale)) * vec(s.FinvT) * vec(s.FinvT).transpose();
    dPdF.diagonal().array() += mu;

    addScaledCofactorMatrixDerivative(s.F, (scale / s.J), dPdF);
}

template <class T, int _dim>
void NeoHookean<T, _dim>::firstPiolaDifferential(const Scratch& s, const TM& dF, TM& dP) const
{
    using namespace EIGEN_EXT;

    cofactorMatrixDifferential(s.F, dF, dP);
    T scale = lambda * s.logJ - mu;
    dP = mu * dF + (lambda - scale) * s.FinvT.cwiseProduct(dF).sum() * s.FinvT + scale / s.J * dP;
}

template <class T, int _dim>
bool NeoHookean<T, _dim>::isC2(const Scratch& s, T tolerance) const
{
    return s.J > tolerance;
}

template <class T, int _dim>
void NeoHookean<T, _dim>::write(std::ostream& out) const
{
    writeEntry(out, mu);
    writeEntry(out, lambda);
}

template <class T, int _dim>
NeoHookean<T, _dim> NeoHookean<T, _dim>::read(std::istream& in)
{
    NeoHookean<T, _dim> model;
    model.mu = readEntry<T>(in);
    model.lambda = readEntry<T>(in);
    return model;
}

template class NeoHookean<double, 1>;
template class NeoHookean<double, 2>;
template class NeoHookean<double, 3>;
template class NeoHookean<float, 1>;
template class NeoHookean<float, 2>;
template class NeoHookean<float, 3>;
} // namespace ZIRAN
