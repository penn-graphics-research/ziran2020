#include <Ziran/CS/Util/BinaryIO.h>
#include <Ziran/Physics/ConstitutiveModel/LinearElasticity.h>
#include <Ziran/Physics/ConstitutiveModel/HyperelasticConstitutiveModel.h>

namespace ZIRAN {

template <class T, int dim>
LinearElasticity<T, dim>::LinearElasticity(const T E, const T nu)
{
    setLameParameters(E, nu);
}

template <class T, int dim>
void LinearElasticity<T, dim>::setLameParameters(const T E, const T nu)
{
    lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
    mu = E / ((T)2 * ((T)1 + nu));
}

template <class T, int dim>
void LinearElasticity<T, dim>::updateScratch(const TM& new_F, Scratch& scratch)
{
    scratch.F = new_F;
    scratch.epsilon = (T)0.5 * (new_F + new_F.transpose()) - TM::Identity();
}

template <class T, int dim>
T LinearElasticity<T, dim>::psi(const Scratch& s) const
{
    return mu * s.epsilon.squaredNorm() + (T).5 * lambda * s.epsilon.trace() * s.epsilon.trace();
}

template <class T, int dim>
void LinearElasticity<T, dim>::firstPiola(const Scratch& s, TM& P) const
{
    P.noalias() = (T)2 * mu * s.epsilon + lambda * s.epsilon.trace() * TM::Identity();
}

template <class T, int dim>
void LinearElasticity<T, dim>::firstPiolaDifferential(const Scratch& s, const TM& dF, TM& dP) const
{
    dP = mu * (dF + dF.transpose()) + lambda * dF.trace() * TM::Identity();
}

template <class T, int dim>
bool LinearElasticity<T, dim>::isC2(const Scratch& s, T tolerance) const
{
    return true;
}

template <class T, int dim>
void LinearElasticity<T, dim>::write(std::ostream& out) const
{
    writeEntry(out, mu);
    writeEntry(out, lambda);
}

template <class T, int dim>
LinearElasticity<T, dim> LinearElasticity<T, dim>::read(std::istream& in)
{
    LinearElasticity<T, dim> model;
    model.mu = readEntry<T>(in);
    model.lambda = readEntry<T>(in);
    return model;
}

template class LinearElasticity<double, 1>;
template class LinearElasticity<double, 2>;
template class LinearElasticity<double, 3>;
template class LinearElasticity<float, 1>;
template class LinearElasticity<float, 2>;
template class LinearElasticity<float, 3>;
} // namespace ZIRAN
