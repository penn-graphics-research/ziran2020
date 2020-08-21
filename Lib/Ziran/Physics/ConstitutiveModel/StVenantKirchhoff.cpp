#include <Ziran/CS/Util/BinaryIO.h>
#include <Ziran/Physics/ConstitutiveModel/StVenantKirchhoff.h>
#include <Ziran/Physics/ConstitutiveModel/HyperelasticConstitutiveModel.h>
#include <Ziran/Math/Linear/DenseExt.h>
#include <Ziran/Math/Linear/ImplicitQRSVD.h>

namespace ZIRAN {

template <class T, int _dim>
StVenantKirchhoff<T, _dim>::StVenantKirchhoff(const T E, const T nu)
{
    setLameParameters(E, nu);
}

template <class T, int _dim>
void StVenantKirchhoff<T, _dim>::setLameParameters(const T E, const T nu)
{
    lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
    mu = E / ((T)2 * ((T)1 + nu));
}

template <class T, int _dim>
void StVenantKirchhoff<T, _dim>::updateScratch(const TM& new_F, Scratch& scratch)
{
    using namespace EIGEN_EXT;
    scratch.F = new_F;
    singularValueDecomposition(scratch.F, scratch.U, scratch.sigma, scratch.V);
    scratch.sigma_square_m1.array() = scratch.sigma.array() * scratch.sigma.array() - 1;
    scratch.trE = (T).5 * (scratch.sigma.squaredNorm() - dim);
}

template <class T, int _dim>
T StVenantKirchhoff<T, _dim>::psi(const Scratch& s) const
{
    return (T).5 * lambda * s.trE * s.trE + mu * (T).25 * s.sigma_square_m1.squaredNorm();
}

template <class T, int _dim>
void StVenantKirchhoff<T, _dim>::firstPiola(const Scratch& s, TM& P) const
{
    TV skew;
    skew.array() = s.sigma_square_m1.array() * s.sigma.array();
    P = s.U * (lambda * s.trE * s.sigma + mu * skew).asDiagonal() * s.V.transpose();
}

template <class T, int _dim>
void StVenantKirchhoff<T, _dim>::firstPiolaDerivative(const Scratch& s, Hessian& dPdF) const
{
    dPdF = lambda * EIGEN_EXT::vec(s.F) * EIGEN_EXT::vec(s.F).transpose();
    for (int j = 0; j < dim; j++) {
        for (int i = 0; i < dim; i++) {
            for (int n = 0; n < dim; n++) {
                for (int m = 0; m < dim; m++) {
                    if (i == m) {
                        for (int h = 0; h < dim; h++)
                            dPdF(i + j * dim, m + n * dim) += mu * s.F(h, n) * s.F(h, j);
                    }
                    dPdF(i + j * dim, m + n * dim) += mu * s.F(i, n) * s.F(m, j);
                    if (j == n) {
                        for (int k = 0; k < dim; k++)
                            dPdF(i + j * dim, m + n * dim) += mu * s.F(i, k) * s.F(m, k);
                    }
                }
            }
        }
    }
    dPdF.diagonal().array() += lambda * s.trE - mu;
}

template <class T>
bool isC2Helper(const StVenantKirchhoffScratch<T, 1>& s, T tolerance)
{
    return true;
}

template <class T>
bool isC2Helper(const StVenantKirchhoffScratch<T, 2>& s, T tolerance)
{
    return !EIGEN_EXT::nearKink(s.F, tolerance);
}

template <class T>
bool isC2Helper(const StVenantKirchhoffScratch<T, 3>& s, T tolerance)
{
    return !EIGEN_EXT::nearKink(s.F, tolerance);
}

template <class T, int _dim>
bool StVenantKirchhoff<T, _dim>::isC2(const Scratch& s, T tolerance) const
{
    return isC2Helper(s, tolerance);
}

template <class T, int _dim>
void StVenantKirchhoff<T, _dim>::write(std::ostream& out) const
{
    writeEntry(out, mu);
    writeEntry(out, lambda);
}

template <class T, int _dim>
StVenantKirchhoff<T, _dim> StVenantKirchhoff<T, _dim>::read(std::istream& in)
{
    StVenantKirchhoff<T, _dim> model;
    model.mu = readEntry<T>(in);
    model.lambda = readEntry<T>(in);
    return model;
}

template class StVenantKirchhoff<double, 1>;
template class StVenantKirchhoff<double, 2>;
template class StVenantKirchhoff<double, 3>;
template class StVenantKirchhoff<float, 1>;
template class StVenantKirchhoff<float, 2>;
template class StVenantKirchhoff<float, 3>;
} // namespace ZIRAN
