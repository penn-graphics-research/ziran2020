#ifndef EQUATION_OF_STATE_H
#define EQUATION_OF_STATE_H

#include <Ziran/CS/Util/Forward.h>
#include <Ziran/CS/Util/BinaryIO.h>

namespace ZIRAN {

template <typename Derived>
struct HyperelasticTraits;

template <class T, int dim>
struct EquationOfStateScratch {
    T J;

    EquationOfStateScratch()
        : J(1)
    {
    }

    // This copy construct exists so that BinaryIO doesn't think this struct
    // is trivially copyable.
    EquationOfStateScratch(const EquationOfStateScratch& other)
    {
        J = other.J;
    }

    static const char* name()
    {
        return "EquationOfStateScratch";
    }
};

/**
   Fluid equation of state consitutive model uses J as strain measure
   Cauchy stress: sigma = -p I
   where p = k((1/J)^gamma - 1)
   k is bulk modulus
   gamma = 7 for water.
   In terms of energy, 
   *****************************************************************************
   \Psi(J) = - k * ( J^{1-gamma}/(1-gamma) - J ), 
   *****************************************************************************
   where J = 1/rho.
   Using P := d \Psi / d F, sigma = (1/J) P F^T, it can be shown this energy density gives the right sigma.
 */
template <class T, int _dim>
class EquationOfState {
public:
    static const int dim = _dim;
    using TM = Matrix<T, dim, dim>;
    using Strain = T;
    using Scalar = T;
    using Scratch = typename HyperelasticTraits<EquationOfState<T, dim>>::ScratchType;

    T bulk, gamma;

    EquationOfState(const T bulk_in = (T)100, const T gamma_in = 7)
    {
        setMaterialParameters(bulk_in, gamma_in);
    }

    void setMaterialParameters(const T bulk_in, const T gamma_in)
    {
        bulk = bulk_in;
        gamma = gamma_in;
    }

    void updateScratch(const T J, Scratch& scratch) const
    {
        scratch.J = J;
    }

    // Psi(J) = - k * ( J^{1-gamma}/(1-gamma) - J ),
    T psi(const Scratch& s) const
    {
        return -bulk * (std::pow(s.J, (T)1 - gamma) / ((T)1 - gamma) - s.J);
    }

    // d psi d J = - k ( J^{-gamma} - 1 )
    void firstDerivative(const Scratch& s, T& dpsi_dJ) const
    {
        dpsi_dJ = -bulk * (std::pow(s.J, -gamma) - (T)1);
    }

    // d^2 psi d J^2 =  k gamma J ^{-gamma-1}
    void secondDerivative(const Scratch& s, T& d2psi_dJ2) const
    {
        d2psi_dJ2 = bulk * gamma * std::pow(s.J, -gamma - (T)1);
    }

    bool isC2(const Scratch& s, T tolerance) const
    {
        return s.J > tolerance;
    }

    void write(std::ostream& out) const
    {
        writeEntry(out, bulk);
        writeEntry(out, gamma);
    }

    static EquationOfState<T, dim> read(std::istream& in)
    {
        EquationOfState<T, dim> model;
        model.bulk = readEntry<T>(in);
        model.gamma = readEntry<T>(in);
        return model;
    }

    static const char* name()
    {
        return "EquationOfState";
    }

    static const char* scratch_name()
    {
        return Scratch::name();
    }
};

template <class T, int dim>
struct HyperelasticTraits<EquationOfState<T, dim>> {
    using ScratchType = EquationOfStateScratch<T, dim>;
};

template <class T, int dim>
struct RW<EquationOfStateScratch<T, dim>> {
    using Tag = NoWriteTag<EquationOfStateScratch<T, dim>>;
};
} // namespace ZIRAN

#endif
