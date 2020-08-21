#ifndef QUADRATIC_VOLUME_PENALTY_H
#define QUADRATIC_VOLUME_PENALTY_H

#include <Ziran/CS/Util/Forward.h>
#include <Ziran/CS/Util/BinaryIO.h>
#include <Ziran/Math/MathTools.h>

namespace ZIRAN {

template <typename Derived>
struct HyperelasticTraits;

template <class T, int dim>
struct QuadraticVolumePenaltyScratch {
    T J;

    QuadraticVolumePenaltyScratch()
        : J(1)
    {
    }

    // This copy construct exists so that BinaryIO doesn't think this struct
    // is trivially copyable.
    QuadraticVolumePenaltyScratch(const QuadraticVolumePenaltyScratch& other)
    {
        J = other.J;
    }

    static const char* name()
    {
        return "QuadraticVolumePenaltyScratch";
    }
};

/**
*****************************************************************************
\Psi(J) = lambda/2 * ( J - 1 )^2
*****************************************************************************
*/
template <class T, int _dim>
class QuadraticVolumePenalty {
public:
    static const int dim = _dim;
    using TM = Matrix<T, dim, dim>;
    using Strain = T;
    using Scalar = T;
    using Scratch = typename HyperelasticTraits<QuadraticVolumePenalty<T, dim>>::ScratchType;

    T lambda;

    QuadraticVolumePenalty(const T E = (T)1, const T nu = (T)0.3)
    {
        setMaterialParameters(E, nu);
    }

    void setMaterialParameters(const T E, const T nu)
    {
        lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
    }

    void updateScratch(const T J, Scratch& scratch) const
    {
        scratch.J = J;
    }

    // Psi(J) = lambda/2 * ( J - 1 )^2
    T psi(const Scratch& s) const
    {
        return lambda / (T)2 * MATH_TOOLS::sqr(s.J - 1);
    }

    // d psi d J = lambda (J - 1)
    void firstDerivative(const Scratch& s, T& dpsi_dJ) const
    {
        dpsi_dJ = lambda * (s.J - 1);
    }

    // d^2 psi d J^2 =  lambda
    void secondDerivative(const Scratch& s, T& d2psi_dJ2) const
    {
        d2psi_dJ2 = lambda;
    }

    bool isC2(const Scratch& s, T tolerance) const
    {
        return true;
    }

    void write(std::ostream& out) const
    {
        writeEntry(out, lambda);
    }

    static QuadraticVolumePenalty<T, dim> read(std::istream& in)
    {
        QuadraticVolumePenalty<T, dim> model;
        model.lambda = readEntry<T>(in);
        return model;
    }

    static const char* name()
    {
        return "QuadraticVolumePenalty";
    }

    static const char* scratch_name()
    {
        return Scratch::name();
    }
};

template <class T, int dim>
struct HyperelasticTraits<QuadraticVolumePenalty<T, dim>> {
    using ScratchType = QuadraticVolumePenaltyScratch<T, dim>;
};

template <class T, int dim>
struct RW<QuadraticVolumePenaltyScratch<T, dim>> {
    using Tag = NoWriteTag<QuadraticVolumePenaltyScratch<T, dim>>;
};
} // namespace ZIRAN

#endif