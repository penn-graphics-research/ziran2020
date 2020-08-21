#ifndef COTANGENT_ORTHOTROPIC_H
#define COTANGENT_ORTHOTROPIC_H

#include <Ziran/Math/Linear/GivensQR.h>

namespace ZIRAN {

template <class Derived>
class HyperelasticConstitutiveModel;

template <class T, int dim>
class CorotatedElasticity;

template <class T, int dim>
struct CorotatedScratch;

template <typename Derived>
struct HyperelasticTraits;

template <class T, int dim>
struct CotangentOrthotropicScratch;

template <class T, int dim>
class CotangentOrthotropic;

template <class T>
struct CotangentOrthotropicScratch<T, 1> {
    static const int dim = 1;

    CotangentOrthotropicScratch() {}

    static const char* name()
    {
        return "CotangentOrthotropicScratch";
    }
};

template <class T>
class CotangentOrthotropic<T, 1> : public HyperelasticConstitutiveModel<CotangentOrthotropic<T, 1>> {
public:
    static const int dim = 1;
    using Base = HyperelasticConstitutiveModel<CotangentOrthotropic<T, dim>>;
    using Scratch = typename HyperelasticTraits<CotangentOrthotropic<T, dim>>::ScratchType;

    CotangentOrthotropic(const T k0 = (T)1, const T k1 = (T)1, const T k2 = (T)1)
    {
        ZIRAN_ASSERT(false);
    }

    bool isC2(const Scratch& s, T tolerance) const
    {
        return false;
    }

    static const char* name()
    {
        return "CotangentOrthotropic";
    }

    static const char* scratch_name()
    {
        return Scratch::name();
    }
};

template <class T>
struct CotangentOrthotropicScratch<T, 2> {
    static const int dim = 2;
    using TM = Matrix<T, dim, dim>;

    T norm_d0;
    T norm_d1;
    TM F;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CotangentOrthotropicScratch()
        : F(TM::Identity())
    {
    }

    static const char* name()
    {
        return "CotangentOrthotropicScratch";
    }
};

template <class T>
class CotangentOrthotropic<T, 2> : public HyperelasticConstitutiveModel<CotangentOrthotropic<T, 2>> {
public:
    static const int dim = 2;
    using Base = HyperelasticConstitutiveModel<CotangentOrthotropic<T, dim>>;
    using TV = typename Base::TV;
    using TM = typename Base::TM;
    using Strain = Matrix<T, dim, dim>;
    using Hessian = typename Base::Hessian;
    using Scalar = typename Base::Scalar;
    using Scratch = typename HyperelasticTraits<CotangentOrthotropic<T, dim>>::ScratchType;
    using Vec = Vector<T, Eigen::Dynamic>;
    using VecBlock = Eigen::VectorBlock<Vec>;

    T k0, k1, k2;

    CotangentOrthotropic(const T k0 = (T)1, const T k1 = (T)1, const T k2 = (T)1)
        : k0(k0)
        , k1(k1)
        , k2(k2)
    {
    }

    void updateScratch(const TM& new_F, Scratch& scratch)
    {
        scratch.F = new_F;
        scratch.norm_d0 = new_F.col(0).norm();
        scratch.norm_d1 = new_F.col(1).norm();
    }

    static constexpr bool diagonalDifferentiable()
    {
        return false;
    }

    // d = (|d1|-1)^2
    // psi = k1 d \sqrt(d + eps)  (when d < 0)
    //     = 0 (when d >= 0)
    T unilaterialEnergyA(const Scratch& s) const
    {
        static const T eps = (T)1e-15;
        using MATH_TOOLS::sqr;
        T local_k1 = s.norm_d1 > 1 ? 0 : k1;
        T d = sqr(s.norm_d1 - 1);
        return local_k1 * d * sqrt(d + eps);
    }

    T unilateralEnergyADerivative(const Scratch& s) const
    {
        static const T eps = (T)1e-15;
        using MATH_TOOLS::cube;
        using MATH_TOOLS::sqr;
        T local_k1 = s.norm_d1 > 1 ? 0 : k1;
        T J = s.norm_d1;
        return local_k1 * (3 * cube(J - 1) + 2 * (J - 1) * eps) / (std::sqrt(sqr(J - 1) + eps));
    }

    T unilateralEnergyASecondDerivative(const Scratch& s) const
    {
        static const T eps = (T)1e-15;
        using MATH_TOOLS::cube;
        using MATH_TOOLS::sqr;
        using std::sqrt;
        T local_k1 = s.norm_d1 > 1 ? 0 : k1;
        T J = s.norm_d1;
        T jm1_sqr = sqr(J - 1);
        return local_k1 * (6 * sqr(jm1_sqr) + 9 * jm1_sqr * eps + 2 * sqr(eps)) / cube(sqrt(jm1_sqr + eps));
    }

    // psi = k1 log(|d1|+eps) (|d1|-1)^3
    T unilaterialEnergyB(const Scratch& s) const
    {
        using MATH_TOOLS::cube;
        static const T eps = (T)1e-15;
        T local_k1 = s.norm_d1 > 1 ? 0 : k1;
        T J = s.norm_d1;
        return local_k1 * std::log(J + eps) * cube(J - 1);
    }

    T unilateralEnergyBDerivative(const Scratch& s) const
    {
        static const T eps = (T)1e-15;
        using MATH_TOOLS::sqr;
        using std::log;
        T local_k1 = s.norm_d1 > 1 ? 0 : k1;
        T J = s.norm_d1;
        return local_k1 * sqr(J - 1) * ((J - 1) / (J + eps) + 3 * log(J + eps));
    }

    T unilateralEnergyBSecondDerivative(const Scratch& s) const
    {
        static const T eps = (T)1e-15;
        using MATH_TOOLS::sqr;
        using std::log;
        T local_k1 = s.norm_d1 > 1 ? 0 : k1;
        T J = s.norm_d1;
        return local_k1 * (J - 1) * ((J - 1) * (1 + 5 * J + 6 * eps) / sqr(J + eps) + 6 * log(J + eps));
    }

    T psi(const Scratch& s) const
    {
        using MATH_TOOLS::sqr;
        TM C = s.F.transpose() * s.F;
        return (T)0.5 * k0 * sqr(s.norm_d0 - 1)
            + unilaterialEnergyB(s)
            + (T)0.5 * k2 * C(0, 1) * C(1, 0) / (C(0, 0) * C(1, 1));
        // Always pretend C is not symmetric when writing psi in terms of C
    }

    void firstPiola(const Scratch& s, TM& P) const
    {
        using MATH_TOOLS::sqr;
        TM C = s.F.transpose() * s.F;
        T gamma = k2 * C(0, 1) / (C(0, 0) * C(1, 1));
        TM S;
        S << -C(0, 1) / C(0, 0), 1, 1, -C(0, 1) / C(1, 1);
        S *= gamma;
        P = s.F * S;
        P.col(0) += k0 * (1 - 1 / s.norm_d0) * s.F.col(0);
        P.col(1) += unilateralEnergyBDerivative(s) * s.F.col(1) / s.norm_d1;
    }

    void firstPiolaDifferential(const Scratch& s, const TM& dF, TM& dP) const
    {
        using MATH_TOOLS::cube;
        using MATH_TOOLS::sqr;
        using std::sqrt;
        TM C = s.F.transpose() * s.F;
        TM dC = dF.transpose() * s.F + s.F.transpose() * dF;
        T gamma = k2 / (C(0, 0) * C(1, 1));
        T dgamma = -gamma * (dC(0, 0) / C(0, 0) + dC(1, 1) / C(1, 1));
        TM M;
        M << -C(0, 1) * C(1, 0) / C(0, 0), C(1, 0),
            C(0, 1), -C(0, 1) * C(1, 0) / C(1, 1);

        TM dM;
        dM << (-dC(0, 1) * C(1, 0) - C(0, 1) * dC(1, 0) + C(0, 1) * C(1, 0) * dC(0, 0) / C(0, 0)) / C(0, 0), dC(1, 0),
            dC(0, 1), (-dC(0, 1) * C(1, 0) - C(0, 1) * dC(1, 0) + C(0, 1) * C(1, 0) * dC(1, 1) / C(1, 1)) / C(1, 1);

        TM S = gamma * M;
        TM dS = dgamma * M + gamma * dM;
        dP = dF * S + s.F * dS;
        T alpha = s.F.col(0).dot(dF.col(0));
        dP.col(0) += k0 * ((1 - 1 / s.norm_d0) * dF.col(0) + alpha / cube(s.norm_d0) * s.F.col(0));

        TV b = s.F.col(1) / s.norm_d1;
        T a = unilateralEnergyBSecondDerivative(s) * b.dot(dF.col(1));
        T c = unilateralEnergyBDerivative(s);
        T x = s.F(0, 1), y = s.F(1, 1), dx = dF(0, 1), dy = dF(1, 1);
        T z = (y * dx - x * dy) / cube(s.norm_d1);
        TV d;
        d << y * z, -x * z;
        dP.col(1) += a * b + c * d;
    }

    bool isC2(const Scratch& s, T tolerance) const
    {
        return s.norm_d0 > tolerance && s.norm_d1 > tolerance;
    }

    static const char* name()
    {
        return "CotangentOrthotropic";
    }

    static const char* scratch_name()
    {
        return Scratch::name();
    }
};

template <class T>
struct CotangentOrthotropicScratch<T, 3> {
    static const int dim = 3;
    using TM = Matrix<T, dim, dim>;

    T norm_d;
    TM F;
    Matrix<T, 3, 2> Q; // the rotation from the 2D plane

    CorotatedScratch<T, 2> corotated_scratch;

    CotangentOrthotropicScratch()
        : F(TM::Identity())
    {
    }

    static const char* name()
    {
        return "CotangentOrthotropicScratch";
    }
};

template <class T>
class CotangentOrthotropic<T, 3> : public HyperelasticConstitutiveModel<CotangentOrthotropic<T, 3>> {
public:
    static const int dim = 3;
    using Base = HyperelasticConstitutiveModel<CotangentOrthotropic<T, dim>>;
    using TV = typename Base::TV;
    using TM = typename Base::TM;
    using Strain = Matrix<T, dim, dim>;
    using Hessian = typename Base::Hessian;
    using Scalar = typename Base::Scalar;
    using Scratch = typename HyperelasticTraits<CotangentOrthotropic<T, dim>>::ScratchType;

    CorotatedElasticity<T, 2> corotated;
    T k;

    CotangentOrthotropic(const T E = (T)1, const T nu = (T)0.3, const T k = (T)0)
        : corotated(E, nu)
        , k(k)
    {
    }

    void updateScratch(const TM& new_F, Scratch& scratch)
    {
        scratch.F = new_F;
        Matrix<T, 2, 2> R;
        Matrix<T, 3, 2> F_tangent = new_F.template topLeftCorner<3, 2>();
        thinGivensQR(F_tangent, scratch.Q, R);

        corotated.updateScratch(R, scratch.corotated_scratch);

        scratch.norm_d = new_F.col(2).norm();
    }

    static constexpr bool diagonalDifferentiable()
    {
        return false;
    }

    T psi(const Scratch& s) const
    {
        static const T eps = (T)1e-15;
        using MATH_TOOLS::sqr;
        T local_k = s.norm_d > 1 ? 0 : k;
        T d = sqr(s.norm_d - 1);
        return (T)0.5 * local_k * d * sqrt(d + eps) + corotated.psi(s.corotated_scratch);
    }

    void firstPiola(const Scratch& s, TM& P) const
    {
        static const T eps = (T)1e-15;
        using MATH_TOOLS::sqr;
        Matrix<T, 2, 2> corotated_P;
        corotated.firstPiola(s.corotated_scratch, corotated_P);
        P.template topLeftCorner<3, 2>() = s.Q * corotated_P;
        T local_k = s.norm_d > 1 ? 0 : k;
        T d = sqr(s.norm_d - 1);
        P.col(2) = local_k * (3 * d + 2 * eps) / (2 * sqrt(d + eps)) * (1 - 1 / s.norm_d) * s.F.col(2);
    }

    void firstPiolaDifferential(const Scratch& s, const TM& dF, TM& dP) const
    {
        static const T eps = (T)1e-15;
        using MATH_TOOLS::cube;
        using MATH_TOOLS::sqr;
        using std::sqrt;

        Matrix<T, 2, 2> corotated_dF = s.Q.transpose() * dF.template topLeftCorner<3, 2>();
        Matrix<T, 2, 2> corotated_dP;
        corotated.firstPiolaDifferential(s.corotated_scratch, corotated_dF, corotated_dP);
        dP.template topLeftCorner<3, 2>() = s.Q * corotated_dP;

        T d = sqr(s.norm_d - 1);
        T local_k = s.norm_d > 1 ? 0 : k;
        TV Q = (1 - 1 / s.norm_d) * s.F.col(2);
        T Q_contract_dF = Q.dot(dF.col(2));
        T R_prime = (3 * d + 4 * eps) / (4 * cube(sqrt(d + eps)));
        T R = (3 * d + 2 * eps) / (2 * sqrt(d + eps));
        T alpha = s.F.col(2).dot(dF.col(2));
        TV dQ = (1 - 1 / s.norm_d) * dF.col(2) + alpha / cube(s.norm_d) * s.F.col(2);
        dP.col(2) = local_k * R_prime * 2 * Q_contract_dF * Q + local_k * R * dQ;
    }

    bool isC2(const Scratch& s, T tolerance) const
    {
        return corotated.isC2(s.corotated_scratch, tolerance) && s.norm_d > tolerance;
    }

    static const char* name()
    {
        return "CotangentOrthotropic";
    }

    static const char* scratch_name()
    {
        return Scratch::name();
    }
};

template <class T, int dim>
struct HyperelasticTraits<CotangentOrthotropic<T, dim>> {
    using ScratchType = CotangentOrthotropicScratch<T, dim>;
};

template <class T, int dim>
struct RW<CotangentOrthotropicScratch<T, dim>> {
    using Tag = NoWriteTag<CotangentOrthotropicScratch<T, dim>>;
};
} // namespace ZIRAN

#endif
