#ifndef QR_STABLE_NEO_HOOKEAN_H
#define QR_STABLE_NEO_HOOKEAN_H

#include <Ziran/Physics/ConstitutiveModel/QrBasedHelper.h>
#include <Ziran/Math/Linear/GivensQR.h>
namespace ZIRAN {

template <class T, int _dim>
struct QRStableNeoHookeanScratch {
    static const int dim = _dim;
    using TM = Matrix<T, dim, dim>;

    TM F;
    TM Q; // from the QR of F
    TM R; // from the QR of F
    TM R_inv;
    T J;
    QrBasedHelper<T, dim> qr_helper;
    T energy_density;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    QRStableNeoHookeanScratch()
        : F(TM::Identity())
    {
    }

    static const char* name()
    {
        return "QRStableNeoHookeanScratch";
    }
};

template <class T, int _dim>
class QRStableNeoHookean : public HyperelasticConstitutiveModel<QRStableNeoHookean<T, _dim>> {
public:
    static const int dim = _dim;
    using Base = HyperelasticConstitutiveModel<QRStableNeoHookean<T, dim>>;
    using TV = typename Base::TV;
    using TM = typename Base::TM;
    using Strain = Matrix<T, dim, dim>;
    using Hessian = typename Base::Hessian;
    using Scalar = typename Base::Scalar;
    using Scratch = typename HyperelasticTraits<QRStableNeoHookean<T, dim>>::ScratchType;

    T mu;
    T lambda;
    T kx, ky;

    T g; // degradation multiplier

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    QRStableNeoHookean(const T E = (T)1, const T nu = (T)0.3)
        : mu(0), lambda(0), kx(0), ky(0), g(1)
    {
        setParameters(E, nu);
    }

    void setParameters(const T E, const T nu)
    {
        lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
        mu = E / ((T)2 * ((T)1 + nu));
    }

    T getSigmaCrit(T percentage) const
    {
        // Create a dummy model that is isotropic and non-degraded
        QRStableNeoHookean<T, dim> model_dummy = *this;
        model_dummy.g = 1;
        model_dummy.kx = 0;
        model_dummy.ky = 0;

        TM F_dummy = Matrix<T, dim, dim>::Identity() * ((T)1 + percentage);
        Scratch scratch_dummy;
        model_dummy.updateScratch(F_dummy, scratch_dummy);
        return model_dummy.psi(scratch_dummy);
    }

    //
    // Get effective Cauchy stress: non-degraded isotropic stress
    //
    void evaluateEffectiveCauchyStress(const TM& F, TM& cauchystress)
    { // let a dummy model copied from the real model call this function
        g = 1; // undegrade.
        kx = 0;
        ky = 0;
        Scratch scratch_dummy;
        updateScratch(F, scratch_dummy);
        TM P;
        firstPiola(scratch_dummy, P);
        cauchystress = P * F.transpose() / scratch_dummy.J;
    }

    void setExtraFiberStiffness(const int component, const T scale)
    {
        if (component == 0)
            kx = scale * mu;
        else if (component == 1) {
            ZIRAN_ASSERT(dim == 3, "Only scale y direction fiber for 3D.");
            ky = scale * mu;
        }
        else {
            ZIRAN_ASSERT(0, "It is not allowed to put fiber at z direction");
        }
    }

    void updateScratch(const TM& new_F, Scratch& s)
    {
        using MATH_TOOLS::sqr;

        s.F = new_F;
        GivensQR(new_F, s.Q, s.R);

        // sign convention
        if constexpr (dim == 2) {
            if (s.R(0, 0) < 0) {
                s.R *= -1;
                s.Q *= -1;
            }
        }
        else if constexpr (dim == 3) {
            if (s.R(0, 0) < 0) {
                s.R(0, 0) *= -1;
                s.Q.col(0) *= -1;
                s.R(0, 1) *= -1;
                s.R(0, 2) *= -1;
                s.R(2, 2) *= -1;
                s.Q.col(2) *= -1;
            }
            if (s.R(1, 1) < 0) {
                s.R(1, 1) *= -1;
                s.Q.col(1) *= -1;
                s.R(1, 2) *= -1;
                s.R(2, 2) *= -1;
                s.Q.col(2) *= -1;
            }
        }

        s.R_inv = s.R.inverse();
        s.J = 1;
        for (int i = 0; i < dim; i++)
            s.J *= s.R(i, i);

        s.energy_density = 0;

        // lambda term contribution to energy
        T degraded_volume_term = 0;
        if (s.J >= 1)
            degraded_volume_term = g * (T)0.5 * lambda * sqr(s.J - 1);
        else
            degraded_volume_term = (T)0.5 * lambda * sqr(s.J - 1);
        s.energy_density += degraded_volume_term;

        // mu term contribution to energy
        T r_squared_sum = 0;
        for (int i = 0; i < dim; i++)
            for (int j = i; j < dim; j++)
                r_squared_sum += sqr(s.R(i, j));
        s.energy_density += g * (mu / 2 * (r_squared_sum - dim) - mu * (s.J - 1));
        // s.energy_density += g* (mu/2*(r_squared_sum - dim)-mu*std::log(s.J));

        // kx fiber term (no degradation)
        s.energy_density += kx / 2 * sqr(s.R(0, 0) - 1);

        // ky fiber term (no degradation)
        if (dim == 3)
            s.energy_density += ky / 2 * sqr(std::sqrt(sqr(s.R(0, 1)) + sqr(s.R(1, 1))) - 1);

        TM& derivative = s.qr_helper.dPsiHatdR;
        derivative = TM::Zero();

        // lambda term contribution to the derivative
        if (s.J >= 1)
            for (int i = 0; i < dim; i++)
                derivative(i, i) += g * lambda * (s.J - 1) * s.J / s.R(i, i);
        else
            for (int i = 0; i < dim; i++)
                derivative(i, i) += lambda * (s.J - 1) * s.J / s.R(i, i);

        // mu term contribution to the derivative
        for (int i = 0; i < dim; i++)
            for (int j = i; j < dim; j++)
                derivative(i, j) += g * mu * s.R(i, j);
        for (int i = 0; i < dim; i++)
            derivative(i, i) -= g * mu * s.J / s.R(i, i);
        //derivative(i,i) -= g * mu / s.R(i,i);

        // kx term contribution to the derivative
        derivative(0, 0) += kx * (s.R(0, 0) - 1);

        // ky term contribution to the derivative
        if (dim == 3) {
            T ss = std::sqrt(sqr(s.R(0, 1)) + sqr(s.R(1, 1)));
            derivative(0, 1) += ky * (ss - 1) * s.R(0, 1) / ss;
            derivative(1, 1) += ky * (ss - 1) * s.R(1, 1) / ss;
        }
    }

    static constexpr bool diagonalDifferentiable()
    {
        return false;
    }

    T psi(const Scratch& s) const
    {
        return s.energy_density;
    }

    void firstPiola(const Scratch& s, TM& P) const
    {
        s.qr_helper.evaluateP(s.Q, s.R, s.R_inv, P);
    }

    void firstPiolaDifferential(const Scratch& s, const TM& dF, TM& dP) const
    {
        using MATH_TOOLS::sqr;
        ZIRAN_ASSERT(g == 1, "implicit currently only implemented for no damage");
        ZIRAN_ASSERT(ky == 0, "implicit currently only implemented for no ky");

        if constexpr (dim == 2) {
            ZIRAN_ASSERT(0, "only 3D");
        }
        else {
            if (std::abs(s.J) < 1e-5) {
                dP = TM::Zero();
                return;
            }
            TM P = TM::Zero();
            firstPiola(s, P);
            TM dQ, dR;
            const TM& R = s.R;
            const TM& Q = s.Q;
            const T& J = s.J;
            EIGEN_EXT::QRDifferential(Q, R, dF, dQ, dR);
            dP = -s.Q * dQ.transpose() * P - P * dR.transpose() * s.R_inv.transpose();

            // dP += Q dA R^{-T}
            /*
                dPsiHatdR00 = mu r00 - mu r11 r22 + lambda (r00 r11 r22 -1) r11 r22 + kx (r00-1)
                dPsiHatdR01 = mu r01
                dPsiHatdR02 = mu r02
                dPsiHatdR11 = mu r11 - mu r00 r22 + lambda (r00 r11 r22 -1) r00 r22
                dPsiHatdR12 = mu r12
                dPsiHatdR22 = mu r22 - mu r00 r11 + lambda (r00 r11 r22 -1) r00 r11
                A(0,0) =  mu r00^2 - mu r00 r11 r22 + lambda (r00 r11 r22 -1) r00 r11 r22 + kx (r00-1) r00
                        + mu r01 r01 + mu r02 r02
                       =  mu r00^2 - mu J + lambda (J-1) J + kx (r00-1) r00 + mu r01^2 + mu r02^2
                A(0,1) =  mu r01 r11 + mu r02 r12
                A(0,2) =  mu r02 r22
                A(1,1) =  mu r11^2 - mu r00 r11 r22 + lambda (r00 r11 r22 -1) r00 r11 r22 + mu r12 r12
                       =  mu r11^2 - mu J + lambda (J-1) J + mu r12^2
                A(1,2) =  mu r12 r22
                A(2,2) =  mu r22^2 - mu r00 r11 r22 + lambda (r00 r11 r22 -1) r00 r11 r22
                       =  mu r22^2 - mu J + lambda (J-1) J
                dJ  = dr00 r11 r22 + r00 dr11 r22 + r00 r11 dr22
                b  =  (J-1) J
                db =  dJ J + (J-1) dJ
                dA(0,0) = 2 mu r00 dr00 - mu dJ + lambda db + kx dr00 r00 + kx (r00-1) dr00 + 2 mu r01 dr01 + 2 mu r02 dr02
                dA(0,1) = mu dr01 r11 + mu r01 dr11 + mu dr02 r12 + mu r02 dr12
                dA(0,2) = mu dr02 r22 + mu r02 dr22
                dA(1,1) = 2 mu r11 dr11 - mu dJ + lambda db + 2 mu r12 dr12
                dA(1,2) = mu dr12 r22 + mu r12 dr22
                dA(2,2) = 2 mu r22 dr22 - mu dJ + lambda db
            */
            TM dA = TM::Zero();
            T dJ = dR(0, 0) * R(1, 1) * R(2, 2) + R(0, 0) * dR(1, 1) * R(2, 2) + R(0, 0) * R(1, 1) * dR(2, 2);
            T db = dJ * s.J + (J - 1) * dJ;
            dA(0, 0) = 2 * mu * R(0, 0) * dR(0, 0) - mu * dJ + lambda * db + kx * dR(0, 0) * R(0, 0) + kx * (R(0, 0) - 1) * dR(0, 0) + 2 * mu * R(0, 1) * dR(0, 1) + 2 * mu * R(0, 2) * dR(0, 2);
            dA(0, 1) = mu * dR(0, 1) * R(1, 1) + mu * R(0, 1) * dR(1, 1) + mu * dR(0, 2) * R(1, 2) + mu * R(0, 2) * dR(1, 2);
            dA(0, 2) = mu * dR(0, 2) * R(2, 2) + mu * R(0, 2) * dR(2, 2);
            dA(1, 1) = 2 * mu * R(1, 1) * dR(1, 1) - mu * dJ + lambda * db + 2 * mu * R(1, 2) * dR(1, 2);
            dA(1, 2) = mu * dR(1, 2) * R(2, 2) + mu * R(1, 2) * dR(2, 2);
            dA(2, 2) = 2 * mu * R(2, 2) * dR(2, 2) - mu * dJ + lambda * db;
            dA(1, 0) = dA(0, 1);
            dA(2, 0) = dA(0, 2);
            dA(2, 1) = dA(1, 2);
            dP += Q * dA * s.R_inv.transpose();
        }
    }

    bool isC2(const Scratch& s, T tolerance) const
    {
        return std::abs(s.J) > (T)2e-5;
    }

    bool hessianImplemented() const
    {
        return true; // derivative test skips dPdF test if this is false
    }

    static const char* name()
    {
        return "QRStableNeoHookean";
    }

    static const char* scratch_name()
    {
        return Scratch::name();
    }
};

template <class T, int dim>
struct HyperelasticTraits<QRStableNeoHookean<T, dim>> {
    using ScratchType = QRStableNeoHookeanScratch<T, dim>;
};

template <class T, int dim>
struct RW<QRStableNeoHookeanScratch<T, dim>> {
    using Tag = NoWriteTag<QRStableNeoHookeanScratch<T, dim>>;
};
} // namespace ZIRAN
#endif
