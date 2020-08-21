#ifndef QR_ANISOTROPIC_H
#define QR_ANISOTROPIC_H
#include <Ziran/Physics/ConstitutiveModel/QrBasedHelper.h>
#include <Ziran/Math/Linear/GivensQR.h>
namespace ZIRAN {

template <class T, int dim>
struct QRAnisotropicScratch {
    using TM = Matrix<T, dim, dim>;

    TM F;
    TM Q; // from the QR of F
    TM R; // from the QR of F
    TM R_inv;
    T J;
    QrBasedHelper<T, dim> qr_helper;
    T energy_density;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    QRAnisotropicScratch()
        : F(TM::Identity())
    {
    }

    static const char* name()
    {
        return "QRAnisotropicScratch";
    }
};

template <class T, int dim>
class QRAnisotropic : public HyperelasticConstitutiveModel<QRAnisotropic<T, dim>> {
public:
    using Base = HyperelasticConstitutiveModel<QRAnisotropic<T, dim>>;
    using TV = typename Base::TV;
    using TM = typename Base::TM;
    using Strain = Matrix<T, dim, dim>;
    using Hessian = typename Base::Hessian;
    using Scalar = typename Base::Scalar;
    using Scratch = typename HyperelasticTraits<QRAnisotropic<T, dim>>::ScratchType;

    T k_fiber_0, k_fiber_1, k_fiber_2;
    T k_shear;
    T k_volume;

    T k_fiber_isotropic; // only used for getting effective isotropic stress (for damage evolution)
    T g; // degradation multiplier

    bool isotropic = false;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    QRAnisotropic(const T E = (T)1, const T nu = (T)0.3, const bool isotropic = false)
        : k_fiber_0(0), k_fiber_1(0), k_fiber_2(0), k_shear(0), k_volume(0), g(1), isotropic(isotropic)
    {
        setParameters(E, nu);
    }

    void setParameters(const T E, const T nu)
    {
        T lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
        T mu = E / ((T)2 * ((T)1 + nu));
        k_fiber_0 = k_fiber_1 = k_fiber_2 = k_fiber_isotropic = mu;
        k_shear = mu;
        k_volume = lambda;
    }

    T getSigmaCrit(T percentage) const
    {
        // Create a dummy model that is isotropic and non-degraded
        QRAnisotropic<T, dim> model_dummy = *this;
        model_dummy.g = 1;
        model_dummy.k_fiber_0 = model_dummy.k_fiber_isotropic;
        model_dummy.k_fiber_1 = model_dummy.k_fiber_isotropic;
        model_dummy.k_fiber_2 = model_dummy.k_fiber_isotropic;

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
        k_fiber_0 = k_fiber_1 = k_fiber_2 = k_fiber_isotropic; // reset all fiber stiffness to isotropic history
        Scratch scratch_dummy;
        updateScratch(F, scratch_dummy);
        TM P;
        firstPiola(scratch_dummy, P);
        cauchystress = P * F.transpose() / scratch_dummy.J;
    }

    void scaleFiberStiffness(const int component, const T scale)
    {
        if (isotropic) {
            return;
        }
        if (component == 0) k_fiber_0 *= scale;
        if (component == 1) k_fiber_1 *= scale;
        if (component == 2) k_fiber_2 *= scale;
    }

    void updateScratch(const TM& new_F, Scratch& s)
    {
        s.F = new_F;
        GivensQR(new_F, s.Q, s.R);
        s.R_inv = s.R.inverse();
        s.J = 1;
        for (int i = 0; i < dim; i++)
            s.J *= s.R(i, i);

        /*
         Psi = 0.5  \sum_i k_fiber_i (rii-1)^2  + 0.5 k_shear (r_12^2 + r13^2 + r23^2) + 0.5 k_volume (J-1)^2
        */
        using MATH_TOOLS::sqr;
        T k_fiber[3] = { k_fiber_0, k_fiber_1, k_fiber_2 };

        s.energy_density = 0;
        // volume term contribution to energy
        T degraded_volume_term = 0;
        if (s.J >= 1)
            degraded_volume_term = g * (T)0.5 * k_volume * sqr(s.J - 1);
        else
            degraded_volume_term = (T)0.5 * k_volume * sqr(s.J - 1);
        s.energy_density += degraded_volume_term;
        // shearing term contribution to energy
        for (int i = 0; i < dim; i++)
            for (int j = i + 1; j < dim; j++)
                s.energy_density += g * (T)0.5 * k_shear * sqr(s.R(i, j));
        // fiber term contribution to energy
        if (!isotropic) {
            s.energy_density += (T)0.5 * k_fiber[0] * sqr(s.R(0, 0) - 1); // fiber gets no degradation
            for (int i = 1; i < dim; i++) // other directions gets degradation
                s.energy_density += g * (T)0.5 * k_fiber[i] * sqr(s.R(i, i) - 1);
        }
        else {
            for (int i = 0; i < dim; i++) // degrade ALL directions if isotropic
                s.energy_density += g * (T)0.5 * k_fiber[i] * sqr(s.R(i, i) - 1);
        }

        TM& derivative = s.qr_helper.dPsiHatdR;
        derivative = TM::Zero();
        // volume term contribution to the derivative
        if (s.J >= 1)
            for (int i = 0; i < dim; i++)
                derivative(i, i) += g * k_volume * (s.J - 1) * s.J / s.R(i, i);
        else
            for (int i = 0; i < dim; i++)
                derivative(i, i) += k_volume * (s.J - 1) * s.J / s.R(i, i);
        // shearing term contribution to the derivative
        for (int i = 0; i < dim; i++)
            for (int j = i + 1; j < dim; j++)
                derivative(i, j) += g * k_shear * s.R(i, j);
        // fiber term contribution to the derivative
        if (!isotropic) {
            derivative(0, 0) += k_fiber[0] * (s.R(0, 0) - 1);
            for (int i = 1; i < dim; i++)
                derivative(i, i) += g * k_fiber[i] * (s.R(i, i) - 1);
        }
        else {
            for (int i = 0; i < dim; i++)
                derivative(i, i) += g * k_fiber[i] * (s.R(i, i) - 1);
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
        ZIRAN_ERR("Implicit QRAnisotropic not implemented yet.");
    }

    void firstPiolaDerivative(const Scratch& s, Hessian& dPdF) const
    {
        ZIRAN_ERR("Implicit QRAnisotropic not implemented yet.");
    }

    bool isC2(const Scratch& s, T tolerance) const
    {
        return std::abs(s.J) > tolerance;
    }

    bool hessianImplemented() const
    {
        return false; // derivative test skips dPdF test if this is false
    }

    static const char* name()
    {
        return "QRAnisotropic";
    }

    static const char* scratch_name()
    {
        return Scratch::name();
    }
};

template <class T, int dim>
struct HyperelasticTraits<QRAnisotropic<T, dim>> {
    using ScratchType = QRAnisotropicScratch<T, dim>;
};

template <class T, int dim>
struct RW<QRAnisotropicScratch<T, dim>> {
    using Tag = NoWriteTag<QRAnisotropicScratch<T, dim>>;
};
} // namespace ZIRAN
#endif
