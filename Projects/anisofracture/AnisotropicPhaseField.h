#ifndef ANISO_PHASE_FIELD_H
#define ANISO_PHASE_FIELD_H

#include <Ziran/Math/MathTools.h>
//#include <Ziran/Physics/ConstitutiveModel/NeoHookeanBorden.h>
//#include <Ziran/CS/Util/BinaryIO.h>

namespace ZIRAN {

template <class T, int dim>
class AnisotropicPhaseField {
public:
    using TV = Vector<T, dim>;

    StdVector<TV> a_0; //structural directors
    StdVector<T> alphas; //weights for the structural directors between -1 and infty
    T residual_phase;
    T d; // 0 is undamaged, 1 is fully damaged
    T l0;
    T sigma_crit;
    T eta;
    T zeta;
    T laplacian;
    T vol;
    bool allow_damage;
    T maxDTilde;

    AnisotropicPhaseField() {}

    AnisotropicPhaseField(const StdVector<TV> a_0, const StdVector<T> alphas, const T residual_phase, const T d, const T l0, const T sigma_crit, const T eta, const T zeta, const T laplacian, const T vol, const bool allow_damage)
        : a_0(a_0), alphas(alphas), residual_phase(residual_phase), d(d), l0(l0), sigma_crit(sigma_crit), eta(eta), zeta(zeta), laplacian(laplacian), vol(vol), allow_damage(allow_damage), maxDTilde(0)
    {
    }

    // Maybe I don't need read/write because the data are trivial.
    void write(std::ostream& out) const
    {
        writeSTDVector(out, a_0);
        writeSTDVector(out, alphas);
        writeEntry(out, residual_phase);
        writeEntry(out, d);
        writeEntry(out, l0);
        writeEntry(out, sigma_crit);
        writeEntry(out, eta);
        writeEntry(out, zeta);
        writeEntry(out, laplacian);
        writeEntry(out, vol);
        writeEntry(out, allow_damage);
        writeEntry(out, maxDTilde);
    }
    static AnisotropicPhaseField<T, dim> read(std::istream& in)
    {
        AnisotropicPhaseField<T, dim> target;
        readSTDVector(in, target.a_0);
        readSTDVector(in, target.alphas);
        target.residual_phase = readEntry<T>(in);
        target.d = readEntry<T>(in);
        target.l0 = readEntry<T>(in);
        target.sigma_crit = readEntry<T>(in);
        target.eta = readEntry<T>(in);
        target.zeta = readEntry<T>(in);
        target.laplacian = readEntry<T>(in);
        target.vol = readEntry<T>(in);
        target.allow_damage = readEntry<bool>(in);
        target.maxDTilde = readEntry<T>(in);
        return target;
    }

    // template <class TCONST>
    // static T getSigmaCritFromPercentageStretch(const T percentage, const TCONST& model)
    // {
    //     typename TCONST::Scratch s;
    //     s.F = Matrix<T, dim, dim>::Identity() * ((T)1 + percentage);
    //     s.J = s.F.determinant();
    //     T G = model.psi(s);
    //     T sigmaCrit = G;
    //     ZIRAN_INFO("SigmaCrit Value: ", G);
    //     return sigmaCrit;
    // }

    /*
    void Update_Phase_Field_Fp(const T psi_pos)
    {
        if (psi_pos > H) {
            H = std::min(psi_pos, H_max);
            pf_Fp = 4 * l0 * (1 - residual_phase) * H * one_over_sigma_c + 1;
        }
    }*/

    static const char* name()
    {
        return "AnisotropicPhaseField";
    }
};

template <class T, int dim>
struct RW<AnisotropicPhaseField<T, dim>> {
    using Tag = CustomTypeTag<AnisotropicPhaseField<T, dim>>;
};

} // namespace ZIRAN

#endif