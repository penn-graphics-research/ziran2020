#ifndef COROTATED_ISOTROPIC_H
#define COROTATED_ISOTROPIC_H
#include <Ziran/Physics/ConstitutiveModel/SvdBasedIsotropicHelper.h>
#include <Ziran/CS/Util/Forward.h>
#include <Ziran/CS/Util/BinaryIO.h>
#include <Ziran/Math/MathTools.h>
#include <Ziran/Math/Linear/DenseExt.h>
#include <Ziran/Math/Linear/ImplicitQRSVD.h>
#include <tick/requires.h>

namespace ZIRAN {

template <class Derived>
class HyperelasticConstitutiveModel;

template <typename Derived>
struct ScratchTrait;

template <class T, int _dim>
class CorotatedIsotropic;

// scratch (non-state) variables for the consitutive model
template <class T, int dim>
struct CorotatedIsotropicScratch {
    using TM = Matrix<T, dim, dim>;
    using TV = Vector<T, dim>;
    T J;
    TM F, U, V, R, JFinvT;
    TV sigma;
    SvdBasedIsotropicHelper<T, dim> isotropic;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CorotatedIsotropicScratch()
        : J(1), isotropic(0)
    {
    }

    static const char* name()
    {
        return "CorotatedIsotropicScratch";
    }
};

template <class T, int _dim>
class CorotatedIsotropic : public HyperelasticConstitutiveModel<CorotatedIsotropic<T, _dim>> {
public:
    static const int dim = _dim;
    static constexpr T eps = (T)1e-6;
    using Base = HyperelasticConstitutiveModel<CorotatedIsotropic<T, dim>>;
    using TM = typename Base::TM;
    using TV = typename Base::TV;
    using Strain = TM;
    using Hessian = typename Base::Hessian;
    using Scalar = typename Base::Scalar;
    using Scratch = typename HyperelasticTraits<CorotatedIsotropic<T, dim>>::ScratchType;
    using Base::firstPiolaDerivative; // TODO: a more efficient version
    using Vec = Vector<T, Eigen::Dynamic>;
    using VecBlock = Eigen::VectorBlock<Vec>;

    bool project{ true };
    T mu, lambda;

    CorotatedIsotropic(const T E = (T)1, const T nu = (T)0.3)
    {
        setLameParameters(E, nu);
    }

    void setLameParameters(const T E, const T nu)
    {
        lambda = E * nu / (((T)1 + nu) * ((T)1 - (T)2 * nu));
        mu = E / ((T)2 * ((T)1 + nu));
    }

    void updateScratchSVD(const TM& new_F, Scratch& s) const // private
    {
        s.F = new_F;
        singularValueDecomposition(s.F, s.U, s.sigma, s.V);
        s.R = s.U * s.V.transpose();
    }

    TICK_MEMBER_REQUIRES(dim == 2)
    void updateScratch(const TM& new_F, Scratch& s) const
    {
        using namespace MATH_TOOLS;
        updateScratchSVD(new_F, s);
        EIGEN_EXT::cofactorMatrix(s.F, s.JFinvT);
        s.J = s.sigma(0) * s.sigma(1);
        T _2mu = mu * 2;
        T _lambda = lambda * (s.J - 1);
        T Sprod[2] = { s.sigma(1), s.sigma(0) };
        s.isotropic.psi0 = _2mu * (s.sigma(0) - 1) + _lambda * Sprod[0];
        s.isotropic.psi1 = _2mu * (s.sigma(1) - 1) + _lambda * Sprod[1];
        s.isotropic.psi00 = _2mu + lambda * Sprod[0] * Sprod[0];
        s.isotropic.psi11 = _2mu + lambda * Sprod[1] * Sprod[1];
        s.isotropic.psi01 = _lambda + lambda * Sprod[0] * Sprod[1];

        // (psi0-psi1)/(sigma0-sigma1)
        s.isotropic.m01 = _2mu - _lambda;

        // (psi0+psi1)/(sigma0+sigma1)
        s.isotropic.p01 = (s.isotropic.psi0 + s.isotropic.psi1) / clamp_small_magnitude(s.sigma(0) + s.sigma(1), eps);

        if (project) {
            s.isotropic.buildMatrixBlock();
            s.isotropic.projectABBlock();
        }
    }

    TICK_MEMBER_REQUIRES(dim == 3)
    void updateScratch(const TM& new_F, Scratch& s) const
    {
        using namespace MATH_TOOLS;
        updateScratchSVD(new_F, s);
        EIGEN_EXT::cofactorMatrix(s.F, s.JFinvT);
        s.J = s.sigma(0) * s.sigma(1) * s.sigma(2);
        T _2mu = mu * 2;
        T _lambda = lambda * (s.J - 1);
        T Sprod[3] = { s.sigma(1) * s.sigma(2), s.sigma(0) * s.sigma(2), s.sigma(0) * s.sigma(1) };
        s.isotropic.psi0 = _2mu * (s.sigma(0) - 1) + _lambda * Sprod[0];
        s.isotropic.psi1 = _2mu * (s.sigma(1) - 1) + _lambda * Sprod[1];
        s.isotropic.psi2 = _2mu * (s.sigma(2) - 1) + _lambda * Sprod[2];
        s.isotropic.psi00 = _2mu + lambda * Sprod[0] * Sprod[0];
        s.isotropic.psi11 = _2mu + lambda * Sprod[1] * Sprod[1];
        s.isotropic.psi22 = _2mu + lambda * Sprod[2] * Sprod[2];
        s.isotropic.psi01 = _lambda * s.sigma(2) + lambda * Sprod[0] * Sprod[1];
        s.isotropic.psi02 = _lambda * s.sigma(1) + lambda * Sprod[0] * Sprod[2];
        s.isotropic.psi12 = _lambda * s.sigma(0) + lambda * Sprod[1] * Sprod[2];

        // (psiA-psiB)/(sigmaA-sigmaB)
        s.isotropic.m01 = _2mu - _lambda * s.sigma(2); // i = 0
        s.isotropic.m02 = _2mu - _lambda * s.sigma(1); // i = 2
        s.isotropic.m12 = _2mu - _lambda * s.sigma(0); // i = 1

        // (psiA+psiB)/(sigmaA+sigmaB)
        s.isotropic.p01 = (s.isotropic.psi0 + s.isotropic.psi1) / clamp_small_magnitude(s.sigma(0) + s.sigma(1), eps);
        s.isotropic.p02 = (s.isotropic.psi0 + s.isotropic.psi2) / clamp_small_magnitude(s.sigma(0) + s.sigma(2), eps);
        s.isotropic.p12 = (s.isotropic.psi1 + s.isotropic.psi2) / clamp_small_magnitude(s.sigma(1) + s.sigma(2), eps);

        if (project) {
            s.isotropic.buildMatrixBlock();
            s.isotropic.projectABBlock();
        }
    }

    static constexpr bool diagonalDifferentiable()
    {
        return true;
    }

    T psi(const Scratch& s) const
    {
        T Jm1 = s.J - 1;
        return mu * ZIRAN::EIGEN_EXT::squaredNorm(s.F - s.R) + (T).5 * lambda * Jm1 * Jm1;
    }

    void firstPiola(const Scratch& s, TM& P) const
    {
        P.noalias() = (T)2 * mu * (s.F - s.R) + lambda * (s.J - 1) * s.JFinvT;
    }

    void firstPiolaDifferential(const Scratch& s, const TM& dF, TM& dP) const
    {
        TM D = s.U.transpose() * dF * s.V;
        TM K;
        if (project)
            s.isotropic.dPdFOfSigmaContractProjected(D, K);
        else
            s.isotropic.dPdFOfSigmaContract(D, K);
        dP = s.U * K * s.V.transpose();
    }

#if 1
    void firstPiolaDerivative(const Scratch& ss, Hessian& dPdF) const
    {
        if constexpr (dim == 2) {
            for (int ij = 0; ij < 4; ++ij) {
                int j = ij / 2;
                int i = ij & 1;
                for (int rs = 0; rs <= ij; ++rs) {
                    int s = rs / 2;
                    int r = rs & 1;
                    if (project) {
#if 0
                        if (j != s) {
                            dPdF(ij, rs) = dPdF(rs, ij) = 0;
                            continue;
                        }
#endif
                        dPdF(ij, rs) = dPdF(rs, ij) = ss.isotropic.Aij(0, 0) * ss.U(i, 0) * ss.V(j, 0) * ss.U(r, 0) * ss.V(s, 0) + ss.isotropic.Aij(0, 1) * ss.U(i, 0) * ss.V(j, 0) * ss.U(r, 1) * ss.V(s, 1) + ss.isotropic.B01(0, 0) * ss.U(i, 0) * ss.V(j, 1) * ss.U(r, 0) * ss.V(s, 1) + ss.isotropic.B01(0, 1) * ss.U(i, 0) * ss.V(j, 1) * ss.U(r, 1) * ss.V(s, 0) + ss.isotropic.B01(1, 0) * ss.U(i, 1) * ss.V(j, 0) * ss.U(r, 0) * ss.V(s, 1) + ss.isotropic.B01(1, 1) * ss.U(i, 1) * ss.V(j, 0) * ss.U(r, 1) * ss.V(s, 0) + ss.isotropic.Aij(1, 0) * ss.U(i, 1) * ss.V(j, 1) * ss.U(r, 0) * ss.V(s, 0) + ss.isotropic.Aij(1, 1) * ss.U(i, 1) * ss.V(j, 1) * ss.U(r, 1) * ss.V(s, 1);
                    }
                    else
                        dPdF(ij, rs) = dPdF(rs, ij) = ss.isotropic.psi00 * ss.U(i, 0) * ss.V(j, 0) * ss.U(r, 0) * ss.V(s, 0) + ss.isotropic.psi01 * ss.U(i, 0) * ss.V(j, 0) * ss.U(r, 1) * ss.V(s, 1) + ss.isotropic.b01(0, 0) * ss.U(i, 0) * ss.V(j, 1) * ss.U(r, 0) * ss.V(s, 1) + ss.isotropic.b01(0, 1) * ss.U(i, 0) * ss.V(j, 1) * ss.U(r, 1) * ss.V(s, 0) + ss.isotropic.b01(1, 0) * ss.U(i, 1) * ss.V(j, 0) * ss.U(r, 0) * ss.V(s, 1) + ss.isotropic.b01(1, 1) * ss.U(i, 1) * ss.V(j, 0) * ss.U(r, 1) * ss.V(s, 0) + ss.isotropic.psi01 * ss.U(i, 1) * ss.V(j, 1) * ss.U(r, 0) * ss.V(s, 0) + ss.isotropic.psi11 * ss.U(i, 1) * ss.V(j, 1) * ss.U(r, 1) * ss.V(s, 1);
                }
            }
        }
        else {
            for (int ij = 0; ij < 9; ++ij) {
                int j = ij / 3;
                int i = ij - j * 3;
                for (int rs = 0; rs <= ij; ++rs) {
                    int s = rs / 3;
                    int r = rs - s * 3;
                    if (project) {
#if 0
                        if (j != s) {
                            dPdF(ij, rs) = dPdF(rs, ij) = 0;
                            continue;
                        }
#endif
                        dPdF(ij, rs) = dPdF(rs, ij) = ss.isotropic.Aij(0, 0) * ss.U(i, 0) * ss.V(j, 0) * ss.U(r, 0) * ss.V(s, 0) + ss.isotropic.Aij(0, 1) * ss.U(i, 0) * ss.V(j, 0) * ss.U(r, 1) * ss.V(s, 1) + ss.isotropic.Aij(0, 2) * ss.U(i, 0) * ss.V(j, 0) * ss.U(r, 2) * ss.V(s, 2) + ss.isotropic.Aij(0, 1) * ss.U(i, 1) * ss.V(j, 1) * ss.U(r, 0) * ss.V(s, 0) + ss.isotropic.Aij(1, 1) * ss.U(i, 1) * ss.V(j, 1) * ss.U(r, 1) * ss.V(s, 1) + ss.isotropic.Aij(1, 2) * ss.U(i, 1) * ss.V(j, 1) * ss.U(r, 2) * ss.V(s, 2) + ss.isotropic.Aij(0, 2) * ss.U(i, 2) * ss.V(j, 2) * ss.U(r, 0) * ss.V(s, 0) + ss.isotropic.Aij(1, 2) * ss.U(i, 2) * ss.V(j, 2) * ss.U(r, 1) * ss.V(s, 1) + ss.isotropic.Aij(2, 2) * ss.U(i, 2) * ss.V(j, 2) * ss.U(r, 2) * ss.V(s, 2) +

                            ss.isotropic.B01(0, 0) * ss.U(i, 0) * ss.V(j, 1) * ss.U(r, 0) * ss.V(s, 1) + ss.isotropic.B01(0, 1) * ss.U(i, 0) * ss.V(j, 1) * ss.U(r, 1) * ss.V(s, 0) + ss.isotropic.B01(1, 0) * ss.U(i, 1) * ss.V(j, 0) * ss.U(r, 0) * ss.V(s, 1) + ss.isotropic.B01(1, 1) * ss.U(i, 1) * ss.V(j, 0) * ss.U(r, 1) * ss.V(s, 0) +

                            ss.isotropic.B12(0, 0) * ss.U(i, 1) * ss.V(j, 2) * ss.U(r, 1) * ss.V(s, 2) + ss.isotropic.B12(0, 1) * ss.U(i, 1) * ss.V(j, 2) * ss.U(r, 2) * ss.V(s, 1) + ss.isotropic.B12(1, 0) * ss.U(i, 2) * ss.V(j, 1) * ss.U(r, 1) * ss.V(s, 2) + ss.isotropic.B12(1, 1) * ss.U(i, 2) * ss.V(j, 1) * ss.U(r, 2) * ss.V(s, 1) +

                            ss.isotropic.B20(1, 1) * ss.U(i, 0) * ss.V(j, 2) * ss.U(r, 0) * ss.V(s, 2) + ss.isotropic.B20(1, 0) * ss.U(i, 0) * ss.V(j, 2) * ss.U(r, 2) * ss.V(s, 0) + ss.isotropic.B20(0, 1) * ss.U(i, 2) * ss.V(j, 0) * ss.U(r, 0) * ss.V(s, 2) + ss.isotropic.B20(0, 0) * ss.U(i, 2) * ss.V(j, 0) * ss.U(r, 2) * ss.V(s, 0);
                    }
                    else
                        dPdF(ij, rs) = dPdF(rs, ij) = ss.isotropic.psi00 * ss.U(i, 0) * ss.V(j, 0) * ss.U(r, 0) * ss.V(s, 0) + ss.isotropic.psi01 * ss.U(i, 0) * ss.V(j, 0) * ss.U(r, 1) * ss.V(s, 1) + ss.isotropic.psi02 * ss.U(i, 0) * ss.V(j, 0) * ss.U(r, 2) * ss.V(s, 2) + ss.isotropic.psi01 * ss.U(i, 1) * ss.V(j, 1) * ss.U(r, 0) * ss.V(s, 0) + ss.isotropic.psi11 * ss.U(i, 1) * ss.V(j, 1) * ss.U(r, 1) * ss.V(s, 1) + ss.isotropic.psi12 * ss.U(i, 1) * ss.V(j, 1) * ss.U(r, 2) * ss.V(s, 2) + ss.isotropic.psi02 * ss.U(i, 2) * ss.V(j, 2) * ss.U(r, 0) * ss.V(s, 0) + ss.isotropic.psi12 * ss.U(i, 2) * ss.V(j, 2) * ss.U(r, 1) * ss.V(s, 1) + ss.isotropic.psi22 * ss.U(i, 2) * ss.V(j, 2) * ss.U(r, 2) * ss.V(s, 2) +

                            ss.isotropic.b01(0, 0) * ss.U(i, 0) * ss.V(j, 1) * ss.U(r, 0) * ss.V(s, 1) + ss.isotropic.b01(0, 1) * ss.U(i, 0) * ss.V(j, 1) * ss.U(r, 1) * ss.V(s, 0) + ss.isotropic.b01(1, 0) * ss.U(i, 1) * ss.V(j, 0) * ss.U(r, 0) * ss.V(s, 1) + ss.isotropic.b01(1, 1) * ss.U(i, 1) * ss.V(j, 0) * ss.U(r, 1) * ss.V(s, 0) +

                            ss.isotropic.b12(0, 0) * ss.U(i, 1) * ss.V(j, 2) * ss.U(r, 1) * ss.V(s, 2) + ss.isotropic.b12(0, 1) * ss.U(i, 1) * ss.V(j, 2) * ss.U(r, 2) * ss.V(s, 1) + ss.isotropic.b12(1, 0) * ss.U(i, 2) * ss.V(j, 1) * ss.U(r, 1) * ss.V(s, 2) + ss.isotropic.b12(1, 1) * ss.U(i, 2) * ss.V(j, 1) * ss.U(r, 2) * ss.V(s, 1) +

                            ss.isotropic.b20(1, 1) * ss.U(i, 0) * ss.V(j, 2) * ss.U(r, 0) * ss.V(s, 2) + ss.isotropic.b20(1, 0) * ss.U(i, 0) * ss.V(j, 2) * ss.U(r, 2) * ss.V(s, 0) + ss.isotropic.b20(0, 1) * ss.U(i, 2) * ss.V(j, 0) * ss.U(r, 0) * ss.V(s, 2) + ss.isotropic.b20(0, 0) * ss.U(i, 2) * ss.V(j, 0) * ss.U(r, 2) * ss.V(s, 0);
                }
            }
        }
    }
#endif

    Matrix<T, 2, 2> Bij(const TV& sigma, int i, int j, T clamp_value) const
    {
        auto mingClampMagnitude = [&](const T input) {
            T magnitude = input > 0 ? input : -input;
            T sign = input > 0 ? 1.f : -1.f;
            T output = magnitude > clamp_value ? magnitude : clamp_value;
            return output * sign;
        };
        if constexpr (dim == 2) {
            TV dE = firstPiolaDiagonal(sigma);
            T B_Pij = 0.5 * (dE[i] + dE[j]) / mingClampMagnitude(sigma[i] + sigma[j]);
            T B_Mij = mu - lambda * (sigma(0) * sigma(1) - 1) * 0.5;

            Matrix<T, 2, 2> B_P_Const;
            B_P_Const << 1, 1, 1, 1;
            Matrix<T, 2, 2> B_M_Const;
            B_M_Const << 1, -1, -1, 1;
            return B_M_Const * B_Pij + B_P_Const * B_Mij;
        }
        else {
            TV dE = firstPiolaDiagonal(sigma);
            T B_Pij = 0.5 * (dE[i] + dE[j]) / mingClampMagnitude(sigma[i] + sigma[j]);

            T J = sigma(0) * sigma(1) * sigma(2);
            T B_Mij = mu - lambda * (J - 1) * 0.5 * J / sigma(i) / sigma(j);

            Matrix<T, 2, 2> B_P_Const;
            B_P_Const << 1, 1, 1, 1;
            Matrix<T, 2, 2> B_M_Const;
            B_M_Const << 1, -1, -1, 1;
            return B_M_Const * B_Pij + B_P_Const * B_Mij;
        }
    }

    TV firstPiolaDiagonal(const TV& sigma) const
    {
        TV dE;
        if constexpr (dim == 1) {
            // TODO: implement, and modify hessianImplemented() function.
            ZIRAN_ASSERT(false, "not implemented");
        }
        else if constexpr (dim == 2) {
            T _lambda = lambda * (sigma(0) * sigma(1) - 1);
            T Sprod[2] = { sigma(1), sigma(0) };
            dE[0] = 2 * mu * (sigma(0) - 1) + _lambda * Sprod[0];
            dE[1] = 2 * mu * (sigma(1) - 1) + _lambda * Sprod[1];
        }
        else {
            T _lambda = lambda * (sigma(0) * sigma(1) * sigma(2) - 1);
            T Sprod[3] = { sigma(1) * sigma(2), sigma(0) * sigma(2), sigma(0) * sigma(1) };
            dE[0] = 2 * mu * (sigma(0) - 1) + _lambda * Sprod[0];
            dE[1] = 2 * mu * (sigma(1) - 1) + _lambda * Sprod[1];
            dE[2] = 2 * mu * (sigma(2) - 1) + _lambda * Sprod[2];
        }
        return dE;
    }

    TM firstPiolaDerivativeDiagonal(const TV& sigma) const
    {
        TM ddE;
        if constexpr (dim == 1) {
            // TODO: implement, and modify hessianImplemented() function.
            ZIRAN_ASSERT(false, "not implemented");
        }
        else if constexpr (dim == 2) {
            T _lambda = lambda * (sigma(0) * sigma(1) - 1);
            T Sprod[2] = { sigma(1), sigma(0) };
            ddE(0, 0) = 2 * mu + lambda * Sprod[0] * Sprod[0];
            ddE(1, 1) = 2 * mu + lambda * Sprod[1] * Sprod[1];
            ddE(0, 1) = ddE(1, 0) = _lambda + lambda * Sprod[0] * Sprod[1];
        }
        else {
            T _lambda = lambda * (sigma(0) * sigma(1) * sigma(2) - 1);
            T Sprod[3] = { sigma(1) * sigma(2), sigma(0) * sigma(2), sigma(0) * sigma(1) };
            ddE(0, 0) = 2 * mu + lambda * Sprod[0] * Sprod[0];
            ddE(1, 1) = 2 * mu + lambda * Sprod[1] * Sprod[1];
            ddE(2, 2) = 2 * mu + lambda * Sprod[2] * Sprod[2];
            ddE(0, 1) = ddE(1, 0) = _lambda * sigma(2) + lambda * Sprod[0] * Sprod[1];
            ddE(0, 2) = ddE(2, 0) = _lambda * sigma(1) + lambda * Sprod[0] * Sprod[2];
            ddE(1, 2) = ddE(2, 1) = _lambda * sigma(0) + lambda * Sprod[1] * Sprod[2];
        }
        return ddE;
    }

    bool isC2(const Scratch& s, T tolerance) const
    {
        return s.sigma.prod() > tolerance; // due to the log sigma term
    }

    /**
       Returns whether dP (or dPdF) is implemented
    */
    bool hessianImplemented() const
    {
        return true;
    }

    void write(std::ostream& out) const
    {
        writeEntry(out, mu);
        writeEntry(out, lambda);
    }

    static CorotatedIsotropic<T, dim> read(std::istream& in)
    {
        CorotatedIsotropic<T, dim> model;
        model.mu = readEntry<T>(in);
        model.lambda = readEntry<T>(in);
        return model;
    }

    static const char* name()
    {
        return "CorotatedIsotropic";
    }

    static const char* scratch_name()
    {
        return Scratch::name();
    }
};

template <class T, int dim>
struct HyperelasticTraits<CorotatedIsotropic<T, dim>> {
    using ScratchType = CorotatedIsotropicScratch<T, dim>;
};

template <class T, int dim>
struct RW<CorotatedIsotropicScratch<T, dim>> {
    using Tag = NoWriteTag<CorotatedIsotropicScratch<T, dim>>;
};
} // namespace ZIRAN

#endif