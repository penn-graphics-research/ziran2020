#ifndef SVD_BASED_ISOTROPIC_HELPER_H
#define SVD_BASED_ISOTROPIC_HELPER_H

#include <Ziran/Math/Linear/DenseExt.h>

namespace ZIRAN {
/**
   This is a helper class for F based consitutive models written 
   in terms of singular values.

   F = U Sigma V'
   Psi(F) = PsiHat(Sigma)
   P = U d_PsiHat_d_Sigma V'
   dP = U ( dPdF_of_Sigma : (U' dF V) ) V'
 */

template <class, int dim, class enable = void>
class SvdBasedIsotropicHelper;

template <class T, int dim>
class SvdBasedIsotropicHelper<T, dim, std::enable_if_t<dim == 1>> {
public:
    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;

    // These should be updated everytime a new SVD is performed to F
    T psi0; // d_PsiHat_d_sigma0
    T psi00; // d^2_PsiHat_d_sigma0_d_sigma0

    SvdBasedIsotropicHelper() {}

    SvdBasedIsotropicHelper(T a)
        : psi0(a)
        , psi00(a)
    {
    }

    ~SvdBasedIsotropicHelper() {}

    void computePHat(TV& P_hat) const
    {
        P_hat(0) = psi0;
    }

    // B = dPdF(Sigma) : A
    void dPdFOfSigmaContract(const TM& A, TM& B) const
    {
        B(0, 0) = psi00 * A(0, 0);
    }
};

template <class T, int dim>
class SvdBasedIsotropicHelper<T, dim, std::enable_if_t<dim == 2>> {
public:
    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;
    using TM2 = Matrix<T, 2, 2>;

    // These should be updated everytime a new SVD is performed to F
    T psi0; // d_PsiHat_d_sigma0
    T psi1; // d_PsiHat_d_sigma1
    T psi00; // d^2_PsiHat_d_sigma0_d_sigma0
    T psi01; // d^2_PsiHat_d_sigma0_d_sigma1
    T psi11; // d^2_PsiHat_d_sigma1_d_sigma1
    T m01; // (psi0-psi1)/(sigma0-sigma1), usually can be computed robustly
    T p01; // (psi0+psi1)/(sigma0+sigma1), need to clamp bottom with 1e-6
    TM Aij;
    TM2 B01;

    SvdBasedIsotropicHelper() {}

    SvdBasedIsotropicHelper(T a)
        : psi0(a)
        , psi1(a)
        , psi00(a)
        , psi01(a)
        , psi11(a)
        , m01(a)
        , p01(a)
    {
    }

    ~SvdBasedIsotropicHelper() {}

    auto a(int i, int j) const
    {
        if (i != j)
            return psi01;
        if (i == 0)
            return psi00;
        else
            return psi11;
    }
    auto b01(int i, int j) const
    {
        return (i == j ? m01 + p01 : m01 - p01) * 0.5;
    }

    void buildMatrixBlock()
    {
        Aij(0, 0) = psi00;
        Aij(0, 1) = Aij(1, 0) = psi01;
        Aij(1, 1) = psi11;
        B01(0, 0) = B01(1, 1) = (m01 + p01) * 0.5;
        B01(0, 1) = B01(1, 0) = (m01 - p01) * 0.5;
    }

    void projectABBlock()
    {
        // proj A
        EIGEN_EXT::makePD(Aij);
        // proj B
        EIGEN_EXT::makePD2d(B01);
    }

    void computePHat(TV& P_hat) const
    {
        P_hat(0) = psi0;
        P_hat(1) = psi1;
    }

    // B = dPdF(Sigma) : A
    void dPdFOfSigmaContract(const TM& A, TM& B) const
    {
        B(0, 0) = psi00 * A(0, 0) + psi01 * A(1, 1);
        B(1, 1) = psi01 * A(0, 0) + psi11 * A(1, 1);
        B(0, 1) = ((m01 + p01) * A(0, 1) + (m01 - p01) * A(1, 0)) / 2;
        B(1, 0) = ((m01 - p01) * A(0, 1) + (m01 + p01) * A(1, 0)) / 2;
    }

    // B = dPdF(Sigma) : A
    void dPdFOfSigmaContractProjected(const TM& A, TM& B) const
    {
        B(0, 0) = Aij(0, 0) * A(0, 0) + Aij(0, 1) * A(1, 1);
        B(1, 1) = Aij(1, 0) * A(0, 0) + Aij(1, 1) * A(1, 1);
        B(0, 1) = B01(0, 0) * A(0, 1) + B01(0, 1) * A(1, 0);
        B(1, 0) = B01(1, 0) * A(0, 1) + B01(1, 1) * A(1, 0);
    }
};

template <class T, int dim>
class SvdBasedIsotropicHelper<T, dim, std::enable_if_t<dim == 3>> {
public:
    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;
    using TM2 = Matrix<T, 2, 2>;

    // These should be updated everytime a new SVD is performed to F
    T psi0; // d_PsiHat_d_sigma0
    T psi1; // d_PsiHat_d_sigma1
    T psi2; // d_PsiHat_d_sigma2
    T psi00; // d^2_PsiHat_d_sigma0_d_sigma0
    T psi11; // d^2_PsiHat_d_sigma1_d_sigma1
    T psi22; // d^2_PsiHat_d_sigma2_d_sigma2
    T psi01; // d^2_PsiHat_d_sigma0_d_sigma1
    T psi02; // d^2_PsiHat_d_sigma0_d_sigma2
    T psi12; // d^2_PsiHat_d_sigma1_d_sigma2

    T m01; // (psi0-psi1)/(sigma0-sigma1), usually can be computed robustly
    T p01; // (psi0+psi1)/(sigma0+sigma1), need to clamp bottom with 1e-6
    T m02; // (psi0-psi2)/(sigma0-sigma2), usually can be computed robustly
    T p02; // (psi0+psi2)/(sigma0+sigma2), need to clamp bottom with 1e-6
    T m12; // (psi1-psi2)/(sigma1-sigma2), usually can be computed robustly
    T p12; // (psi1+psi2)/(sigma1+sigma2), need to clamp bottom with 1e-6
    TM Aij;
    TM2 B01, B12, B20;

    SvdBasedIsotropicHelper() {}

    SvdBasedIsotropicHelper(T a)
        : psi0(a)
        , psi1(a)
        , psi2(a)
        , psi00(a)
        , psi11(a)
        , psi22(a)
        , psi01(a)
        , psi02(a)
        , psi12(a)
        , m01(a)
        , p01(a)
        , m02(a)
        , p02(a)
        , m12(a)
        , p12(a)
    {
    }

    ~SvdBasedIsotropicHelper() {}

    auto a(int i, int j) const
    {
        if (i == j) {
            if (i == 0)
                return psi00;
            else if (i == 1)
                return psi11;
            else
                return psi22;
        }
        if (i > j)
            std::swap(i, j);
        if (i == 1)
            return psi12;
        else if (j == 1)
            return psi01;
        else
            return psi02;
    }
    auto b01(int i, int j) const
    {
        return (i == j ? m01 + p01 : m01 - p01) * 0.5;
    }
    auto b12(int i, int j) const
    {
        return (i == j ? m12 + p12 : m12 - p12) * 0.5;
    }
    auto b20(int i, int j) const
    {
        return (i == j ? m02 + p02 : m02 - p02) * 0.5;
    }

    void buildMatrixBlock()
    {
        Aij(0, 0) = psi00;
        Aij(1, 1) = psi11;
        Aij(2, 2) = psi22;
        Aij(0, 1) = Aij(1, 0) = psi01;
        Aij(0, 2) = Aij(2, 0) = psi02;
        Aij(1, 2) = Aij(2, 1) = psi12;
        B01(0, 0) = B01(1, 1) = (m01 + p01) * 0.5;
        B01(0, 1) = B01(1, 0) = (m01 - p01) * 0.5;
        B12(0, 0) = B12(1, 1) = (m12 + p12) * 0.5;
        B12(0, 1) = B12(1, 0) = (m12 - p12) * 0.5;
        B20(0, 0) = B20(1, 1) = (m02 + p02) * 0.5;
        B20(0, 1) = B20(1, 0) = (m02 - p02) * 0.5;
    }

    void projectABBlock()
    {
        // proj A
        EIGEN_EXT::makePD(Aij);
        // proj B
        EIGEN_EXT::makePD2d(B01);
        EIGEN_EXT::makePD2d(B12);
        EIGEN_EXT::makePD2d(B20);
    }

    void computePHat(TV& P_hat) const
    {
        P_hat(0) = psi0;
        P_hat(1) = psi1;
        P_hat(2) = psi2;
    }

    // B = dPdF(Sigma) : A
    void dPdFOfSigmaContract(const TM& A, TM& B) const
    {
        B(0, 0) = psi00 * A(0, 0) + psi01 * A(1, 1) + psi02 * A(2, 2);
        B(1, 1) = psi01 * A(0, 0) + psi11 * A(1, 1) + psi12 * A(2, 2);
        B(2, 2) = psi02 * A(0, 0) + psi12 * A(1, 1) + psi22 * A(2, 2);
        B(0, 1) = ((m01 + p01) * A(0, 1) + (m01 - p01) * A(1, 0)) / 2;
        B(1, 0) = ((m01 - p01) * A(0, 1) + (m01 + p01) * A(1, 0)) / 2;
        B(0, 2) = ((m02 + p02) * A(0, 2) + (m02 - p02) * A(2, 0)) / 2;
        B(2, 0) = ((m02 - p02) * A(0, 2) + (m02 + p02) * A(2, 0)) / 2;
        B(1, 2) = ((m12 + p12) * A(1, 2) + (m12 - p12) * A(2, 1)) / 2;
        B(2, 1) = ((m12 - p12) * A(1, 2) + (m12 + p12) * A(2, 1)) / 2;
    }

    // B = dPdF(Sigma) : A
    void dPdFOfSigmaContractProjected(const TM& A, TM& B) const
    {
        B(0, 0) = Aij(0, 0) * A(0, 0) + Aij(0, 1) * A(1, 1) + Aij(0, 2) * A(2, 2);
        B(1, 1) = Aij(1, 0) * A(0, 0) + Aij(1, 1) * A(1, 1) + Aij(1, 2) * A(2, 2);
        B(2, 2) = Aij(2, 0) * A(0, 0) + Aij(2, 1) * A(1, 1) + Aij(2, 2) * A(2, 2);
        B(0, 1) = B01(0, 0) * A(0, 1) + B01(0, 1) * A(1, 0);
        B(1, 0) = B01(1, 0) * A(0, 1) + B01(1, 1) * A(1, 0);
        B(0, 2) = B20(0, 0) * A(0, 2) + B20(0, 1) * A(2, 0);
        B(2, 0) = B20(1, 0) * A(0, 2) + B20(1, 1) * A(2, 0);
        B(1, 2) = B12(0, 0) * A(1, 2) + B12(0, 1) * A(2, 1);
        B(2, 1) = B12(1, 0) * A(1, 2) + B12(1, 1) * A(2, 1);
    }
};
} // namespace ZIRAN
#endif