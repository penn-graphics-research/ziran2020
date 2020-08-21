#ifndef QR_BASED_HELPER_H
#define QR_BASED_HELPER_H

namespace ZIRAN {
/**
   This is a helper class for QR based consitutive models written 
   in terms of entries of R

   F = QR
   Psi(F) = PsiHat(Sigma)
*/

template <class T, int dim>
class QrBasedHelper {
public:
    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;

    TM dPsiHatdR;

    QrBasedHelper()
        : dPsiHatdR(TM::Zero())
    {
    }

    ~QrBasedHelper() {}

    // P = QAR^{-T}
    // A is symmetric, and has uppertriangular part same as dPsiHatdR R^T
    void evaluateP(const TM& Q, const TM& R, const TM& R_inv, TM& P) const
    {
        TM A = dPsiHatdR * (R.transpose());

        // Modify the entries below diagonal
        for (int i = 1; i < dim; i++)
            for (int j = 0; j < i; j++)
                A(i, j) = A(j, i);

        P = Q * A * (R_inv.transpose());
    }
};

} // namespace ZIRAN
#endif
