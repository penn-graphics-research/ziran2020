#include "PlasticityApplier.h"

namespace ZIRAN {

template <class T>
SnowPlasticity<T>::SnowPlasticity(T psi_in, T theta_c_in, T theta_s_in, T min_Jp_in, T max_Jp_in)
    : Jp(1)
    , psi(psi_in)
    , theta_c(theta_c_in)
    , theta_s(theta_s_in)
    , min_Jp(min_Jp_in)
    , max_Jp(max_Jp_in)
{
}

template <class T>
template <class TConst>
bool SnowPlasticity<T>::projectStrain(TConst& c, Matrix<T, TConst::dim, TConst::dim>& strain)
{
    static const int dim = TConst::dim;
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    TM U, V;
    TV sigma;

    // TODO: this is inefficient because next time step updateState will do the svd again!
    singularValueDecomposition(strain, U, sigma, V);

    T Fe_det = (T)1;
    for (int i = 0; i < dim; i++) {
        sigma(i) = std::max(std::min(sigma(i), (T)1 + theta_s), (T)1 - theta_c);
        Fe_det *= sigma(i);
    }

    Eigen::DiagonalMatrix<T, dim, dim> sigma_m(sigma);
    TM Fe = U * sigma_m * V.transpose();
    // T Jp_new = std::max(std::min(Jp * strain.determinant() / Fe_det, max_Jp), min_Jp);
    T Jp_new = Jp * strain.determinant() / Fe_det;
    if (!(Jp_new <= max_Jp))
        Jp_new = max_Jp;
    if (!(Jp_new >= min_Jp))
        Jp_new = min_Jp;

    strain = Fe;
    c.mu *= std::exp(psi * (Jp - Jp_new));
    c.lambda *= std::exp(psi * (Jp - Jp_new));
    Jp = Jp_new;

    return false;
}

template <class T>
template <class TConst>
void SnowPlasticity<T>::projectStrainDiagonal(TConst& c, Vector<T, TConst::dim>& sigma)
{
    static const int dim = TConst::dim;

    for (int i = 0; i < dim; i++) {
        sigma(i) = std::max(std::min(sigma(i), (T)1 + theta_s), (T)1 - theta_c);
    }
}

template <class T>
template <class TConst>
void SnowPlasticity<T>::computeSigmaPInverse(TConst& c, const Vector<T, TConst::dim>& sigma_e, Vector<T, TConst::dim>& sigma_p_inv)
{
    using TV = typename TConst::TV;
    TV sigma_proj = sigma_e;
    projectStrainDiagonal(c, sigma_proj);
    sigma_p_inv.array() = sigma_proj.array() / sigma_e.array();

    ZIRAN_WARN("Snow lambda step not fully implemented yet. ");
}

template <class T>
const char* SnowPlasticity<T>::name()
{
    return "SnowPlasticity";
}

///////////////////////////////////////////////////////////////////////////////
/**
   This is NonAssociativeCamClay
 */
///////////////////////////////////////////////////////////////////////////////

//Hardening Mode
// 0 -> CD-MPM fake p hardening
// 1 -> our new q hardening
// 2 -> zhao2019 exponential hardening

template <class T>
NonAssociativeCamClay<T>::NonAssociativeCamClay(T logJp, T friction_angle, T beta, T xi, int dim, bool hardeningOn, bool qHard)
    : logJp(logJp)
    , beta(beta)
    , xi(xi)
    , hardeningOn(hardeningOn)
    , qHard(qHard)
{
    T sin_phi = std::sin(friction_angle / (T)180 * (T)3.141592653);
    T mohr_columb_friction = std::sqrt((T)2 / (T)3) * (T)2 * sin_phi / ((T)3 - sin_phi);
    M = mohr_columb_friction * (T)dim / std::sqrt((T)2 / ((T)6 - dim));
}

template <class T, int dim>
void Compare_With_Phybam_Numerical_Check(
    const Vector<T, dim>& Se1,
    const Vector<T, dim>& Se2,
    const T logJp1,
    const T logJp2)
{

    ZIRAN_ASSERT(std::abs(logJp1 - logJp2) < 1e-4, logJp1, logJp2);
    for (int i = 0; i < dim; ++i)
        ZIRAN_ASSERT(std::abs(Se1(i) - Se2(i)) < 1e-4, Se1(i), Se2(i));
}

template <class T, int dim>
void Compare_With_Physbam(
    const T mu,
    const T kappa,
    const T cam_clay_M,
    const T cam_clay_beta,
    const T p0,
    Vector<T, dim>& Se,
    T& cam_clay_logJp)
{
    using namespace EIGEN_EXT;
    using namespace MATH_TOOLS;
    //typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;

    T a = ((T)1 + (T)2 * cam_clay_beta) * ((T)6 - (T)dim) / (T)2;
    T b = cam_clay_beta * p0;
    T c = p0;
    T M2 = sqr(cam_clay_M);

    T Je = 1.;
    for (int i = 0; i < dim; ++i) Je *= Se(i);

    TV Se2;
    for (int i = 0; i < dim; ++i)
        Se2(i) = sqr(Se(i));

    TV s_hat_trial = mu * std::pow(Je, -(T)2 / (T)dim) * deviatoric(Se2);

    TV tau_hat;
    T Uprime = kappa / (T)2 * (Je - 1 / Je);
    T p_trial = -Uprime * Je;

    // Projecting to the tips
    if (p_trial > c) {
        T Je_new = sqrt(-2 * c / kappa + 1);
        Se = TV::Ones() * pow(Je_new, (T)1 / dim);
        cam_clay_logJp += log(Je / Je_new);
        return;
    }
    else if (p_trial < -b) {
        T Je_new = sqrt(2 * b / kappa + 1);
        Se = TV::Ones() * pow(Je_new, (T)1 / dim);
        cam_clay_logJp += log(Je / Je_new);
        return;
    }
#if 1
    T k = sqrt(-M2 * (p_trial + b) * (p_trial - c) / a);

    T s_hat_trial_norm = s_hat_trial.norm();
    T y = a * sqr(s_hat_trial_norm) + M2 * (p_trial + b) * (p_trial - c);

    if (y < 1e-4) return; // inside the yield surface

    // Fake hardening by computing intersection to center
    T pc = ((T)1 - cam_clay_beta) * p0 / 2;
    if (p0 > 1e-4 && p_trial < p0 - (1e-4) && p_trial > -cam_clay_beta * p0 + (1e-4)) {
        T aa = M2 * sqr(p_trial - pc) / (a * sqr(s_hat_trial_norm));
        T dd = 1 + aa;
        T ff = aa * cam_clay_beta * p0 - aa * p0 - 2 * pc;
        T gg = sqr(pc) - aa * cam_clay_beta * sqr(p0);
        T zz = std::sqrt(std::abs(sqr(ff) - 4 * dd * gg));
        T p1 = (-ff + zz) / (2 * dd);
        T p2 = (-ff - zz) / (2 * dd);
        T p_fake = (p_trial - pc) * (p1 - pc) > 0 ? p1 : p2;
        T Je_new_fake = sqrt(std::abs(-2 * p_fake / kappa + 1));
        if (Je_new_fake > 1e-4)
            cam_clay_logJp += log(Je / Je_new_fake);
    }

    TV be_new = k / mu * std::pow(Je, (T)2 / (T)dim) * s_hat_trial / s_hat_trial_norm + (T)1 / dim * Se2.sum() * TV::Ones();

    for (int i = 0; i < dim; ++i)
        Se(i) = std::sqrt(be_new(i));
#endif
}

template <class T>
template <class TConst>
bool NonAssociativeCamClay<T>::projectStrain(TConst& c, Matrix<T, TConst::dim, TConst::dim>& strain)
{
    using namespace EIGEN_EXT;
    static const int dim = TConst::dim;
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    TM U, V;
    TV sigma;
    T dAlpha; //change in logJp, or the change in volumetric plastic strain
    T dOmega; //change in logJp from q hardening (only for q hardening)

    // TODO: this is inefficient because next time step updateState will do the svd again!
    singularValueDecomposition(strain, U, sigma, V);

    T p0 = c.kappa * (T(0.00001) + std::sinh(xi * std::max(-logJp, (T)0)));

    // debug with physbam only
    // TV XXXSe = sigma;
    // T XXXcam_clay_logJp = logJp;
    // Compare_With_Physbam(c.mu, c.kappa, M, beta, p0, XXXSe, XXXcam_clay_logJp);
    // logJp = XXXcam_clay_logJp;
    // strain = U * XXXSe.asDiagonal() * V.transpose();
    // return true;

    T J = 1.;
    for (int i = 0; i < dim; ++i) J *= sigma(i);

    //Step 1, compute pTrial and see if case 1, 2, or 3
    TV B_hat_trial;
    for (int i = 0; i < dim; ++i)
        B_hat_trial(i) = sigma(i) * sigma(i);
    TV s_hat_trial = c.mu * std::pow(J, -(T)2 / (T)dim) * deviatoric(B_hat_trial);

    T prime = c.kappa / (T)2 * (J - 1 / J);
    T p_trial = -prime * J;

    //Cases 1 and 2 (Ellipsoid Tips)
    //Project to the tips
    T pMin = beta * p0;
    T pMax = p0;
    if (p_trial > pMax) {
        T Je_new = std::sqrt(-2 * pMax / c.kappa + 1);
        sigma = TV::Ones() * std::pow(Je_new, (T)1 / dim);
        Eigen::DiagonalMatrix<T, dim, dim> sigma_m(sigma);
        TM Fe = U * sigma_m * V.transpose();
        strain = Fe;
        if (hardeningOn) {
            logJp += log(J / Je_new);
        }
        return false;
    }
    else if (p_trial < -pMin) {
        T Je_new = std::sqrt(2 * pMin / c.kappa + 1);
        sigma = TV::Ones() * std::pow(Je_new, (T)1 / dim);
        Eigen::DiagonalMatrix<T, dim, dim> sigma_m(sigma);
        TM Fe = U * sigma_m * V.transpose();
        strain = Fe;
        if (hardeningOn) {
            logJp += log(J / Je_new);
        }
        return false;
    }

    //Case 3 --> check if inside or outside YS
    T y_s_half_coeff = ((T)6 - dim) / (T)2 * ((T)1 + (T)2 * beta);
    T y_p_half = M * M * (p_trial + pMin) * (p_trial - pMax);
    T y = y_s_half_coeff * s_hat_trial.squaredNorm() + y_p_half;

    //Case 3a (Inside Yield Surface)
    //Do nothing
    if (y < 1e-4) return false;

    //Case 3b (Outside YS)
    // project to yield surface
    TV B_hat_new = std::pow(J, (T)2 / (T)dim) / c.mu * std::sqrt(-y_p_half / y_s_half_coeff) * s_hat_trial / s_hat_trial.norm();
    B_hat_new += (T)1 / dim * B_hat_trial.sum() * TV::Ones();

    for (int i = 0; i < dim; ++i)
        sigma(i) = std::sqrt(B_hat_new(i));
    Eigen::DiagonalMatrix<T, dim, dim> sigma_m(sigma);
    TM Fe = U * sigma_m * V.transpose();
    strain = Fe;

    //Step 2: Hardening
    //Three approaches to hardening:
    //0 -> hack the hardening by computing a fake delta_p
    //1 -> q based
    if (p0 > 1e-4 && p_trial < pMax - 1e-4 && p_trial > 1e-4 - pMin) {
        T p_center = p0 * ((1 - beta) / (T)2);
        T q_trial = std::sqrt(((T)6 - (T)dim) / (T)2) * s_hat_trial.norm();
        Vector<T, 2> direction;
        direction(0) = p_center - p_trial;
        direction(1) = 0 - q_trial;
        direction = direction / direction.norm();

        T C = M * M * (p_center + beta * p0) * (p_center - p0);
        T B = M * M * direction(0) * (2 * p_center - p0 + beta * p0);
        T A = M * M * direction(0) * direction(0) + (1 + 2 * beta) * direction(1) * direction(1);

        T l1 = (-B + std::sqrt(B * B - 4 * A * C)) / (2 * A);
        T l2 = (-B - std::sqrt(B * B - 4 * A * C)) / (2 * A);

        T p1 = p_center + l1 * direction(0);
        T p2 = p_center + l2 * direction(0);
        T p_fake = (p_trial - p_center) * (p1 - p_center) > 0 ? p1 : p2;

        //Only for pFake Hardening
        T Je_new_fake = sqrt(std::abs(-2 * p_fake / c.kappa + 1));
        dAlpha = log(J / Je_new_fake);

        //Only for q Hardening
        T qNPlus = sqrt(M * M * (p_trial + pMin) * (pMax - p_trial) / ((T)1 + (T)2 * beta));
        T Jtrial = J;
        T zTrial = sqrt(((q_trial * pow(Jtrial, ((T)2 / (T)dim))) / (c.mu * sqrt(((T)6 - (T)dim) / (T)2))) + 1);
        T zNPlus = sqrt(((qNPlus * pow(Jtrial, ((T)2 / (T)dim))) / (c.mu * sqrt(((T)6 - (T)dim) / (T)2))) + 1);
        if (p_trial > p_fake) {
            dOmega = -1 * log(zTrial / zNPlus);
        }
        else {
            dOmega = log(zTrial / zNPlus);
        }

        if (hardeningOn) {
            if (!qHard) {
                if (Je_new_fake > 1e-4) {
                    logJp += dAlpha;
                }
            }
            else if (qHard) {
                if (zNPlus > 1e-4) {
                    logJp += dOmega;
                }
            }
        }
    }

    return false;
}

//#########################################################################
// Function: fillAttributesToVec3
//
// This is for PartioIO purposes so we can dump out attributes out from the plasticity model.
//#########################################################################
template <class T>
void NonAssociativeCamClay<T>::fillAttributesToVec3(Vector<T, 3>& data)
{
    for (int d = 0; d < 3; d++) { data(d) = logJp; }
}

template <class T>
const char* NonAssociativeCamClay<T>::name()
{
    return "NonAssociativeCamClay";
}

///////////////////////////////////////////////////////////////////////////////
/**
   This is Non Associative Von Mises plasticity.
 */
///////////////////////////////////////////////////////////////////////////////

template <class T>
NonAssociativeVonMises<T>::NonAssociativeVonMises(T tauY, T alpha, T hardeningCoeff, int dim)
    : tauY(tauY), alpha(alpha), hardeningCoeff(hardeningCoeff)
{
}

template <class T>
template <class TConst>
bool NonAssociativeVonMises<T>::projectStrain(TConst& c, Matrix<T, TConst::dim, TConst::dim>& strain)
{
    using namespace EIGEN_EXT;
    static const int dim = TConst::dim;
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    TM U, V;
    TV sigma;

    // TODO: this is inefficient because next time step updateState will do the
    // svd again!
    singularValueDecomposition(strain, U, sigma, V);

    T g = 1; // dummies for now
    T gp = 1;

    // Compute scaled tauY
    T scaledTauY = std::sqrt((T)2 / ((T)6 - (T)dim)) * gp * (tauY + (hardeningCoeff * alpha));

    // compute the value of J (multiply all sing values of F out)
    T J = 1.;
    for (int i = 0; i < dim; ++i)
        J *= sigma(i);

    // Compute B hat trial based on the singular values from F^trial --> sigma^2
    // are the sing vals of be hat trial
    TV B_hat_trial;
    for (int i = 0; i < dim; ++i)
        B_hat_trial(i) = sigma(i) * sigma(i);

    // Compute s hat trial using b hat trial
    TV s_hat_trial = c.mu * std::pow(J, -(T)2 / (T)dim) * g * deviatoric(B_hat_trial);

    // Compute s hat trial's L2 norm (since it appears in our expression for
    // y(tau)
    T s_hat_trial_norm = s_hat_trial.norm();

    // Compute y using sqrt(s:s) and scaledTauY
    T y = s_hat_trial_norm - scaledTauY;

    if (y < 1e-4)
        return false; // inside the yield surface if true

    T trace_B_trial = B_hat_trial.sum(); // sum sing vals to get trace
    T z = y / (c.mu * std::pow(J, -(T)2 / (T)dim) * g);

    // Compute new Bhat
    TV B_hat_new = B_hat_trial - ((z / s_hat_trial_norm) * s_hat_trial);

    // Now compute new sigmas by taking sqrt of B hat new, then set strain to be
    // this new F^{n+1} value
    for (int i = 0; i < dim; ++i)
        sigma(i) = std::sqrt(B_hat_new(i));
    Eigen::DiagonalMatrix<T, dim, dim> sigma_m(sigma);
    TM Fe = U * sigma_m * V.transpose();
    strain = Fe;

    // Hardening computations
    // First compute delta gamma, then update alpha based on the result
    T deltaGamma = z / (((T)2 / (T)dim) * trace_B_trial);
    alpha += deltaGamma * (std::sqrt((T)2 / ((T)6 - (T)dim)));

    return false;
}

template <class T>
const char* NonAssociativeVonMises<T>::name()
{
    return "NonAssociativeVonMises";
}

///////////////////////////////////////////////////////////////////////////////
/**
   This is Non Associative Drucker Prager plasticity.
 */
///////////////////////////////////////////////////////////////////////////////

template <class T>
NonAssociativeDruckerPrager<T>::NonAssociativeDruckerPrager(T frictionAngle, T cohesionCoeff, int dim)
    : frictionAngle(frictionAngle), cohesionCoeff(cohesionCoeff)
{
    T sinPhi = std::sin((frictionAngle / (T)180) * M_PI);
    frictionCoeff = std::sqrt((T)2 / (T)3) * (((T)2 * sinPhi) / ((T)3 - sinPhi));
}

template <class T>
template <class TConst>
bool NonAssociativeDruckerPrager<T>::projectStrain(TConst& c, Matrix<T, TConst::dim, TConst::dim>& strain)
{

    using namespace EIGEN_EXT;
    static const int dim = TConst::dim;
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    TM U, V;
    TV sigma;

    if (c.mu == 0) { //don't want to apply plasticity in this case of course
        return false;
    }

    // TODO: this is inefficient because next time step updateState will do the
    // svd again!
    singularValueDecomposition(strain, U, sigma, V);

    T g = 1; // eventually fill with actual phase calculation
    if (g < 1e-6) {
        g = 1e-6;
    }
    //T gp = 1;

    // compute the value of J (multiply all sing values of F out)
    T J = 1.;
    for (int i = 0; i < dim; ++i)
        J *= sigma(i);

    // Compute B hat trial based on the singular values from F^trial --> sigma^2
    // are the sing vals of be hat trial
    TV B_hat_trial;
    for (int i = 0; i < dim; ++i)
        B_hat_trial(i) = sigma(i) * sigma(i);

    // Compute s hat trial using b hat trial
    TV s_hat_trial = c.mu * std::pow(J, -(T)2 / (T)dim) * g * deviatoric(B_hat_trial);

    // Compute s hat trial's L2 norm (since it appears in our expression for
    // y(tau)
    T s_hat_trial_norm = s_hat_trial.norm();

    T uPrime = (c.kappa / (T)2) * (J - ((T)1 / J));

    T gStar = g; //init with g
    if (J < 1) {
        gStar = 1; //chage to 1 if J < 1
    }

    //Compute the trace of tau hat using a reduced formula
    T traceTauHat = uPrime * J * dim * gStar;

    //If trace of tauhat is greater than the cohesion coefficient we must project to the tip of the yield surface
    if (traceTauHat > cohesionCoeff) {

        T tip = (((cohesionCoeff / (T)dim) * ((T)2 / (gStar * c.kappa))) + 1);
        tip = std::pow(tip, ((T)1 / (2 * (T)dim)));

        //Now set the singular values of the new strain to be each equal to this tip value!
        for (int i = 0; i < dim; ++i)
            sigma(i) = tip;
        Eigen::DiagonalMatrix<T, dim, dim> sigma_m(sigma);
        TM Fe = U * sigma_m * V.transpose();
        strain = Fe;
        return false;
    }

    // Compute y using sqrt(s:s) and scaledTauY
    T y = s_hat_trial_norm - (cohesionCoeff - (frictionCoeff * traceTauHat));

    if (y < 1e-4)
        return false; // inside the yield surface if true

    T z = y / (c.mu * std::pow(J, -(T)2 / (T)dim) * g);

    // Compute new Bhat
    TV B_hat_new = B_hat_trial - ((z / s_hat_trial_norm) * s_hat_trial);

    // Now compute new sigmas by taking sqrt of B hat new, then set strain to be
    // this new F^{n+1} value
    for (int i = 0; i < dim; ++i)
        sigma(i) = std::sqrt(B_hat_new(i));
    Eigen::DiagonalMatrix<T, dim, dim> sigma_m(sigma);
    TM Fe = U * sigma_m * V.transpose();
    strain = Fe;

    // TODO: Hardening computations?

    return false;
}

template <class T>
const char* NonAssociativeDruckerPrager<T>::name()
{
    return "NonAssociativeDruckerPrager";
}

///////////////////////////////////////////////////////////////////////////////
/**
   This is Smudge Plasticity. Use this to get dry sand with semi-implicit time
   integration, meaning: do elasticity first, followed by plasticity.
   The assumption here is that we use the full quartic constitutive model for
   elasticity.
 */
///////////////////////////////////////////////////////////////////////////////
template <class T>
SmudgePlasticity<T>::SmudgePlasticity(const T friction_angle, const T beta, const T cohesion)
    : beta(beta)
    , logJp(0)
    , cohesion(cohesion)

{
    T sin_phi = std::sin(friction_angle / (T)180 * (T)3.141592653);
    alpha = std::sqrt((T)2 / (T)3) * (T)2 * sin_phi / ((T)3 - sin_phi);
}

template <class T>
void SmudgePlasticity<T>::setParameters(const T friction_angle_in, const T beta_in, const T cohesion_in)
{
    beta = beta_in;
    cohesion = cohesion_in;
    T sin_phi = std::sin(friction_angle_in / (T)180 * (T)3.141592653);
    alpha = std::sqrt((T)2 / (T)3) * (T)2 * sin_phi / ((T)3 - sin_phi);
}

template <class T>
template <class TConst>
bool SmudgePlasticity<T>::projectStrain(TConst& c, Matrix<T, 2, 2>& strain)
{
    static const int dim = 2;
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    TM U, V;
    TV sigma;
    using MATH_TOOLS::sqr;
    using std::pow;
    using std::sqrt;

    // TODO: this is inefficient because next time step updateState will do the svd again!
    singularValueDecomposition(strain, U, sigma, V);

    TV epsilon = sigma.array().abs().max(1e-4).log() - cohesion;
    T trace_epsilon = epsilon.sum() + logJp;
    //TV epsilon_hat = epsilon - (trace_epsilon / (T)dim) * TV::Ones();

    if (trace_epsilon >= 0) // case II: project to tip
    {
        strain = U * std::exp(cohesion) * V.transpose();
        logJp = epsilon.sum() + logJp;
        return false;
    }
    else if (c.mu == 0) {
        return false;
    }
    logJp = 0;
    // Do projection
    T mu = c.mu;
    T lambda = c.lambda;
    T h0t = epsilon(0);
    T h1t = epsilon(1);

    T he1 = sqrt(pow(h0t + h1t, 6) * pow(mu, 4) * (pow(mu, 2) + 4 * pow(alpha, 4) * mu * (4 * lambda + mu) + pow(alpha, 2) * (8 * pow(lambda, 2) - 8 * lambda * mu - 4 * pow(mu, 2))));
    T he2 = pow(sqrt(2) * he1 - 2 * alpha * pow(h0t + h1t, 3) * pow(mu, 2) * (2 * lambda - mu + 2 * pow(alpha, 2) * mu), 0.3333333333333333);
    T he3 = -1 + 2 * sqr(alpha);

    //T dg = -(sqrt(2) * h0t * mu + 2 * alpha * h0t * mu - sqrt(2) * h1t * mu + 2 * alpha * h1t * mu) / (2. * mu) + (3 * pow(2, 0.3333333333333333) * (-(pow(h0t, 2) * pow(mu, 2)) + 2 * pow(alpha, 2) * pow(h0t, 2) * pow(mu, 2) - 2 * h0t * h1t * pow(mu, 2) + 4 * pow(alpha, 2) * h0t * h1t * pow(mu, 2) - pow(h1t, 2) * pow(mu, 2) + 2 * pow(alpha, 2) * pow(h1t, 2) * pow(mu, 2))) / (mu * pow(-432 * alpha * pow(h0t, 3) * lambda * pow(mu, 2) - 1296 * alpha * pow(h0t, 2) * h1t * lambda * pow(mu, 2) - 1296 * alpha * h0t * pow(h1t, 2) * lambda * pow(mu, 2) - 432 * alpha * pow(h1t, 3) * lambda * pow(mu, 2) + 216 * alpha * pow(h0t, 3) * pow(mu, 3) - 432 * pow(alpha, 3) * pow(h0t, 3) * pow(mu, 3) + 648 * alpha * pow(h0t, 2) * h1t * pow(mu, 3) - 1296 * pow(alpha, 3) * pow(h0t, 2) * h1t * pow(mu, 3) + 648 * alpha * h0t * pow(h1t, 2) * pow(mu, 3) - 1296 * pow(alpha, 3) * h0t * pow(h1t, 2) * pow(mu, 3) + 216 * alpha * pow(h1t, 3) * pow(mu, 3) - 432 * pow(alpha, 3) * pow(h1t, 3) * pow(mu, 3) + sqrt(-23328 * pow(-(pow(h0t, 2) * pow(mu, 2)) + 2 * pow(alpha, 2) * pow(h0t, 2) * pow(mu, 2) - 2 * h0t * h1t * pow(mu, 2) + 4 * pow(alpha, 2) * h0t * h1t * pow(mu, 2) - pow(h1t, 2) * pow(mu, 2) + 2 * pow(alpha, 2) * pow(h1t, 2) * pow(mu, 2), 3) + pow(-432 * alpha * pow(h0t, 3) * lambda * pow(mu, 2) - 1296 * alpha * pow(h0t, 2) * h1t * lambda * pow(mu, 2) - 1296 * alpha * h0t * pow(h1t, 2) * lambda * pow(mu, 2) - 432 * alpha * pow(h1t, 3) * lambda * pow(mu, 2) + 216 * alpha * pow(h0t, 3) * pow(mu, 3) - 432 * pow(alpha, 3) * pow(h0t, 3) * pow(mu, 3) + 648 * alpha * pow(h0t, 2) * h1t * pow(mu, 3) - 1296 * pow(alpha, 3) * pow(h0t, 2) * h1t * pow(mu, 3) + 648 * alpha * h0t * pow(h1t, 2) * pow(mu, 3) - 1296 * pow(alpha, 3) * h0t * pow(h1t, 2) * pow(mu, 3) + 216 * alpha * pow(h1t, 3) * pow(mu, 3) - 432 * pow(alpha, 3) * pow(h1t, 3) * pow(mu, 3), 2)), 0.3333333333333333)) + pow(-432 * alpha * pow(h0t, 3) * lambda * pow(mu, 2) - 1296 * alpha * pow(h0t, 2) * h1t * lambda * pow(mu, 2) - 1296 * alpha * h0t * pow(h1t, 2) * lambda * pow(mu, 2) - 432 * alpha * pow(h1t, 3) * lambda * pow(mu, 2) + 216 * alpha * pow(h0t, 3) * pow(mu, 3) - 432 * pow(alpha, 3) * pow(h0t, 3) * pow(mu, 3) + 648 * alpha * pow(h0t, 2) * h1t * pow(mu, 3) - 1296 * pow(alpha, 3) * pow(h0t, 2) * h1t * pow(mu, 3) + 648 * alpha * h0t * pow(h1t, 2) * pow(mu, 3) - 1296 * pow(alpha, 3) * h0t * pow(h1t, 2) * pow(mu, 3) + 216 * alpha * pow(h1t, 3) * pow(mu, 3) - 432 * pow(alpha, 3) * pow(h1t, 3) * pow(mu, 3) + sqrt(-23328 * pow(-(pow(h0t, 2) * pow(mu, 2)) + 2 * pow(alpha, 2) * pow(h0t, 2) * pow(mu, 2) - 2 * h0t * h1t * pow(mu, 2) + 4 * pow(alpha, 2) * h0t * h1t * pow(mu, 2) - pow(h1t, 2) * pow(mu, 2) + 2 * pow(alpha, 2) * pow(h1t, 2) * pow(mu, 2), 3) + pow(-432 * alpha * pow(h0t, 3) * lambda * pow(mu, 2) - 1296 * alpha * pow(h0t, 2) * h1t * lambda * pow(mu, 2) - 1296 * alpha * h0t * pow(h1t, 2) * lambda * pow(mu, 2) - 432 * alpha * pow(h1t, 3) * lambda * pow(mu, 2) + 216 * alpha * pow(h0t, 3) * pow(mu, 3) - 432 * pow(alpha, 3) * pow(h0t, 3) * pow(mu, 3) + 648 * alpha * pow(h0t, 2) * h1t * pow(mu, 3) - 1296 * pow(alpha, 3) * pow(h0t, 2) * h1t * pow(mu, 3) + 648 * alpha * h0t * pow(h1t, 2) * pow(mu, 3) - 1296 * pow(alpha, 3) * h0t * pow(h1t, 2) * pow(mu, 3) + 216 * alpha * pow(h1t, 3) * pow(mu, 3) - 432 * pow(alpha, 3) * pow(h1t, 3) * pow(mu, 3), 2)), 0.3333333333333333) / (6. * pow(2, 0.3333333333333333) * mu);
    T numer = mu * (-(sqrt(2) * h0t * he2) + sqrt(2) * h1t * he2 - 2 * alpha * (h0t + h1t) * he2 + pow(2, 0.6666666666666666) * pow(h0t + h1t, 2) * he3 * mu) + pow(2, 0.3333333333333333) * pow(sqrt(2) * he1 - 2 * alpha * pow(h0t + h1t, 3) * pow(mu, 2) * (2 * lambda + he3 * mu), 0.6666666666666666);
    T denom = 2 * mu * pow(sqrt(2) * he1 - 2 * alpha * pow(h0t + h1t, 3) * pow(mu, 2) * (2 * lambda + he3 * mu), 0.3333333333333333);
    ZIRAN_ASSERT(numer == numer, "numerator is not finite");
    ZIRAN_ASSERT(denom == denom, "denominator is not finite");
    T dg = numer / denom;

    TV H;
    if (dg > 0) // case I: inside yield surface
    {
        H = epsilon + TV::Constant(cohesion);
    }
    else {
        H(0) = epsilon(0) + sqrt(2) / 2 * dg;
        H(1) = epsilon(1) - sqrt(2) / 2 * dg;
    }
    TV exp_H = H.array().exp();
    strain = U * exp_H.asDiagonal() * V.transpose();
    return false;
}

template <class T>
T SmudgePlasticity<T>::yieldFunction(const Vector<T, 3>& strain, T mu, T lambda)
{
    using TV = Vector<T, 3>;
    using MATH_TOOLS::cube;
    using MATH_TOOLS::sqr;
    T trace = strain.sum();
    TV trace_vec = TV::Constant(cube(trace));
    TV cube_vec = strain.array().cube();
    TV st = 2 * qscale * lambda * trace_vec + 4 * qscale * mu * cube_vec; // stress trial
    ZIRAN_ASSERT(st(0) == st(0) && st(1) == st(1) && st(2) == st(2), "\tst = (", st(0), ", ", st(1), ", ", st(2), ").\n");
    ZIRAN_ASSERT(st(0) < 1e10 && st(1) < 1e10 && st(2) < 1e10, "\tst = (", st(0), ", ", st(1), ", ", st(2), ").\nstrain = (", strain(0), " ,", strain(1), " ,", strain(2), ")\n");
    T yield = alpha * st.sum() + sqrt(((T)1. / (T)3.) * (sqr(st(0) - st(1)) + sqr(st(0) - st(2)) + sqr(st(1) - st(2)))); // this is where the previous one dies
    return yield;
}

template <class T>
void SmudgePlasticity<T>::qHelper(double& yield, double& dyield, const double& q, const Vector<double, 3>& dev, T mu, T lambda, T p, const Vector<T, 3>& strain, int newton_step)
{
    using MATH_TOOLS::cube;
    using MATH_TOOLS::sqr;
    using std::sqrt;

    double dev0 = (double)(dev(0));
    double dev1 = (double)(dev(1));
    double dev2 = (double)(dev(2));

    double b0 = p + dev0 * q;
    double b1 = p + dev1 * q;
    double b2 = p + dev2 * q;
    double c0 = cube(b0);
    double c1 = cube(b1);
    double c2 = cube(b2);

    //T under_sqrt =pow(p + dev0*q,6) - pow(p + dev0*q,3)*pow(p + dev1*q,3) + pow(p + dev1*q,6) - pow(p + dev0*q,3)*pow(p + dev2*q,3) - pow(p + dev1*q,3)*pow(p + dev2*q,3) + pow(p + dev2*q,6);
    //T b0_6 = sqr(cube(p+dev0*q)); //T b1_6 = sqr(cube(p+dev1*q)); //T b2_6 = sqr(cube(p+dev2*q)); //T b01_33 = cube(p + dev0 * q) * cube(p + dev1 * q); //T b02_33 = cube(p + dev0 * q) * cube(p + dev2 * q); //T b12_33 = cube(p + dev1 * q) * cube(p + dev2 * q); //T under_sqrt =pow(p + dev0*q,6) - b01_33 + b1_6 - b02_33 - b12_33 + b2_6;
    double under_sqrt = (double)1 / 2 * (sqr(c0 - c1) + sqr(c0 - c2) + sqr(c1 - c2));
    yield = 2 * alpha * qscale * (81 * lambda * pow(p, 3) + 2 * mu * pow(p + dev0 * q, 3) + 2 * mu * pow(p + dev1 * q, 3) + 2 * mu * pow(p + dev2 * q, 3));
    yield += 4 * sqrt((T)2. / 3) * mu * qscale * sqrt(under_sqrt);

    //T d0 = dev0 * (6 * sqr(b0) * cube(b0) - 3 * sqr(b0) * cube(b1) - 3 * sqr(b0) * cube(b2));
    //T d1 = dev1 * (6 * sqr(b1) * cube(b1) - 3 * sqr(b1) * cube(b0) - 3 * sqr(b1) * cube(b2));
    //T d2 = dev2 * (6 * sqr(b2) * cube(b2) - 3 * sqr(b2) * cube(b0) - 3 * sqr(b2) * cube(b1));

    //T numer = sqrt((T)6) * (d0 + d1 + d2);
    //T denom = std::max(sqrt(under_sqrt), (T)1e-8);
    //ZIRAN_ASSERT(std::abs(denom) > 1e-10, "newton_step = ", newton_step, "denom = ", denom, "\nstrain = (", strain(0), ", ", strain(1), ", ", strain(2), ")\ndev0 = ", dev0, "\ndev1 = ", dev1, "\ndev2 = ", dev2, "\np = ", p, "\nq = ", q, "\nb0 = ", b0, "\nb1 = ", b1, "\nb2 = ", b2);
    //T aa = 18 * alpha * (dev0 * sqr(b0) + dev1 * sqr(b1) + dev2 * sqr(b2)) + numer / denom;
    //dyield = ((T)2 / 3) * mu * qscale * aa;

    double f_b0 = c0 * sqr(b0);
    double f_b1 = c1 * sqr(b1);
    double f_b2 = c2 * sqr(b2);
    double s_b0 = sqr(c0);
    double s_b1 = sqr(c1);
    double s_b2 = sqr(c2);

    double first_term = 18 * alpha * (dev0 + dev1 + dev2) * lambda * sqr(3 * p + dev0 * q + dev1 * q + dev2 * q) * qscale + 4 * alpha * mu * (3 * dev0 * sqr(b0) + 3 * dev1 * sqr(b1) + 3 * dev2 * sqr(b2)) * qscale;
    double numerator = 2 * sqrt((double)2. / (double)3.) * sqr(mu) * (6 * dev0 * f_b0 + 6 * dev1 * f_b1 - 3 * dev2 * c1 * sqr(b2) - 3 * dev1 * sqr(b1) * c2 + 6 * dev2 * f_b2 - c0 * (3 * dev1 * sqr(b1) + 3 * dev2 * sqr(b2)) - 3 * dev0 * sqr(b0) * (c1 + c2)) * sqr(qscale);
    double denominator = mu * qscale * sqrt((s_b0 + s_b1 - c1 * c2 + s_b2 - c0 * (c1 + c2)));
    dyield = first_term + numerator / denominator;
}

template <class T>
template <class TConst>
bool SmudgePlasticity<T>::projectStrain(TConst& c, Matrix<T, 3, 3>& strain)
{
    static const int dim = 3;
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    TM U, V;
    TV sigma;
    using MATH_TOOLS::sqr;
    using std::pow;
    using std::sqrt;

    // other parameters
    T lambda = c.lambda;
    T mu = c.mu;

    // compute singular values decomposition of strain
    singularValueDecomposition(strain, U, sigma, V);
    TV epsilon = sigma.array().abs().max(1e-4).log() - cohesion;

    ZIRAN_ASSERT(epsilon(0) < 1e10 && epsilon(1) < 1e10 && epsilon(2) < 1e10, "\tepsilon = (", epsilon(0), ", ", epsilon(1), ", ", epsilon(2), ").\n");
    // check if strain is inside the yield function: if it is, then no need to do any projection
    T yield_value = yieldFunction(epsilon, mu, lambda);

    // check the value of yield function in (h0t + h1t + h2t) / sqrt(3)
    T h0t = epsilon(0);
    T h1t = epsilon(1);
    T h2t = epsilon(2);
    T p = (h0t + h1t + h2t) / 3;
    TV H = p * TV(1, 1, 1);
    T trace_epsilon = h0t + h1t + h2t + logJp;

    if (yield_value <= 0) {
        logJp = 0;
        return false;
    }
    else if (trace_epsilon >= 0) {
        logJp = epsilon.sum() + logJp;
        H = TV(0, 0, 0);
        TV exp_H = H.array().exp();
        strain = U * exp_H.asDiagonal() * V.transpose();
        return true;
    }
    else {
        logJp = 0;

        T a0 = (-2 * h0t + h1t + h2t);
        T a1 = (h0t - 2 * h1t + h2t);
        T a2 = (h0t + h1t - 2 * h2t);
        Vector<double, 3> dev_double((double)(a0), (double)(a1), (double)(a2));
        TV dev(a0, a1, a2);

        double q = (T)-1. / (T)3.;
        double yield, dyield;

        int newton_step = 0;
        const int THRESHOLD = 100;
        for (; newton_step < THRESHOLD; newton_step++) {
            qHelper(yield, dyield, q, dev_double, mu, lambda, p, epsilon, newton_step);
            //std::cout << "newton step = " << newton_step << ", q = " << q << ", yield = " << yield << std::endl
            //          << std::flush;
            if (std::abs(yield) < 1e-4)
                break;
            q = q - yield / dyield;
        }
        H = p * TV(1, 1, 1) + q * dev;
        double yield_check;
        qHelper(yield_check, dyield, q, dev_double, mu, lambda, p, epsilon, newton_step);
        if (newton_step >= THRESHOLD - 1) {
            ZIRAN_WARN("NEWTON STEP AT LEAST ", THRESHOLD - 1);
        }
        TV exp_H = H.array().exp();
        strain = U * exp_H.asDiagonal() * V.transpose();
        return true;
    }
}

template <class T>
template <class TConst>
void SmudgePlasticity<T>::computeSigmaPInverse(TConst& c, const Vector<T, TConst::dim>& sigma_e, Vector<T, TConst::dim>& sigma_p_inv)
{
    ZIRAN_WARN("SmudgePlasticity lambda step not fully implemented yet. ");
}

template <class T>
const char* SmudgePlasticity<T>::name()
{
    return "SmudgePlasticity";
}
///////////////////////////////////////////////////////////////////////////////
/**
 * End of Smudge Plasticity
 */
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/**
   This is the Drucker Prager plasticity model from

   Drucker-Prager Elastoplasticity for Sand Animation,
   G. Klar, T. Gast, A. Pradhana, C. Fu, C. Schroeder, C. Jiang, J. Teran,
   ACM Transactions on Graphics (SIGGRAPH 2016).

   It assumes the StvkHencky elasticity consitutive model.
 */
///////////////////////////////////////////////////////////////////////////////
template <class T>
DruckerPragerStvkHencky<T>::DruckerPragerStvkHencky(const T friction_angle, const T beta, const T cohesion, const bool volume_correction)
    : beta(beta)
    , logJp(0)
    , cohesion(cohesion)
    , volume_correction(volume_correction)
{
    T sin_phi = std::sin(friction_angle / (T)180 * (T)3.141592653);
    alpha = std::sqrt((T)2 / (T)3) * (T)2 * sin_phi / ((T)3 - sin_phi);
}

template <class T>
void DruckerPragerStvkHencky<T>::setParameters(const T friction_angle_in, const T beta_in, const T cohesion_in)
{
    beta = beta_in;
    cohesion = cohesion_in;
    T sin_phi = std::sin(friction_angle_in / (T)180 * (T)3.141592653);
    alpha = std::sqrt((T)2 / (T)3) * (T)2 * sin_phi / ((T)3 - sin_phi);
}

// bool is for fracture
template <class T>
template <class TConst>
bool DruckerPragerStvkHencky<T>::projectStrain(TConst& c, Matrix<T, TConst::dim, TConst::dim>& strain)
{
    static const int dim = TConst::dim;
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    TM U, V;
    TV sigma;

    // TODO: this is inefficient because next time step updateState will do the svd again!
    singularValueDecomposition(strain, U, sigma, V);

    TV epsilon = sigma.array().abs().max(1e-4).log() - cohesion;
    T trace_epsilon = epsilon.sum() + logJp;
    TV epsilon_hat = epsilon - (trace_epsilon / (T)dim) * TV::Ones();
    T epsilon_hat_squared_norm = epsilon_hat.squaredNorm();

    if (trace_epsilon >= (T)0) // case II: project to tip
    {
        strain = U * std::exp(cohesion) * V.transpose();
        if (volume_correction) {
            logJp = beta * epsilon.sum() + logJp;
        }
        return false;
    }
    else if (c.mu == 0) {
        return false;
    }
    logJp = 0;
    T epsilon_hat_norm = std::sqrt(epsilon_hat_squared_norm);
    T delta_gamma = epsilon_hat_norm + (dim * c.lambda + 2 * c.mu) / (2 * c.mu) * trace_epsilon * alpha;
    TV H;
    if (delta_gamma <= 0) // case I: inside yield surface
    {
        H = epsilon + TV::Constant(cohesion);
    }
    else {
        H = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat + TV::Constant(cohesion); // case III: projection
    }
    TV exp_H = H.array().exp();
    strain = U * exp_H.asDiagonal() * V.transpose();
    return false;
}

// bool is for fracture
template <class T>
template <class TConst>
Vector<T, TConst::dim> DruckerPragerStvkHencky<T>::projectSigma(TConst& c, const Vector<T, TConst::dim>& sigma)
{
    static const int dim = TConst::dim;
    typedef Vector<T, dim> TV;
    TV epsilon = sigma.array().abs().max(1e-4).log() - cohesion;
    T trace_epsilon = epsilon.sum();
    TV epsilon_hat = epsilon - (trace_epsilon / (T)dim) * TV::Ones();
    T epsilon_hat_squared_norm = epsilon_hat.squaredNorm();
    if (trace_epsilon >= 0) // case II: project to tip
    {
        TV ret = std::exp(cohesion) * TV::Ones();
        return ret;
    }
    T epsilon_hat_norm = std::sqrt(epsilon_hat_squared_norm);
    T delta_gamma = epsilon_hat_norm + (dim * c.lambda + 2 * c.mu) / (2 * c.mu) * trace_epsilon * alpha;
    TV H;
    if (delta_gamma <= 0) // case I: inside yield surface
    {
        H = epsilon + TV::Constant(cohesion);
    }
    else {
        H = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat + TV::Constant(cohesion); // case III: projection
    }
    TV ret = H.array().exp();
    return ret;
}

// bool is for fracture
template <class T>
template <class TConst>
Matrix<T, TConst::dim, TConst::dim> DruckerPragerStvkHencky<T>::projectSigmaDerivative(TConst& c, const Vector<T, TConst::dim>& sigma)
{
    // const T eps = (T)1e-6;
    static const int dim = TConst::dim;
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    TV epsilon = sigma.array().abs().max(1e-4).log() - cohesion;
    T trace_epsilon = epsilon.sum();
    TV epsilon_hat = epsilon - (trace_epsilon / (T)dim) * TV::Ones();
    T epsilon_hat_squared_norm = epsilon_hat.squaredNorm();
    if (trace_epsilon >= (T)0) // case II: project to tip
    {
        TM ret = TM::Zero();
        return ret;
    }
    T epsilon_hat_norm = std::sqrt(epsilon_hat_squared_norm);
    T delta_gamma = epsilon_hat_norm + (dim * c.lambda + 2 * c.mu) / (2 * c.mu) * trace_epsilon * alpha;
    TV H;
    if (delta_gamma <= 0) // case I: inside yield surface
    {
        TM ret = TM::Identity();
        return ret;
    }
    else {
        TV w = sigma.array().inverse();
        T k = trace_epsilon;
        TV s = epsilon - k / dim * TV::Ones();
        TV s_hat = s / s.norm();
        T p = alpha * k * (dim * c.lambda + 2 * c.mu) / (2 * c.mu * s.norm());
        H = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat + TV::Constant(cohesion); // case III: projection
        TV Z_hat = H.array().exp();
        TM ret = Z_hat.asDiagonal() * ((((T)1 + 2 * p) / dim * TV::Ones() - p / k * epsilon) * w.transpose() - p * (TM::Identity() - s_hat * s_hat.transpose()) * w.asDiagonal());
        return ret;
    }
}

template <class T>
template <class TConst>
void DruckerPragerStvkHencky<T>::projectSigmaAndDerivative(TConst& c, const Vector<T, TConst::dim>& sigma, Vector<T, TConst::dim>& projectedSigma, Matrix<T, TConst::dim, TConst::dim>& projectedSigmaDerivative)
{
    // const T eps = (T)1e-6;
    static const int dim = TConst::dim;
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    TV epsilon = sigma.array().abs().max(1e-4).log() - cohesion;
    T trace_epsilon = epsilon.sum();
    TV epsilon_hat = epsilon - (trace_epsilon / (T)dim) * TV::Ones();
    T epsilon_hat_squared_norm = epsilon_hat.squaredNorm();
    T epsilon_hat_norm = std::sqrt(epsilon_hat_squared_norm);
    T delta_gamma = epsilon_hat_norm + (dim * c.lambda + 2 * c.mu) / (2 * c.mu) * trace_epsilon * alpha;
    TV H;
    if (delta_gamma <= 0) // case I: inside yield surface
    {
        H = epsilon + TV::Constant(cohesion);
        projectedSigma = H.array().exp();
        projectedSigmaDerivative = TM::Identity();
    }
    else if (trace_epsilon > (T)0 || epsilon_hat_norm == 0) // case II: project to tip
    {
        projectedSigma = std::exp(cohesion) * TV::Ones();
        projectedSigmaDerivative = TM::Zero();
    }
    else {
        TV w = sigma.array().inverse();
        T k = trace_epsilon;
        TV s = epsilon - k / dim * TV::Ones();
        TV s_hat = s / s.norm();
        T p = alpha * k * (dim * c.lambda + 2 * c.mu) / (2 * c.mu * s.norm());
        H = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat + TV::Constant(cohesion); // case III: projection
        projectedSigma = H.array().exp();
        projectedSigmaDerivative = projectedSigma.asDiagonal() * ((((T)1 + 2 * p) / dim * TV::Ones() - p / k * epsilon) * w.transpose() - p * (TM::Identity() - s_hat * s_hat.transpose()) * w.asDiagonal());
    }
}

template <class T>
template <class TConst>
void DruckerPragerStvkHencky<T>::computeSigmaPInverse(TConst& c, const Vector<T, TConst::dim>& sigma_e, Vector<T, TConst::dim>& sigma_p_inv)
{
    using TV = typename TConst::TV;
    TV sigma_proj = sigma_e;
    projectStrainDiagonal(c, sigma_proj);
    sigma_p_inv.array() = sigma_proj.array() / sigma_e.array();

    ZIRAN_WARN("Drucker Prager lambda step not fully implemented yet. ");
}

template <class T>
const char* DruckerPragerStvkHencky<T>::name()
{
    return "DruckerPragerStvkHencky";
}
///////////////////////////////////////////////////////////////////////////////
/**
 * END OF DruckerPragerStvkWithHencky
 */
///////////////////////////////////////////////////////////////////////////////

template <class T>
DruckerPragerStrainSoftening<T>::DruckerPragerStrainSoftening(const T cohesion_peak, const T cohesion_residual,
    const T friction_angle_peak, const T friction_angle_residual,
    const T softening_rate)
    : logJp(0)
    , logJp_damage(0)
    , cohesion_peak(cohesion_peak)
    , cohesion_residual(cohesion_residual)
    , friction_angle_peak(friction_angle_peak)
    , friction_angle_residual(friction_angle_residual)
    , softening_rate(softening_rate)

{
    friction_angle = friction_angle_residual + (friction_angle_peak - friction_angle_residual) * std::exp(-softening_rate * logJp_damage);
    cohesion = cohesion_residual + (cohesion_peak - cohesion_residual) * std::exp(-softening_rate * logJp_damage);
    T sin_phi = std::sin(friction_angle / (T)180 * (T)3.141592653);
    alpha = std::sqrt((T)2 / (T)3) * (T)2 * sin_phi / ((T)3 - sin_phi);
}

// bool is for fracture
template <class T>
template <class TConst>
bool DruckerPragerStrainSoftening<T>::projectStrain(TConst& c, Matrix<T, TConst::dim, TConst::dim>& strain)
{
    static const int dim = TConst::dim;
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    TM U, V;
    TV sigma;

    singularValueDecomposition(strain, U, sigma, V);

    // softerning
    friction_angle = friction_angle_residual + (friction_angle_peak - friction_angle_residual) * std::exp(-softening_rate * logJp_damage);
    cohesion = cohesion_residual + (cohesion_peak - cohesion_residual) * std::exp(-softening_rate * logJp_damage);
    T sin_phi = std::sin(friction_angle / (T)180 * (T)3.141592653);
    alpha = std::sqrt((T)2 / (T)3) * (T)2 * sin_phi / ((T)3 - sin_phi);

    TV epsilon = sigma.array().abs().max(1e-4).log() - cohesion;
    T trace_epsilon = epsilon.sum() + logJp;
    TV epsilon_hat = epsilon - (trace_epsilon / (T)dim) * TV::Ones();
    T epsilon_hat_squared_norm = epsilon_hat.squaredNorm();

    if (trace_epsilon >= 0) // case II: project to tip
    {
        strain = U * std::exp(cohesion) * V.transpose();
        logJp += epsilon.sum();
        logJp_damage += epsilon.sum();
        return false;
    }
    else if (c.mu == 0) {
        return false;
    }
    logJp = 0;
    T epsilon_hat_norm = std::sqrt(epsilon_hat_squared_norm);
    T delta_gamma = epsilon_hat_norm + (dim * c.lambda + 2 * c.mu) / (2 * c.mu) * trace_epsilon * alpha;
    TV H;
    if (delta_gamma <= 0) // case I: inside yield surface
    {
        H = epsilon + TV::Constant(cohesion);
    }
    else {
        H = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat + TV::Constant(cohesion); // case III: projection
    }
    TV exp_H = H.array().exp();
    strain = U * exp_H.asDiagonal() * V.transpose();

    return false;
}

template <class T>
const char* DruckerPragerStrainSoftening<T>::name()
{
    return "DruckerPragerStrainSoftening";
}

template <class T>
template <int dim>
bool UnilateralJ<T>::projectStrain(EquationOfState<T, dim>& c, T& strain)
{
    T corrected_strain = strain * Jp;

    if (corrected_strain > 1) {
        Jp = corrected_strain;
        if (Jp > max_Jp)
            Jp = max_Jp;
        strain = 1;
    }
    else {
        strain = corrected_strain;
        Jp = 1;
    }
    return false;
}

template <class T>
const char* UnilateralJ<T>::name()
{
    return "UnilateralJ";
}

template <class T>
ModifiedCamClay<T>::ModifiedCamClay(const T M, const T beta, const T Jp, const T xi, const bool hardeningOn)
    : M(M)
    , beta(beta)
    , logJp(std::log(Jp))
    , xi(xi)
    , hardeningOn(hardeningOn)
{
}

template <class T>
const char* ModifiedCamClay<T>::name()
{
    return "ModifiedCamClay";
}

template <class T>
bool ModifiedCamClay<T>::projectStrain(StvkWithHencky<T, 2>& c, Matrix<T, 2, 2>& strain)
{
    using MATH_TOOLS::sqr;
    typedef StvkWithHencky<T, 2> TConst;
    static const int dim = TConst::dim;
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    TM U, V;
    TV sigma;

    // TODO: this is inefficient because next time step updateState will do the svd again!
    singularValueDecomposition(strain, U, sigma, V);

    // hardening
    T mu = c.mu, lambda = c.lambda;
    T p0 = (lambda + 2 * mu) * std::sinh(std::max(-logJp, (T)0));

    TV epsilon = sigma.array().abs().max(1e-4).log();
    T trace_epsilon = epsilon.sum();
    TV tau = (2 * mu) * epsilon + (lambda * trace_epsilon) * TV::Ones();
    T trace_tau = tau.sum();
    T p = -trace_tau / (T)dim;
    T q2 = sqr(tau(0)) + sqr(tau(1)) - (T)2 * tau(0) * tau(1); // Mises equivalent stress squared

    T y = q2 * (1 + 2 * beta) + sqr(M) * (p + beta * p0) * (p - p0);
    if (y <= 0)
        return false;

    // ZIRAN_ASSERT(false, sigma.transpose(), ' ', epsilon.transpose(), ' ', y, ' ', tau.transpose());

    Vector<T, 3> x;
    x << tau(0), tau(1), 0;
    for (int newton_step = 0; newton_step < 100; newton_step++) {
        Vector<T, 3> r;

        r(0) = -epsilon(0) + ((lambda + 2. * mu) * x(0)) / (4. * lambda * mu + 4. * (mu * mu)) + x(2) * (-0.5 * (M * M) * (-p0 + 0.5 * (-x(0) - x(1))) - 0.5 * (M * M) * (beta * p0 + 0.5 * (-x(0) - x(1))) + (1. + 2. * beta) * (2. * x(0) - 2. * x(1))) - (0.25 * lambda * x(1)) / (mu * (lambda + mu));
        r(1) = -epsilon(1) - (0.25 * lambda * x(0)) / (mu * (lambda + mu)) + ((lambda + 2. * mu) * x(1)) / (4. * lambda * mu + 4. * (mu * mu)) + x(2) * (-0.5 * (M * M) * (-p0 + 0.5 * (-x(0) - x(1))) - 0.5 * (M * M) * (beta * p0 + 0.5 * (-x(0) - x(1))) + (1. + 2. * beta) * (-2. * x(0) + 2. * x(1)));
        r(2) = M * M * (-p0 + 0.5 * (-x(0) - x(1))) * (beta * p0 + 0.5 * (-x(0) - x(1))) + (1. + 2. * beta) * (x(0) * x(0) - 2. * x(0) * x(1) + x(1) * x(1));

        if (r.squaredNorm() < (T)1e-14)
            break;

        Matrix<T, 3, 3> J;

        J(0, 0) = x(2) * (2. * (1. + 2. * beta) + 0.5 * (M * M)) + (lambda + 2. * mu) / (4. * lambda * mu + 4. * (mu * mu));
        J(0, 1) = x(2) * (-2. * (1. + 2. * beta) + 0.5 * (M * M)) - (0.25 * lambda) / (mu * (lambda + mu));
        J(0, 2) = -0.5 * (M * M) * (-p0 + 0.5 * (-x(0) - x(1))) - 0.5 * (M * M) * (beta * p0 + 0.5 * (-x(0) - x(1))) + (1. + 2. * beta) * (2. * x(0) - 2. * x(1));
        J(1, 0) = J(0, 1);
        J(1, 1) = x(2) * (2. * (1. + 2. * beta) + 0.5 * (M * M)) + (lambda + 2. * mu) / (4. * lambda * mu + 4. * (mu * mu));
        J(1, 2) = -0.5 * (M * M) * (-p0 + 0.5 * (-x(0) - x(1))) - 0.5 * (M * M) * (beta * p0 + 0.5 * (-x(0) - x(1))) + (1. + 2. * beta) * (-2. * x(0) + 2. * x(1));
        J(2, 0) = J(0, 2);
        J(2, 1) = J(1, 2);
        J(2, 2) = 0;

        x -= J.inverse() * r;
    }

    tau(0) = x(0);
    tau(1) = x(1);

    T z1 = lambda + 2 * mu;
    T z2 = 4 * lambda * mu + 4 * mu * mu;
    TV H;
    H(0) = z1 * tau(0) - lambda * tau(1);
    H(1) = -lambda * tau(0) + z1 * tau(1);
    H /= z2;

    // check yield
    trace_tau = tau.sum();
    p = -trace_tau / (T)dim;
    q2 = sqr(tau(0)) + sqr(tau(1)) - 2 * tau(0) * tau(1); // Mises equivalent stress squared
    y = q2 * (1 + 2 * beta) + sqr(M) * (p + beta * p0) * (p - p0);
    ZIRAN_ASSERT(y <= (T)1e-6, y);

    // tracking logJp for hardening
    if (hardeningOn) {
        logJp += trace_epsilon - H.sum();
    }

    TV exp_H = H.array().exp();
    strain = U * exp_H.asDiagonal() * V.transpose();
    return false;
}

template <class T>
bool ModifiedCamClay<T>::projectStrain(StvkWithHenckyIsotropic<T, 2>& c, Matrix<T, 2, 2>& strain)
{
    using MATH_TOOLS::sqr;
    typedef StvkWithHenckyIsotropic<T, 2> TConst;
    static const int dim = TConst::dim;
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    TM U, V;
    TV sigma;

    // TODO: this is inefficient because next time step updateState will do the svd again!
    singularValueDecomposition(strain, U, sigma, V);

    // hardening
    T mu = c.mu, lambda = c.lambda;
    T p0 = (lambda + 2 * mu) * std::sinh(std::max(-logJp, (T)0));

    TV epsilon = sigma.array().abs().max(1e-4).log();
    T trace_epsilon = epsilon.sum();
    TV tau = (2 * mu) * epsilon + (lambda * trace_epsilon) * TV::Ones();
    T trace_tau = tau.sum();
    T p = -trace_tau / (T)dim;
    T q2 = sqr(tau(0)) + sqr(tau(1)) - (T)2 * tau(0) * tau(1); // Mises equivalent stress squared

    T y = q2 * (1 + 2 * beta) + sqr(M) * (p + beta * p0) * (p - p0);
    if (y <= 0)
        return false;

    // ZIRAN_ASSERT(false, sigma.transpose(), ' ', epsilon.transpose(), ' ', y, ' ', tau.transpose());

    Vector<T, 3> x;
    x << tau(0), tau(1), 0;
    for (int newton_step = 0; newton_step < 100; newton_step++) {
        Vector<T, 3> r;

        r(0) = -epsilon(0) + ((lambda + 2. * mu) * x(0)) / (4. * lambda * mu + 4. * (mu * mu)) + x(2) * (-0.5 * (M * M) * (-p0 + 0.5 * (-x(0) - x(1))) - 0.5 * (M * M) * (beta * p0 + 0.5 * (-x(0) - x(1))) + (1. + 2. * beta) * (2. * x(0) - 2. * x(1))) - (0.25 * lambda * x(1)) / (mu * (lambda + mu));
        r(1) = -epsilon(1) - (0.25 * lambda * x(0)) / (mu * (lambda + mu)) + ((lambda + 2. * mu) * x(1)) / (4. * lambda * mu + 4. * (mu * mu)) + x(2) * (-0.5 * (M * M) * (-p0 + 0.5 * (-x(0) - x(1))) - 0.5 * (M * M) * (beta * p0 + 0.5 * (-x(0) - x(1))) + (1. + 2. * beta) * (-2. * x(0) + 2. * x(1)));
        r(2) = M * M * (-p0 + 0.5 * (-x(0) - x(1))) * (beta * p0 + 0.5 * (-x(0) - x(1))) + (1. + 2. * beta) * (x(0) * x(0) - 2. * x(0) * x(1) + x(1) * x(1));

        if (r.squaredNorm() < (T)1e-14)
            break;

        Matrix<T, 3, 3> J;

        J(0, 0) = x(2) * (2. * (1. + 2. * beta) + 0.5 * (M * M)) + (lambda + 2. * mu) / (4. * lambda * mu + 4. * (mu * mu));
        J(0, 1) = x(2) * (-2. * (1. + 2. * beta) + 0.5 * (M * M)) - (0.25 * lambda) / (mu * (lambda + mu));
        J(0, 2) = -0.5 * (M * M) * (-p0 + 0.5 * (-x(0) - x(1))) - 0.5 * (M * M) * (beta * p0 + 0.5 * (-x(0) - x(1))) + (1. + 2. * beta) * (2. * x(0) - 2. * x(1));
        J(1, 0) = J(0, 1);
        J(1, 1) = x(2) * (2. * (1. + 2. * beta) + 0.5 * (M * M)) + (lambda + 2. * mu) / (4. * lambda * mu + 4. * (mu * mu));
        J(1, 2) = -0.5 * (M * M) * (-p0 + 0.5 * (-x(0) - x(1))) - 0.5 * (M * M) * (beta * p0 + 0.5 * (-x(0) - x(1))) + (1. + 2. * beta) * (-2. * x(0) + 2. * x(1));
        J(2, 0) = J(0, 2);
        J(2, 1) = J(1, 2);
        J(2, 2) = 0;

        x -= J.inverse() * r;
    }

    tau(0) = x(0);
    tau(1) = x(1);

    T z1 = lambda + 2 * mu;
    T z2 = 4 * lambda * mu + 4 * mu * mu;
    TV H;
    H(0) = z1 * tau(0) - lambda * tau(1);
    H(1) = -lambda * tau(0) + z1 * tau(1);
    H /= z2;

    // check yield
    trace_tau = tau.sum();
    p = -trace_tau / (T)dim;
    q2 = sqr(tau(0)) + sqr(tau(1)) - 2 * tau(0) * tau(1); // Mises equivalent stress squared
    y = q2 * (1 + 2 * beta) + sqr(M) * (p + beta * p0) * (p - p0);
    ZIRAN_ASSERT(y <= (T)1e-6, y);

    // tracking logJp for hardening
    if (hardeningOn) {
        logJp += trace_epsilon - H.sum();
    }

    TV exp_H = H.array().exp();
    strain = U * exp_H.asDiagonal() * V.transpose();
    return false;
}

template <class T>
void CamClayReturnMapping(T& p, T& q, T trace_epsilon, T norm_eps_hat, T M, T p0, T beta, T mu, T lambda)
{
    using TV = Vector<T, 3>;
    using MATH_TOOLS::rsqrt;
    using std::abs;
    using std::max;
    using std::min;
    using std::sqrt;
    T bulk_modulus = lambda + T(2) / 3 * mu;
    p = -bulk_modulus * trace_epsilon;
    q = sqrt(T(6)) * norm_eps_hat * mu;

    T y = M * M * (p - p0) * (p + beta * p0) + (1 + 2 * beta) * (q * q);

    if (y > 0) {
        T max_p = p0;
        T min_p = -beta * max_p;
        T e = 1 + 2 * beta;
        T max_q = T(0.5) * M * max_p * (1 + beta) * rsqrt(e);

        p = max(min(p, max_p), min_p);
        q = min(q, max_q);
        if (max_q < T(1e-10)) {
            // Too small to project properly
            return;
        }
        T scale = max(-min_p, max(max_p, max_q));
        T scale_inverse = 1 / scale;

        p0 *= scale_inverse;
        trace_epsilon *= scale_inverse;
        norm_eps_hat *= scale_inverse;
        p *= scale_inverse;
        q *= scale_inverse;

        // Project
        T a = 1 / (3 * mu);
        T b = 1 / bulk_modulus;
        T c = trace_epsilon;
        T d = -sqrt(T(2) / 3) * norm_eps_hat;
        T f = M * M;

        T gamma = 0;

        TV r;
        for (int iter = 0; iter < 40; iter++) {
            T d1 = (p - p0);
            T d2 = (p + beta * p0);
            T A13 = f * (d1 + d2);
            T A22 = a + 2 * e * gamma;
            r = TV(c + b * p + A13 * gamma,
                d + q * A22,
                f * d1 * d2 + e * (q * q));

            T A11 = b + 2 * f * gamma;
            T A23 = 2 * e * q;
            T neg_det = (A13 * A13 * A22 + A11 * A23 * A23);
            TV step(-(A23 * A23 * r(0)) + A13 * A23 * r(1) - A13 * A22 * r(2),
                A13 * A23 * r(0) - A13 * A13 * r(1) - A11 * A23 * r(2),
                -(A13 * A22 * r(0)) - A11 * A23 * r(1) + A11 * A22 * r(2));

            if (abs(neg_det) <= T(1e-6))
                step = T(-0.001) * r;
            else
                step = step / neg_det;
            p += step(0);
            q += step(1);
            gamma += step(2);
        }
        p0 *= scale;
        p *= scale;
        q *= scale;

        p = max(min(p, max_p), min_p);
        q = min(abs(q), max_q);
        assert((M * M * (p - p0) * (p + beta * p0) + (1 + 2 * beta) * (q * q)) <= T(1e-3));
    }
    assert(std::isfinite(p));
    assert(std::isfinite(q));
}

template <class T>
bool ModifiedCamClay<T>::projectStrain(StvkWithHencky<T, 3>& c, Matrix<T, 3, 3>& strain)
{
    using MATH_TOOLS::sqr;
    using std::sqrt;
    typedef StvkWithHencky<T, 3> TConst;
    static const int dim = TConst::dim;
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    TM U, V;
    TV sigma;

    // TODO: this is inefficient because next time step updateState will do the svd again!
    singularValueDecomposition(strain, U, sigma, V);

    // hardening
    T mu = c.mu, lambda = c.lambda;
    T bulk_modulus = (3 * lambda + 2 * mu) / 3;
    T p0 = bulk_modulus * (T(0.00001) + std::sinh(xi * std::max(-logJp, (T)0)));

    TV epsilon = sigma.array().abs().max(1e-4).log();
    T trace_epsilon = epsilon.sum();
    TV eps_hat = epsilon - TV::Constant(trace_epsilon / 3);
    T norm_eps_hat = eps_hat.norm();
    if (norm_eps_hat > 0)
        eps_hat /= norm_eps_hat;

    T p, q;
    CamClayReturnMapping(p, q, trace_epsilon, norm_eps_hat, M, p0, beta, mu, lambda);

    T ep = p / (3 * lambda + 2 * mu);
    // tracking logJp for hardening
    if (hardeningOn) {
        logJp += trace_epsilon + 3 * ep;
    }
    TV H = sqrt(T(2) / 3) * q / (2 * mu) * eps_hat - TV::Constant(ep, ep, ep);

    TV exp_H = H.array().exp();
    strain = U * exp_H.asDiagonal() * V.transpose();
    return false;
}

template <class T>
bool ModifiedCamClay<T>::projectStrain(StvkWithHenckyIsotropic<T, 3>& c, Matrix<T, 3, 3>& strain)
{
    using MATH_TOOLS::sqr;
    using std::sqrt;
    typedef StvkWithHenckyIsotropic<T, 3> TConst;
    static const int dim = TConst::dim;
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    TM U, V;
    TV sigma;

    // TODO: this is inefficient because next time step updateState will do the svd again!
    singularValueDecomposition(strain, U, sigma, V);

    // hardening
    T mu = c.mu, lambda = c.lambda;
    T bulk_modulus = (3 * lambda + 2 * mu) / 3;
    T p0 = bulk_modulus * (T(0.00001) + std::sinh(xi * std::max(-logJp, (T)0)));

    TV epsilon = sigma.array().abs().max(1e-4).log();
    T trace_epsilon = epsilon.sum();
    TV eps_hat = epsilon - TV::Constant(trace_epsilon / 3);
    T norm_eps_hat = eps_hat.norm();
    if (norm_eps_hat > 0)
        eps_hat /= norm_eps_hat;

    T p, q;
    CamClayReturnMapping(p, q, trace_epsilon, norm_eps_hat, M, p0, beta, mu, lambda);

    T ep = p / (3 * lambda + 2 * mu);
    // tracking logJp for hardening
    if (hardeningOn) {
        logJp += trace_epsilon + 3 * ep;
    }
    TV H = sqrt(T(2) / 3) * q / (2 * mu) * eps_hat - TV::Constant(ep, ep, ep);

    TV exp_H = H.array().exp();
    strain = U * exp_H.asDiagonal() * V.transpose();
    return false;
}

template <class T, int dim>
VonMisesStvkHencky<T, dim>::VonMisesStvkHencky(const T yield_stress, const T fail_stress, const T xi)
    : yield_stress(yield_stress)
    , xi(xi)
    , fail_stress(fail_stress)
{
}

template <class T, int dim>
void VonMisesStvkHencky<T, dim>::setParameters(const T yield_stress_in, const T xi_in, const T fail_stress_in)
{
    yield_stress = yield_stress_in;
    xi = xi_in;
    if (fail_stress_in > -1)
        fail_stress = fail_stress_in;
}

// strain s is deformation F
//TODO no support for secondary cuts yet.
template <class T, int dim>
template <class TConst>
bool VonMisesStvkHencky<T, dim>::projectStrain(TConst& c, Matrix<T, TConst::dim, TConst::dim>& strain)
{
    static_assert(dim == TConst::dim, "Plasticity model has a different dimension as the Constitutive model!");
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    TM U, V;
    TV sigma;

    // TODO: this is inefficient because next time step updateState will do the svd again!
    singularValueDecomposition(strain, U, sigma, V);

    //TV epsilon = sigma.array().log();
    TV epsilon = sigma.array().max(1e-4).log(); //TODO: need the max part?
    T trace_epsilon = epsilon.sum();
    TV epsilon_hat = epsilon - (trace_epsilon / (T)dim) * TV::Ones();
    T epsilon_hat_squared_norm = epsilon_hat.squaredNorm();
    T epsilon_hat_norm = std::sqrt(epsilon_hat_squared_norm);
    T delta_gamma = epsilon_hat_norm - yield_stress / (2 * c.mu);
    if (delta_gamma <= 0) // case I
    {
        return false;
    }
    //hardening
    yield_stress -= xi * delta_gamma; //supposed to only increase yield_stress
    //yield_stress = std::max((T)0, yield_stress);

    TV H = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat; // case II
    T tau_0 = 2 * c.mu * H(0) + c.lambda * H.sum();
    TV exp_H = H.array().exp();
    strain = U * exp_H.asDiagonal() * V.transpose();

    if (tau_0 >= fail_stress) {
        broken = true;
        crack_normal = V.col(0);
    }

    return false;
}

// strain s is deformation F
template <class T, int dim>
bool VonMisesStvkHencky<T, dim>::projectStrain(StvkWithHenckyWithFp<T, dim>& c, Matrix<T, dim, dim>& strain)
{
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    TM U, V;
    TV sigma;

    if (broken) {
        return false;
    }

    TM Fe_tr = strain * c.Fp_inv;
    // TODO: this is inefficient because next time step updateState will do the svd again!
    singularValueDecomposition(Fe_tr, U, sigma, V);

    //TV epsilon = sigma.array().log();
    TV epsilon = sigma.array().max(1e-4).log();
    T trace_epsilon = epsilon.sum();
    TV epsilon_hat = epsilon - (trace_epsilon / (T)dim) * TV::Ones();
    T epsilon_hat_squared_norm = epsilon_hat.squaredNorm();
    T epsilon_hat_norm = std::sqrt(epsilon_hat_squared_norm);
    T delta_gamma = epsilon_hat_norm - yield_stress / (2 * c.mu);
    if (delta_gamma <= 0) // case I
    {
        return false;
    }
    TV H = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat; // case II
    T tau_0 = 2 * c.mu * H(0) + c.lambda * H.sum();

    //hardening
    yield_stress -= xi * delta_gamma;

    TV exp_H = H.array().exp();
    TM F_e = U * exp_H.asDiagonal() * V.transpose();
    c.Fp_inv = strain.inverse() * F_e;

    if (tau_0 >= fail_stress) {
        broken = true;
        crack_normal = V.col(0);
        c.damage_scale = 0;
    }
    return false;
}

//TODO this is not up to date
template <class T, int dim>
template <class TConst>
void VonMisesStvkHencky<T, dim>::projectStrainDiagonal(TConst& c, Vector<T, TConst::dim>& sigma)
{
    static_assert(dim == TConst::dim, "Plasticity model has a different dimensiona s the Constitutive model!");
    typedef Vector<T, dim> TV;

    //TV epsilon = sigma.array().log();
    TV epsilon = sigma.array().max(1e-4).log();
    T trace_epsilon = epsilon.sum();
    TV epsilon_hat = epsilon - (trace_epsilon / (T)dim) * TV::Ones();
    T epsilon_hat_squared_norm = epsilon_hat.squaredNorm();
    T epsilon_hat_norm = std::sqrt(epsilon_hat_squared_norm);
    T delta_gamma = epsilon_hat_norm - yield_stress / (2 * c.mu);
    if (delta_gamma <= 0) // case I
    {
        return;
    }
    TV H = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat; // case II
    sigma = H.array().exp();
}

template <class T, int dim>
template <class TConst>
Vector<T, TConst::dim> VonMisesStvkHencky<T, dim>::projectSigma(TConst& c, const Vector<T, TConst::dim>& sigma)
{
    static_assert(dim == TConst::dim, "Plasticity model has a different dimensiona s the Constitutive model!");
    typedef Vector<T, dim> TV;

    //TV epsilon = sigma.array().log();
    TV epsilon = sigma.array().max(1e-4).log();
    T trace_epsilon = epsilon.sum();
    TV epsilon_hat = epsilon - (trace_epsilon / (T)dim) * TV::Ones();
    T epsilon_hat_squared_norm = epsilon_hat.squaredNorm();
    T epsilon_hat_norm = std::sqrt(epsilon_hat_squared_norm);
    T delta_gamma = epsilon_hat_norm - yield_stress / (2 * c.mu);
    if (delta_gamma <= 0) // case I
    {
        return sigma;
    }
    TV H = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat; // case II
    TV ret = H.array().exp();
    return ret;
}

template <class T, int dim>
template <class TConst>
Matrix<T, TConst::dim, TConst::dim> VonMisesStvkHencky<T, dim>::projectSigmaDerivative(TConst& c, const Vector<T, TConst::dim>& sigma)
{
    static_assert(dim == TConst::dim, "Plasticity model has a different dimensiona s the Constitutive model!");
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;

    //TV epsilon = sigma.array().log();
    TV epsilon = sigma.array().max(1e-4).log();
    T trace_epsilon = epsilon.sum();
    TV epsilon_hat = epsilon - (trace_epsilon / (T)dim) * TV::Ones();
    T epsilon_hat_squared_norm = epsilon_hat.squaredNorm();
    T epsilon_hat_norm = std::sqrt(epsilon_hat_squared_norm);
    T delta_gamma = epsilon_hat_norm - yield_stress / (2 * c.mu);
    if (delta_gamma <= 0) // case I
    {
        return TM::Identity();
    }
    TV w = sigma.array().inverse();
    T k = trace_epsilon;
    TV s = epsilon - k / dim * TV::Ones();
    TV s_hat = s / s.norm();
    TV H = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat; // case II
    TV Z_hat = H.array().exp();
    TM ret = Z_hat.asDiagonal() * (TM::Identity() * w.asDiagonal() - ((T)1 - yield_stress / (2 * c.mu * s.norm())) * (TM::Identity() * w.asDiagonal() - TV::Ones() * w.transpose() / (T)dim) - yield_stress / (2 * c.mu * s.norm()) * s_hat * s_hat.transpose() * w.asDiagonal());
    return ret;
}

template <class T, int dim>
template <class TConst>
void VonMisesStvkHencky<T, dim>::projectSigmaAndDerivative(TConst& c, const Vector<T, TConst::dim>& sigma, Vector<T, TConst::dim>& projectedSigma, Matrix<T, TConst::dim, TConst::dim>& projectedSigmaDerivative)
{
    static_assert(dim == TConst::dim, "Plasticity model has a different dimensiona s the Constitutive model!");
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;

    //TV epsilon = sigma.array().log();
    TV epsilon = sigma.array().max(1e-4).log();
    T trace_epsilon = epsilon.sum();
    TV epsilon_hat = epsilon - (trace_epsilon / (T)dim) * TV::Ones();
    T epsilon_hat_squared_norm = epsilon_hat.squaredNorm();
    T epsilon_hat_norm = std::sqrt(epsilon_hat_squared_norm);
    T delta_gamma = epsilon_hat_norm - yield_stress / (2 * c.mu);
    if (delta_gamma <= 0) // case I
    {
        projectedSigma = sigma;
        projectedSigmaDerivative = TM::Identity();
        return;
    }
    TV w = sigma.array().inverse();
    T k = trace_epsilon;
    TV s = epsilon - k / dim * TV::Ones();
    TV s_hat = s / s.norm();
    TV H = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat; // case II
    projectedSigma = H.array().exp();
    projectedSigmaDerivative = projectedSigma.asDiagonal() * (TM::Identity() * w.asDiagonal() - ((T)1 - yield_stress / (2 * c.mu * s.norm())) * (TM::Identity() * w.asDiagonal() - TV::Ones() * w.transpose() / (T)dim) - yield_stress / (2 * c.mu * s.norm()) * s_hat * s_hat.transpose() * w.asDiagonal());
}

template <class T, int dim>
template <class TConst>
void VonMisesStvkHencky<T, dim>::computeSigmaPInverse(TConst& c, const Vector<T, TConst::dim>& sigma_e, Vector<T, TConst::dim>& sigma_p_inv)
{
    using TV = typename TConst::TV;
    TV eps = sigma_e.array().log();
    TV tau = c.lambda * eps.sum() + 2 * c.mu * eps.array();
    TV tau_trfree;
    tau_trfree.array() = tau.array() - ((T)1 / TConst::dim) * tau.sum();

    T tau_trfree_norm = tau_trfree.norm();
    if (tau_trfree_norm > yield_stress) {
        // Lambda = 1 / (2*dt*mu*(1+ k/(tau_trefree.norm()-k)))*tau_trfree
        sigma_p_inv = tau_trfree / tau_trfree_norm;
        sigma_p_inv *= (yield_stress - tau_trfree_norm) / (2.0 * c.mu);
        sigma_p_inv = sigma_p_inv.array().exp();
    }
    else
        sigma_p_inv = TV::Ones();
}

template <class T, int dim>
const char* VonMisesStvkHencky<T, dim>::name()
{
    return "VonMisesStvkHencky";
}

template <class T>
VonMisesCapped<T>::VonMisesCapped(const T k1_compress, const T k1_stretch, const T k2)
    : k1_compress(k1_compress)
    , k1_stretch(k1_stretch)
    , k2(k2)
{
}

// strain s is deformation F
template <class T>
template <class TConst>
bool VonMisesCapped<T>::projectStrain(TConst& c, Matrix<T, TConst::dim, TConst::dim>& strain)
{
    static const int dim = TConst::dim;
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    TM U, V;
    TV sigma;

    // TODO: this is inefficient because next time step updateState will do the svd again!
    singularValueDecomposition(strain, U, sigma, V);

    TV epsilon = sigma.array().log();
    T trace_epsilon = epsilon.sum();
    TV epsilon_hat = epsilon - (trace_epsilon / (T)dim) * TV::Ones();
    T epsilon_hat_squared_norm = epsilon_hat.squaredNorm();
    T epsilon_hat_norm = std::sqrt(epsilon_hat_squared_norm);
    T delta_gamma = epsilon_hat_norm - k2 / (2 * c.mu);

    if (delta_gamma > 0) {
        TV H = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat;
        sigma = H.array().exp();
    }

    if (trace_epsilon > k1_stretch / (dim * c.lambda + 2 * c.mu))
        sigma *= exp(k1_stretch / (dim * dim * c.lambda + 2 * dim * c.mu) - trace_epsilon / dim);
    else if (trace_epsilon < -k1_compress / (dim * c.lambda + 2 * c.mu))
        sigma *= exp(-k1_compress / (dim * dim * c.lambda + 2 * dim * c.mu) - trace_epsilon / dim);

    strain = U * sigma.asDiagonal() * V.transpose();
    return false;
}

template <class T>
template <class TConst>
void VonMisesCapped<T>::projectStrainDiagonal(TConst& c, Vector<T, TConst::dim>& sigma)
{
    static const int dim = TConst::dim;
    typedef Vector<T, dim> TV;

    TV epsilon = sigma.array().log();
    T trace_epsilon = epsilon.sum();
    TV epsilon_hat = epsilon - (trace_epsilon / (T)dim) * TV::Ones();
    T epsilon_hat_squared_norm = epsilon_hat.squaredNorm();
    T epsilon_hat_norm = std::sqrt(epsilon_hat_squared_norm);
    T delta_gamma = epsilon_hat_norm - k2 / (2 * c.mu);

    if (delta_gamma > 0) {
        TV H = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat;
        sigma = H.array().exp();
    }

    if (trace_epsilon > k1_stretch / (dim * c.lambda + 2 * c.mu))
        sigma *= exp(k1_stretch / (dim * dim * c.lambda + 2 * dim * c.mu) - trace_epsilon / dim);
    else if (trace_epsilon < -k1_compress / (dim * c.lambda + 2 * c.mu))
        sigma *= exp(-k1_compress / (dim * dim * c.lambda + 2 * dim * c.mu) - trace_epsilon / dim);

    return;
}

template <class T>
template <class TConst>
void VonMisesCapped<T>::computeSigmaPInverse(TConst& c, const Vector<T, TConst::dim>& sigma_e, Vector<T, TConst::dim>& sigma_p_inv)
{
    using TV = typename TConst::TV;
    TV epsilon = sigma_e.array().log();
    T trace_epsilon = epsilon.sum();
    TV epsilon_hat = epsilon - TV::Constant(trace_epsilon / (T)TConst::dim);
    T epsilon_hat_norm = epsilon.norm();
    T delta_gamma = epsilon_hat_norm - k2 / ((T)2 * c.mu);

    sigma_p_inv = TV::Ones();
    if (delta_gamma > 0) {
        TV H = (-delta_gamma / epsilon_hat_norm) * epsilon_hat;
        sigma_p_inv = H.array().exp();
    }

    if (trace_epsilon > k1_stretch / (TConst::dim * c.lambda + 2 * c.mu))
        sigma_p_inv *= exp(k1_stretch / (TConst::dim * TConst::dim * c.lambda + 2 * TConst::dim * c.mu) - trace_epsilon / TConst::dim);
    else if (trace_epsilon < -k1_compress / (TConst::dim * c.lambda + 2 * c.mu))
        sigma_p_inv *= exp(-k1_compress / (TConst::dim * TConst::dim * c.lambda + 2 * TConst::dim * c.mu) - trace_epsilon / TConst::dim);
}

template <class T>
const char* VonMisesCapped<T>::name()
{
    return "VonMisesCapped";
}

template class ModifiedCamClay<double>;
template class ModifiedCamClay<float>;
template class UnilateralJ<double>;
template class UnilateralJ<float>;
template class SnowPlasticity<double>;
template class SnowPlasticity<float>;
template class NonAssociativeCamClay<double>;
template class NonAssociativeCamClay<float>;
template class NonAssociativeVonMises<double>;
template class NonAssociativeVonMises<float>;
template class NonAssociativeDruckerPrager<double>;
template class NonAssociativeDruckerPrager<float>;

template class DruckerPragerStvkHencky<double>;
template class DruckerPragerStvkHencky<float>;

template class SmudgePlasticity<double>;
template class SmudgePlasticity<float>;

template class DruckerPragerStrainSoftening<double>;
template class DruckerPragerStrainSoftening<float>;

template class VonMisesStvkHencky<double, 2>;
template class VonMisesStvkHencky<float, 2>;
template class VonMisesStvkHencky<double, 3>;
template class VonMisesStvkHencky<float, 3>;

template class VonMisesCapped<double>;
template class VonMisesCapped<float>;

template class PlasticityApplier<CorotatedIsotropic<double, 2>, SnowPlasticity<double>, Eigen::Matrix<double, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<CorotatedIsotropic<double, 3>, SnowPlasticity<double>, Eigen::Matrix<double, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<CorotatedIsotropic<float, 2>, SnowPlasticity<float>, Eigen::Matrix<float, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<CorotatedIsotropic<float, 3>, SnowPlasticity<float>, Eigen::Matrix<float, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<CorotatedElasticity<double, 2>, SnowPlasticity<double>, Eigen::Matrix<double, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<CorotatedElasticity<double, 3>, SnowPlasticity<double>, Eigen::Matrix<double, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<CorotatedElasticity<float, 2>, SnowPlasticity<float>, Eigen::Matrix<float, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<CorotatedElasticity<float, 3>, SnowPlasticity<float>, Eigen::Matrix<float, 3, 3, 0, 3, 3>>;

template class PlasticityApplier<LinearCorotated<double, 2>, SnowPlasticity<double>, Eigen::Matrix<double, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<LinearCorotated<double, 3>, SnowPlasticity<double>, Eigen::Matrix<double, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<LinearCorotated<float, 2>, SnowPlasticity<float>, Eigen::Matrix<float, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<LinearCorotated<float, 3>, SnowPlasticity<float>, Eigen::Matrix<float, 3, 3, 0, 3, 3>>;

template class PlasticityApplier<NeoHookean<double, 2>, SnowPlasticity<double>, Eigen::Matrix<double, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<NeoHookean<double, 3>, SnowPlasticity<double>, Eigen::Matrix<double, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<NeoHookean<float, 2>, SnowPlasticity<float>, Eigen::Matrix<float, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<NeoHookean<float, 3>, SnowPlasticity<float>, Eigen::Matrix<float, 3, 3, 0, 3, 3>>;

template class PlasticityApplier<NeoHookeanBorden<double, 2>, NonAssociativeCamClay<double>, Eigen::Matrix<double, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<NeoHookeanBorden<double, 3>, NonAssociativeCamClay<double>, Eigen::Matrix<double, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<NeoHookeanBorden<float, 2>, NonAssociativeCamClay<float>, Eigen::Matrix<float, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<NeoHookeanBorden<float, 3>, NonAssociativeCamClay<float>, Eigen::Matrix<float, 3, 3, 0, 3, 3>>;

template class PlasticityApplier<NeoHookeanBorden<double, 2>, NonAssociativeVonMises<double>, Eigen::Matrix<double, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<NeoHookeanBorden<double, 3>, NonAssociativeVonMises<double>, Eigen::Matrix<double, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<NeoHookeanBorden<float, 2>, NonAssociativeVonMises<float>, Eigen::Matrix<float, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<NeoHookeanBorden<float, 3>, NonAssociativeVonMises<float>, Eigen::Matrix<float, 3, 3, 0, 3, 3>>;

template class PlasticityApplier<NeoHookeanBorden<double, 2>, NonAssociativeDruckerPrager<double>, Eigen::Matrix<double, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<NeoHookeanBorden<double, 3>, NonAssociativeDruckerPrager<double>, Eigen::Matrix<double, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<NeoHookeanBorden<float, 2>, NonAssociativeDruckerPrager<float>, Eigen::Matrix<float, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<NeoHookeanBorden<float, 3>, NonAssociativeDruckerPrager<float>, Eigen::Matrix<float, 3, 3, 0, 3, 3>>;

template class PlasticityApplier<StvkWithHencky<double, 2>, ModifiedCamClay<double>, Eigen::Matrix<double, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<StvkWithHencky<float, 2>, ModifiedCamClay<float>, Eigen::Matrix<float, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<StvkWithHencky<double, 3>, ModifiedCamClay<double>, Eigen::Matrix<double, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<StvkWithHencky<float, 3>, ModifiedCamClay<float>, Eigen::Matrix<float, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<EquationOfState<double, 2>, UnilateralJ<double>, double>;
template class PlasticityApplier<EquationOfState<float, 2>, UnilateralJ<float>, float>;
template class PlasticityApplier<EquationOfState<double, 3>, UnilateralJ<double>, double>;
template class PlasticityApplier<EquationOfState<float, 3>, UnilateralJ<float>, float>;

template class PlasticityApplier<StvkWithHencky<double, 2>, DruckerPragerStvkHencky<double>, Eigen::Matrix<double, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<StvkWithHencky<double, 2>, DruckerPragerStrainSoftening<double>, Eigen::Matrix<double, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<StvkWithHencky<double, 2>, VonMisesStvkHencky<double, 2>, Eigen::Matrix<double, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<QRStableNeoHookean<double, 2>, VonMisesStvkHencky<double, 2>, Eigen::Matrix<double, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<StvkWithHencky<double, 2>, VonMisesCapped<double>, Eigen::Matrix<double, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<StvkWithHencky<double, 3>, DruckerPragerStvkHencky<double>, Eigen::Matrix<double, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<StvkWithHencky<double, 3>, DruckerPragerStrainSoftening<double>, Eigen::Matrix<double, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<StvkWithHencky<double, 3>, VonMisesStvkHencky<double, 3>, Eigen::Matrix<double, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<QRStableNeoHookean<double, 3>, VonMisesStvkHencky<double, 3>, Eigen::Matrix<double, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<StvkWithHencky<double, 3>, VonMisesCapped<double>, Eigen::Matrix<double, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<StvkWithHencky<float, 2>, DruckerPragerStvkHencky<float>, Eigen::Matrix<float, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<StvkWithHencky<float, 2>, DruckerPragerStrainSoftening<float>, Eigen::Matrix<float, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<StvkWithHenckyDecoupled<double, 2>, VonMisesStvkHencky<double, 2>, Eigen::Matrix<double, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<StvkWithHenckyDecoupled<double, 3>, VonMisesStvkHencky<double, 3>, Eigen::Matrix<double, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<StvkWithHenckyDecoupled<float, 2>, VonMisesStvkHencky<float, 2>, Eigen::Matrix<float, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<StvkWithHenckyDecoupled<float, 3>, VonMisesStvkHencky<float, 3>, Eigen::Matrix<float, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<StvkWithHenckyDecoupled<double, 2>, DruckerPragerStvkHencky<double>, Eigen::Matrix<double, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<StvkWithHenckyDecoupled<double, 3>, DruckerPragerStvkHencky<double>, Eigen::Matrix<double, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<StvkWithHenckyDecoupled<float, 2>, DruckerPragerStvkHencky<float>, Eigen::Matrix<float, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<StvkWithHenckyDecoupled<float, 3>, DruckerPragerStvkHencky<float>, Eigen::Matrix<float, 3, 3, 0, 3, 3>>;

template class PlasticityApplier<StvkWithHencky<float, 2>, VonMisesStvkHencky<float, 2>, Eigen::Matrix<float, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<QRStableNeoHookean<float, 2>, VonMisesStvkHencky<float, 2>, Eigen::Matrix<float, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<StvkWithHencky<float, 2>, VonMisesCapped<float>, Eigen::Matrix<float, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<StvkWithHencky<float, 3>, DruckerPragerStvkHencky<float>, Eigen::Matrix<float, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<StvkWithHencky<float, 3>, DruckerPragerStrainSoftening<float>, Eigen::Matrix<float, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<StvkWithHencky<float, 3>, VonMisesStvkHencky<float, 3>, Eigen::Matrix<float, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<QRStableNeoHookean<float, 3>, VonMisesStvkHencky<float, 3>, Eigen::Matrix<float, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<StvkWithHencky<float, 3>, VonMisesCapped<float>, Eigen::Matrix<float, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<StvkWithHenckyIsotropic<double, 2>, DruckerPragerStvkHencky<double>, Eigen::Matrix<double, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<StvkWithHenckyIsotropic<double, 2>, DruckerPragerStrainSoftening<double>, Eigen::Matrix<double, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<StvkWithHenckyIsotropic<double, 2>, VonMisesStvkHencky<double, 2>, Eigen::Matrix<double, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<StvkWithHenckyIsotropic<double, 2>, VonMisesCapped<double>, Eigen::Matrix<double, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<StvkWithHenckyIsotropic<double, 2>, ModifiedCamClay<double>, Eigen::Matrix<double, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<StvkWithHenckyIsotropic<double, 3>, DruckerPragerStvkHencky<double>, Eigen::Matrix<double, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<StvkWithHenckyIsotropic<double, 3>, DruckerPragerStrainSoftening<double>, Eigen::Matrix<double, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<StvkWithHenckyIsotropic<double, 3>, VonMisesStvkHencky<double, 3>, Eigen::Matrix<double, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<StvkWithHenckyIsotropic<double, 3>, VonMisesCapped<double>, Eigen::Matrix<double, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<StvkWithHenckyIsotropic<double, 3>, ModifiedCamClay<double>, Eigen::Matrix<double, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<StvkWithHenckyIsotropic<float, 2>, DruckerPragerStvkHencky<float>, Eigen::Matrix<float, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<StvkWithHenckyIsotropic<float, 2>, DruckerPragerStrainSoftening<float>, Eigen::Matrix<float, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<StvkWithHenckyIsotropic<float, 2>, VonMisesStvkHencky<float, 2>, Eigen::Matrix<float, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<StvkWithHenckyIsotropic<float, 2>, VonMisesCapped<float>, Eigen::Matrix<float, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<StvkWithHenckyIsotropic<float, 3>, DruckerPragerStvkHencky<float>, Eigen::Matrix<float, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<StvkWithHenckyIsotropic<float, 3>, DruckerPragerStrainSoftening<float>, Eigen::Matrix<float, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<StvkWithHenckyIsotropic<float, 3>, VonMisesStvkHencky<float, 3>, Eigen::Matrix<float, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<StvkWithHenckyIsotropic<float, 3>, VonMisesCapped<float>, Eigen::Matrix<float, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<StvkWithHenckyIsotropicUnilateral<double, 2>, DruckerPragerStvkHencky<double>, Eigen::Matrix<double, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<StvkWithHenckyIsotropicUnilateral<double, 3>, DruckerPragerStvkHencky<double>, Eigen::Matrix<double, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<StvkWithHenckyIsotropicUnilateral<float, 2>, DruckerPragerStvkHencky<float>, Eigen::Matrix<float, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<StvkWithHenckyIsotropicUnilateral<float, 3>, DruckerPragerStvkHencky<float>, Eigen::Matrix<float, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<Smudge<double, 2>, DruckerPragerStvkHencky<double>, Eigen::Matrix<double, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<Smudge<double, 3>, DruckerPragerStvkHencky<double>, Eigen::Matrix<double, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<Smudge<float, 2>, DruckerPragerStvkHencky<float>, Eigen::Matrix<float, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<Smudge<float, 3>, DruckerPragerStvkHencky<float>, Eigen::Matrix<float, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<Smudge<double, 2>, SmudgePlasticity<double>, Eigen::Matrix<double, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<Smudge<double, 3>, SmudgePlasticity<double>, Eigen::Matrix<double, 3, 3, 0, 3, 3>>;
template class PlasticityApplier<Smudge<float, 2>, SmudgePlasticity<float>, Eigen::Matrix<float, 2, 2, 0, 2, 2>>;
template class PlasticityApplier<Smudge<float, 3>, SmudgePlasticity<float>, Eigen::Matrix<float, 3, 3, 0, 3, 3>>;

template bool UnilateralJ<double>::projectStrain<2>(EquationOfState<double, 2>&, double&);
template bool UnilateralJ<double>::projectStrain<3>(EquationOfState<double, 3>&, double&);
template bool UnilateralJ<float>::projectStrain<2>(EquationOfState<float, 2>&, float&);
template bool UnilateralJ<float>::projectStrain<3>(EquationOfState<float, 3>&, float&);

template void SnowPlasticity<double>::projectStrainDiagonal<CorotatedElasticity<double, 2>>(CorotatedElasticity<double, 2>&, Eigen::Matrix<double, CorotatedElasticity<double, 2>::dim, 1, 0, CorotatedElasticity<double, 2>::dim, 1>&);
template void SnowPlasticity<double>::projectStrainDiagonal<CorotatedElasticity<double, 3>>(CorotatedElasticity<double, 3>&, Eigen::Matrix<double, CorotatedElasticity<double, 3>::dim, 1, 0, CorotatedElasticity<double, 3>::dim, 1>&);
template void SnowPlasticity<double>::projectStrainDiagonal<NeoHookean<double, 2>>(NeoHookean<double, 2>&, Eigen::Matrix<double, NeoHookean<double, 2>::dim, 1, 0, NeoHookean<double, 2>::dim, 1>&);
template void SnowPlasticity<double>::projectStrainDiagonal<NeoHookean<double, 3>>(NeoHookean<double, 3>&, Eigen::Matrix<double, NeoHookean<double, 3>::dim, 1, 0, NeoHookean<double, 3>::dim, 1>&);
template void SnowPlasticity<double>::projectStrainDiagonal<StVenantKirchhoff<double, 2>>(StVenantKirchhoff<double, 2>&, Eigen::Matrix<double, StVenantKirchhoff<double, 2>::dim, 1, 0, StVenantKirchhoff<double, 2>::dim, 1>&);
template void SnowPlasticity<double>::projectStrainDiagonal<StVenantKirchhoff<double, 3>>(StVenantKirchhoff<double, 3>&, Eigen::Matrix<double, StVenantKirchhoff<double, 3>::dim, 1, 0, StVenantKirchhoff<double, 3>::dim, 1>&);
template void SnowPlasticity<double>::projectStrainDiagonal<StvkWithHencky<double, 2>>(StvkWithHencky<double, 2>&, Eigen::Matrix<double, StvkWithHencky<double, 2>::dim, 1, 0, StvkWithHencky<double, 2>::dim, 1>&);
template void SnowPlasticity<double>::projectStrainDiagonal<StvkWithHencky<double, 3>>(StvkWithHencky<double, 3>&, Eigen::Matrix<double, StvkWithHencky<double, 3>::dim, 1, 0, StvkWithHencky<double, 3>::dim, 1>&);
template void SnowPlasticity<float>::projectStrainDiagonal<CorotatedElasticity<float, 2>>(CorotatedElasticity<float, 2>&, Eigen::Matrix<float, CorotatedElasticity<float, 2>::dim, 1, 0, CorotatedElasticity<float, 2>::dim, 1>&);
template void SnowPlasticity<float>::projectStrainDiagonal<CorotatedElasticity<float, 3>>(CorotatedElasticity<float, 3>&, Eigen::Matrix<float, CorotatedElasticity<float, 3>::dim, 1, 0, CorotatedElasticity<float, 3>::dim, 1>&);
template void SnowPlasticity<float>::projectStrainDiagonal<NeoHookean<float, 2>>(NeoHookean<float, 2>&, Eigen::Matrix<float, NeoHookean<float, 2>::dim, 1, 0, NeoHookean<float, 2>::dim, 1>&);
template void SnowPlasticity<float>::projectStrainDiagonal<NeoHookean<float, 3>>(NeoHookean<float, 3>&, Eigen::Matrix<float, NeoHookean<float, 3>::dim, 1, 0, NeoHookean<float, 3>::dim, 1>&);
template void SnowPlasticity<float>::projectStrainDiagonal<StVenantKirchhoff<float, 2>>(StVenantKirchhoff<float, 2>&, Eigen::Matrix<float, StVenantKirchhoff<float, 2>::dim, 1, 0, StVenantKirchhoff<float, 2>::dim, 1>&);
template void SnowPlasticity<float>::projectStrainDiagonal<StVenantKirchhoff<float, 3>>(StVenantKirchhoff<float, 3>&, Eigen::Matrix<float, StVenantKirchhoff<float, 3>::dim, 1, 0, StVenantKirchhoff<float, 3>::dim, 1>&);
template void SnowPlasticity<float>::projectStrainDiagonal<StvkWithHencky<float, 2>>(StvkWithHencky<float, 2>&, Eigen::Matrix<float, StvkWithHencky<float, 2>::dim, 1, 0, StvkWithHencky<float, 2>::dim, 1>&);
template void SnowPlasticity<float>::projectStrainDiagonal<StvkWithHencky<float, 3>>(StvkWithHencky<float, 3>&, Eigen::Matrix<float, StvkWithHencky<float, 3>::dim, 1, 0, StvkWithHencky<float, 3>::dim, 1>&);

template void VonMisesStvkHencky<double, 2>::projectStrainDiagonal<StvkWithHencky<double, 2>>(StvkWithHencky<double, 2>&, Eigen::Matrix<double, StvkWithHencky<double, 2>::dim, 1, 0, StvkWithHencky<double, 2>::dim, 1>&);
template void VonMisesStvkHencky<double, 3>::projectStrainDiagonal<StvkWithHencky<double, 3>>(StvkWithHencky<double, 3>&, Eigen::Matrix<double, StvkWithHencky<double, 3>::dim, 1, 0, StvkWithHencky<double, 3>::dim, 1>&);
template void VonMisesStvkHencky<float, 2>::projectStrainDiagonal<StvkWithHencky<float, 2>>(StvkWithHencky<float, 2>&, Eigen::Matrix<float, StvkWithHencky<float, 2>::dim, 1, 0, StvkWithHencky<float, 2>::dim, 1>&);
template void VonMisesStvkHencky<float, 3>::projectStrainDiagonal<StvkWithHencky<float, 3>>(StvkWithHencky<float, 3>&, Eigen::Matrix<float, StvkWithHencky<float, 3>::dim, 1, 0, StvkWithHencky<float, 3>::dim, 1>&);
template void VonMisesStvkHencky<double, 2>::computeSigmaPInverse<StvkWithHencky<double, 2>>(StvkWithHencky<double, 2>&, Eigen::Matrix<double, StvkWithHencky<double, 2>::dim, 1, 0, StvkWithHencky<double, 2>::dim, 1> const&, Eigen::Matrix<double, StvkWithHencky<double, 2>::dim, 1, 0, StvkWithHencky<double, 2>::dim, 1>&);
template void VonMisesStvkHencky<double, 3>::computeSigmaPInverse<StvkWithHencky<double, 3>>(StvkWithHencky<double, 3>&, Eigen::Matrix<double, StvkWithHencky<double, 3>::dim, 1, 0, StvkWithHencky<double, 3>::dim, 1> const&, Eigen::Matrix<double, StvkWithHencky<double, 3>::dim, 1, 0, StvkWithHencky<double, 3>::dim, 1>&);
template void VonMisesStvkHencky<float, 2>::computeSigmaPInverse<StvkWithHencky<float, 2>>(StvkWithHencky<float, 2>&, Eigen::Matrix<float, StvkWithHencky<float, 2>::dim, 1, 0, StvkWithHencky<float, 2>::dim, 1> const&, Eigen::Matrix<float, StvkWithHencky<float, 2>::dim, 1, 0, StvkWithHencky<float, 2>::dim, 1>&);
template void VonMisesStvkHencky<float, 3>::computeSigmaPInverse<StvkWithHencky<float, 3>>(StvkWithHencky<float, 3>&, Eigen::Matrix<float, StvkWithHencky<float, 3>::dim, 1, 0, StvkWithHencky<float, 3>::dim, 1> const&, Eigen::Matrix<float, StvkWithHencky<float, 3>::dim, 1, 0, StvkWithHencky<float, 3>::dim, 1>&);

template void VonMisesStvkHencky<double, 2>::projectStrainDiagonal<QRStableNeoHookean<double, 2>>(QRStableNeoHookean<double, 2>&, Eigen::Matrix<double, QRStableNeoHookean<double, 2>::dim, 1, 0, QRStableNeoHookean<double, 2>::dim, 1>&);
template void VonMisesStvkHencky<double, 3>::projectStrainDiagonal<QRStableNeoHookean<double, 3>>(QRStableNeoHookean<double, 3>&, Eigen::Matrix<double, QRStableNeoHookean<double, 3>::dim, 1, 0, QRStableNeoHookean<double, 3>::dim, 1>&);
template void VonMisesStvkHencky<float, 2>::projectStrainDiagonal<QRStableNeoHookean<float, 2>>(QRStableNeoHookean<float, 2>&, Eigen::Matrix<float, QRStableNeoHookean<float, 2>::dim, 1, 0, QRStableNeoHookean<float, 2>::dim, 1>&);
template void VonMisesStvkHencky<float, 3>::projectStrainDiagonal<QRStableNeoHookean<float, 3>>(QRStableNeoHookean<float, 3>&, Eigen::Matrix<float, QRStableNeoHookean<float, 3>::dim, 1, 0, QRStableNeoHookean<float, 3>::dim, 1>&);
template void VonMisesStvkHencky<double, 2>::computeSigmaPInverse<QRStableNeoHookean<double, 2>>(QRStableNeoHookean<double, 2>&, Eigen::Matrix<double, QRStableNeoHookean<double, 2>::dim, 1, 0, QRStableNeoHookean<double, 2>::dim, 1> const&, Eigen::Matrix<double, QRStableNeoHookean<double, 2>::dim, 1, 0, QRStableNeoHookean<double, 2>::dim, 1>&);
template void VonMisesStvkHencky<double, 3>::computeSigmaPInverse<QRStableNeoHookean<double, 3>>(QRStableNeoHookean<double, 3>&, Eigen::Matrix<double, QRStableNeoHookean<double, 3>::dim, 1, 0, QRStableNeoHookean<double, 3>::dim, 1> const&, Eigen::Matrix<double, QRStableNeoHookean<double, 3>::dim, 1, 0, QRStableNeoHookean<double, 3>::dim, 1>&);
template void VonMisesStvkHencky<float, 2>::computeSigmaPInverse<QRStableNeoHookean<float, 2>>(QRStableNeoHookean<float, 2>&, Eigen::Matrix<float, QRStableNeoHookean<float, 2>::dim, 1, 0, QRStableNeoHookean<float, 2>::dim, 1> const&, Eigen::Matrix<float, QRStableNeoHookean<float, 2>::dim, 1, 0, QRStableNeoHookean<float, 2>::dim, 1>&);
template void VonMisesStvkHencky<float, 3>::computeSigmaPInverse<QRStableNeoHookean<float, 3>>(QRStableNeoHookean<float, 3>&, Eigen::Matrix<float, QRStableNeoHookean<float, 3>::dim, 1, 0, QRStableNeoHookean<float, 3>::dim, 1> const&, Eigen::Matrix<float, QRStableNeoHookean<float, 3>::dim, 1, 0, QRStableNeoHookean<float, 3>::dim, 1>&);

template void VonMisesCapped<double>::projectStrainDiagonal<StvkWithHencky<double, 2>>(StvkWithHencky<double, 2>&, Eigen::Matrix<double, StvkWithHencky<double, 2>::dim, 1, 0, StvkWithHencky<double, 2>::dim, 1>&);
template void VonMisesCapped<double>::projectStrainDiagonal<StvkWithHencky<double, 3>>(StvkWithHencky<double, 3>&, Eigen::Matrix<double, StvkWithHencky<double, 3>::dim, 1, 0, StvkWithHencky<double, 3>::dim, 1>&);
template void VonMisesCapped<float>::projectStrainDiagonal<StvkWithHencky<float, 2>>(StvkWithHencky<float, 2>&, Eigen::Matrix<float, StvkWithHencky<float, 2>::dim, 1, 0, StvkWithHencky<float, 2>::dim, 1>&);
template void VonMisesCapped<float>::projectStrainDiagonal<StvkWithHencky<float, 3>>(StvkWithHencky<float, 3>&, Eigen::Matrix<float, StvkWithHencky<float, 3>::dim, 1, 0, StvkWithHencky<float, 3>::dim, 1>&);
template void VonMisesCapped<double>::computeSigmaPInverse<StvkWithHencky<double, 2>>(StvkWithHencky<double, 2>&, Eigen::Matrix<double, StvkWithHencky<double, 2>::dim, 1, 0, StvkWithHencky<double, 2>::dim, 1> const&, Eigen::Matrix<double, StvkWithHencky<double, 2>::dim, 1, 0, StvkWithHencky<double, 2>::dim, 1>&);
template void VonMisesCapped<double>::computeSigmaPInverse<StvkWithHencky<double, 3>>(StvkWithHencky<double, 3>&, Eigen::Matrix<double, StvkWithHencky<double, 3>::dim, 1, 0, StvkWithHencky<double, 3>::dim, 1> const&, Eigen::Matrix<double, StvkWithHencky<double, 3>::dim, 1, 0, StvkWithHencky<double, 3>::dim, 1>&);
template void VonMisesCapped<float>::computeSigmaPInverse<StvkWithHencky<float, 2>>(StvkWithHencky<float, 2>&, Eigen::Matrix<float, StvkWithHencky<float, 2>::dim, 1, 0, StvkWithHencky<float, 2>::dim, 1> const&, Eigen::Matrix<float, StvkWithHencky<float, 2>::dim, 1, 0, StvkWithHencky<float, 2>::dim, 1>&);
template void VonMisesCapped<float>::computeSigmaPInverse<StvkWithHencky<float, 3>>(StvkWithHencky<float, 3>&, Eigen::Matrix<float, StvkWithHencky<float, 3>::dim, 1, 0, StvkWithHencky<float, 3>::dim, 1> const&, Eigen::Matrix<float, StvkWithHencky<float, 3>::dim, 1, 0, StvkWithHencky<float, 3>::dim, 1>&);

template bool SmudgePlasticity<double>::projectStrain<Smudge<double, 3>>(Smudge<double, 3>&, Eigen::Matrix<double, 3, 3, 0, 3, 3>&);
template bool SmudgePlasticity<float>::projectStrain<Smudge<float, 3>>(Smudge<float, 3>&, Eigen::Matrix<float, 3, 3, 0, 3, 3>&);
template bool SmudgePlasticity<double>::projectStrain<Smudge<double, 2>>(Smudge<double, 2>&, Eigen::Matrix<double, 2, 2, 0, 2, 2>&);
template bool SmudgePlasticity<float>::projectStrain<Smudge<float, 2>>(Smudge<float, 2>&, Eigen::Matrix<float, 2, 2, 0, 2, 2>&);
template bool SmudgePlasticity<double>::projectStrain<StvkWithHenckyIsotropic<double, 3>>(StvkWithHenckyIsotropic<double, 3>&, Eigen::Matrix<double, 3, 3, 0, 3, 3>&);
template bool SmudgePlasticity<float>::projectStrain<StvkWithHenckyIsotropic<float, 3>>(StvkWithHenckyIsotropic<float, 3>&, Eigen::Matrix<float, 3, 3, 0, 3, 3>&);
template bool SmudgePlasticity<double>::projectStrain<StvkWithHenckyIsotropic<double, 2>>(StvkWithHenckyIsotropic<double, 2>&, Eigen::Matrix<double, 2, 2, 0, 2, 2>&);
template bool SmudgePlasticity<float>::projectStrain<StvkWithHenckyIsotropic<float, 2>>(StvkWithHenckyIsotropic<float, 2>&, Eigen::Matrix<float, 2, 2, 0, 2, 2>&);

template Eigen::Matrix<double, 3, 1, 0, 3, 1> DruckerPragerStvkHencky<double>::projectSigma<StvkWithHencky<double, 3>>(StvkWithHencky<double, 3>&, const Eigen::Matrix<double, 3, 1, 0, 3, 1>&);
template Eigen::Matrix<float, 3, 1, 0, 3, 1> DruckerPragerStvkHencky<float>::projectSigma<StvkWithHencky<float, 3>>(StvkWithHencky<float, 3>&, const Eigen::Matrix<float, 3, 1, 0, 3, 1>&);
template Eigen::Matrix<double, 2, 1, 0, 2, 1> DruckerPragerStvkHencky<double>::projectSigma<StvkWithHencky<double, 2>>(StvkWithHencky<double, 2>&, const Eigen::Matrix<double, 2, 1, 0, 2, 1>&);
template Eigen::Matrix<float, 2, 1, 0, 2, 1> DruckerPragerStvkHencky<float>::projectSigma<StvkWithHencky<float, 2>>(StvkWithHencky<float, 2>&, const Eigen::Matrix<float, 2, 1, 0, 2, 1>&);
template Eigen::Matrix<double, 3, 3, 0, 3, 3> DruckerPragerStvkHencky<double>::projectSigmaDerivative<StvkWithHencky<double, 3>>(StvkWithHencky<double, 3>&, const Eigen::Matrix<double, 3, 1, 0, 3, 1>&);
template Eigen::Matrix<float, 3, 3, 0, 3, 3> DruckerPragerStvkHencky<float>::projectSigmaDerivative<StvkWithHencky<float, 3>>(StvkWithHencky<float, 3>&, const Eigen::Matrix<float, 3, 1, 0, 3, 1>&);
template Eigen::Matrix<double, 2, 2, 0, 2, 2> DruckerPragerStvkHencky<double>::projectSigmaDerivative<StvkWithHencky<double, 2>>(StvkWithHencky<double, 2>&, const Eigen::Matrix<double, 2, 1, 0, 2, 1>&);
template Eigen::Matrix<float, 2, 2, 0, 2, 2> DruckerPragerStvkHencky<float>::projectSigmaDerivative<StvkWithHencky<float, 2>>(StvkWithHencky<float, 2>&, const Eigen::Matrix<float, 2, 1, 0, 2, 1>&);
template void DruckerPragerStvkHencky<double>::projectSigmaAndDerivative<StvkWithHencky<double, 3>>(StvkWithHencky<double, 3>&, const Eigen::Matrix<double, 3, 1, 0, 3, 1>&, Eigen::Matrix<double, 3, 1, 0, 3, 1>&, Eigen::Matrix<double, 3, 3, 0, 3, 3>&);
template void DruckerPragerStvkHencky<float>::projectSigmaAndDerivative<StvkWithHencky<float, 3>>(StvkWithHencky<float, 3>&, const Eigen::Matrix<float, 3, 1, 0, 3, 1>&, Eigen::Matrix<float, 3, 1, 0, 3, 1>&, Eigen::Matrix<float, 3, 3, 0, 3, 3>&);
template void DruckerPragerStvkHencky<double>::projectSigmaAndDerivative<StvkWithHencky<double, 2>>(StvkWithHencky<double, 2>&, const Eigen::Matrix<double, 2, 1, 0, 2, 1>&, Eigen::Matrix<double, 2, 1, 0, 2, 1>&, Eigen::Matrix<double, 2, 2, 0, 2, 2>&);
template void DruckerPragerStvkHencky<float>::projectSigmaAndDerivative<StvkWithHencky<float, 2>>(StvkWithHencky<float, 2>&, const Eigen::Matrix<float, 2, 1, 0, 2, 1>&, Eigen::Matrix<float, 2, 1, 0, 2, 1>&, Eigen::Matrix<float, 2, 2, 0, 2, 2>&);

template Eigen::Matrix<double, 3, 1, 0, 3, 1> VonMisesStvkHencky<double, 3>::projectSigma<StvkWithHencky<double, 3>>(StvkWithHencky<double, 3>&, const Eigen::Matrix<double, 3, 1, 0, 3, 1>&);
template Eigen::Matrix<float, 3, 1, 0, 3, 1> VonMisesStvkHencky<float, 3>::projectSigma<StvkWithHencky<float, 3>>(StvkWithHencky<float, 3>&, const Eigen::Matrix<float, 3, 1, 0, 3, 1>&);
template Eigen::Matrix<double, 2, 1, 0, 2, 1> VonMisesStvkHencky<double, 2>::projectSigma<StvkWithHencky<double, 2>>(StvkWithHencky<double, 2>&, const Eigen::Matrix<double, 2, 1, 0, 2, 1>&);
template Eigen::Matrix<float, 2, 1, 0, 2, 1> VonMisesStvkHencky<float, 2>::projectSigma<StvkWithHencky<float, 2>>(StvkWithHencky<float, 2>&, const Eigen::Matrix<float, 2, 1, 0, 2, 1>&);
template Eigen::Matrix<double, 3, 3, 0, 3, 3> VonMisesStvkHencky<double, 3>::projectSigmaDerivative<StvkWithHencky<double, 3>>(StvkWithHencky<double, 3>&, const Eigen::Matrix<double, 3, 1, 0, 3, 1>&);
template Eigen::Matrix<float, 3, 3, 0, 3, 3> VonMisesStvkHencky<float, 3>::projectSigmaDerivative<StvkWithHencky<float, 3>>(StvkWithHencky<float, 3>&, const Eigen::Matrix<float, 3, 1, 0, 3, 1>&);
template Eigen::Matrix<double, 2, 2, 0, 2, 2> VonMisesStvkHencky<double, 2>::projectSigmaDerivative<StvkWithHencky<double, 2>>(StvkWithHencky<double, 2>&, const Eigen::Matrix<double, 2, 1, 0, 2, 1>&);
template Eigen::Matrix<float, 2, 2, 0, 2, 2> VonMisesStvkHencky<float, 2>::projectSigmaDerivative<StvkWithHencky<float, 2>>(StvkWithHencky<float, 2>&, const Eigen::Matrix<float, 2, 1, 0, 2, 1>&);
template void VonMisesStvkHencky<double, 3>::projectSigmaAndDerivative<StvkWithHencky<double, 3>>(StvkWithHencky<double, 3>&, const Eigen::Matrix<double, 3, 1, 0, 3, 1>&, Eigen::Matrix<double, 3, 1, 0, 3, 1>&, Eigen::Matrix<double, 3, 3, 0, 3, 3>&);
template void VonMisesStvkHencky<float, 3>::projectSigmaAndDerivative<StvkWithHencky<float, 3>>(StvkWithHencky<float, 3>&, const Eigen::Matrix<float, 3, 1, 0, 3, 1>&, Eigen::Matrix<float, 3, 1, 0, 3, 1>&, Eigen::Matrix<float, 3, 3, 0, 3, 3>&);
template void VonMisesStvkHencky<double, 2>::projectSigmaAndDerivative<StvkWithHencky<double, 2>>(StvkWithHencky<double, 2>&, const Eigen::Matrix<double, 2, 1, 0, 2, 1>&, Eigen::Matrix<double, 2, 1, 0, 2, 1>&, Eigen::Matrix<double, 2, 2, 0, 2, 2>&);
template void VonMisesStvkHencky<float, 2>::projectSigmaAndDerivative<StvkWithHencky<float, 2>>(StvkWithHencky<float, 2>&, const Eigen::Matrix<float, 2, 1, 0, 2, 1>&, Eigen::Matrix<float, 2, 1, 0, 2, 1>&, Eigen::Matrix<float, 2, 2, 0, 2, 2>&);

template Eigen::Matrix<double, 3, 1, 0, 3, 1> VonMisesStvkHencky<double, 3>::projectSigma<QRStableNeoHookean<double, 3>>(QRStableNeoHookean<double, 3>&, const Eigen::Matrix<double, 3, 1, 0, 3, 1>&);
template Eigen::Matrix<float, 3, 1, 0, 3, 1> VonMisesStvkHencky<float, 3>::projectSigma<QRStableNeoHookean<float, 3>>(QRStableNeoHookean<float, 3>&, const Eigen::Matrix<float, 3, 1, 0, 3, 1>&);
template Eigen::Matrix<double, 2, 1, 0, 2, 1> VonMisesStvkHencky<double, 2>::projectSigma<QRStableNeoHookean<double, 2>>(QRStableNeoHookean<double, 2>&, const Eigen::Matrix<double, 2, 1, 0, 2, 1>&);
template Eigen::Matrix<float, 2, 1, 0, 2, 1> VonMisesStvkHencky<float, 2>::projectSigma<QRStableNeoHookean<float, 2>>(QRStableNeoHookean<float, 2>&, const Eigen::Matrix<float, 2, 1, 0, 2, 1>&);
template Eigen::Matrix<double, 3, 3, 0, 3, 3> VonMisesStvkHencky<double, 3>::projectSigmaDerivative<QRStableNeoHookean<double, 3>>(QRStableNeoHookean<double, 3>&, const Eigen::Matrix<double, 3, 1, 0, 3, 1>&);
template Eigen::Matrix<float, 3, 3, 0, 3, 3> VonMisesStvkHencky<float, 3>::projectSigmaDerivative<QRStableNeoHookean<float, 3>>(QRStableNeoHookean<float, 3>&, const Eigen::Matrix<float, 3, 1, 0, 3, 1>&);
template Eigen::Matrix<double, 2, 2, 0, 2, 2> VonMisesStvkHencky<double, 2>::projectSigmaDerivative<QRStableNeoHookean<double, 2>>(QRStableNeoHookean<double, 2>&, const Eigen::Matrix<double, 2, 1, 0, 2, 1>&);
template Eigen::Matrix<float, 2, 2, 0, 2, 2> VonMisesStvkHencky<float, 2>::projectSigmaDerivative<QRStableNeoHookean<float, 2>>(QRStableNeoHookean<float, 2>&, const Eigen::Matrix<float, 2, 1, 0, 2, 1>&);
template void VonMisesStvkHencky<double, 3>::projectSigmaAndDerivative<QRStableNeoHookean<double, 3>>(QRStableNeoHookean<double, 3>&, const Eigen::Matrix<double, 3, 1, 0, 3, 1>&, Eigen::Matrix<double, 3, 1, 0, 3, 1>&, Eigen::Matrix<double, 3, 3, 0, 3, 3>&);
template void VonMisesStvkHencky<float, 3>::projectSigmaAndDerivative<QRStableNeoHookean<float, 3>>(QRStableNeoHookean<float, 3>&, const Eigen::Matrix<float, 3, 1, 0, 3, 1>&, Eigen::Matrix<float, 3, 1, 0, 3, 1>&, Eigen::Matrix<float, 3, 3, 0, 3, 3>&);
template void VonMisesStvkHencky<double, 2>::projectSigmaAndDerivative<QRStableNeoHookean<double, 2>>(QRStableNeoHookean<double, 2>&, const Eigen::Matrix<double, 2, 1, 0, 2, 1>&, Eigen::Matrix<double, 2, 1, 0, 2, 1>&, Eigen::Matrix<double, 2, 2, 0, 2, 2>&);
template void VonMisesStvkHencky<float, 2>::projectSigmaAndDerivative<QRStableNeoHookean<float, 2>>(QRStableNeoHookean<float, 2>&, const Eigen::Matrix<float, 2, 1, 0, 2, 1>&, Eigen::Matrix<float, 2, 1, 0, 2, 1>&, Eigen::Matrix<float, 2, 2, 0, 2, 2>&);

template Eigen::Matrix<double, 3, 1, 0, 3, 1> DummyPlasticity<double>::projectSigma<StvkWithHencky<double, 3>>(StvkWithHencky<double, 3>&, const Eigen::Matrix<double, 3, 1, 0, 3, 1>&);
template Eigen::Matrix<float, 3, 1, 0, 3, 1> DummyPlasticity<float>::projectSigma<StvkWithHencky<float, 3>>(StvkWithHencky<float, 3>&, const Eigen::Matrix<float, 3, 1, 0, 3, 1>&);
template Eigen::Matrix<double, 2, 1, 0, 2, 1> DummyPlasticity<double>::projectSigma<StvkWithHencky<double, 2>>(StvkWithHencky<double, 2>&, const Eigen::Matrix<double, 2, 1, 0, 2, 1>&);
template Eigen::Matrix<float, 2, 1, 0, 2, 1> DummyPlasticity<float>::projectSigma<StvkWithHencky<float, 2>>(StvkWithHencky<float, 2>&, const Eigen::Matrix<float, 2, 1, 0, 2, 1>&);
template Eigen::Matrix<double, 3, 3, 0, 3, 3> DummyPlasticity<double>::projectSigmaDerivative<StvkWithHencky<double, 3>>(StvkWithHencky<double, 3>&, const Eigen::Matrix<double, 3, 1, 0, 3, 1>&);
template Eigen::Matrix<float, 3, 3, 0, 3, 3> DummyPlasticity<float>::projectSigmaDerivative<StvkWithHencky<float, 3>>(StvkWithHencky<float, 3>&, const Eigen::Matrix<float, 3, 1, 0, 3, 1>&);
template Eigen::Matrix<double, 2, 2, 0, 2, 2> DummyPlasticity<double>::projectSigmaDerivative<StvkWithHencky<double, 2>>(StvkWithHencky<double, 2>&, const Eigen::Matrix<double, 2, 1, 0, 2, 1>&);
template Eigen::Matrix<float, 2, 2, 0, 2, 2> DummyPlasticity<float>::projectSigmaDerivative<StvkWithHencky<float, 2>>(StvkWithHencky<float, 2>&, const Eigen::Matrix<float, 2, 1, 0, 2, 1>&);
template void DummyPlasticity<double>::projectSigmaAndDerivative<StvkWithHencky<double, 3>>(StvkWithHencky<double, 3>&, const Eigen::Matrix<double, 3, 1, 0, 3, 1>&, Eigen::Matrix<double, 3, 1, 0, 3, 1>&, Eigen::Matrix<double, 3, 3, 0, 3, 3>&);
template void DummyPlasticity<float>::projectSigmaAndDerivative<StvkWithHencky<float, 3>>(StvkWithHencky<float, 3>&, const Eigen::Matrix<float, 3, 1, 0, 3, 1>&, Eigen::Matrix<float, 3, 1, 0, 3, 1>&, Eigen::Matrix<float, 3, 3, 0, 3, 3>&);
template void DummyPlasticity<double>::projectSigmaAndDerivative<StvkWithHencky<double, 2>>(StvkWithHencky<double, 2>&, const Eigen::Matrix<double, 2, 1, 0, 2, 1>&, Eigen::Matrix<double, 2, 1, 0, 2, 1>&, Eigen::Matrix<double, 2, 2, 0, 2, 2>&);
template void DummyPlasticity<float>::projectSigmaAndDerivative<StvkWithHencky<float, 2>>(StvkWithHencky<float, 2>&, const Eigen::Matrix<float, 2, 1, 0, 2, 1>&, Eigen::Matrix<float, 2, 1, 0, 2, 1>&, Eigen::Matrix<float, 2, 2, 0, 2, 2>&);

template Eigen::Matrix<double, 3, 1, 0, 3, 1> DummyPlasticity<double>::projectSigma<CorotatedElasticity<double, 3>>(CorotatedElasticity<double, 3>&, const Eigen::Matrix<double, 3, 1, 0, 3, 1>&);
template Eigen::Matrix<float, 3, 1, 0, 3, 1> DummyPlasticity<float>::projectSigma<CorotatedElasticity<float, 3>>(CorotatedElasticity<float, 3>&, const Eigen::Matrix<float, 3, 1, 0, 3, 1>&);
template Eigen::Matrix<double, 2, 1, 0, 2, 1> DummyPlasticity<double>::projectSigma<CorotatedElasticity<double, 2>>(CorotatedElasticity<double, 2>&, const Eigen::Matrix<double, 2, 1, 0, 2, 1>&);
template Eigen::Matrix<float, 2, 1, 0, 2, 1> DummyPlasticity<float>::projectSigma<CorotatedElasticity<float, 2>>(CorotatedElasticity<float, 2>&, const Eigen::Matrix<float, 2, 1, 0, 2, 1>&);
template Eigen::Matrix<double, 3, 3, 0, 3, 3> DummyPlasticity<double>::projectSigmaDerivative<CorotatedElasticity<double, 3>>(CorotatedElasticity<double, 3>&, const Eigen::Matrix<double, 3, 1, 0, 3, 1>&);
template Eigen::Matrix<float, 3, 3, 0, 3, 3> DummyPlasticity<float>::projectSigmaDerivative<CorotatedElasticity<float, 3>>(CorotatedElasticity<float, 3>&, const Eigen::Matrix<float, 3, 1, 0, 3, 1>&);
template Eigen::Matrix<double, 2, 2, 0, 2, 2> DummyPlasticity<double>::projectSigmaDerivative<CorotatedElasticity<double, 2>>(CorotatedElasticity<double, 2>&, const Eigen::Matrix<double, 2, 1, 0, 2, 1>&);
template Eigen::Matrix<float, 2, 2, 0, 2, 2> DummyPlasticity<float>::projectSigmaDerivative<CorotatedElasticity<float, 2>>(CorotatedElasticity<float, 2>&, const Eigen::Matrix<float, 2, 1, 0, 2, 1>&);

template bool SnowPlasticity<double>::projectStrain<CorotatedIsotropic<double, 2>>(CorotatedIsotropic<double, 2>&, Eigen::Matrix<double, CorotatedIsotropic<double, 2>::dim, CorotatedIsotropic<double, 2>::dim, 0, CorotatedIsotropic<double, 2>::dim, CorotatedIsotropic<double, 2>::dim>&);
template bool SnowPlasticity<double>::projectStrain<CorotatedElasticity<double, 2>>(CorotatedElasticity<double, 2>&, Eigen::Matrix<double, CorotatedElasticity<double, 2>::dim, CorotatedElasticity<double, 2>::dim, 0, CorotatedElasticity<double, 2>::dim, CorotatedElasticity<double, 2>::dim>&);
template bool SnowPlasticity<double>::projectStrain<LinearCorotated<double, 2>>(LinearCorotated<double, 2>&, Eigen::Matrix<double, LinearCorotated<double, 2>::dim, LinearCorotated<double, 2>::dim, 0, LinearCorotated<double, 2>::dim, LinearCorotated<double, 2>::dim>&);
template bool SnowPlasticity<double>::projectStrain<NeoHookean<double, 2>>(NeoHookean<double, 2>&, Eigen::Matrix<double, NeoHookean<double, 2>::dim, NeoHookean<double, 2>::dim, 0, NeoHookean<double, 2>::dim, NeoHookean<double, 2>::dim>&);
template bool SnowPlasticity<float>::projectStrain<CorotatedElasticity<float, 2>>(CorotatedElasticity<float, 2>&, Eigen::Matrix<float, CorotatedElasticity<float, 2>::dim, CorotatedElasticity<float, 2>::dim, 0, CorotatedElasticity<float, 2>::dim, CorotatedElasticity<float, 2>::dim>&);
template bool SnowPlasticity<float>::projectStrain<LinearCorotated<float, 2>>(LinearCorotated<float, 2>&, Eigen::Matrix<float, LinearCorotated<float, 2>::dim, LinearCorotated<float, 2>::dim, 0, LinearCorotated<float, 2>::dim, LinearCorotated<float, 2>::dim>&);
template bool SnowPlasticity<float>::projectStrain<NeoHookean<float, 2>>(NeoHookean<float, 2>&, Eigen::Matrix<float, NeoHookean<float, 2>::dim, NeoHookean<float, 2>::dim, 0, NeoHookean<float, 2>::dim, NeoHookean<float, 2>::dim>&);

template bool NonAssociativeCamClay<double>::projectStrain<NeoHookeanBorden<double, 2>>(NeoHookeanBorden<double, 2>&, Eigen::Matrix<double, NeoHookeanBorden<double, 2>::dim, NeoHookeanBorden<double, 2>::dim, 0, NeoHookeanBorden<double, 2>::dim, NeoHookeanBorden<double, 2>::dim>&);
template bool NonAssociativeCamClay<float>::projectStrain<NeoHookeanBorden<float, 2>>(NeoHookeanBorden<float, 2>&, Eigen::Matrix<float, NeoHookeanBorden<float, 2>::dim, NeoHookeanBorden<float, 2>::dim, 0, NeoHookeanBorden<float, 2>::dim, NeoHookeanBorden<float, 2>::dim>&);
template bool NonAssociativeVonMises<double>::projectStrain<NeoHookeanBorden<double, 2>>(NeoHookeanBorden<double, 2>&, Eigen::Matrix<double, NeoHookeanBorden<double, 2>::dim, NeoHookeanBorden<double, 2>::dim, 0, NeoHookeanBorden<double, 2>::dim, NeoHookeanBorden<double, 2>::dim>&);
template bool NonAssociativeVonMises<float>::projectStrain<NeoHookeanBorden<float, 2>>(NeoHookeanBorden<float, 2>&, Eigen::Matrix<float, NeoHookeanBorden<float, 2>::dim, NeoHookeanBorden<float, 2>::dim, 0, NeoHookeanBorden<float, 2>::dim, NeoHookeanBorden<float, 2>::dim>&);
template bool NonAssociativeDruckerPrager<double>::projectStrain<NeoHookeanBorden<double, 2>>(NeoHookeanBorden<double, 2>&, Eigen::Matrix<double, NeoHookeanBorden<double, 2>::dim, NeoHookeanBorden<double, 2>::dim, 0, NeoHookeanBorden<double, 2>::dim, NeoHookeanBorden<double, 2>::dim>&);
template bool NonAssociativeDruckerPrager<float>::projectStrain<NeoHookeanBorden<float, 2>>(NeoHookeanBorden<float, 2>&, Eigen::Matrix<float, NeoHookeanBorden<float, 2>::dim, NeoHookeanBorden<float, 2>::dim, 0, NeoHookeanBorden<float, 2>::dim, NeoHookeanBorden<float, 2>::dim>&);

} // namespace ZIRAN
