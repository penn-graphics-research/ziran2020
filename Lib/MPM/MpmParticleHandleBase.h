#ifndef MPM_PARTICLE_HANDLE_BASE_H
#define MPM_PARTICLE_HANDLE_BASE_H

#include <functional>
#include <memory>
#include <string>
#include <Ziran/Math/Geometry/Rotation.h>
#include <Ziran/CS/DataStructure/DisjointRanges.h>
#include <Ziran/CS/DataStructure/Ref.h>
#include <Ziran/CS/Util/Forward.h>
#include <Ziran/CS/Util/AttributeNamesForward.h>
#include <Ziran/CS/Util/AttributeNamesForward.h>
#include <Ziran/Physics/ConstitutiveModel/ConstitutiveModel.h>
#include <MPM/Force/MpmForceBase.h>

namespace ZIRAN {

template <class T, int dim>
class AnalyticLevelSet;
template <class T, int dim>
class Particles;
class PlasticityApplierBase;
template <class T, int dim>
class Scene;
template <class T, int dim>
class MpmSimulation;
template <class T, int dim>
class AnalyticLevelSet;
template <class T, int dim>
class Particles;
template <class T, int dim>
class Scene;

template <class T, int dim>
class MpmParticleHandleBase {
public:
    static constexpr int interpolation_degree = ZIRAN_MPM_DEGREE;
    static const int splat_size = MATH_TOOLS::power(interpolation_degree + 1, dim);

    using TV = Vector<T, dim>;
    typedef Matrix<T, dim, dim> TM;

    Particles<T, dim>& particles;
    Scene<T, dim>& scene;
    MpmForceBase<T, dim>* mpmforce;
    StdVector<std::unique_ptr<PlasticityApplierBase>>& plasticity_appliers;
    StdVector<TV>& scratch_xp;
    T& dt;
    Range particle_range;
    T total_volume;
    int cotangent_manifold_dim;

    bool already_pinned_particles = false;
    bool already_parallel_pinned_particles = false;

    MpmParticleHandleBase(Particles<T, dim>& particles, Scene<T, dim>& scene, MpmForceBase<T, dim>* mpmforce,
        StdVector<std::unique_ptr<PlasticityApplierBase>>& plasticity_appliers,
        StdVector<TV>& scratch_xp, T& dt, Range particle_range, T total_volume, int cotangent_manifold_dim = 0);

    // Creates a copy with new particles
    MpmParticleHandleBase copy();

    void transform(const std::function<void(int, Ref<T>, Vector<T, dim>&, Vector<T, dim>&)>& mapping);

    void addBulkViscosity(const T b);

    void addVolumeFraction(const T b);

    void addImplicitViscosity(const T mu);

    void addEtherDrag(const T drag);

    template <class TCONST>
    void addFBasedMpmForce(const TCONST& model);

    template <class TCONST>
    void addFBasedMpmForceWithPhaseField(const T& percentage, const T& l0, const TCONST& model, bool allow_damage = true, const T random_fluctuation_percentage = 0);

    template <class TCONST>
    void addFBasedMpmForceWithAnisotropicPhaseField(const StdVector<Vector<T, dim>> a_0, const StdVector<T> alphas, const T& percentage, const T& l0, const TCONST& model, const T nu, const T zeta, bool allow_damage = true, const T residual = 0.001);

    Matrix<T, dim, dim> initializeRotatedFHelper(const StdVector<Vector<T, dim>> a_0);

    void radialFibers(const TV center, const int zeroDimension = 0);

    void spiralFibers(const TV center, const int zeroDimension = 0, const T theta = 45);

    void longitudinalFibers(const TV center, const T radius);

    void fibersVaryWithY(const T yVal);

    void addOriginalPositionAsAttribute();

    void randomizePhaseFieldSigmaC(const T random_fluctuation_percentage);

    //void voronoiSigmaC(std::string voronoiPointsFile, T radius, T minPercent, T maxPercent);
    void voronoiSigmaC(std::string voronoiPointsFile, T radius);

    void yDimSigmaCInit(T yMin, T yMax, T maxScale);

    template <class TCONST>
    void addFDecoupledBasedMpmForce(const TCONST& model, T neighbor_search_h);

    void scaleF(const T scale);

    void scaleF(const TV scale);

    void scaleJ(const T scale);

    void updatePinningTarget(const Vector<T, 4>& R_in, const Vector<T, 3>& omega_in, const TV& b, const TV& dbdt);

    void scaleFCurve(int frame, const std::function<T(int)>& growCurve);

    void manipulateF(const std::function<Matrix<T, dim, dim>(Matrix<T, dim, dim>, Vector<T, dim>)>& op);

    void standingPoolInitialization(const EquationOfState<T, dim> model, const TV& gravity, const TV& seaLevel);

    void prescoreSnow(AnalyticLevelSet<T, dim>& levelset, T grain_size, T density_min, T Jp_min);

    void prescoreWetSand2D(AnalyticLevelSet<T, dim>& levelset, T grain_size, T density_min, T Jp_min, T scale_min = (T)0.5);

    void prescorePhaseFieldSigmaC(AnalyticLevelSet<T, dim>& levelset, T grain_size, T scale_lower_clamp = 0);

    void prescoreModifiedCamClay(T grain_size, T M_max, T beta_max, T Jp_min);

    void resetDeformation();

    template <class TCONST>
    void addJBasedMpmForce(const TCONST& model);

    template <class FCONST, class JCONST>
    void addFJMixedMpmForce(const FCONST& f_model, const JCONST& j_model, MATERIAL_PHASE_ENUM phase, MATERIAL_PROPERTY_ENUM property, bool linear_corotated = false);

    void addDensitySummationFluidMpmForce(const T h);

    void addLinearCorotatedMpmForce(const LinearCorotated<T, dim>& model);

    void addElementMeasure();

    // this should only be called once for a handle
    void pinParticlesInLevelSet(AnalyticLevelSet<T, dim>& levelset, T stiffness, T damping);

    // this should only be called once for a handle
    void pinParticlesInVdbLevelSet(std::string filename, T stiffness, T damping);

    // this should only be called once for a handle
    void parallelPinParticlesInLevelSet(AnalyticLevelSet<T, dim>& levelset, T stiffness, T damping, const bool is_collider_spring = false);

    // this should only be called once for a handle
    void parallelPinParticlesInVdbLevelSet(std::string filename, T stiffness, T damping);

    template <class TConst, class TPlastic>
    void addPlasticity(const TConst& cons, const TPlastic& plasticity, std::string strain_name = "F");

    void addDummyPlasticity();

    void pinParticles(const DisjointRanges& ranges, const T stiffness, const T damping);

    void parallelPinParticles(const DisjointRanges& ranges, const T stiffness, const T damping, const bool is_collider_spring = false);

    template <class TCONST, int manifold_dim>
    void addCotangentBasedMpmForce(const TCONST& model, const Range& range);

    template <class TCONST>
    void addFBasedMpmForceWithMeasure(const TCONST& model, const Range& range, T total_volume);

    template <class TCONST>
    void addFElasticNonequilibratedBasedMpmForce(const TCONST& model, T viscosity_d_input, T viscosity_v_input);

    void setMassFromDensity(const T density);

    void addInextensibility(const TM& F = TM::Identity());
};
} // namespace ZIRAN
#endif /* ifndef MPM_PARTICLE_HANDLE */
