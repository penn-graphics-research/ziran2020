// 3D Parameter Comparison Tests: uses notched mode 1 fracture with 45deg fibers
// ./anisofracture -test 13

//FINAL PARAMS FROM PYTHON
//see the script for final parameters!

sim.end_frame = 216; //72 for p and FS, 216 for eta, and 192 for youngs!!
T frameRate = 72; //24 for percent, FS, and youngs, 72fps for eta!!
sim.step.frame_dt = (T)1 / frameRate;
sim.dx = 0.01; //use 0.0025 for high res
sim.gravity = 0 * TV::Unit(1); //no gravity
sim.step.max_dt = /*1e-3*/ /*3e-4*/ 2e-4; //with test params can use 8e-5
sim.newton.max_iterations = 5;
sim.newton.tolerance = 1e-3;
sim.objective.minres.max_iterations = 10000;
sim.objective.minres.tolerance = 1e-4;
sim.quasistatic = false;
sim.symplectic = true; // want explicit!
sim.objective.matrix_free = true;
sim.verbose = false;
sim.cfl = 0.4;
sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
sim.apic_rpic_ratio = 0; //full RPIC

//Test params
T Youngs = Python::E;
T nu = 0.25;
T rho = 2;
QRStableNeoHookean<T, dim> model(Youngs, nu);

//Anisotropic fracture params, grab these from the flags
StdVector<TV> a_0;
StdVector<T> alphas;
TV a_1;
a_1[0] = 1;
a_1[1] = 1; //45 deg
a_1[2] = 0;
a_1.normalize();
a_0.push_back(a_1);
alphas.push_back(-1);

T percentage = Python::percent;
T l0 = 0.5 * sim.dx;
T eta = Python::eta; //set this as nu since it's what we used before
T zeta = 1;
bool allow_damage = true;

//Construct the file path

std::string alpha_str = std::to_string(Python::alpha1);
alpha_str.erase(alpha_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "Alpha = " << Python::alpha1 << std::endl;

std::string E_str = std::to_string(Youngs);
E_str.erase(E_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "Youngs = " << Youngs << std::endl;

std::string percent_str = std::to_string(percentage);
percent_str.erase(percent_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "Percentage = " << percentage << std::endl;

std::string eta_str = std::to_string(eta);
eta_str.erase(eta_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "Eta = " << eta << std::endl;

std::string dx_str = std::to_string(sim.dx);
dx_str.erase(dx_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "SimDx = " << sim.dx << std::endl;

std::string scale_str = std::to_string(Python::fiberScale);
scale_str.erase(scale_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "FiberScale = " << Python::fiberScale << std::endl;

if (Python::isotropic) {
    std::string path("output/3D_ParamEffects/3D_ParamEffects_Isotropic_Youngs" + E_str + "_percent" + percent_str + "_eta" + eta_str + "_dx" + dx_str + "_fiberScale" + scale_str);
    sim.output_dir.path = path;
}
else {
    //std::string path("output/2D_NotchedMode1Fracture/SearchFor45/aaa/");
    std::string path("output/3D_ParamEffects/3D_ParamEffects_Youngs" + E_str + "_percent" + percent_str + "_eta" + eta_str + "_dx" + dx_str + "_fiberScale" + scale_str /*+ "computeSigmaCrit"*/);
    sim.output_dir.path = path;
    model.setExtraFiberStiffness(0, Python::fiberScale); //only scale elasticity if anisotropic!
}

init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

//Setup the box material
T materialToHold = 0.02;
T depth = 0.2;
TV boxMin(2, 2 - materialToHold, 2 - depth / 2.0); //needs extra material at bottom to hold onto for mode 1
TV boxMax(2.4, 2.4 + materialToHold, 2 + depth / 2.0);
AxisAlignedAnalyticBox<T, dim> boxLevelSet(boxMin, boxMax);

TV plateCenter(2.2, 2.2, 2); //define the three points of the triangular notch
T halfNotchWidth = 0.018;

T theta = std::atan2(halfNotchWidth, 0.2); //get needed rotation for the boxes we'll use to cut out notch
TM rotateTheta, rotateNegTheta;
rotateTheta << std::cos(theta), -1 * std::sin(theta), 0, std::sin(theta), std::cos(theta), 0, 0, 0, 1;
rotateNegTheta << std::cos(-1 * theta), -1 * std::sin(-1 * theta), 0, std::sin(-1 * theta), std::cos(-1 * theta), 0, 0, 0, 1;
TV bottomRightCorner = rotateTheta * TV(0.2, -0.2, 0);
TV topRightCorner = rotateNegTheta * TV(0.2, 0.2, 0);

TV half_edge(0.2, 0.2, 0.2);
Vector<T, 4> rot1(-theta, 0, 0, 1);
Vector<T, 4> rot2(theta, 0, 0, 1);
TV translation1 = plateCenter - bottomRightCorner;
TV translation2 = plateCenter - topRightCorner;
translation1[0] += 0.1;
translation2[0] += 0.1;
AnalyticBox<T, dim> box1LS(half_edge, rot1, translation1);
AnalyticBox<T, dim> box2LS(half_edge, rot2, translation2);

IntersectionLevelSet<T, dim> cutoutLS;
cutoutLS.add(box1LS, box2LS); //take intersection of the two wedges

DifferenceLevelSet<T, dim> cut2LS;
cut2LS.add(boxLevelSet, cutoutLS); //use the intersection to cut out the wedge

//Sample particles in the desired level set
int ppc = 4; //4 for p, eta, and FS, 7 for youngs
MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleInAnalyticLevelSet(cut2LS, rho, ppc);

//Elasticity Handler
particles_handle.addFBasedMpmForceWithAnisotropicPhaseField(a_0, alphas, percentage, l0, model, eta, zeta, allow_damage, Python::residual);

//Now setup the boundaries

//Top Box (up (mode 1) or moves right (mode 2))
TV pushMin(1.9, 2.4, 2 - depth);
TV pushMax(2.5, 3, 2 + depth);
AxisAlignedAnalyticBox<T, dim> pushLevelSet(pushMin, pushMax);
auto upperTransformMode1 = [](T time, AnalyticCollisionObject<T, dim>& object) {
    T pushSpeed = 0.05;
    TV translation_velocity(0, 1 * pushSpeed, 0);
    TV translation(0, 1 * pushSpeed * time, 0); //multiply each velocity by dt to get dx!
    object.setTranslation(translation, translation_velocity);
};
//Choose transform based on which fracture mode!
AnalyticCollisionObject<T, dim> pushObject(upperTransformMode1, pushLevelSet, AnalyticCollisionObject<T, dim>::STICKY);
init_helper.addAnalyticCollisionObject(pushObject);

//Bottom Box (moves down (mode1) or stationary (mode2))
TV pullMin(1.9, 1.5, 2 - depth);
TV pullMax(2.5, 2.0, 2 + depth);
AxisAlignedAnalyticBox<T, dim> pullLevelSet(pullMin, pullMax);
auto lowerTransformMode1 = [](T time, AnalyticCollisionObject<T, dim>& object) {
    T pushSpeed = 0.05;
    TV translation_velocity(0, -1 * pushSpeed, 0);
    TV translation(0, -1 * pushSpeed * time, 0); //multiply each velocity by dt to get dx!
    object.setTranslation(translation, translation_velocity);
};
AnalyticCollisionObject<T, dim> pullObject(lowerTransformMode1, pullLevelSet, AnalyticCollisionObject<T, dim>::STICKY); //this sphere moves
init_helper.addAnalyticCollisionObject(pullObject);
