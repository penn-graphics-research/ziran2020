//Disk Tear -- can be made radial, longitudinal, etc.
//./anisofracture -test 6

//PARAMS ARE FINALIZED DO NOT CHANGE!
//PYTHON PARAMS
//fiberScale = 50
//residual = 0.01
//percent = 0.35
//eta = 0.1

sim.end_frame = 240;
T frameRate = 48; //need super high fps for this super small dt and high E
sim.step.frame_dt = (T)1 / frameRate;
sim.dx = 0.01; //use 0.0025 for high res
sim.gravity = 0 * TV::Unit(1); //no gravity
sim.step.max_dt = /*1e-3*/ /*3e-4*/ 5.1e-4; //with test params can use 8e-5
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
T Youngs = 100;
T nu = 0.25;
T rho = 2;
QRStableNeoHookean<T, dim> model(Youngs, nu);

T suggested_dt = evaluateTimestepLinearElasticityAnalysis(Youngs, nu, rho, sim.dx, sim.cfl);
if (sim.symplectic) { ZIRAN_ASSERT(sim.step.max_dt <= suggested_dt, suggested_dt); }

//Anisotropic fracture params, grab these from the flags
StdVector<TV> a_0;
StdVector<T> alphas;
TV a_1, a_2;
a_1[0] = (T)Python::ax;
a_1[1] = (T)Python::ay;
a_1.normalize();
a_0.push_back(a_1);
alphas.push_back((T)Python::alpha1);
T theta2 = std::atan2(a_1[1], a_1[0]) * (180 / M_PI);
if ((T)Python::bx != 0 && (T)Python::by != 0) {
    a_2[0] = (T)Python::bx;
    a_2[1] = (T)Python::by;
    a_2.normalize();
    a_0.push_back(a_2);
    alphas.push_back((T)Python::alpha2);
}

T percentage = Python::percent; //.22 was too high I think
T l0 = 0.5 * sim.dx;
T eta = Python::eta;
T zeta = 1;
bool allow_damage = true;

//Make disk level set
TV center(3, 2.5);
T radius = 0.25;
Sphere<T, dim> diskLS(center, radius);

//Construct the file path
std::string theta_str = std::to_string(theta2);
theta_str.erase(theta_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "Theta = " << theta2 << std::endl;

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

std::string res_str = std::to_string(Python::residual);
res_str.erase(res_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "Residual = " << Python::residual << std::endl;

//Sample particles in the desired level set
int ppc = 4;
MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleInAnalyticLevelSet(diskLS, rho, ppc);

if (Python::useRadial) {
    std::string path("output/2D_DiskTear/2D_DiskTear_RadialFibers_withalpha" + alpha_str + "_Youngs" + E_str + "_percent" + percent_str + "_eta" + eta_str + "_dx" + dx_str + "_fiberScale" + scale_str + "_residual" + res_str);
    sim.output_dir.path = path;

    model.setExtraFiberStiffness(0, Python::fiberScale); //only scale elasticity if anisotropic!
    particles_handle.addFBasedMpmForceWithAnisotropicPhaseField(a_0, alphas, percentage, l0, model, eta, zeta, allow_damage, Python::residual);
    particles_handle.radialFibers(center);
}
else if (Python::useLongitudinal) {
    std::string path("output/2D_DiskTear/2D_DiskTear_LongitudinalFibers_withalpha" + alpha_str + "_Youngs" + E_str + "_percent" + percent_str + "_eta" + eta_str + "_dx" + dx_str + "_fiberScale" + scale_str + "_residual" + res_str);
    sim.output_dir.path = path;

    model.setExtraFiberStiffness(0, Python::fiberScale); //only scale elasticity if anisotropic!
    particles_handle.addFBasedMpmForceWithAnisotropicPhaseField(a_0, alphas, percentage, l0, model, eta, zeta, allow_damage, Python::residual);
    particles_handle.longitudinalFibers(center, radius);
}
else if (Python::isotropic) {
    std::string path("output/2D_DiskTear/2D_DiskTear_Isotropic_Youngs" + E_str + "_percent" + percent_str + "_eta" + eta_str + "_dx" + dx_str + "_fiberScale" + scale_str + "_residual" + res_str);
    sim.output_dir.path = path;
    particles_handle.addFBasedMpmForceWithAnisotropicPhaseField(a_0, alphas, percentage, l0, model, eta, zeta, allow_damage, Python::residual);
}
else {
    std::string path("output/2D_DiskTear/2D_DiskTear_" + theta_str + "deg_withalpha" + alpha_str + "_Youngs" + E_str + "_percent" + percent_str + "_eta" + eta_str + "_dx" + dx_str + "_fiberScale" + scale_str + "_residual" + res_str);
    sim.output_dir.path = path;
    model.setExtraFiberStiffness(0, Python::fiberScale); //only scale elasticity if anisotropic!
    particles_handle.addFBasedMpmForceWithAnisotropicPhaseField(a_0, alphas, percentage, l0, model, eta, zeta, allow_damage, Python::residual);
}

// Ground is right below disk;
TV ground_origin(0, 1);
TV ground_normal(0, 1);
HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::SLIP);
init_helper.addAnalyticCollisionObject(ground_object);

//Add left and right pullers
T fingerDepth = 0.05;
T fingerRadius = 0.05;
TV leftCenter(center[0] - radius + fingerRadius + fingerDepth, center[1]);
TV rightCenter(center[0] + radius - fingerRadius - fingerDepth, center[1]);
Sphere<T, dim> leftLS(leftCenter, fingerRadius);
Sphere<T, dim> rightLS(rightCenter, fingerRadius);
auto leftTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
    T pullSpeed = 0.2;
    TV translation_velocity(-1 * pullSpeed, 0);
    TV translation(-1 * pullSpeed * time, 0); //multiply each velocity by dt to get dx!
    object.setTranslation(translation, translation_velocity);
};
auto rightTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
    T pullSpeed = 0.2;
    TV translation_velocity(pullSpeed, 0);
    TV translation(pullSpeed * time, 0); //multiply each velocity by dt to get dx!
    object.setTranslation(translation, translation_velocity);
};
AnalyticCollisionObject<T, dim> leftObject(leftTransform, leftLS, AnalyticCollisionObject<T, dim>::STICKY); //this sphere moves
AnalyticCollisionObject<T, dim> rightObject(rightTransform, rightLS, AnalyticCollisionObject<T, dim>::STICKY); //this sphere moves
init_helper.addAnalyticCollisionObject(leftObject);
init_helper.addAnalyticCollisionObject(rightObject);