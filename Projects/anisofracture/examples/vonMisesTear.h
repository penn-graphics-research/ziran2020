//Von Mises Tearing -- add von Mises plasticity to anisotropic fracture to see unique patterns and ductile fracture
//./anisofracture -test 22

//FINAL PARAMS
// residual = 0.01 #final param is 0.01
// percent = 0.183 #final param is 0.183
// tau = 4 #final param is 4
// eta = 0.1 #final param is 0.1
// fiberScale = 3 #final param is 3

bool usePlasticity = Python::orthotropic; //use orthotropic flag for plasticity toggle!

sim.end_frame = 72;
T frameRate = 24; //need super high fps for this super small dt and high E
sim.step.frame_dt = (T)1 / frameRate;
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
sim.dump_F_for_meshing = true;
TV helper_fiber = TV(1, 1, 1); // will overwrite with radial fiber
T helper_alpha = -1;

//Test params
T Youngs = 100;
T nu = 0.25;
T rho = 2;
QRStableNeoHookean<T, dim> model(Youngs, nu);

T suggested_dt = evaluateTimestepLinearElasticityAnalysis(Youngs, nu, rho, sim.dx, sim.cfl);
if (sim.symplectic) { ZIRAN_ASSERT(sim.step.max_dt <= suggested_dt, suggested_dt); }

//Anisotropic fracture params, grab these from the flags
// StdVector<TV> a_0;
// StdVector<T> alphas;
// TV a_1;
// a_1[0] = 0;
// a_1[1] = 1;
// a_1[2] = 0;
// a_1.normalize();
// a_0.push_back(a_1);
// if (Python::isotropic) {
//     alphas.push_back(0);
// }
// else {
//     alphas.push_back(-1);
// }

//Anisotropic fracture params
StdVector<TV> a_0;
StdVector<T> alphas;
TV a_1;
a_1 = helper_fiber;
a_1.normalize();
a_0.push_back(a_1);
if (!Python::isotropic) {
    alphas.push_back(helper_alpha); //alpha should be -1 if not iso
}
else {
    alphas.push_back(0); //set alpha to 0 if isotropic
}

T percentage = Python::percent; //.22 was too high I think
T eta = Python::eta;
T zeta = 1;
bool allow_damage = true;

// sample particles from a .mesh file
int ppc = 7; //was 7, testing 4 now
//std::string filename = "/home/joshwolper/Desktop/Link_to_output/3D_DiskTear/disk200k.mesh"; //50k is plenty for nice dynamics and no debris
std::string filename = "TetMesh/heart500k.mesh";
MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleFromTetWildFile(filename, rho);
T total_volume = particles_handle.total_volume;
T particle_count = particles_handle.particle_range.length();
T per_particle_volume = total_volume / particle_count;

// set dx
sim.dx = std::pow(ppc * per_particle_volume, (T)1 / (T)3);
T l0 = 0.5 * sim.dx;

StdVector<TV> node_wise_fiber;
if (1) {
    StdVector<TV> samples;
    StdVector<Vector<int, 4>> indices;
    std::string absolute_path = DataDir().absolutePath(filename);
    readTetMeshTetWild(absolute_path, samples, indices);
    std::function<bool(TV, int)> inflow = [](TV X, int vertex) {
        return (X[1] > 2.69);
    };
    std::function<bool(TV, int)> outflow = [](TV X, int vertex) {
        return (X[1] < 2.003);
    };
    StdVector<TV> tet_wise_fiber;
    fiberGen(samples, indices, inflow, outflow, tet_wise_fiber, node_wise_fiber);
}

//Construct the file path
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

std::string res_str = std::to_string(Python::residual);
res_str.erase(res_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "Residual = " << Python::residual << std::endl;

std::string tau_str = std::to_string(Python::tau);
tau_str.erase(tau_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "Tau = " << Python::tau << std::endl;

std::string fs_str = std::to_string(Python::fiberScale);
fs_str.erase(fs_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "FiberScale = " << Python::fiberScale << std::endl;

if (Python::useLongitudinal) {
    std::string path("output/3D_VonMisesTear/3D_VonMisesTear_LongitudinalFibers_Youngs" + E_str + "_percent" + percent_str + "_eta" + eta_str + "_dx" + dx_str + "_residual" + res_str + "_tau" + tau_str + "_fiberScale" + fs_str);
    sim.output_dir.path = path;
    model.setExtraFiberStiffness(0, Python::fiberScale); //only scale elasticity if anisotropic!
    particles_handle.addFBasedMpmForceWithAnisotropicPhaseField(a_0, alphas, percentage, l0, model, eta, zeta, allow_damage, Python::residual);
}
else if (Python::isotropic) {
    std::string path("output/3D_VonMisesTear/3D_VonMisesTear_IsotropicDamage_IsoElasticity_Youngs" + E_str + "_percent" + percent_str + "_eta" + eta_str + "_dx" + dx_str + "_residual" + res_str + "_tau" + tau_str + "_fiberScale" + fs_str);
    sim.output_dir.path = path;
    particles_handle.addFBasedMpmForceWithAnisotropicPhaseField(a_0, alphas, percentage, l0, model, eta, zeta, allow_damage, Python::residual);
}
else if (usePlasticity) {
    std::string path("output/3D_VonMisesTear/3D_VonMisesTear_Heart_Plastic_Youngs" + E_str + "_percent" + percent_str + "_eta" + eta_str + "_dx" + dx_str + "_residual" + res_str + "_tau" + tau_str + "_fiberScale" + fs_str);
    sim.output_dir.path = path;
    model.setExtraFiberStiffness(0, Python::fiberScale); //only scale elasticity if anisotropic!
    particles_handle.addFBasedMpmForceWithAnisotropicPhaseField(a_0, alphas, percentage, l0, model, eta, zeta, allow_damage, Python::residual);
    VonMisesStvkHencky<T, dim> p(Python::tau, FLT_MAX, 0);
    particles_handle.addPlasticity(model, p, "F");
}
else {
    std::string path("output/3D_VonMisesTear/3D_VonMisesTear_Heart_Elastic_Youngs" + E_str + "_percent" + percent_str + "_eta" + eta_str + "_dx" + dx_str + "_residual" + res_str + "_tau" + tau_str + "_fiberScale" + fs_str);
    sim.output_dir.path = path;
    model.setExtraFiberStiffness(0, Python::fiberScale); //only scale elasticity if anisotropic!
    particles_handle.addFBasedMpmForceWithAnisotropicPhaseField(a_0, alphas, percentage, l0, model, eta, zeta, allow_damage, Python::residual);
}

if (!Python::isotropic && Python::useLongitudinal) {
    //Rotate F for longitudinal fibers
    TV center(3, 2.5, 3);
    T radius = 0.25;
    particles_handle.longitudinalFibers(center, radius);
}

if (1) {
    StdVector<TV> samples;
    StdVector<Vector<int, 4>> indices;
    std::string absolute_path = DataDir().absolutePath(filename);
    readTetMeshTetWild(absolute_path, samples, indices);
    sim.output_dir.createPath();
    std::string vtk_path = DataDir().path + "/../Projects/anisofracture/" + sim.output_dir.path + "/tet.vtk";
    writeTetmeshVtk(vtk_path, samples, indices);
}

//Rotate F to match fiber direction
int i = 0;
for (auto iter = particles_handle.particles.subsetIter(DisjointRanges{ particles_handle.particle_range }, F_name<T, dim>()); iter; ++iter) {
    auto& F = iter.template get<0>();
    TV fiber = node_wise_fiber[i++];
    StdVector<TV> a0;
    a0.emplace_back(fiber);
    F = particles_handle.initializeRotatedFHelper(a0);
}

// Ground is right below disk;
TV ground_origin(0, 1, 0);
TV ground_normal(0, 1, 0);
HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::SLIP);
init_helper.addAnalyticCollisionObject(ground_object);

//Add left and right pullers
VdbLevelSet<T, dim> leftLS("LevelSets/leftPull_heart.vdb");
VdbLevelSet<T, dim> rightLS("LevelSets/rightPull_heart.vdb");
auto leftTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
    T pullSpeed = -0.2;
    TV translation_velocity(pullSpeed, 0, 0);
    TV translation(pullSpeed * time, 0, 0); //multiply each velocity by dt to get dx!
    object.setTranslation(translation, translation_velocity);
};
auto rightTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
    T pullSpeed = 0.2;
    TV translation_velocity(pullSpeed, 0, 0);
    TV translation(pullSpeed * time, 0, 0); //multiply each velocity by dt to get dx!
    object.setTranslation(translation, translation_velocity);
};
AnalyticCollisionObject<T, dim> leftObject(leftTransform, leftLS, AnalyticCollisionObject<T, dim>::STICKY); //this sphere moves
AnalyticCollisionObject<T, dim> rightObject(rightTransform, rightLS, AnalyticCollisionObject<T, dim>::STICKY); //this sphere moves
init_helper.addAnalyticCollisionObject(leftObject);
init_helper.addAnalyticCollisionObject(rightObject);