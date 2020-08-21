//Hanging Torus -- hang toruses that get more and more heavy to show off inextensibility and large time steps
// ./anisofracture -test 16 [OPTIONS]

//FINAL PARAMETERS!! See python script for remaining finalized params

bool inextensible = Python::inextensible;

sim.end_frame = 120;
T frameRate = 24;
sim.step.frame_dt = (T)1 / frameRate;
sim.gravity = -3 * TV::Unit(1);
sim.step.max_dt = 1e-3; //1.5 for fanfu
sim.symplectic = true;
sim.verbose = false;
sim.cfl = 0.6;
sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
sim.apic_rpic_ratio = 0;
// sim.transfer_scheme = MpmSimulationBase<T, dim>::FLIP_blend_PIC;
// sim.flip_pic_ratio = 0.0; //0 is full PIC
//sim.rpic_damping_iteration = 5; //THis looks better without damping!
sim.dump_F_for_meshing = true;
T particle_per_cell = 8; //was 7 before

T youngsTorus = Python::E;
T nuTorus = 0.3;
T rhoTorus = Python::rho;
T youngsRopes = 10;
T nuRopes = 0.3;
T rhoRopes = 2;

// sample particles from a .mesh file
std::string filename = "TetMesh/torus90k.mesh"; //50k is plenty for nice dynamics and no debris
std::string filename2 = "TetMesh/ropes20k.mesh";
MpmParticleHandleBase<T, dim> particles_handle_torus = init_helper.sampleFromTetWildFile(filename, rhoTorus);
MpmParticleHandleBase<T, dim> particles_handle_ropes = init_helper.sampleFromTetWildFile(filename2, rhoRopes);
T total_volume = particles_handle_ropes.total_volume;
T particle_count = particles_handle_ropes.particle_range.length();
T per_particle_volume = total_volume / particle_count;

// set dx
sim.dx = 2 * std::pow(particle_per_cell * per_particle_volume, (T)1 / (T)3);

//Sample from Analytic Level Sets to get particle count we want in each handle
//sim.dx = 0.012;
//int ppc = 8;
//Torus<T,dim> torusLS(0.2, 0.075, Vector<T,4>(1,0,0,0), TV(2,2,2));
//MpmParticleHandleBase<T, dim> particles_handle_torus = init_helper.sampleInAnalyticLevelSet(torusLS, rhoTorus, ppc);
//CappedCylinder<T, dim> cylinder1(0.03, 0.6, Vector<T, 4>(1, 0, 0, 0), TV(2.4, 2.35, 2));
//MpmParticleHandleBase<T, dim> particles_handle_ropes = init_helper.sampleInAnalyticLevelSet(cylinder1, rhoRopes, 8);

//Anisotropic fracture params, grab these from the flags
StdVector<TV> a_0;
StdVector<T> alphas;
TV a_1;
a_1[0] = 0;
a_1[1] = 1; //fiber direction should always just be straight up
a_1[2] = 0;
a_1.normalize();
a_0.push_back(a_1);
alphas.push_back(0); //no aniso damage

T percentage = 99999; //no aniso damage!!
T l0 = 0.5 * sim.dx;
T eta = 99999; //no aniso damage!!
T zeta = 1;
bool allow_damage = true;

//Construct the file path

std::string E_str = std::to_string(youngsTorus);
E_str.erase(E_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "Youngs = " << youngsTorus << std::endl;

std::string percent_str = std::to_string(percentage);
percent_str.erase(percent_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "Percentage = " << percentage << std::endl;

std::string eta_str = std::to_string(eta);
eta_str.erase(eta_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "Eta = " << eta << std::endl;

std::string scale_str = std::to_string(Python::fiberScale);
scale_str.erase(scale_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "FiberScale = " << Python::fiberScale << std::endl;

std::string res_str = std::to_string(Python::residual);
res_str.erase(res_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "Residual = " << Python::residual << std::endl;

std::string dx_str = std::to_string(sim.dx);
dx_str.erase(dx_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "SimDx = " << sim.dx << std::endl;

std::string rho_str = std::to_string(rhoTorus);
rho_str.erase(rho_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "Torus Rho = " << rhoTorus << std::endl;

if (inextensible) { //inextensible
    std::string path("output/3D_HangingTorus/3D_HangingTorus_Inextensible_torusYoungs" + E_str + "_torusRho" + rho_str + "_percent" + percent_str + "_eta" + eta_str + "_dx" + dx_str + "_residual" + res_str);
    sim.output_dir.path = path;
}
else { //anisotropic elasticity
    std::string path("output/3D_HangingTorus/3D_HangingTorus_AnisoElasticity_torusYoungs" + E_str + "_torusRho" + rho_str + "_percent" + percent_str + "_eta" + eta_str + "_dx" + dx_str + "_fiberScale" + scale_str + "_residual" + res_str);
    sim.output_dir.path = path;
}

//// dump a vtk file in the output folder for mpmmeshing
if (1) {
    StdVector<TV> samples;
    StdVector<Vector<int, 4>> indices;
    std::string absolute_path = DataDir().absolutePath(filename);
    readTetMeshTetWild(absolute_path, samples, indices);
    sim.output_dir.createPath();
    std::string vtk_path = DataDir().path + "/../Projects/anisofracture/" + sim.output_dir.path + "/tet1.vtk";
    writeTetmeshVtk(vtk_path, samples, indices);
}
if (1) {
    StdVector<TV> samples2;
    StdVector<Vector<int, 4>> indices2;
    std::string absolute_path2 = DataDir().absolutePath(filename2);
    readTetMeshTetWild(absolute_path2, samples2, indices2);
    sim.output_dir.createPath();
    std::string vtk_path2 = DataDir().path + "/../Projects/anisofracture/" + sim.output_dir.path + "/tet2.vtk";
    writeTetmeshVtk(vtk_path2, samples2, indices2);
}

init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

//Add constitutive model, scale elasticity if needed, and add mpm force
QRStableNeoHookean<T, dim> modelTorus(youngsTorus, nuTorus);
QRStableNeoHookean<T, dim> modelRopes(youngsRopes, nuRopes);
if (!inextensible) {
    modelRopes.setExtraFiberStiffness(0, Python::fiberScale); //only scale elasticity if anisotropic elasticity!
}
particles_handle_torus.addFBasedMpmForceWithAnisotropicPhaseField(a_0, alphas, percentage, l0, modelTorus, eta, zeta, allow_damage, Python::residual);
particles_handle_ropes.addFBasedMpmForceWithAnisotropicPhaseField(a_0, alphas, percentage, l0, modelRopes, eta, zeta, allow_damage, Python::residual);

//Construct rotated F for cheese fiber direction
if (inextensible) {
    TM rotatedF;
    rotatedF << 0, -1, 0, 1, 0, 0, 0, 0, 1;
    particles_handle_ropes.addInextensibility(rotatedF); //only add to the ropes
}

//Set up a ground plane
TV ground_origin = TV(2, 0.8, 2);
TV ground_normal(0, 1, 0);
HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::STICKY);
init_helper.addAnalyticCollisionObject(ground_object);

//Set up the ceiling
TV ceiling_origin = TV(2, 2.6, 2);
TV ceiling_normal(0, -1, 0);
HalfSpace<T, dim> ceiling_ls(ceiling_origin, ceiling_normal);
AnalyticCollisionObject<T, dim> ceiling_object(ceiling_ls, AnalyticCollisionObject<T, dim>::STICKY);
init_helper.addAnalyticCollisionObject(ceiling_object);