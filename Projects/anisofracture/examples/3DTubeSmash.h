// ./anisofracture -test 21 [OPTIONS]

bool inextensible = Python::inextensible;
bool isotropic = Python::isotropic;

sim.end_frame = 140;
T frameRate = 24;
sim.step.frame_dt = (T)1 / frameRate;
sim.gravity = -10 * TV::Unit(1); //
sim.step.max_dt = 2e-4; //1.5 for fanfu//1.25e-4 before
sim.symplectic = true;
sim.verbose = false;
sim.cfl = 0.6;
//sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
//sim.apic_rpic_ratio = 0;
sim.transfer_scheme = MpmSimulationBase<T, dim>::FLIP_blend_PIC;
sim.flip_pic_ratio = 0; //0 is full PIC
//sim.pic_damping_iteration = 5;//need to add pic damping first!!!
sim.dump_F_for_meshing = true;
T particle_per_cell = 7; //was 7 before

//Important:Parameters used: Youngs=5000 residual = 0.01  percent = 0.15  eta = 0.3  fiberScale = 10
T youngs = Python::E;
T nu = 0.366;
T rho = 2;

// sample particles from a .mesh file
std::string filename = "TetMesh/tube400k.mesh";
MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleFromTetWildFile(filename, rho);
T total_volume = particles_handle.total_volume;
T particle_count = particles_handle.particle_range.length();
T per_particle_volume = total_volume / particle_count;

// set dx
sim.dx = std::pow(particle_per_cell * per_particle_volume, (T)1 / (T)3);

StdVector<TV> a_0;
StdVector<T> alphas;
TV helper_fiber = TV(0, 1, 0);
T helper_alpha = -1;
TV a_1;
a_1 = helper_fiber;
a_1.normalize();
a_0.push_back(a_1);
if (!isotropic) {
    alphas.push_back(helper_alpha); //alpha should be -1 if not iso
}
else {
    alphas.push_back(0); //set alpha to 0 if isotropic
}

T percentage = Python::percent; //no aniso damage!!
T l0 = 0.5 * sim.dx;
T eta = Python::eta; //no aniso damage!!
T zeta = 1;
bool allow_damage = true;

//Construct the file path

std::string E_str = std::to_string(youngs);
E_str.erase(E_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "Youngs = " << youngs << std::endl;

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

std::string rho_str = std::to_string(rho);
rho_str.erase(rho_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "Pumpkin Rho = " << rho << std::endl;

if (isotropic) { //isotropic
    std::string path("output/3D_TubeCompress/3D_Tube_Isotropic_Youngs" + E_str + "_Rho" + rho_str + "_percent" + percent_str + "_eta" + eta_str + "_dx" + dx_str + "_fiberScale" + scale_str + "_residual" + res_str);
    sim.output_dir.path = path;
}
else { //anisotropic
    std::string path("output/3D_TubeCompress/3D_Tube_Anisotropic_Youngs" + E_str + "_Rho" + rho_str + "_percent" + percent_str + "_eta" + eta_str + "_dx" + dx_str + "_fiberScale" + scale_str + "_residual" + res_str);
    sim.output_dir.path = path;
}

//// dump a vtk file in the output folder for mpmmeshing
if (1) {
    StdVector<TV> samples;
    StdVector<Vector<int, 4>> indices;
    std::string absolute_path = DataDir().absolutePath(filename);
    readTetMeshTetWild(absolute_path, samples, indices);
    sim.output_dir.createPath();
    std::string vtk_path = DataDir().path + "/../Projects/anisofracture/" + sim.output_dir.path + "/tet.vtk";
    writeTetmeshVtk(vtk_path, samples, indices);
}

init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

//Add constitutive model, scale elasticity if needed, and add mpm force
QRStableNeoHookean<T, dim> model(youngs, nu);
if (!inextensible) {
    if (!isotropic) {
        model.setExtraFiberStiffness(0, Python::fiberScale); //only do this if not isotropic or inextensible
    }
}
particles_handle.addFBasedMpmForceWithAnisotropicPhaseField(a_0, alphas, percentage, l0, model, eta, zeta, allow_damage, Python::residual);

// ****************************************************************************
// Collision objects
// ****************************************************************************
//TV ground_origin = TV(0, 4.771, 0);
//TV ground_origin = TV(0, 4.819, 0);
TV ground_origin = TV(0, 4.49, 0);
TV ground_normal(0, 1, 0);
HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::SEPARATE);
ground_object.setFriction(1);
init_helper.addAnalyticCollisionObject(ground_object);

//Add CRUSHER at top
TV crusherMin(0, 5.505, 0);
TV crusherMax(10, 5.701, 10);
AxisAlignedAnalyticBox<T, dim> crusherLevelSet(crusherMin, crusherMax);
auto crusherTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
    T pushSpeed = -0.5;
    T endTime = 1.2;
    if (time < endTime) {
        TV translation_velocity(0, pushSpeed, 0);
        TV translation(0, pushSpeed * time, 0); //multiply each velocity by dt to get dx!
        object.setTranslation(translation, translation_velocity);
    }
    else {
        TV translation = TV(0, pushSpeed * endTime, 0);
        TV translation_velocity(0, 0, 0);
        object.setTranslation(translation, translation_velocity);
    }
};
AnalyticCollisionObject<T, dim> crusherObject(crusherTransform, crusherLevelSet, AnalyticCollisionObject<T, dim>::SEPARATE); //this sphere moves
init_helper.addAnalyticCollisionObject(crusherObject);