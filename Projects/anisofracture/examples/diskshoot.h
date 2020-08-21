//Disk Shoot -- sphere smashing through disk
// ./anisofracture -test 3 [OPTIONS]

//THESE PARAMETERS ARE FINAL, DO NOT CHANGE!
//FINAL PYTHON PARAMS:
//residual = 0.01
//percent = 0.07
//eta = 0.1
//using 50k disk

sim.end_frame = 144;
T frameRate = 24;
sim.step.frame_dt = (T)1 / frameRate;
sim.gravity = -1 * TV::Unit(1);
sim.step.max_dt = 1e-3;
sim.symplectic = true;
sim.verbose = false;
sim.cfl = 0.4;
sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
sim.apic_rpic_ratio = 0.01;
//sim.rpic_damping_iteration = 5; //THis looks better without damping!
//sim.dump_F_for_meshing = true;
T particle_per_cell = 7;

T Youngs = 40000;
T nu = 0.45;
T rho = 500;

// sample particles from a .mesh file
std::string filename = "TetMesh/disk50k.mesh"; //50k is plenty for nice dynamics and no debris
MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleFromTetWildFile(filename, rho);
T total_volume = particles_handle.total_volume;
T particle_count = particles_handle.particle_range.length();
T per_particle_volume = total_volume / particle_count;

// set dx
sim.dx = std::pow(particle_per_cell * per_particle_volume, (T)1 / (T)3);

//Anisotropic fracture params, grab these from the flags
StdVector<TV> a_0;
StdVector<T> alphas;
TV a_1, a_2;
a_1[0] = (T)Python::ax;
a_1[1] = (T)Python::ay;
a_1[2] = (T)Python::az;
a_1.normalize();
a_0.push_back(a_1);
alphas.push_back((T)Python::alpha1);
T theta2 = std::atan2(a_1[1], a_1[2]) * (180 / M_PI);
if ((T)Python::bx != 0 || (T)Python::by != 0 || (T)Python::bz != 0) {
    a_2[0] = (T)Python::bx;
    a_2[1] = (T)Python::by;
    a_2[2] = (T)Python::bz;
    a_2.normalize();
    a_0.push_back(a_2);
    alphas.push_back((T)Python::alpha2);
}

T percentage = Python::percent; //0.07 is good
T l0 = 0.5 * sim.dx;
T eta = Python::eta; //0.08 is good
T zeta = 1;
bool allow_damage = true;

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

std::string scale_str = std::to_string(Python::fiberScale);
scale_str.erase(scale_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "FiberScale = " << Python::fiberScale << std::endl;

std::string res_str = std::to_string(Python::residual);
res_str.erase(res_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "Residual = " << Python::residual << std::endl;

std::string dx_str = std::to_string(sim.dx);
dx_str.erase(dx_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "SimDx = " << sim.dx << std::endl;

if (Python::useRadial) {
    std::string path("output/3D_DiskShoot/3D_DiskShoot_RadialFibers_withalpha" + alpha_str + "_Youngs" + E_str + "_percent" + percent_str + "_eta" + eta_str + "_dx" + dx_str + "_fiberScale" + scale_str + "_residual" + res_str);
    sim.output_dir.path = path;
}
else if (Python::isotropic) {
    std::string path("output/3D_DiskShoot/3D_DiskShoot_Isotropic_Youngs" + E_str + "_percent" + percent_str + "_eta" + eta_str + "_dx" + dx_str + "_fiberScale" + scale_str + "_residual" + res_str);
    sim.output_dir.path = path;
}
else {
    std::string path("output/3D_DiskShoot/3D_DiskShoot_" + theta_str + "deg_withalpha" + alpha_str + "_Youngs" + E_str + "_percent" + percent_str + "_eta" + eta_str + "_dx" + dx_str + "_fiberScale" + scale_str + "_residual" + res_str);
    sim.output_dir.path = path;
}

// dump a vtk file in the output folder for mpmmeshing
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
QRStableNeoHookean<T, dim> model(Youngs, nu);
if (!Python::isotropic) {
    model.setExtraFiberStiffness(0, Python::fiberScale); //only scale elasticity if anisotropic!
}
particles_handle.addFBasedMpmForceWithAnisotropicPhaseField(a_0, alphas, percentage, l0, model, eta, zeta, allow_damage, Python::residual);

if (Python::useRadial) {
    TV center(2, 2, 2);
    int zeroDim = 0; //zero the x dimension
    particles_handle.radialFibers(center, zeroDim);
}

//Set up a ground plane
TV ground_origin = TV(0, -0.8, 0) + TV(2, 2, 2);
TV ground_normal(0, 1, 0);
HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::SEPARATE);
ground_object.setFriction(0.5);
init_helper.addAnalyticCollisionObject(ground_object);

// Torus holder
T theta = M_PI / (T)2;
Vector<T, 4> rotation(std::cos(theta / 2), 0, 0, std::sin(theta / 2));
TV trans = TV(2, 2, 2);
Torus<T, dim> torus(.4, .03, rotation, trans);
AnalyticCollisionObject<T, dim> torus_object(torus, AnalyticCollisionObject<T, dim>::STICKY);
init_helper.addAnalyticCollisionObject(torus_object);

auto leftTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
    T speed = 0.1;
    TV translation = TV(speed * time, 0, 0) + TV(2, 2, 2);
    TV translation_velocity(speed, 0, 0);
    object.setTranslation(translation, translation_velocity);
};

Sphere<T, dim> sphere(TV(-0.2, 0, 0), 0.1);
AnalyticCollisionObject<T, dim> leftObject(leftTransform, sphere, AnalyticCollisionObject<T, dim>::SEPARATE);
init_helper.addAnalyticCollisionObject(leftObject);
