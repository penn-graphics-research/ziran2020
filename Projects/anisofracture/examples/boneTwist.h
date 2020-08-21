// 3D Bone Twist Test -- twist bone and top and bottom to see if we can get some nice torsion fractures
// ./anisofracture -test 25

int fracture_type = 1; //0:twisting 1:bending 2:pulling

sim.end_frame = 144;
T frameRate = 48;
sim.step.frame_dt = (T)1 / frameRate;
sim.dx = 0.005;
sim.gravity = 0 * TV::Unit(1); //no gravity
sim.step.max_dt = 1e-6; //3e-5;2e-6;6e-6;4e-6;1e-6;1e-5;2e-5;5e-6  //bending:200k->1e-6 50k->5e-6 //pulling:50k->7e-6 400k->2.9e-6 //twisting: 200k->5e-6
sim.newton.max_iterations = 5;
sim.newton.tolerance = 1e-3;
sim.objective.minres.max_iterations = 10000;
sim.objective.minres.tolerance = 1e-4;
sim.quasistatic = false;
sim.symplectic = true; // want explicit!
sim.objective.matrix_free = true;
sim.verbose = false;
sim.dump_F_for_meshing = true;
sim.cfl = 0.4;
sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
sim.apic_rpic_ratio = 0; //full RPIC
//sim.rpic_damping_iteration = 5;

//Test params
T Youngs = Python::E; //need to crank this for bone
T nu = 0.25;
T rho = 800; //2
QRStableNeoHookean<T, dim> model(Youngs, nu);

// sample particles from a .mesh file
std::string filename = "TetMesh/bone200k.mesh"; //50k is plenty for nice dynamics and no debris
MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleFromTetWildFile(filename, rho);
T total_volume = particles_handle.total_volume;
T particle_count = particles_handle.particle_range.length();
T per_particle_volume = total_volume / particle_count;

// set dx
int particle_per_cell = 6;
sim.dx = std::pow(particle_per_cell * per_particle_volume, (T)1 / (T)3);

T suggested_dt = evaluateTimestepLinearElasticityAnalysis(Youngs, nu, rho, sim.dx, sim.cfl);
if (sim.symplectic) { ZIRAN_ASSERT(sim.step.max_dt <= suggested_dt, suggested_dt); }

StdVector<TV> node_wise_fiber;
if (fracture_type == 2) {
    StdVector<TV> samples;
    StdVector<Vector<int, 4>> indices;
    std::string absolute_path = DataDir().absolutePath(filename);
    readTetMeshTetWild(absolute_path, samples, indices);
    std::function<bool(TV, int)> inflow = [](TV X, int vertex) {
        return ((X - TV(5.13, 5.83, 5.18)).norm() < 0.1);
    };
    std::function<bool(TV, int)> outflow = [](TV X, int vertex) {
        return ((X - TV(5.16, 5.14, 5.17)).norm() < 0.1);
    };
    StdVector<TV> tet_wise_fiber;
    fiberGen(samples, indices, inflow, outflow, tet_wise_fiber, node_wise_fiber);
}

//Anisotropic fracture params, grab these from the flags
StdVector<TV> a_0;
StdVector<T> alphas;
TV a_1, a_2;
a_1[0] = 0; //y direction fiber
a_1[1] = 1;
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
    std::string path("output/3D_BoneTwist/3D_BoneTwist_Isotropic_Youngs" + E_str + "_percent" + percent_str + "_eta" + eta_str + "_dx" + dx_str + "_fiberScale" + scale_str);
    sim.output_dir.path = path;
}
else {
    std::string path("output/3D_BoneTwist/3D_BoneTwist_Youngs" + E_str + "_percent" + percent_str + "_eta" + eta_str + "_dx" + dx_str + "_fiberScale" + scale_str /*+ "computeSigmaCrit"*/);
    sim.output_dir.path = path;
    model.setExtraFiberStiffness(0, Python::fiberScale); //only scale elasticity if anisotropic!
}

particles_handle.addFBasedMpmForceWithAnisotropicPhaseField(a_0, alphas, percentage, l0, model, eta, zeta, allow_damage, Python::residual);

init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

if (1) {
    StdVector<TV> samples;
    StdVector<Vector<int, 4>> indices;
    std::string absolute_path = DataDir().absolutePath(filename);
    readTetMeshTetWild(absolute_path, samples, indices);
    sim.output_dir.createPath();
    std::string vtk_path = DataDir().path + "/../Projects/anisofracture/" + sim.output_dir.path + "/tet.vtk";
    writeTetmeshVtk(vtk_path, samples, indices);
}

if (fracture_type == 2) {
    //Rotate F to match fiber direction
    int i = 0;
    for (auto iter = particles_handle.particles.subsetIter(DisjointRanges{ particles_handle.particle_range }, F_name<T, dim>()); iter; ++iter) {
        auto& F = iter.template get<0>();
        TV fiber = node_wise_fiber[i++];
        StdVector<TV> a0;
        a0.emplace_back(fiber);
        F = particles_handle.initializeRotatedFHelper(a0);
    }
}

//Setup boundary condition

if (fracture_type == 0) //twist
{
    TV center(0, 0, 0);
    T radius = 0.2;
    auto topTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
        T velocity = 0.02;
        T theta = (T)-40.0 / 180 * M_PI;
        T t = time;
        Vector<T, 4> rotation(std::cos(theta * t / 2), 0, std::sin(theta * t / 2), 0);
        object.setRotation(rotation);
        TV omega(0, theta, 0);
        object.setAngularVelocity(omega);
        TV translation_velocity(0, velocity, 0);
        TV translation(5.13, 5.83 + velocity * t, 5.18);
        object.setTranslation(translation, translation_velocity);
    };
    auto bottomTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
        T velocity = -0.02;
        //T endTime = 1.2;
        T theta = (T)40.0 / 180 * M_PI;
        T t = time;
        Vector<T, 4> rotation(std::cos(theta * t / 2), 0, std::sin(theta * t / 2), 0);
        object.setRotation(rotation);
        TV omega(0, theta, 0);
        object.setAngularVelocity(omega);
        TV translation_velocity(0, velocity, 0);
        TV translation(5.16, 5.14 + velocity * t, 5.17);
        object.setTranslation(translation, translation_velocity);
    };

    Sphere<T, dim> topLS(center, radius);
    Sphere<T, dim> bottomLS(center, radius);
    AnalyticCollisionObject<T, dim> topObject(topTransform, topLS, AnalyticCollisionObject<T, dim>::STICKY);
    AnalyticCollisionObject<T, dim> bottomObject(bottomTransform, bottomLS, AnalyticCollisionObject<T, dim>::STICKY);
    init_helper.addAnalyticCollisionObject(topObject);
    init_helper.addAnalyticCollisionObject(bottomObject);
}

else if (fracture_type == 1) //bending
{
    TV center(0, 0, 0);
    T radius = 0.3;
    auto topTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
        T theta = (T)40.0 / 180 * M_PI;
        T t = time;
        Vector<T, 4> rotation(std::cos(theta * t / 2), std::sin(theta * t / 2), 0, 0);
        object.setRotation(rotation);
        TV omega(theta, 0, 0);
        object.setAngularVelocity(omega);
        TV translation_velocity(0, 0, 0);
        TV translation(5.13, 5.83, 5.18);
        object.setTranslation(translation, translation_velocity);
    };
    auto bottomTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
        T theta = (T)-40.0 / 180 * M_PI;
        T t = time;
        Vector<T, 4> rotation(std::cos(theta * t / 2), std::sin(theta * t / 2), 0, 0);
        object.setRotation(rotation);
        TV omega(theta, 0, 0);
        object.setAngularVelocity(omega);
        TV translation_velocity(0, 0, 0);
        TV translation(5.16, 5.14, 5.17);
        object.setTranslation(translation, translation_velocity);
    };
    Sphere<T, dim> topLS(center, radius);
    Sphere<T, dim> bottomLS(center, radius);
    AnalyticCollisionObject<T, dim> topObject(topTransform, topLS, AnalyticCollisionObject<T, dim>::STICKY);
    AnalyticCollisionObject<T, dim> bottomObject(bottomTransform, bottomLS, AnalyticCollisionObject<T, dim>::STICKY);
    init_helper.addAnalyticCollisionObject(topObject);
    init_helper.addAnalyticCollisionObject(bottomObject);
}
else if (fracture_type == 2) //pulling
{
    TV center(0, 0, 0);
    T radius = 0.3;
    auto topTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
        T velocity = 0.04; //0.02
        T theta = (T)-0 / 180 * M_PI;
        T t = time;
        Vector<T, 4> rotation(std::cos(theta * t / 2), 0, std::sin(theta * t / 2), 0);
        object.setRotation(rotation);
        TV omega(0, theta, 0);
        object.setAngularVelocity(omega);
        TV translation_velocity(0, velocity, 0);
        TV translation(5.13, 5.83 + velocity * t, 5.18);
        object.setTranslation(translation, translation_velocity);
    };
    auto bottomTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
        T velocity = -0.04; //-0.02
        //T endTime = 1.2;
        T theta = (T)0 / 180 * M_PI;
        T t = time;
        Vector<T, 4> rotation(std::cos(theta * t / 2), 0, std::sin(theta * t / 2), 0);
        object.setRotation(rotation);
        TV omega(0, theta, 0);
        object.setAngularVelocity(omega);
        TV translation_velocity(0, velocity, 0);
        TV translation(5.16, 5.14 + velocity * t, 5.17);
        object.setTranslation(translation, translation_velocity);
    };
    Sphere<T, dim> topLS(center, radius);
    Sphere<T, dim> bottomLS(center, radius);
    AnalyticCollisionObject<T, dim> topObject(topTransform, topLS, AnalyticCollisionObject<T, dim>::STICKY);
    AnalyticCollisionObject<T, dim> bottomObject(bottomTransform, bottomLS, AnalyticCollisionObject<T, dim>::STICKY);
    init_helper.addAnalyticCollisionObject(topObject);
    init_helper.addAnalyticCollisionObject(bottomObject);
}