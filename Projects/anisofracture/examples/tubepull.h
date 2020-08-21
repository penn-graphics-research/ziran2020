// Tube pull
// ./anisofracture -test 2 -helper 1
// ./anisofracture -test 2 -helper 2

std::string helper_output;
bool helper_isotropic;
TV helper_fiber;
T helper_alpha = 0;
if (CmdHelper::helper == 1) {
    helper_output = "output/tubePull/tubepull_isotropic";
    helper_isotropic = true;
    helper_fiber = TV(1, 0, 0);
    helper_alpha = 0; //no damage
}

else if (CmdHelper::helper == 2) {
    helper_output = "output/tubePull/tubepull_anisotropic";
    helper_isotropic = false;
    helper_fiber = TV(1, 1, 0);
    helper_alpha = -1; //allow damage
}

else if (CmdHelper::helper == 3) {
    helper_output = "output/tubePull/tubepull_anisoDamageOnly";
    helper_isotropic = true; //isotropic elasticity
    helper_fiber = TV(1, 1, 0);
    helper_alpha = -1; //allow damage
}

else if (CmdHelper::helper == 4) {
    helper_output = "output/tubePull/tubepull_anisoElasticityOnly";
    helper_isotropic = false; //anisotropic elasticity
    helper_fiber = TV(1, 1, 0);
    helper_alpha = 0; //no damage
}

else
    ZIRAN_ASSERT(0);

sim.output_dir.path = helper_output;
sim.end_frame = 168;
T frameRate = 24;
sim.step.frame_dt = (T)1 / frameRate;
sim.gravity = -1 * TV::Unit(1);
sim.step.max_dt = 1e-3;
sim.symplectic = true;
sim.verbose = false;
sim.cfl = 0.4;
sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
sim.apic_rpic_ratio = 0.01;
sim.dump_F_for_meshing = true;
T particle_per_cell = 7;

T Youngs = 40000;
T nu = 0.45;
T rho = 500;

// sample particles from a .mesh file
std::string filename = "TetMesh/tubePull.mesh";
MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleFromTetWildFile(filename, rho);
T total_volume = particles_handle.total_volume;
T particle_count = particles_handle.particle_range.length();
T per_particle_volume = total_volume / particle_count;

// set dx
sim.dx = std::pow(particle_per_cell * per_particle_volume, (T)1 / (T)3);

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
particles_handle.transform([&](int index, Ref<T> mass, TV& X, TV& V) {
    X += TV(2, 2, 2);
});

init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

StdVector<TV> a_0;
StdVector<T> alphas;
TV a_1, a_2;
a_1 = helper_fiber;
a_1.normalize();
a_0.push_back(a_1);
alphas.push_back(helper_alpha);
T theta2 = std::atan2(a_1[1], a_1[2]) * (180 / M_PI);

T percentage = 0.2; //.22 was too high I think
T l0 = 0.5 * sim.dx;
T eta = 0.45; //was passing nu = 0.45 before, so setting this 0.45
T zeta = 1;
bool allow_damage = true;
T residual_stress = 0.01;

if (helper_isotropic) {
    QRStableNeoHookean<T, dim> model(Youngs, nu);
    particles_handle.addFBasedMpmForceWithAnisotropicPhaseField(a_0, alphas, percentage, l0, model, eta, zeta, allow_damage, residual_stress);
}
else {
    QRStableNeoHookean<T, dim> model(Youngs, nu);
    model.setExtraFiberStiffness(0, 10);
    particles_handle.addFBasedMpmForceWithAnisotropicPhaseField(a_0, alphas, percentage, l0, model, eta, zeta, allow_damage, residual_stress);
}

//Set up a ground plane
TV ground_origin = TV(0, -0.8, 0) + TV(2, 2, 2);
TV ground_normal(0, 1, 0);
HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::SEPARATE);
ground_object.setFriction(0.5);
init_helper.addAnalyticCollisionObject(ground_object);

{
    auto leftTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
        T t = time;
        T theta = M_PI * (T)0.3 * 0;
        Vector<T, 4> rotation(std::cos(theta * t / 2), std::sin(theta * t / 2), 0, 0);
        TV omega(theta, 0, 0);
        object.setRotation(rotation);
        object.setAngularVelocity(omega);
        TV translation = TV(-0.1 * time, 0, 0) + TV(2 - 0.6, 2, 2);
        TV translation_velocity(-0.1, 0, 0);
        object.setTranslation(translation, translation_velocity);
    };
    T radius = 0.3;
    T height = 0.4;
    T theta = M_PI / 2;
    Vector<T, 4> cylinder_rotation(0, 0, std::cos(theta / 2), std::sin(theta / 2));

    //Vector<T, 4> cylinder_rotation(std::cos(theta / 2), 0,0,std::sin(theta / 2));
    //TV translation(0, 0, 0);
    //CappedCylinder<T, dim> cylinder(radius, height, cylinder_rotation, translation);
    Sphere<T, dim> cylinder(TV::Zero(), radius);

    AnalyticCollisionObject<T, dim> leftObject(leftTransform, cylinder, AnalyticCollisionObject<T, dim>::STICKY);
    init_helper.addAnalyticCollisionObject(leftObject);
}

{
    auto leftTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
        T t = time;
        T theta = M_PI * (T)0.3 * 0;
        Vector<T, 4> rotation(std::cos(theta * t / 2), std::sin(theta * t / 2), 0, 0);
        TV omega(theta, 0, 0);
        object.setRotation(rotation);
        object.setAngularVelocity(omega);
        TV translation = TV(0.1 * time, 0, 0) + TV(2 + 0.6, 2, 2);
        TV translation_velocity(0.1, 0, 0);
        object.setTranslation(translation, translation_velocity);
    };
    T radius = 0.3;
    T height = 0.4;
    T theta = M_PI / 2;
    Vector<T, 4> cylinder_rotation(0, 0, std::cos(theta / 2), std::sin(theta / 2));

    //TV translation(0, 0, 0);
    // CappedCylinder<T, dim> cylinder(radius, height, cylinder_rotation, translation);
    Sphere<T, dim> cylinder(TV::Zero(), radius);
    AnalyticCollisionObject<T, dim> leftObject(leftTransform, cylinder, AnalyticCollisionObject<T, dim>::STICKY);
    init_helper.addAnalyticCollisionObject(leftObject);
}
