std::string helper_output;
bool helper_isotropic;
TV helper_fiber;
T helper_alpha = 0;
if (CmdHelper::helper == 1) {
    helper_output = "output/meatTear_anisotropic";
    helper_isotropic = false;
    helper_fiber = TV(0.9487, 0, -0.3156);
    helper_alpha = -1;
}
else if (CmdHelper::helper == 2) {
    helper_output = "output/meatTear_isotropic";
    helper_isotropic = true;
    helper_fiber = TV(1, 0, 0);
    helper_alpha = 0;
}
else
    ZIRAN_ASSERT(0);

sim.output_dir.path = helper_output;
sim.write_partio = false; // don't dumpout partio$F.bgeo
sim.end_frame = 72;
T frameRate = 24;
sim.step.frame_dt = (T)1 / frameRate;
sim.gravity = -9.8 * TV::Unit(1);
sim.step.max_dt = 5e-4;
sim.symplectic = true;
sim.verbose = false;
sim.cfl = 0.4;
// sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
//sim.apic_rpic_ratio = 0.01;
sim.transfer_scheme = MpmSimulationBase<T, dim>::FLIP_blend_PIC;
sim.flip_pic_ratio = 0; // FULL PIC for damping
//sim.rpic_damping_iteration = 5;
sim.dump_F_for_meshing = true;
T particle_per_cell = 7;

T Youngs = 5000;
T nu = 0.45;
T rho = 500;

// sample particles from a .mesh file
std::string filename = "TetMesh/meat_500k.mesh";
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

init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.

// QRAnisotropic<T, dim> model(Youngs, nu, helper_isotropic);

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
T eta = 0.45; //you will seg fault if this is too low
T zeta = 1;
bool allow_damage = true;
T residual_stress = 0.1; // 0.08;

// model.scaleFiberStiffness(0, 10); //does nothing if isotropic is set in the model as true

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
TV ground_origin = TV(0, 1, 0);
TV ground_normal(0, 1, 0);
HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::SEPARATE);
ground_object.setFriction(.2);
init_helper.addAnalyticCollisionObject(ground_object);

auto leftTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
    TV translation = TV(0 * time + (time < 0.5) * 9, 0, 0);
    TV translation_velocity(0, 0, 0);
    object.setTranslation(translation, translation_velocity);
};
// Sphere<T, dim> sphere(TV(2.37, 1.03, 1.95), 0.1);
AxisAlignedAnalyticBox<T, dim> sphere(TV(2.32, 0.93, 1.83), TV(2.42, 1.13, 2.13));
AnalyticCollisionObject<T, dim> leftObject(leftTransform, sphere, AnalyticCollisionObject<T, dim>::STICKY);
init_helper.addAnalyticCollisionObject(leftObject);

auto rightTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
    if (time > 0.5) {
        TV translation = TV(0.5 / 3 * (time - 0.5), 0.5 / 3 * (time - 0.5), 0);
        TV translation_velocity(0.5 / 3, .5 / 3, 0);
        object.setTranslation(translation, translation_velocity);
    }
    else {
        TV translation = TV(9, 9, 9);
        TV translation_velocity(0, 0, 0);
        object.setTranslation(translation, translation_velocity);
    }
};
// Sphere<T, dim> sphere2(TV(2.82, 1.03, 1.95), 0.1);
CappedCylinder<T, dim> cylinder1(0.1, 0.1, Vector<T, 4>(1, 0, 0, 0), TV(2.82, 1.04, 1.95));
HalfSpace<T, dim> board2(TV(0, 1, 0), TV(0, 1, 0));
DifferenceLevelSet<T, dim> cutsphere2;
cutsphere2.add(cylinder1, board2);
AnalyticCollisionObject<T, dim> rightObject(rightTransform, cutsphere2, AnalyticCollisionObject<T, dim>::STICKY);
init_helper.addAnalyticCollisionObject(rightObject);
