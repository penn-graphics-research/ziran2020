std::string helper_output;
helper_output = "output/orange";

sim.output_dir.path = helper_output;
sim.end_frame = 80;
T frameRate = 24;
sim.step.frame_dt = (T)1 / frameRate;
sim.gravity = -9.8 * TV::Unit(1);
sim.step.max_dt = 1e-3;
sim.symplectic = true;
sim.verbose = false;
sim.cfl = 0.4;
sim.transfer_scheme = MpmSimulationBase<T, dim>::FLIP_blend_PIC;
sim.flip_pic_ratio = 0; // FULL PIC for damping
sim.dump_F_for_meshing = true;
T particle_per_cell = 20;
sim.rpic_damping_iteration = 0;

// ****************************************************************************
// Interior
// ****************************************************************************
if (1) {
    T Youngs = 10000;
    T nu = 0.4;
    T rho = 500;
    T helper_isotropic = false;
    TV helper_fiber = TV(1, 1, 1); // will overwrite with radial fiber
    T helper_alpha = -1;

    std::string filename = "TetMesh/orange_50k.mesh";
    MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleFromTetWildFile(filename, rho);
    T total_volume = particles_handle.total_volume;
    T particle_count = particles_handle.particle_range.length();
    T per_particle_volume = total_volume / particle_count;
    sim.dx = std::pow(particle_per_cell * per_particle_volume, (T)1 / (T)3);
    if (1) {
        StdVector<TV> samples;
        StdVector<Vector<int, 4>> indices;
        std::string absolute_path = DataDir().absolutePath(filename);
        readTetMeshTetWild(absolute_path, samples, indices);
        sim.output_dir.createPath();
        std::string vtk_path = DataDir().path + "/../Projects/anisofracture/" + sim.output_dir.path + "/tet1.vtk";
        writeTetmeshVtk(vtk_path, samples, indices);
    }

    QRAnisotropic<T, dim> model(Youngs, nu, helper_isotropic);
    StdVector<TV> a_0;
    StdVector<T> alphas;
    TV a_1, a_2;
    a_1 = helper_fiber;
    a_1.normalize();
    a_0.push_back(a_1);
    alphas.push_back(helper_alpha);
    T theta2 = std::atan2(a_1[1], a_1[2]) * (180 / M_PI);
    T percentage = 0.15;
    T l0 = 0.5 * sim.dx;
    T eta = 0.01;
    T zeta = 1;
    bool allow_damage = true;
    T residual_stress = 0.001;
    // model.scaleFiberStiffness(0, 2);
    particles_handle.transform([&](int index, Ref<T> mass, TV& X, TV& V) { X += TV(2, 2, 2); });
    particles_handle.addFBasedMpmForceWithAnisotropicPhaseField(a_0, alphas, percentage, l0, model, eta, zeta, allow_damage, residual_stress);

    TV center(2, 2, 2);
    int zeroDim = 1;
    particles_handle.radialFibers(center, zeroDim);

    // SnowPlasticity<T> p(0, 1, 0.5);
    //particles_handle.addPlasticity(model, p, "F");
    std::cout << "Particle count: " << sim.particles.count << std::endl;
}

// ****************************************************************************
// Peel
// ****************************************************************************
if (1) {
    T Youngs = 50000;
    T nu = 0.4;
    T rho = 500;
    T helper_isotropic = true;
    TV helper_fiber = TV(1, 1, 1);
    T helper_alpha = 0;

    std::string filename = "TetMesh/orange_peel_notched_30k.mesh";
    MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleFromTetWildFile(filename, rho);
    T total_volume = particles_handle.total_volume;
    T particle_count = particles_handle.particle_range.length();
    T per_particle_volume = total_volume / particle_count;
    sim.dx = std::pow(particle_per_cell * per_particle_volume, (T)1 / (T)3);
    if (1) {
        StdVector<TV> samples;
        StdVector<Vector<int, 4>> indices;
        std::string absolute_path = DataDir().absolutePath(filename);
        readTetMeshTetWild(absolute_path, samples, indices);
        sim.output_dir.createPath();
        std::string vtk_path = DataDir().path + "/../Projects/anisofracture/" + sim.output_dir.path + "/tet2.vtk";
        writeTetmeshVtk(vtk_path, samples, indices);
    }

    QRAnisotropic<T, dim> model(Youngs, nu, helper_isotropic);
    StdVector<TV> a_0;
    StdVector<T> alphas;
    TV a_1, a_2;
    a_1 = helper_fiber;
    a_1.normalize();
    a_0.push_back(a_1);
    alphas.push_back(helper_alpha);
    T theta2 = std::atan2(a_1[1], a_1[2]) * (180 / M_PI);
    T percentage = 999;
    T l0 = 0.5 * sim.dx;
    T eta = 0.1;
    T zeta = 1;
    bool allow_damage = true;
    T residual_stress = 0.005;
    // model.scaleFiberStiffness(0, 2);
    particles_handle.transform([&](int index, Ref<T> mass, TV& X, TV& V) { X += TV(2, 2, 2); });
    particles_handle.addFBasedMpmForceWithAnisotropicPhaseField(a_0, alphas, percentage, l0, model, eta, zeta, allow_damage, residual_stress);
}

// ****************************************************************************
// Collision objects
// ****************************************************************************
TV ground_origin = TV(0, 1.984, 0);
TV ground_normal(0, 1, 0);
HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::SLIP);
ground_object.setFriction(1);
init_helper.addAnalyticCollisionObject(ground_object);

{
    auto leftTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
        T t = time;
        TV translation = TV(-0.1 * time, 0, 0) + TV(2, 2, 2);
        TV translation_velocity(-0.1, 0, 0);
        object.setTranslation(translation, translation_velocity);
    };
    Sphere<T, dim> leftLS(TV(-0.08, 0, -0.2), 0.05);
    AnalyticCollisionObject<T, dim> leftObject(leftTransform, leftLS, AnalyticCollisionObject<T, dim>::STICKY);
    init_helper.addAnalyticCollisionObject(leftObject);
}

{
    auto leftTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
        T t = time;
        TV translation = TV(-0.1 * time, 0, 0) + TV(2, 2, 2);
        TV translation_velocity(-0.1, 0, 0);
        object.setTranslation(translation, translation_velocity);
    };
    Sphere<T, dim> leftLS(TV(-0.19, 0, -0.12), 0.05);
    AnalyticCollisionObject<T, dim> leftObject(leftTransform, leftLS, AnalyticCollisionObject<T, dim>::STICKY);
    init_helper.addAnalyticCollisionObject(leftObject);
}
{
    auto leftTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
        T t = time;
        TV translation = TV(-0.1 * time, 0, 0) + TV(2, 2, 2);
        TV translation_velocity(-0.1, 0, 0);
        object.setTranslation(translation, translation_velocity);
    };
    Sphere<T, dim> leftLS(TV(-0.22, 0, 0.01), 0.05);
    AnalyticCollisionObject<T, dim> leftObject(leftTransform, leftLS, AnalyticCollisionObject<T, dim>::STICKY);
    init_helper.addAnalyticCollisionObject(leftObject);
}

{
    auto rightTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
        TV translation = TV(0 * time, 0, 0) + TV(2, 2, 2);
        TV translation_velocity(0, 0, 0);
        object.setTranslation(translation, translation_velocity);
    };
    Sphere<T, dim> sphere2(TV(0.12, 0, 0), 0.05);
    AnalyticCollisionObject<T, dim> rightObject(rightTransform, sphere2, AnalyticCollisionObject<T, dim>::STICKY);
    init_helper.addAnalyticCollisionObject(rightObject);
}

{
    auto rightTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
        TV translation = TV(0 * time, 0, 0) + TV(2, 2, 2);
        TV translation_velocity(0, 0, 0);
        object.setTranslation(translation, translation_velocity);
    };
    Sphere<T, dim> sphere2(TV(-0.12, 0, 0.19), 0.05);
    AnalyticCollisionObject<T, dim> rightObject(rightTransform, sphere2, AnalyticCollisionObject<T, dim>::STICKY);
    init_helper.addAnalyticCollisionObject(rightObject);
}

init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.