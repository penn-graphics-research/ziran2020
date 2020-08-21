std::string helper_output;
helper_output = "output/fishskin";
sim.output_dir.path = helper_output;
sim.end_frame = 104;
T frameRate = 48;
sim.step.frame_dt = (T)1 / frameRate;
sim.gravity = -9.8 * TV::Unit(1);
sim.step.max_dt = 1e-3;
sim.symplectic = true;
sim.verbose = false;
sim.cfl = 0.4;
sim.dump_F_for_meshing = true;
sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
sim.apic_rpic_ratio = 0;
T particle_per_cell = 7;

// ****************************************************************************
// Interior
// ****************************************************************************
if (1) {
    T Youngs = 100000;
    T nu = .4;
    T rho = 1000;
    T helper_isotropic = false;
    TV helper_fiber = TV(1, 1, 1); // will overwrite with radial fiber
    T helper_alpha = -1;

    std::string filename = "TetMesh/fish_50k.mesh";
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
        std::string vtk_path = DataDir().path + "/../Projects/anisofracture/" + sim.output_dir.path + "/tet.vtk";
        writeTetmeshVtk(vtk_path, samples, indices);
    }

    // geenrate fiber direcitons
    StdVector<TV> node_wise_fiber;
    if (1) {
        StdVector<TV> samples;
        StdVector<Vector<int, 4>> indices;
        std::string absolute_path = DataDir().absolutePath(filename);
        readTetMeshTetWild(absolute_path, samples, indices);
        std::function<bool(TV, int)> inflow = [](TV X, int vertex) { return (X - TV(-1.1574, -0.05, 0.01)).norm() < 0.06; };
        std::function<bool(TV, int)> outflow = [](TV X, int vertex) { return (X - TV(0.56, 0.16, 0.05)).norm() < 0.06; };
        StdVector<TV> tet_wise_fiber;
        fiberGen(samples, indices, inflow, outflow, tet_wise_fiber, node_wise_fiber);
    }

    QRStableNeoHookean<T, dim> model(Youngs, nu);
    model.setExtraFiberStiffness(0, 8);
    StdVector<TV> a_0;
    StdVector<T> alphas;
    TV a_1, a_2;
    a_1 = helper_fiber;
    a_1.normalize();
    a_0.push_back(a_1);
    alphas.push_back(helper_alpha);
    T theta2 = std::atan2(a_1[1], a_1[2]) * (180 / M_PI);
    T percentage = 0.2;
    T l0 = 0.5 * sim.dx;
    T eta = 0.01;
    T zeta = 1;
    bool allow_damage = true;
    T residual_stress = .1;
    particles_handle.transform([&](int index, Ref<T> mass, TV& X, TV& V) { X += TV(2, 2, 2); });
    particles_handle.addFBasedMpmForceWithAnisotropicPhaseField(a_0, alphas, percentage, l0, model, eta, zeta, allow_damage, residual_stress);
    std::cout << "Particle count: " << sim.particles.count << std::endl;

    // rotate F to match fiber direction
    int i = 0;
    for (auto iter = particles_handle.particles.subsetIter(DisjointRanges{ particles_handle.particle_range }, F_name<T, dim>()); iter; ++iter) {
        auto& F = iter.template get<0>();
        TV fiber = node_wise_fiber[i++];
        StdVector<TV> a0;
        a0.emplace_back(fiber);
        F = particles_handle.initializeRotatedFHelper(a0);
    }
}

// ****************************************************************************
// Collision objects
// ****************************************************************************
TV ground_origin = TV(0, 1.5874, 0);
TV ground_normal(0, 1, 0);
HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::SEPARATE);
ground_object.setFriction(1);
init_helper.addAnalyticCollisionObject(ground_object);

{
    auto leftTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
        T t = time;
        TV translation = TV(0, 1 * time, 0) + TV(2, 2, 2);
        TV translation_velocity(0, 1, 0);
        object.setTranslation(translation, translation_velocity);
    };
    Sphere<T, dim> leftLS(TV(-0.961, 0.12, 0.11), 0.1);
    AnalyticCollisionObject<T, dim> leftObject(leftTransform, leftLS, AnalyticCollisionObject<T, dim>::STICKY);
    leftObject.setFriction(1);
    init_helper.addAnalyticCollisionObject(leftObject);
}
{
    auto leftTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
        T t = time;
        TV translation = TV(0, 1 * time, 0) + TV(2, 2, 2);
        TV translation_velocity(0, 1, 0);
        object.setTranslation(translation, translation_velocity);
    };
    Sphere<T, dim> leftLS(TV(-0.521, 0.23, 0.11), 0.1);
    AnalyticCollisionObject<T, dim> leftObject(leftTransform, leftLS, AnalyticCollisionObject<T, dim>::STICKY);
    leftObject.setFriction(1);
    init_helper.addAnalyticCollisionObject(leftObject);
}

{
    auto leftTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
        T t = time;
        TV translation = TV(0, 1 * time, 0) + TV(2, 2, 2);
        TV translation_velocity(0, 1, 0);
        object.setTranslation(translation, translation_velocity);
    };
    Sphere<T, dim> leftLS(TV(0.219, 0.1, 0.12), 0.1);
    AnalyticCollisionObject<T, dim> leftObject(leftTransform, leftLS, AnalyticCollisionObject<T, dim>::STICKY);
    leftObject.setFriction(1);
    init_helper.addAnalyticCollisionObject(leftObject);
}

{
    T radius = 0.14;
    Sphere<T, dim> leftLS(TV(0, 0, 0) + TV(-0.497, -0.36, 0.13) + TV(2, 2, 2), radius);
    AnalyticCollisionObject<T, dim> leftObject(leftLS, AnalyticCollisionObject<T, dim>::STICKY);
    init_helper.addAnalyticCollisionObject(leftObject);
}
{
    T radius = 0.2;
    Sphere<T, dim> leftLS(TV(0, 0, 0) + TV(0.063, -0.36, 0.13) + TV(2, 2, 2), radius);
    AnalyticCollisionObject<T, dim> leftObject(leftLS, AnalyticCollisionObject<T, dim>::STICKY);
    init_helper.addAnalyticCollisionObject(leftObject);
}
{
    T radius = 0.2;
    Sphere<T, dim> leftLS(TV(0, 0, 0) + TV(-0.937, -0.36, 0.13) + TV(2, 2, 2), radius);
    AnalyticCollisionObject<T, dim> leftObject(leftLS, AnalyticCollisionObject<T, dim>::STICKY);
    init_helper.addAnalyticCollisionObject(leftObject);
}

init_helper.addAllWallsInDomain(4096 * sim.dx, 5 * sim.dx, AnalyticCollisionObject<T, dim>::STICKY); // add safety domain walls for SPGrid.