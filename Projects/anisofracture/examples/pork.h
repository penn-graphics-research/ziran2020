// DongPo Pork Belly
// ./anisofracture -test 14

//FINAL PARAMS FROM PYTHON

bool orthotropic = Python::orthotropic;

sim.end_frame = 384; //72
T frameRate = 24; //24
sim.step.frame_dt = (T)1 / frameRate;
sim.dx = 0.01; //use 0.0025 for high res
sim.gravity = -3 * TV::Unit(1); //no gravity
sim.step.max_dt = /*1e-3*/ /*3e-4*/ 2.5e-4; //with test params can use 8e-5
sim.newton.max_iterations = 5;
sim.newton.tolerance = 1e-3;
sim.objective.minres.max_iterations = 10000;
sim.objective.minres.tolerance = 1e-4;
sim.quasistatic = false;
sim.symplectic = true; // want explicit!
sim.objective.matrix_free = true;
sim.verbose = false;
sim.cfl = 0.4;
// sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;  //orthotropic and trans iso use full RPIC
// sim.apic_rpic_ratio = 0; //full RPIC
sim.transfer_scheme = MpmSimulationBase<T, dim>::FLIP_blend_PIC;
sim.flip_pic_ratio = 0.0; //isotropic uses full PIC
sim.dump_F_for_meshing = true;

//Test params
T Youngs = Python::E;
T nu = 0.25;
T rho = 2;
QRStableNeoHookean<T, dim> model(Youngs, nu);

// sample particles from a .mesh file
std::string filename = "TetMesh/porkBelly500k.mesh"; //50k is plenty for nice dynamics and no debris
MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleFromTetWildFile(filename, rho);
T total_volume = particles_handle.total_volume;
T particle_count = particles_handle.particle_range.length();
T per_particle_volume = total_volume / particle_count;

int particle_per_cell = 7;

// set dx
sim.dx = std::pow(particle_per_cell * per_particle_volume, (T)1 / (T)3);

//Orthotropic parameters!
StdVector<TV> a_0;
StdVector<T> alphas;
TV a_1, a_2;
if (orthotropic) { //orthotropic
    a_1[0] = 1;
    a_1[1] = 0;
    a_1[2] = 0;
    a_1.normalize();
    a_0.push_back(a_1);
    alphas.push_back(-1);
    a_2[0] = 0;
    a_2[1] = 0;
    a_2[2] = 1;
    a_2.normalize();
    a_0.push_back(a_2);
    alphas.push_back(-1);
}
else if (!Python::isotropic) { //transverse isotropic
    a_1[0] = 1;
    a_1[1] = 0; //45 deg in XZ plane
    a_1[2] = 1;
    a_1.normalize();
    a_0.push_back(a_1);
    alphas.push_back(-1);
}
else { //isotropic
    a_1[0] = 1;
    a_1[1] = 0;
    a_1[2] = 0;
    a_1.normalize();
    a_0.push_back(a_1);
    alphas.push_back(0);
}

T percentage = Python::percent;
T l0 = 0.5 * sim.dx;
T eta = Python::eta; //set this as nu since it's what we used before
T zeta = 1;
bool allow_damage = true;

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

std::string scale_str = std::to_string(Python::fiberScale);
scale_str.erase(scale_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "FiberScale = " << Python::fiberScale << std::endl;

if (Python::isotropic) {
    std::string path("output/3D_Pork/3D_Pork_Isotropic_Youngs" + E_str + "_percent" + percent_str + "_eta" + eta_str + "_dx" + dx_str + "_fiberScale" + scale_str);
    sim.output_dir.path = path;
}
else if (orthotropic) {
    std::string path("output/3D_Pork/3D_Pork_Orthotropic_Youngs" + E_str + "_percent" + percent_str + "_eta" + eta_str + "_dx" + dx_str + "_fiberScale" + scale_str);
    sim.output_dir.path = path;
    model.setExtraFiberStiffness(0, Python::fiberScale); //only scale elasticity if anisotropic!
    model.setExtraFiberStiffness(1, Python::fiberScale);
}
else {
    std::string path("output/3D_Pork/3D_Pork_TransverseIsotropic_Youngs" + E_str + "_percent" + percent_str + "_eta" + eta_str + "_dx" + dx_str + "_fiberScale" + scale_str);
    sim.output_dir.path = path;
    model.setExtraFiberStiffness(0, Python::fiberScale); //only scale elasticity if anisotropic!
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

//Setup the box material
T xSize = 0.4;
T ySize = 0.2;
T zSize = 0.4;
T materialToHold = 0.03;
TV boxMin(2 - (xSize / 2.0), 2 - (ySize / 2.0) - materialToHold, 2 - (zSize / 2.0)); //needs extra material at bottom to hold onto for mode 1
TV boxMax(2 + (xSize / 2.0), 2 + (ySize / 2.0), 2 + (zSize / 2.0));
//AxisAlignedAnalyticBox<T, dim> boxLevelSet(boxMin, boxMax);

//Sample particles in the desired level set
//int ppc = 7;
//MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleInAnalyticLevelSet(boxLevelSet, rho, ppc);

//Elasticity Handler
particles_handle.addFBasedMpmForceWithAnisotropicPhaseField(a_0, alphas, percentage, l0, model, eta, zeta, allow_damage, Python::residual);

//Now setup the boundaries
//Holder Box
TV holderMin = boxMin - TV(0.5, 1.0, 0.5);
TV holderMax(boxMax[0] + 0.5, boxMin[1] + materialToHold, boxMax[2] + 0.5);
AxisAlignedAnalyticBox<T, dim> holderLS(holderMin, holderMax);
AnalyticCollisionObject<T, dim> holderObj(holderLS, AnalyticCollisionObject<T, dim>::STICKY);
init_helper.addAnalyticCollisionObject(holderObj);

//Sphere puller
//TV sphereCenter(1.8, 2.1, 1.8);
//T sphereRadius = 0.05;
TV sphereCenter(1.865, 2.198, 1.865);
T sphereRadius = 0.05;

T startTime = 6; //wait some time to let it rest

auto sphereTransform = [=](T time, AnalyticCollisionObject<T, dim>& object) {
    T speed = 0.05;
    T xVel = speed;
    T yVel = speed;
    T zVel = speed;

    if (time < startTime) { //don't move yet
        TV translation = TV(10, 10, 10);
        TV translation_velocity(0, 0, 0);
        object.setTranslation(translation, translation_velocity);
    }
    else { //normal translation
        TV translation = TV(xVel * (time - startTime), yVel * (time - startTime), zVel * (time - startTime));
        TV translation_velocity(xVel, yVel, zVel);
        object.setTranslation(translation, translation_velocity);
    }
};

Sphere<T, dim> sphere(sphereCenter, sphereRadius);
AnalyticCollisionObject<T, dim> sphereObject(sphereTransform, sphere, AnalyticCollisionObject<T, dim>::STICKY);
init_helper.addAnalyticCollisionObject(sphereObject);

//Holder Wall
//T wallX = 2 + 0.37 - (0.5 / 2.0);
//HalfSpace<T, dim> holderLS(TV(wallX, 2, 2), TV(-1, 0, 0)); //origin, normal
//AnalyticCollisionObject<T, dim> wallObj(holderLS, AnalyticCollisionObject<T, dim>::STICKY);
//init_helper.addAnalyticCollisionObject(wallObj);

//TV sphereCenter(1.61, 1.74, 2);
//T sphereRadius = 0.2;
//
//auto sphereTransform = [](T time, AnalyticCollisionObject<T, dim>& object) {
//    T speed = 0.1;
//    TV translation_velocity(0, 1 * speed, 0);
//    TV translation(0, 1 * speed * time, 0); //multiply each velocity by dt to get dx!
//    object.setTranslation(translation, translation_velocity);
//};
//
//Sphere<T, dim> sphere(sphereCenter, sphereRadius);
//AnalyticCollisionObject<T, dim> sphereObject(sphereTransform, sphere, AnalyticCollisionObject<T, dim>::SEPARATE);
//sphereObject.setFriction(0.9); //need friction to try and separate layers!
//init_helper.addAnalyticCollisionObject(sphereObject);
