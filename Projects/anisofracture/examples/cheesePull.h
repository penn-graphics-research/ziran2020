//Cheese Pull -- peeling apart a cheese stick with a sphere boundary
// ./anisofracture -test 9 [OPTIONS]

//THESE PARAMETERS ARE FINAL, DO NOT CHANGE!!
//Python params
//residual = 0.01
//percent = 0.19
//eta = 0.1
//fiberScale = 10

sim.end_frame = 160;
T frameRate = 24;
sim.step.frame_dt = (T)1 / frameRate;
sim.gravity = -3 * TV::Unit(1);
sim.step.max_dt = 2e-4; //1.5 for fanfu
sim.symplectic = true;
sim.verbose = false;
sim.cfl = 0.4;
//sim.transfer_scheme = MpmSimulationBase<T, dim>::APIC_blend_RPIC;
//sim.apic_rpic_ratio = 0.01;
sim.transfer_scheme = MpmSimulationBase<T, dim>::FLIP_blend_PIC;
sim.flip_pic_ratio = 0.0; //0 is full PIC
//sim.rpic_damping_iteration = 5; //THis looks better without damping!
sim.dump_F_for_meshing = true;
T particle_per_cell = 7; //was 7 before

T Youngs = 5000;
T nu = 0.45;
T rho = 500;

// sample particles from a .mesh file
std::string filename = "TetMesh/cheeseStick500k.mesh"; //50k is plenty for nice dynamics and no debris
MpmParticleHandleBase<T, dim> particles_handle = init_helper.sampleFromTetWildFile(filename, rho);
T total_volume = particles_handle.total_volume;
T particle_count = particles_handle.particle_range.length();
T per_particle_volume = total_volume / particle_count;

// set dx
sim.dx = std::pow(particle_per_cell * per_particle_volume, (T)1 / (T)3);

//T suggested_dt = evaluateTimestepLinearElasticityAnalysis(Youngs, nu, rho, sim.dx, sim.cfl);
//if (sim.symplectic) { ZIRAN_ASSERT(sim.step.max_dt <= suggested_dt, suggested_dt); }

//Anisotropic fracture params, grab these from the flags
StdVector<TV> a_0;
StdVector<T> alphas;
TV a_1, a_2;
a_1[0] = 0;
a_1[1] = 1; //fiber direction should always just be straight up
a_1[2] = 0;
a_1.normalize();
a_0.push_back(a_1);
if (Python::isotropic) {
    alphas.push_back(0); //for isotropic set alpha 0
}
else {
    alphas.push_back(-1); //for aniso set alpha -1
}

T percentage = Python::percent; //0.07 is good
T l0 = 0.5 * sim.dx;
T eta = Python::eta; //0.08 is good
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

std::string scale_str = std::to_string(Python::fiberScale);
scale_str.erase(scale_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "FiberScale = " << Python::fiberScale << std::endl;

std::string res_str = std::to_string(Python::residual);
res_str.erase(res_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "Residual = " << Python::residual << std::endl;

std::string dx_str = std::to_string(sim.dx);
dx_str.erase(dx_str.find_last_not_of('0') + 1, std::string::npos);
std::cout << "SimDx = " << sim.dx << std::endl;

if (Python::isotropic) {
    std::string path("output/3D_CheesePull/3D_CheesePull_Isotropic_Youngs" + E_str + "_percent" + percent_str + "_eta" + eta_str + "_dx" + dx_str + "_fiberScale" + scale_str + "_residual" + res_str);
    sim.output_dir.path = path;
}
else {
    std::string path("output/3D_CheesePull/3D_CheesePull_Youngs" + E_str + "_percent" + percent_str + "_eta" + eta_str + "_dx" + dx_str + "_fiberScale" + scale_str + "_residual" + res_str + "_slackANDholdAndRelease_andBetterSpacedCuts");
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

//Set up a ground plane
TV ground_origin = TV(2, 1.75, 2);
TV ground_normal(0, 1, 0);
HalfSpace<T, dim> ground_ls(ground_origin, ground_normal);
AnalyticCollisionObject<T, dim> ground_object(ground_ls, AnalyticCollisionObject<T, dim>::SEPARATE);
ground_object.setFriction(0.5);
init_helper.addAnalyticCollisionObject(ground_object);

// Cheese Spine
T spineRadius = sim.dx * 1.01; //make sure the spine is at least 2*dx wide!
TV spineCenter(2, 2, 2);
T spineHeight = 0.6;
Vector<T, 4> q(1, 0, 0, 0);
CappedCylinder<T, dim> spineLS(spineRadius, spineHeight, q, spineCenter);

AnalyticCollisionObject<T, dim> spine_object(spineLS, AnalyticCollisionObject<T, dim>::STICKY);
init_helper.addAnalyticCollisionObject(spine_object);

T cutTheta = 26.57; //26.57 is BEST!! -- degrees down from x-axis that we will pull the cheese
TV cutDir1(1.0, std::tan(cutTheta*(M_PI / 180.0)), 0); //this gives a sense of how much down and outward each cut should move based on theta
cutDir1.normalize();
TV cheeseCenter(2, 2, 2);

TV base(2, 2.25, 2);
TV startCenter(2, 2.255, 2.031); //2.031 is BEST!!!
T sphereRadius = 0.015;
TV baseCenter = startCenter - base;

//define 12 cutters, three rounds of four cuts!
//StdVector<T> cutList = { 0, 90, 180, 270, 30, 120, 210, 300 }; // 60, 150, 240, 330 }; //IN DEGREES!!!! this list determines the rotation of the cutter around the cheese, startCenter is considered theta = 0
StdVector<T> cutList = { 0, 90, 180, 270, 45, 135, 225, 315 };

//Define cutter behavior
T cutTime = 1.25; //1.5, how long should cutter pull for?
T cutDelay = 1; //1, how long should next cutter wait to start cutting?
T slackDuration = 0.5; //0.5, how long to give the strand slack before releasing it
T holdDuration = 0.5; //0.25 hold
StdVector<T> startTimes = { 0, 0, 0, 0, cutDelay, cutDelay, cutDelay, cutDelay }; // cutDelay * 2, cutDelay * 2, cutDelay * 2, cutDelay * 2 };

for (unsigned int i = 0; i < cutList.size(); i++) {

    //Define next sphere center to cut with
    T rotateTheta = cutList[i] * (M_PI / 180.0); //now we specifically define each cut!
    TM rotateAroundYAxis;
    rotateAroundYAxis << std::cos(rotateTheta), 0, -1 * std::sin(rotateTheta), 0, 1, 0, std::sin(rotateTheta), 0, std::cos(rotateTheta);
    TV sphereCenter = (rotateAroundYAxis * baseCenter) + base;

    TV cutDir2 = sphereCenter - cheeseCenter;
    cutDir2[1] = 0; //zero out yDim
    cutDir2.normalize(); //normalize after zeroing y

    auto sphereTransform = [=](T time, AnalyticCollisionObject<T, dim>& object) {
        //These parameters control the speed and scheduling of the cutters
        T speed = 0.1; //how fast should cutter move?

        T startTime = startTimes[i];
        T endTime = startTime + cutTime;
        T slackTime = endTime + slackDuration;
        T holdTime = slackTime + holdDuration;

        T xVel = speed * cutDir1[0] * cutDir2[0];
        T yVel = -speed * cutDir1[1];
        T zVel = speed * cutDir1[0] * cutDir2[2];
        //        T x0 = sphereCenter[0];
        //        T y0 = sphereCenter[1];
        //        T z0 = sphereCenter[2];
        T x0 = 0;
        T y0 = 0;
        T z0 = 0;

        if (time < startTime) { //don't move yet
            TV translation = TV(10, 10, 10);
            TV translation_velocity(0, 0, 0);
            object.setTranslation(translation, translation_velocity);
        }
        else if (time < endTime) { //normal translation
            TV translation = TV(x0 + (xVel * (time - startTime)), y0 + (yVel * (time - startTime)), z0 + (zVel * (time - startTime)));
            TV translation_velocity(xVel, yVel, zVel);
            object.setTranslation(translation, translation_velocity);
        }
        else if (time < slackTime) { //slacken the strands before releasing
            TV translation = TV(x0 + (xVel * (endTime - startTime)) - ((time - endTime) * (xVel * 0.5)), y0 + (yVel * (endTime - startTime)) - ((time - endTime) * (yVel * 0.5)), z0 + (zVel * (endTime - startTime)) - ((time - endTime) * (zVel * 0.5))); //move slightly in towards cheese
            TV translation_velocity(-1 * xVel * 0.5, -1 * yVel * 0.5, -1 * zVel * 0.5);
            object.setTranslation(translation, translation_velocity);
        }
        else if (time < holdTime) {
            TV translation = TV(x0 + (xVel * (endTime - startTime)) - ((slackTime - endTime) * (xVel * 0.5)), y0 + (yVel * (endTime - startTime)) - ((slackTime - endTime) * (yVel * 0.5)), z0 + (zVel * (endTime - startTime)) - ((slackTime - endTime) * (zVel * 0.5))); //move slightly in towards cheese
            TV translation_velocity(0, 0, 0); //trying slack time = hold still
            object.setTranslation(translation, translation_velocity);
        }
        else { //teleport far away
            TV translation = TV(99999, 99999, 99999);
            TV translation_velocity(0, 0, 0);
            object.setTranslation(translation, translation_velocity);
        }
        /*
             *
             *  if(time < 0.5) {
             *      tranlsation = TV(time * vx, time * vy, time* vz)
         *      }
             *  else if (time < 0.8) {
             *      translation  =TV(0.5 * vx - (t-0.5)* vx/3, 0.5 * vy, 0.5 *vz - (t-0.5)*vz/3)
             *  }
             *  else {
             *      translation = TV(9,99,99)
             *
             *  }
             *
             *  }
             *
             *
             */
    };

    Sphere<T, dim> sphere(sphereCenter, sphereRadius);
    //Sphere<T, dim> sphere(TV(10,10,10), sphereRadius); //start all cutters far from cheese
    AnalyticCollisionObject<T, dim> sphereObject(sphereTransform, sphere, AnalyticCollisionObject<T, dim>::STICKY);
    init_helper.addAnalyticCollisionObject(sphereObject);
}
