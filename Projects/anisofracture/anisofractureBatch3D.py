import subprocess

"""
Instructions
1. Set controls in TEST CONTROL CENTER and TEST CONTROL SUBSTATION below
2. Download any necessary files and place them in Data/TetMesh (.mesh) or Data/LevelSets (.vdb)
3. Compile
4. run python anisofractureBatch3D.py 
"""

#TEST CONTROL CENTER
#Set which tests you want to run in the following three lists of demos, then see next section of controls
sectorA = [0,0,0,0,0]                #[diskShoot, cheese, paramEffects, dongpoPork, hangingTorus]
sectorB = [1,0,0,0,0]                #[implicitVsExplicit, brokenHeart, boneTwist, tubeCompress, meatTear]
sectorC = [0,0,0]                    #[tubePull, orange, fish]

#TEST CONTROL SUBSTATION
#Set what runs you want for each demo (e.g. run 0 degree and 90 degree fibers whenever diskShoot is run)
test1 = [0,0,0,1,0]                 #diskShoot: [0deg, 45deg, 90deg, radial, isotropic]
test2 = [1,0]                       #cheese: [aniso, iso]
test3 = [0,0,0,1]                   #paramEffects: [percent, eta, FS, E]
test4 = [1,0,0]                     #dongpoPork: [orthotropic, transverse isotropic, isotropic]
test5 = [0,1]                       #hangingTorus: [anisoElasticity, inextensibility]

test6 = [0,1]                       #implicitVsExplicit: [implicit, explicit]
test7 = [0,1]                       #brokenHeart: [plastic heart, elastic heart]
test8 = [0,0,0,1]                   #boneTwist: [params, twist, pull, bending]
test9 = [0,1]                       #tubeCompress: [aniso, iso]
test10 = [0,1]                      #meatTear: [aniso, iso]

test11 = [0,0,0,1]                  #tubePull: [aniso, iso, anisoDamage only, anisoElasticity only]
test12 = [1]                        #orange: [radialFibers]
test13 = [1]                        #fish: [flowFibers]

##########################################################################

#3D Disk Shoot Test
if sectorA[0]:
    ax_array = [0, 0, 0] #these arrays have fibers for 2D 0deg, 45deg, and 90deg
    ay_array = [0, 1, 1]
    az_array = [1, 1, 0]
    alpha_array = [-1, -1, -1]
    scale_array = [10,10,10,10]
    residual = 0.01
    percent_array = [0.07] #these params are final do not change!
    eta_array = [0.1]
    for j in range(len(percent_array)):
        for k in range(len(eta_array)):
            for i in range(3):
                if test1[i]:
                    runCommand = './anisofracture -test 3 -ax ' + str(ax_array[i]) + ' -ay ' + str(ay_array[i]) + ' -az ' + str(az_array[i]) + ' -alpha1 ' + str(alpha_array[i]) + ' -fiberScale ' + str(scale_array[i]) + ' -residual ' + str(residual) + ' -percent ' + str(percent_array[j]) + ' -eta ' + str(eta_array[k])
                    subprocess.call([runCommand], shell=True)
            if test1[3]:
                runCommand = './anisofracture -test 3 -ax ' + str(1) + ' -ay ' + str(1) +  ' -az ' + str(1) + ' -alpha1 ' + str(-1) + ' -fiberScale ' + str(scale_array[3]) + ' -residual ' + str(residual) + ' -percent ' + str(percent_array[j]) + ' -eta ' + str(eta_array[k]) + ' --useRadial'
                subprocess.call([runCommand], shell=True)
            if test1[4]:
                runCommand = './anisofracture -test 3 -ax ' + str(1) + ' -ay ' + str(1) +  ' -az ' + str(1) + ' -alpha1 ' + str(0) + ' -fiberScale ' + str(1) + ' -residual ' + str(residual) + ' -percent ' + str(percent_array[j]) + ' -eta ' + str(eta_array[k]) + ' --isotropic'
                subprocess.call([runCommand], shell=True)

#3D Cheese Pull Test
if sectorA[1]:
    fiberScale = 10
    residual = 0.01
    percent = 0.19 #these params are final
    eta = 0.1
    if test2[0]:
        runCommand = './anisofracture -test 9 -fiberScale ' + str(fiberScale) + ' -residual ' + str(residual) + ' -percent ' + str(percent) + ' -eta ' + str(eta)
        subprocess.call([runCommand], shell=True)
    if test2[1]:
        runCommand = './anisofracture -test 9 -residual ' + str(residual) + ' -percent ' + str(percent) + ' -eta ' + str(eta) + ' --isotropic'
        subprocess.call([runCommand], shell=True)

#Parameter Effect Demos
if sectorA[2]:
    bestConfig = [0.2, 0.1, 10, 200] #percent, eta, FS, E
    residual = 0.01
    percent_array = [0.05, 0.1, 0.15, 0.3, 0.6] #FINAL: 0.05, 0.1, 0.15, 0.3, 0.6
    eta_array = [0.001, 0.05, 0.2, 1, 100]      #FINAL: 0.001, 0.05, 0.2, 1, 100
    FS_array = [0, 0.5, 1, 20, 200]             #FINAL: 0, 0.5, 1, 20, 100
    E_array = [1, 10, 100, 500, 1000]           #FINAL: 1, 10, 100, 500, 1000
    if test3[0]:
        for percent in percent_array:
            runCommand = './anisofracture -test 13 -fiberScale ' + str(bestConfig[2]) + ' -residual ' + str(residual) + ' -percent ' + str(percent) + ' -eta ' + str(bestConfig[1]) + ' -E ' + str(bestConfig[3])
            subprocess.call([runCommand], shell=True)
    if test3[1]:
        for eta in eta_array:
            runCommand = './anisofracture -test 13 -fiberScale ' + str(bestConfig[2]) + ' -residual ' + str(residual) + ' -percent ' + str(bestConfig[0]) + ' -eta ' + str(eta) + ' -E ' + str(bestConfig[3])
            subprocess.call([runCommand], shell=True)
    if test3[2]:
        for FS in FS_array:
            runCommand = './anisofracture -test 13 -fiberScale ' + str(FS) + ' -residual ' + str(residual) + ' -percent ' + str(bestConfig[0]) + ' -eta ' + str(bestConfig[1]) + ' -E ' + str(bestConfig[3])
            subprocess.call([runCommand], shell=True)
    if test3[3]:
        for E in E_array:
            E_percent = 0.45
            runCommand = './anisofracture -test 13 -fiberScale ' + str(bestConfig[2]) + ' -residual ' + str(residual) + ' -percent ' + str(E_percent) + ' -eta ' + str(bestConfig[1]) + ' -E ' + str(E)
            subprocess.call([runCommand], shell=True)

#Dongpo Pork
if sectorA[3]:
    residual = 0.01
    E = 200
    percent = 0.21
    eta = 0.1
    fiberScale = 10
    if test4[0]:
        runCommand = './anisofracture -test 14 -fiberScale ' + str(fiberScale) + ' -residual ' + str(residual) + ' -percent ' + str(percent) + ' -eta ' + str(eta) + ' -E ' + str(E) + ' --orthotropic'
        subprocess.call([runCommand], shell=True)
    if test4[1]:
        runCommand = './anisofracture -test 14 -fiberScale ' + str(fiberScale) + ' -residual ' + str(residual) + ' -percent ' + str(percent) + ' -eta ' + str(eta) + ' -E ' + str(E)
        subprocess.call([runCommand], shell=True)
    if test4[2]:
        runCommand = './anisofracture -test 14 -fiberScale ' + str(fiberScale) + ' -residual ' + str(residual) + ' -percent ' + str(percent) + ' -eta ' + str(eta) + ' -E ' + str(E) + ' --isotropic'
        subprocess.call([runCommand], shell=True)

#Hanging Torus
if sectorA[4]:
    residual = 0.01
    torusE = 100
    rho_array = [0.75, 1, 2, 3, 5, 10, 20, 25, 30] #FINAL PARAMS
    fiberScale = 100
    if test5[0]:
        for torusRho in rho_array:
            runCommand = './anisofracture -test 16 -residual ' + str(residual) + ' -E ' + str(torusE) + ' -rho ' + str(torusRho) + ' -fiberScale ' + str(fiberScale)
            subprocess.call([runCommand], shell=True)
    if test5[1]:
        inextRho = 0.75 #We ran this with both 0.75 and 20!
        runCommand = './anisofracture -test 16 -residual ' + str(residual) + ' -E ' + str(torusE) + ' -rho ' + str(inextRho) + ' --inextensible'
        subprocess.call([runCommand], shell=True)

#Implicit vs Explicit
if sectorB[0]:
    bestConfig = [0.45, 1e-9, 10, 200] #percent, eta, FS, E
    residual = 0.01
    if test6[0]:
        runCommand = './anisofracture -test 19 -fiberScale ' + str(bestConfig[2]) + ' -residual ' + str(residual) + ' -percent ' + str(bestConfig[0]) + ' -eta ' + str(bestConfig[1]) + ' -E ' + str(bestConfig[3]) + ' --implicit_damage'
        subprocess.call([runCommand], shell=True)
    if test6[1]:
        runCommand = './anisofracture -test 19 -fiberScale ' + str(bestConfig[2]) + ' -residual ' + str(residual) + ' -percent ' + str(bestConfig[0]) + ' -eta ' + str(bestConfig[1]) + ' -E ' + str(bestConfig[3])
        subprocess.call([runCommand], shell=True)

#Broken Heart (Von Mises Plasticity)
if sectorB[1]:
    residual = 0.01 #final param is 0.01
    percent = 0.183 #final param is 0.183
    tau = 4 #final param is 4
    eta = 0.1 #final param is 0.1
    fiberScale = 3 #final param is 3
    percentArray = [0.1, 0.2, 0.3, 0.4]
    tauArray = [1, 10, 100, 1000]
    if test7[0]:
        runCommand = './anisofracture -test 22 -residual ' + str(residual) + ' -percent ' + str(percent) + ' -eta ' + str(eta) + ' -fiberScale ' + str(fiberScale) + ' -tau ' + str(tau) + ' --orthotropic'
        subprocess.call([runCommand], shell=True)
    if test7[1]:
        runCommand = './anisofracture -test 22 -residual ' + str(residual) + ' -percent ' + str(percent) + ' -eta ' + str(eta) + ' -fiberScale ' + str(fiberScale) + ' -tau ' + str(tau)
        subprocess.call([runCommand], shell=True)

# Bone
if sectorB[2]:
    E = 7e7#10000
    percent = 0.1 #try between 0.1 and 0.15
    eta = 0.1
    fiberScale = 0 #try between 10 and 0
    residual = 0.0001 #try between 0.01 and 0.0000001 #tried:1e-4;
    percentArray = [0.085, 0.09, 0.095]
    if test8[0]:#params
        for p in percentArray:
            runCommand = './anisofracture -test 25 -residual ' + str(residual) + ' -percent ' + str(p) + ' -eta ' + str(eta) + ' -fiberScale ' + str(fiberScale) + ' -E ' + str(E)
            subprocess.call([runCommand], shell=True)
    if test8[1]:#twisting
        runCommand = './anisofracture -test 25 -residual ' + str(residual) + ' -percent ' + str(0.1) + ' -eta ' + str(0.1) + ' -fiberScale ' + str(0) + ' -E ' + str(7e7)
        subprocess.call([runCommand], shell=True)
    if test8[2]:#pulling
        runCommand = './anisofracture -test 25 -residual ' + str(residual) + ' -percent ' + str(0.05) + ' -eta ' + str(0.1) + ' -fiberScale ' + str(0) + ' -E ' + str(7e7)
        subprocess.call([runCommand], shell=True)
    if test8[3]:#bending
        runCommand = './anisofracture -test 25 -residual ' + str(residual) + ' -percent ' + str(0.05) + ' -eta ' + str(0.1) + ' -fiberScale ' + str(0) + ' -E ' + str(7e8)
        subprocess.call([runCommand], shell=True)

#3D tube compress
if sectorB[3]:
    residual = 0.01
    percent = 0.15
    eta = 0.3
    fiberScale = 10
    youngs = 5000
    if test9[0]:
        runCommand = './anisofracture -test 21 -residual ' + str(residual) + ' -percent ' + str(percent) + ' -eta ' + str(eta) + ' -fiberScale ' + str(fiberScale) + ' -E ' + str(youngs)
        subprocess.call([runCommand], shell=True)
    if test9[1]:
        runCommand = './anisofracture -test 21 -residual ' + str(residual) + ' -percent ' + str(percent) + ' -eta ' + str(eta) + ' -E ' + str(youngs) + ' --isotropic'
        subprocess.call([runCommand], shell=True)

#Meat Tear
if sectorB[4]:
    if test10[0]:
        runCommand = './anisofracture -test 5 -helper 1' 
        subprocess.call([runCommand], shell=True)
    if test10[1]:
        runCommand = './anisofracture -test 5 -helper 2' 
        subprocess.call([runCommand], shell=True)

#Tube Pull
if sectorC[0]:
    if test11[0]:
        runCommand = './anisofracture -test 2 -helper 2' 
        subprocess.call([runCommand], shell=True)
    if test11[1]:
        runCommand = './anisofracture -test 2 -helper 1' 
        subprocess.call([runCommand], shell=True)
    if test11[2]:
        runCommand = './anisofracture -test 2 -helper 3' 
        subprocess.call([runCommand], shell=True)
    if test11[3]:
        runCommand = './anisofracture -test 2 -helper 4' 
        subprocess.call([runCommand], shell=True)

#Orange
if sectorC[1]:
    if test12[0]:
        runCommand = './anisofracture -test 8' 
        subprocess.call([runCommand], shell=True)

#Fish
if sectorC[2]:
    if test13[0]:
        runCommand = './anisofracture -test 12' 
        subprocess.call([runCommand], shell=True)