import subprocess

#TEST CONTROL CENTER
test2 = [1,1,1,1,1,1] # [0, 45, 90, radial, longitudinal, isotropic]

#2D Disk Tear Test (0, 45, 90, radial, longitudinal, and isotropic)
ax_array = [1, 1, 0] #these arrays have fibers for 2D 0deg, 45deg, and 90deg
ay_array = [0, 1, 1]
alpha_array = [-1, -1, -1]
scale_array = [50, 50, 50, 50, 50]
residual = 0.01
percent = 0.35
eta = 0.1
for i in range(3):
    if test2[i]:
        runCommand = './anisofracture -test 6 -ax ' + str(ax_array[i]) + ' -ay ' + str(ay_array[i]) + ' -alpha1 ' + str(alpha_array[i]) + ' -fiberScale ' + str(scale_array[i]) + ' -residual ' + str(residual) + ' -percent ' + str(percent) + ' -eta ' + str(eta)
        subprocess.call([runCommand], shell=True)
if test2[3]:
    runCommand = './anisofracture -test 6 -ax ' + str(1) + ' -ay ' + str(1) + ' -alpha1 ' + str(-1) + ' -fiberScale ' + str(scale_array[3]) + ' -residual ' + str(residual) + ' -percent ' + str(percent) + ' -eta ' + str(eta) + ' --useRadial'
    subprocess.call([runCommand], shell=True)
if test2[4]:
    runCommand = './anisofracture -test 6 -ax ' + str(1) + ' -ay ' + str(1) + ' -alpha1 ' + str(-1) + ' -fiberScale ' + str(scale_array[4]) + ' -residual ' + str(residual) + ' -percent ' + str(percent) + ' -eta ' + str(eta) + ' --useLongitudinal'
    subprocess.call([runCommand], shell=True)
if test2[5]:
    runCommand = './anisofracture -test 6 -ax ' + str(1) + ' -ay ' + str(1) + ' -alpha1 ' + str(0) + ' -fiberScale ' + str(1) + ' -residual ' + str(residual) + ' -percent ' + str(percent) + ' -eta ' + str(eta) + ' --isotropic'
    subprocess.call([runCommand], shell=True)