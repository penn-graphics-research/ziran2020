Welcome to AnisoMPM!

This code distribution contains one 2D demo and 13 3D demos as follows:
2D: diskTear
3D: diskShoot
    cheese
    parameterEffects
    dongpoPork
    hangingTorus
    implicitVsExplicit
    brokenHeart
    boneTwist
    tubeCompress
    meatTear
    tubePull
    orange
    fish

Instructions for Use:
0. Download (and unzip in root directory of ziran2020) all of the data directory found here: https://www.seas.upenn.edu/~cffjiang/research/ziran2020/Data.zip
1. To switch between 2D/3D demos, go into anisofracture.cpp and change line 14 to define the problem dimension.
2. Open anisofractureBatch2D.py or anisofractureBatch3D.py and set the controls within them to dictate which demos will be run.
3. Make sure all files are downloaded and correctly organized in either Data/TetMesh (for .mesh) or Data/LevelSets (for .vdb)
4. Compile in build directory as "make anisofracture -j8"
5. To run your selected demos, run "python anisofractureBatch3D.py" (for 3D).

**Each demo can also be changed directly in the examples folder, but the Python approach requires minimal
compile time: simply compile once and all demos should be runnable! 
