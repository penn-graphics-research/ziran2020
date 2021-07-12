## Code description 

This is the opensource code for the following papers:

(1) AnisoMPM: Animating Anisotropic Damage Mechanics, Joshuah Wolper, Yunuo Chen, Minchen Li, Yu Fang, Ziyin Qu, Jiecong Lu, Meggie Cheng, Chenfanfu Jiang (SIGGRAPH 2020)
Project Page: https://joshuahwolper.com/anisompm

(2) IQ-MPM: An Interface Quadrature Material Point Method for Non-sticky Strongly Two-way Coupled Nonlinear Solids and Fluids, Yu Fang*, Ziyin Qu* (equal contributions), Minchen Li, Xinxin Zhang, Yixin Zhu, Mridul Aanjaneya, Chenfanfu Jiang (SIGGRAPH 2020)

## Troubleshooting Compiling

If anyone encounters compiling errors with GNU 9.3.0 or other versions, please switch to GNU 7.5.0 to compile which we've verified to work.

Use the following command to install GNU 7.5.0.
```
sudo apt-get install g++-7
```

When executing cmake, users should manually choose the compiler to use as well.

```
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc-7 -DCMAKE_CXX_COMPILER=/usr/bin/g++-7
```

## Unzip Data

Download and unzip the following file in the root directory.

https://drive.google.com/file/d/1f22ObJmiq8H1_YObTz5BuE4IAWLQhyuB/view?usp=sharing

You need to do this due to the github single file size limit.

## Dependencies Installation

    sudo apt-get install make cmake g++ libeigen3-dev gfortran libmetis-dev
    sudo apt-get install libopenvdb-dev libboost-all-dev libilmbase-dev libopenexr-dev
    sudo apt-get install libtbb2 libtbb-dev libz-dev clang-format-6.0 clang-format
   
## Building in Ziran

    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j 4

## Running Demos

    Check folder Projects/anisofracture for AnisoMPM
    Check folder Projects/coupling for IQ-MPM

## Sampling Particles

We used TetWild to compute tetrahedralized volumes to sample particles from, check it out here: https://github.com/Yixin-Hu/TetWild

## Visualization

We dump out .bgeo files for visualization and use Houdini to visualize our outputs. For converting a point cloud to a surface we use the meshing algorithm seen here: https://dl.acm.org/doi/10.1145/3340259, but you also have the option to use Houdini's VDB support to directly compute a VDB level set surface from the output particles (see https://www.sidefx.com/docs/houdini/nodes/sop/vdbfromparticles.html)

## BibTeX

Please cite our papers if you use this code for your research: 
```
@article{wolper2020anisompm,
  title={AnisoMPM: Animating Anisotropic Damage Mechanics},
  author={Wolper, Joshuah and Chen, Yunuo and Li, Minchen and Fang, Yu and Qu, Ziyin and Lu, Jiecong and Cheng, Meggie and Jiang, Chenfanfu},
  journal={ACM Trans. Graph.},
  volume={39},
  number={4},
  year={2020},
  publisher={ACM}
}
```
```
@article{fang2020iqmpm,
  title={IQ-MPM: An Interface Quadrature Material Point Method for Non-Sticky Strongly Two-Way Coupled Nonlinear Solids and Fluids},
  author={Fang, Yu and Qu, Ziyin and Li, Minchen and Zhang, Xinxin and Zhu, Yixin and Aanjaneya, Mridul and Jiang, Chenfanfu},
  volume={39},
  journal={ACM Trans. Graph.},
  number={4},
  year={2020},
  publisher={ACM},
}


```
