## Code description 

This is the opensource code for the following papers:

(1) AnisoMPM: Animating Anisotropic Damage Mechanics, Joshuah Wolper, Yunuo Chen, Minchen Li, Yu Fang, Ziyin Qu, Jiecong Lu, Meggie Cheng, Chenfanfu Jiang (SIGGRAPH 2020)

(2) IQ-MPM: An Interface Quadrature Material Point Method for Non-sticky Strongly Two-way Coupled Nonlinear Solids and Fluids, Yu Fang*, Ziyin Qu* (equal contributions), Minchen Li, Xinxin Zhang, Yixin Zhu, Mridul Aanjaneya, Chenfanfu Jiang (SIGGRAPH 2020)

## Unzip Data

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

## Bibtex

Please cite our papers if you use this code for your research: 
```
@article{fang2019silly,
  title={Silly rubber: an implicit material point method for simulating non-equilibrated viscoelastic and elastoplastic solids},
  author={Fang, Yu and Li, Minchen and Gao, Ming and Jiang, Chenfanfu},
  journal={ACM Transactions on Graphics (TOG)},
  volume={39},
  number={4},
  pages={118},
  year={2019},
  publisher={ACM}
}
```
```
@article{wolper2019cd,
  title={CD-MPM: Continuum damage material point methods for dynamic fracture animation},
  author={Wolper, Joshuah and Fang, Yu and Li, Minchen and Lu, Jiecong and Gao, Ming and Jiang, Chenfanfu},
  journal={ACM Transactions on Graphics (TOG)},
  volume={38},
  number={4},
  pages={119},
  year={2019},
  publisher={ACM}
}
```
