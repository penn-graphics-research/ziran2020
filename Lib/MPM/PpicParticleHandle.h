#ifndef PPIC_PARTICLE_HANDLE_H
#define PPIC_PARTICLE_HANDLE_H
#include <functional>

#include <MPM/MpmParticleHandleBase.h>
#include <sol.hpp>
#include <MPM/PpicSimulation.h>

namespace ZIRAN {

template <class T, int dim>
class PpicParticleHandle : public MpmParticleHandleBase<T, dim> {
public:
    using Base = MpmParticleHandleBase<T, dim>;

    PpicParticleHandle(const MpmParticleHandleBase<T, dim>& base)
        : Base(base)
    {
    }
};
} // namespace ZIRAN

#endif
