#include "../Force/FBasedMpmForceHelper.cpp"
#include "../Force/FDecoupledBasedMpmForceHelper.h"
namespace ZIRAN {
template class FBasedMpmForceHelper<CorotatedElasticity<float, 3>>;
template class FBasedMpmForceHelper<CorotatedIsotropic<float, 3>>;
template class FBasedMpmForceHelper<LinearElasticity<float, 3>>;
template class FBasedMpmForceHelper<LinearCorotated<float, 3>>;
template class FBasedMpmForceHelper<NeoHookean<float, 3>>;
template class FBasedMpmForceHelper<NeoHookeanBorden<float, 3>>;
template class FBasedMpmForceHelper<NeoHookeanIsotropic<float, 3>>;
template class FBasedMpmForceHelper<Smudge<float, 3>>;
template class FBasedMpmForceHelper<StVenantKirchhoff<float, 3>>;
template class FBasedMpmForceHelper<StvkWithHencky<float, 3>>;
template class FBasedMpmForceHelper<StvkWithHenckyDecoupled<float, 3>>;
template class FBasedMpmForceHelper<StvkWithHenckyIsotropic<float, 3>>;
template class FBasedMpmForceHelper<StvkWithHenckyIsotropicUnilateral<float, 3>>;
template class FBasedMpmForceHelper<SurfaceTension<float, 3>>;
template class FBasedMpmForceHelper<CotangentOrthotropic<float, 3>>;
template class FBasedMpmForceHelper<QRAnisotropic<float, 3>>;
template class FBasedMpmForceHelper<QRStableNeoHookean<float, 3>>;
template class FDecoupledBasedMpmForceHelper<StvkWithHenckyDecoupled<float, 3>>;
} // namespace ZIRAN
