#include "../Force/FBasedMpmForceHelper.cpp"
#include "../Force/FDecoupledBasedMpmForceHelper.h"
namespace ZIRAN {
template class FBasedMpmForceHelper<CorotatedElasticity<float, 2>>;
template class FBasedMpmForceHelper<CorotatedIsotropic<float, 2>>;
template class FBasedMpmForceHelper<CorotatedDistortionalDilational<float, 2>>;
template class FBasedMpmForceHelper<LinearElasticity<float, 2>>;
template class FBasedMpmForceHelper<LinearCorotated<float, 2>>;
template class FBasedMpmForceHelper<NeoHookean<float, 2>>;
template class FBasedMpmForceHelper<NeoHookeanBorden<float, 2>>;
template class FBasedMpmForceHelper<NeoHookeanIsotropic<float, 2>>;
template class FBasedMpmForceHelper<Smudge<float, 2>>;
template class FBasedMpmForceHelper<StVenantKirchhoff<float, 2>>;
template class FBasedMpmForceHelper<StvkWithHencky<float, 2>>;
template class FBasedMpmForceHelper<StvkWithHenckyDecoupled<float, 2>>;
template class FBasedMpmForceHelper<StvkWithHenckyIsotropic<float, 2>>;
template class FBasedMpmForceHelper<StvkWithHenckyIsotropicUnilateral<float, 2>>;
template class FBasedMpmForceHelper<SurfaceTension<float, 2>>;
template class FBasedMpmForceHelper<CotangentOrthotropic<float, 2>>;
template class FBasedMpmForceHelper<QRAnisotropic<float, 2>>;
template class FBasedMpmForceHelper<QRStableNeoHookean<float, 2>>;
template class FDecoupledBasedMpmForceHelper<StvkWithHenckyDecoupled<float, 2>>;
} // namespace ZIRAN
