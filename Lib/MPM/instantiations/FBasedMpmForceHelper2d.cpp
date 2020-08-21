#include "../Force/FBasedMpmForceHelper.cpp"
#include "../Force/FDecoupledBasedMpmForceHelper.h"
namespace ZIRAN {
template class FBasedMpmForceHelper<CorotatedElasticity<double, 2>>;
template class FBasedMpmForceHelper<CorotatedIsotropic<double, 2>>;
template class FBasedMpmForceHelper<CorotatedDistortionalDilational<double, 2>>;
template class FBasedMpmForceHelper<LinearElasticity<double, 2>>;
template class FBasedMpmForceHelper<LinearCorotated<double, 2>>;
template class FBasedMpmForceHelper<NeoHookean<double, 2>>;
template class FBasedMpmForceHelper<NeoHookeanBorden<double, 2>>;
template class FBasedMpmForceHelper<NeoHookeanIsotropic<double, 2>>;
template class FBasedMpmForceHelper<Smudge<double, 2>>;
template class FBasedMpmForceHelper<StVenantKirchhoff<double, 2>>;
template class FBasedMpmForceHelper<StvkWithHencky<double, 2>>;
template class FBasedMpmForceHelper<StvkWithHenckyDecoupled<double, 2>>;
template class FBasedMpmForceHelper<StvkWithHenckyIsotropic<double, 2>>;
template class FBasedMpmForceHelper<StvkWithHenckyIsotropicUnilateral<double, 2>>;
template class FBasedMpmForceHelper<SurfaceTension<double, 2>>;
template class FBasedMpmForceHelper<CotangentOrthotropic<double, 2>>;
template class FBasedMpmForceHelper<QRAnisotropic<double, 2>>;
template class FBasedMpmForceHelper<QRStableNeoHookean<double, 2>>;
template class FDecoupledBasedMpmForceHelper<StvkWithHenckyDecoupled<double, 2>>;
} // namespace ZIRAN
