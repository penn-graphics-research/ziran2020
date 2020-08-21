#include "../Force/FBasedMpmForceHelper.cpp"
#include "../Force/FDecoupledBasedMpmForceHelper.h"
namespace ZIRAN {
template class FBasedMpmForceHelper<CorotatedElasticity<double, 3>>;
template class FBasedMpmForceHelper<CorotatedIsotropic<double, 3>>;
template class FBasedMpmForceHelper<LinearElasticity<double, 3>>;
template class FBasedMpmForceHelper<LinearCorotated<double, 3>>;
template class FBasedMpmForceHelper<NeoHookean<double, 3>>;
template class FBasedMpmForceHelper<NeoHookeanBorden<double, 3>>;
template class FBasedMpmForceHelper<NeoHookeanIsotropic<double, 3>>;
template class FBasedMpmForceHelper<Smudge<double, 3>>;
template class FBasedMpmForceHelper<StVenantKirchhoff<double, 3>>;
template class FBasedMpmForceHelper<StvkWithHencky<double, 3>>;
template class FBasedMpmForceHelper<StvkWithHenckyDecoupled<double, 3>>;
template class FBasedMpmForceHelper<StvkWithHenckyIsotropic<double, 3>>;
template class FBasedMpmForceHelper<StvkWithHenckyIsotropicUnilateral<double, 3>>;
template class FBasedMpmForceHelper<SurfaceTension<double, 3>>;
template class FBasedMpmForceHelper<CotangentOrthotropic<double, 3>>;
template class FBasedMpmForceHelper<QRAnisotropic<double, 3>>;
template class FBasedMpmForceHelper<QRStableNeoHookean<double, 3>>;
template class FDecoupledBasedMpmForceHelper<StvkWithHenckyDecoupled<double, 3>>;
} // namespace ZIRAN
