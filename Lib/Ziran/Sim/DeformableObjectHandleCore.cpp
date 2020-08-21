#include <Ziran/CS/Util/DataDir.h>
#include <Ziran/Math/Geometry/ObjIO.h>
#include <Ziran/Math/Geometry/Rotation.h>
#include <Ziran/Math/Geometry/SimplexMesh.h>
#include <Ziran/Math/Geometry/Particles.h>
#include <Ziran/Math/Geometry/Elements.h>
#include <Ziran/Physics/LagrangianForce/LagrangianForce.h>
#include "DeformableObjectHandleCore.h"

namespace ZIRAN {

template <class T, int dim, int manifold_dim>
DeformableObjectHandle<T, dim, SimplexMesh<manifold_dim>>::DeformableObjectHandle(const MeshHandle<T, dim, SimplexMesh<manifold_dim>>& undeformed, ElementManagerFor<T, TMesh, dim>& elements, StdVector<std::unique_ptr<LagrangianForce<T, dim>>>& forces)
    : Base(undeformed)
    , elements(elements)
    , element_range(elements.addUndeformedMesh(*mesh, particles.X.array, undeformed.particle_index_offset))
    , forces(forces)
{
}

template <class T, int dim, int manifold_dim>
DeformableObjectHandle<T, dim, SimplexMesh<manifold_dim>>::DeformableObjectHandle(const MeshHandle<T, dim, SimplexMesh<manifold_dim>>& undeformed, ElementManagerFor<T, TMesh, dim>& elements, Range element_range, StdVector<std::unique_ptr<LagrangianForce<T, dim>>>& forces)
    : Base(undeformed)
    , elements(elements)
    , element_range(element_range)
    , forces(forces)
{
}

template <class T, int dim, int manifold_dim>
MeshHandle<T, dim, SimplexMesh<manifold_dim>> DeformableObjectHandle<T, dim, SimplexMesh<manifold_dim>>::copy()
{
    ZIRAN_INFO("Note: calling copy() on DeformableObject results in a copy of mesh.");
    ZIRAN_INFO("addDeformable, setMass etc still need to be called.");
    return Base::copy();
}

template <class T, int dim, int manifold_dim>
T DeformableObjectHandle<T, dim, SimplexMesh<manifold_dim>>::totalVolume() const
{
    return elements.totalMeasure(element_range);
}

template <class T, int dim, int manifold_dim>
DeformableObjectHandle<T, dim, SimplexMesh<manifold_dim>> DeformableObjectHandle<T, dim, SimplexMesh<manifold_dim>>::subset(Range& subrange)
{
    return DeformableObjectHandle<T, dim, SimplexMesh<manifold_dim>>(*this, elements, subrange, forces);
}

// scales DmInverse by a constant
// this function is for isoptropic cases,
template <class T, int dim, int manifold_dim>
void DeformableObjectHandle<T, dim, SimplexMesh<manifold_dim>>::scaleDmInverse(int frame, const std::function<T(int)>& growCurve)
{
    for (auto iter = elements.subsetIter(DisjointRanges({ element_range }), elements.Dm_inv_name()); iter; ++iter) {
        auto& Dm_inverse = iter.template get<0>();
        Dm_inverse /= growCurve(frame);
        // if (frame != 0)
        //     Dm_inverse *= growCurve(frame - 1);
    }
}

// reset Dm_inverse so that F is identity
template <class T, int dim, int manifold_dim>
void DeformableObjectHandle<T, dim, SimplexMesh<manifold_dim>>::resetDmInverse()
{
    for (auto iter = elements.subsetIter(DisjointRanges({ element_range }), elements.Dm_inv_name(), elements.F_name()); iter; ++iter) {
        auto& Dm_inverse = iter.template get<0>();
        auto F = iter.template get<1>();
        Dm_inverse.noalias() = Dm_inverse * F.inverse();
    }
}

// Add 'no_write' label to the elements in this handle.
// This is for tracking things like bending springs, where we don't want them to be in 'segmesh_to_write'
template <class T, int dim, int manifold_dim>
void DeformableObjectHandle<T, dim, SimplexMesh<manifold_dim>>::labelNoWrite()
{
    AttributeName<int> no_write_name("no_write");
    int no_write = 1;
    elements.add(no_write_name, element_range, no_write);
    ZIRAN_INFO("element [", element_range.lower, ",", element_range.upper, ") labeled no write.");
}

template class DeformableObjectHandle<double, 2, SimplexMesh<1>>;
template class DeformableObjectHandle<double, 2, SimplexMesh<2>>;
template class DeformableObjectHandle<float, 2, SimplexMesh<1>>;
template class DeformableObjectHandle<float, 2, SimplexMesh<2>>;
template class DeformableObjectHandle<double, 3, SimplexMesh<1>>;
template class DeformableObjectHandle<double, 3, SimplexMesh<2>>;
template class DeformableObjectHandle<double, 3, SimplexMesh<3>>;
template class DeformableObjectHandle<float, 3, SimplexMesh<1>>;
template class DeformableObjectHandle<float, 3, SimplexMesh<2>>;
template class DeformableObjectHandle<float, 3, SimplexMesh<3>>;
} // namespace ZIRAN
