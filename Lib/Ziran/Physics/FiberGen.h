#ifndef FIBER_GEN
#define FIBER_GEN
#include <Ziran/CS/DataStructure/KdTree.h>
#include <Ziran/CS/DataStructure/HashTable.h>
#include <Ziran/Math/Geometry/MeshConstruction.h>
#include <Ziran/Math/Geometry/ObjIO.h>
#include <Ziran/Math/Geometry/VtkIO.h>
#include <Ziran/Physics/SimplexMeshPoisson.h>

namespace ZIRAN {

// generate fibers by solving potential flow
template <class T, int dim>
void fiberGen(
    const StdVector<Vector<T, dim>>& verticies,
    const StdVector<Vector<int, 4>>& indices,
    std::function<bool(Vector<T, dim>, int)> inflow_boundary,
    std::function<bool(Vector<T, dim>, int)> outflow_boundary,
    StdVector<Vector<T, dim>>& tet_wise_fiber,
    StdVector<Vector<T, dim>>& node_wise_fiber)
{
    using TV = Vector<T, dim>;

    StdVector<int> dirichlet;
    StdVector<T> dirichlet_value;
    SimplexMesh<dim> tet_mesh;
    tet_mesh.indices = indices;
    tet_mesh.initializeBoundaryElements();
    for (auto i : tet_mesh.boundary_nodes) {
        if (inflow_boundary(verticies[i], i)) {
            dirichlet.emplace_back(i);
            dirichlet_value.emplace_back(0);
        }
        else if (outflow_boundary(verticies[i], i)) {
            dirichlet.emplace_back(i);
            dirichlet_value.emplace_back((T)1);
        }
    }
    ZIRAN_ASSERT(dirichlet.size());

    Vector<T, Eigen::Dynamic> x, b;
    x.resize(verticies.size());
    x.setZero();
    b.resize(verticies.size());
    b.setZero();
    StdVector<Vector<T, dim>> verticies_copy = verticies;
    SimplexMeshPoisson<T, dim> poisson(tet_mesh, verticies_copy, x, b, dirichlet, dirichlet_value);
    poisson.solve();
    poisson.findNormalizedGradient(tet_wise_fiber);

    node_wise_fiber.resize(verticies.size());
    for (size_t i = 0; i < node_wise_fiber.size(); i++)
        node_wise_fiber[i].setZero();
    for (size_t i = 0; i < indices.size(); i++) {
        auto tet = indices[i];
        for (int k = 0; k < 4; k++) {
            node_wise_fiber[tet(k)] += tet_wise_fiber[i];
        }
    }
    for (size_t i = 0; i < node_wise_fiber.size(); i++) {
        node_wise_fiber[i].normalize();
        if (node_wise_fiber[i].norm() < 0.5)
            node_wise_fiber[i] = TV(1, 0, 0);
    }
}

} // namespace ZIRAN

#endif
