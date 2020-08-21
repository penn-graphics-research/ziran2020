#ifndef NQUAD_MESH_H
#define NQUAD_MESH_H

#include <Ziran/CS/Util/Forward.h>
#include <Ziran/CS/DataStructure/HashTable.h>
#include <Ziran/Math/Geometry/Elements.h>
#include <Ziran/Math/Geometry/AnalyticLevelSet.h>

#include <vector>

namespace ZIRAN {
/**
    This is a class for keeping track of quad/cuboid mesh topology: e, i, j, k, l (element e has nodes i,j,k,l), etc. The class also has basic helpers like routines for computing the boundary quad/cuboid mesh
*/
template <int dim>
class NQuadMesh {
public:
    static constexpr int manifold_dim = dim;
    static constexpr int num_vertices = 2 << (dim - 1);
    static constexpr int num_faces = dim * 2;
    static constexpr int boundary_vertices = 1 << (dim - 1);
    typedef Eigen::Vector2i NIV;
    typedef Vector<int, boundary_vertices> BIV;
    typedef Vector<int, 2> EIV;
    typedef Matrix<int, num_vertices, 1> IV;
    typedef std::array<BIV, num_faces> Faces;
    StdVector<IV> indices;
    StdVector<BIV> boundary_indices;
    HashTable<BIV, BIV> boundary_hash;
    bool boundary_initialized = false;

    // used for creating beamelements. assumes dim = 2
    /*
       indices of the meshes in the array when the particle is at the center 
     
         _________________
        |        |        |
        |   3    |   2    |
        |        |        |
        |________|________|
        |        |        |
        |   0    |   1    |
        |        |        |
        |________|________|
      */
    StdVector<std::array<int, 4>> vertex_to_mesh;
    // for each mesh store the particles influenced by the mesh under cubic BSpline
    StdVector<std::array<int, 16>> cubicBSplineInfluencedParticles;

    // Mesh Class owned neighboring infos
    bool is_neighbor_list_created = false;
    bool neighbor_list_sorted;
    StdVector<NIV> neighbor_list;
    StdVector<BIV> neighbor_vertex_list;
    HashTable<int, StdVector<int>> vertex_to_neighbor_vertices;

    NQuadMesh()
    {
    }

    NQuadMesh(StdVector<IV> indices)
        : indices(std::move(indices))
    {
    }

    inline int index(const size_t e, const int i) const { return indices[e](i); }

    inline size_t numberElements() const { return indices.size(); }

    /**
        Find boundary elements for the mesh using hash table
        Any (dim -1) dimension element (which is stored in vector of size dim) traversed only once is a boundary element.
        Any (dim -1) dimension element traversed twice is an interior element.
    */

    void initializeBoundaryElements();

    inline static void getFaces(const IV& element, Faces& result);

    void constructListOfFaceNeighboringElements(const bool sort = false, HashTable<BIV, Eigen::Vector2i>* face_hash_ptr = nullptr);

    void constructListOfFaces(StdVector<BIV>& face_list) const;

    void initializeVertexToNeighbouringVertices();

    //return the two/three vertices of the common face, sorted w.r.t. vertex ID
    static bool findSharedFace(const IV& element1, const IV& element2, BIV& face);

    static void findSharedFaceAndIndex(const IV& element1, const IV& element2, BIV& sorted, BIV* local_index1_ptr = nullptr, BIV* local_index2_ptr = nullptr);

    static void findUnsortedFace(const IV& element, const BIV& sorted, BIV& face);

    inline static int getInternalIndex(const IV& element, int index);

    // construct list of pair of endpoints for the interior edges
    void constructNeighborVertexList();

    static const char* name();

    void findIndexBounds(int& min, int& max) const;

    void findVertexToMesh();

    void findCubicBSplineInfluencedParticles();
};

template <int dim>
inline int NQuadMesh<dim>::getInternalIndex(const IV& element, int index)
{
    for (int i = 0; i < num_vertices; i++) {
        if (element(i) == index)
            return i;
    }
    return -1;
}

template <>
inline void NQuadMesh<1>::getFaces(
    const IV& element,
    std::array<Vector<int, 1>, 2>& result)
{
    int n0 = element(0);
    int n1 = element(1);
    result[0](0) = n0;
    result[1](0) = n1;
}

template <>
inline void NQuadMesh<2>::getFaces(
    const IV& element,
    std::array<Vector<int, 2>, 4>& result)
{
    int n0 = element(0);
    int n1 = element(1);
    int n2 = element(2);
    int n3 = element(3);
    result[0] = Vector<int, 2>(n0, n1);
    result[1] = Vector<int, 2>(n1, n2);
    result[2] = Vector<int, 2>(n2, n3);
    result[3] = Vector<int, 2>(n3, n0);
}

/*
  The following shows unfolded regular cuboid and its vertex indices and face indices
0: origin
1: x-axis
2: y-axis
3: z-axis
(by xyz axes I mean the mathematician xyz axes not the Houdini xyz axes)
  The faces are oriented such that right hand rule gives outward normal.
 
            5 ________ 7           
             |        |            
             |  f4    |            
             |        |            
    5_______2|________|4_______7   
    |        |        |        |   
    |  f1    |  f0    |  f3    |   
    |        |        |        |   
    |________|________|________|   
    3       0|        |1       6   
             |  f2    |            
             |        |            
            3|________|6           
             |        |            
             |  f5    |            
             |        |            
             |________|            
             5        7            
  */

template <>
inline void NQuadMesh<3>::getFaces(
    const IV& element,
    std::array<Vector<int, 4>, 6>& result)
{
    int n0 = element(0);
    int n1 = element(1);
    int n2 = element(2);
    int n3 = element(3);
    int n4 = element(4);
    int n5 = element(5);
    int n6 = element(6);
    int n7 = element(7);
    result[0] = Vector<int, 4>(n0, n2, n4, n1);
    result[1] = Vector<int, 4>(n0, n3, n5, n2);
    result[2] = Vector<int, 4>(n0, n1, n6, n3);
    result[3] = Vector<int, 4>(n1, n4, n7, n6);
    result[4] = Vector<int, 4>(n2, n5, n7, n4);
    result[5] = Vector<int, 4>(n3, n6, n7, n5);
}
} // namespace ZIRAN

#endif
