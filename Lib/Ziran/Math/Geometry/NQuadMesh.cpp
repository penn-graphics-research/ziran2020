#include <Ziran/CS/Util/Debug.h>
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <Ziran/Math/Geometry/NQuadMesh.h>

namespace ZIRAN {

template <int dim>
void NQuadMesh<dim>::initializeBoundaryElements()
{
    size_t number_elements = indices.size();
    Faces faces;

    boundary_hash.clear();
    for (size_t i = 0; i < number_elements; i++) {
        //go through each element
        //and get all faces for each element
        getFaces(indices[i], faces);

        //for each face, sort indices
        //if the set of indices does not exist, put in hash table
        // if the indices already exists, delete from hash table
        // all faces left are boundary faces
        for (size_t j = 0; j < faces.size(); j++) {
            BIV sorted = faces[j];
            //ZIRAN_DBUG("sorted", sorted);
            std::sort(sorted.data(), sorted.data() + sorted.size());

            if (!boundary_hash.erase(sorted))
                boundary_hash.insert(sorted, faces[j]);
        }
    }
    boundary_indices.clear();
    boundary_indices.reserve(boundary_hash.size());
    for (auto it = boundary_hash.cbegin(); it != boundary_hash.cend(); it++) {
        boundary_indices.emplace_back(it->value);
        //ZIRAN_DBUG("it->value", it->value);
    }
    // ZIRAN_VERB_IF(boundary_initialized, "NQuadMesh::initializeBoundaryElements(): boundary had been initialized. Reinitializing.");
    boundary_initialized = true;
}

//TODO: need justification: only the tips of the simplexmesh vertices actually count???
template <int dim>
void NQuadMesh<dim>::initializeVertexToNeighbouringVertices()
{
    int number_elements = int(indices.size());
    Faces faces;
    for (int i = 0; i < number_elements; i++) {
        for (int p = 0; p < num_vertices; p++) {
            int my_index = indices[i](p);
            for (int q = 0; q < num_vertices; q++) {
                int your_index = indices[i](q);
                if (p != q) {
                    StdVector<int>& list = vertex_to_neighbor_vertices[my_index];
                    auto found = std::find(std::begin(list), std::end(list), your_index);
                    if (found == list.end())
                        list.emplace_back(your_index);
                }
            }
        }
    }
}

template <int dim>
bool NQuadMesh<dim>::findSharedFace(const IV& element1, const IV& element2, BIV& face)
{
    int e = 0;
    bool job_done = false;
    for (int i = 0; i < num_vertices && !job_done; i++) {
        for (int j = 0; j < num_vertices && !job_done; j++) {
            if (element1(i) == element2(j) && !job_done) {
                face(e) = element1(i);
                e++;
                if (e == boundary_vertices)
                    job_done = true;
            }
        }
    }
    if (e == boundary_vertices)
        job_done = true;

    std::sort(face.data(), face.data() + face.size());
    return job_done;
}

template <int dim>
void NQuadMesh<dim>::findSharedFaceAndIndex(const IV& element1, const IV& element2, BIV& sorted, BIV* local_index1_ptr, BIV* local_index2_ptr)
{
    BIV inner_local_index1, inner_local_index2;
    if (!local_index1_ptr)
        local_index1_ptr = &inner_local_index1;
    if (!local_index2_ptr)
        local_index2_ptr = &inner_local_index2;

    int find = 0;
    for (int k = 0; k < num_vertices && find < boundary_vertices; k++) {
        for (int l = 0; l < num_vertices && find < boundary_vertices; l++) {
            if (element1(k) == element2(l)) {
                (*local_index1_ptr)(find) = k;
                (*local_index2_ptr)(find) = l;
                sorted(find) = element1(k);
                find++;
                break;
            }
        }
    }
    std::sort(local_index1_ptr->data(), local_index1_ptr->data() + local_index1_ptr->size());
    std::sort(local_index2_ptr->data(), local_index2_ptr->data() + local_index2_ptr->size());
    std::sort(sorted.data(), sorted.data() + sorted.size());
}

template <>
void NQuadMesh<1>::findUnsortedFace(const typename NQuadMesh<1>::IV& element, const typename NQuadMesh<1>::BIV& sorted, typename NQuadMesh<1>::BIV& face)
{
    int find = 0;
    for (int k = 0; k < num_vertices && find < boundary_vertices; k++) {
        for (int l = 0; l < boundary_vertices && find < boundary_vertices; l++) {
            if (element(k) == sorted(l)) {
                face(find) = element(k);
                find++;
                break;
            }
        }
    }
}

template <>
void NQuadMesh<2>::findUnsortedFace(const typename NQuadMesh<2>::IV& element, const typename NQuadMesh<2>::BIV& sorted, typename NQuadMesh<2>::BIV& face)
{
    int find = 0;
    for (int k = 0; k < num_vertices && find < boundary_vertices; k++) {
        for (int l = 0; l < boundary_vertices && find < boundary_vertices; l++) {
            if (element(k) == sorted(l)) {
                face(find) = element(k);
                find++;
                break;
            }
        }
    }
}

template <>
void NQuadMesh<3>::findUnsortedFace(const typename NQuadMesh<3>::IV& element, const typename NQuadMesh<3>::BIV& sorted, typename NQuadMesh<3>::BIV& face)
{
    StdVector<BIV> faces;
    faces.emplace_back(element(0), element(1), element(4), element(2));
    faces.emplace_back(element(0), element(2), element(5), element(3));
    faces.emplace_back(element(0), element(3), element(6), element(1));
    faces.emplace_back(element(1), element(6), element(7), element(4));
    faces.emplace_back(element(2), element(4), element(7), element(5));
    faces.emplace_back(element(3), element(5), element(7), element(6));

    bool find = false;
    for (int i = 0; i < num_faces && !find; i++) {
        BIV compare = faces[i];
        std::sort(compare.data(), compare.data() + compare.size());
        if (sorted.isApprox(compare)) {
            face = faces[i];
            find = true;
            break;
        }
    }
}

template <int dim>
void NQuadMesh<dim>::constructNeighborVertexList()
{
    // if the neighbor_list hasn't been created, then create one.
    if (!is_neighbor_list_created) {
        constructListOfFaceNeighboringElements(false);
    }

    neighbor_vertex_list.clear();
    neighbor_vertex_list.reserve(neighbor_list.size());
    for (int n = 0; n < int(neighbor_list.size()); n++) {
        NIV face = neighbor_list[n];
        IV element1 = indices[face(0)];
        IV element2 = indices[face(1)];
        BIV vertices;
        findSharedFace(element1, element2, vertices);
        neighbor_vertex_list.emplace_back(vertices);
    }
}

template <int dim>
const char* NQuadMesh<dim>::name()
{
    switch (dim) {
    case 1:
        return "SegQ";
        break;
    case 2:
        return "Quad";
        break;
    case 3:
        return "Cuboid";
        break;
    default:
        ZIRAN_ASSERT("Unsupported Element dimension ", dim);
    }
}

template <int dim>
void NQuadMesh<dim>::constructListOfFaceNeighboringElements(const bool sort, HashTable<BIV, Eigen::Vector2i>* face_hash_ptr)
{
    neighbor_list_sorted = sort;
    HashTable<BIV, Eigen::Vector2i> inner_face_hash;
    if (!face_hash_ptr)
        face_hash_ptr = &inner_face_hash;

    size_t number_elements = indices.size();
    Faces faces;

    for (size_t i = 0; i < number_elements; i++) {
        getFaces(indices[i], faces);
        for (size_t j = 0; j < faces.size(); j++) {
            BIV sorted = faces[j];
            std::sort(sorted.data(), sorted.data() + sorted.size());

            Eigen::Vector2i* v = face_hash_ptr->get(sorted);
            if (v == nullptr)
                face_hash_ptr->insert(sorted, Eigen::Vector2i(i, -1));
            else {
                ZIRAN_ASSERT(((*v)(0) != -1 && (*v)(0) != (int)i && (*v)(1) == -1), (*v)(0), ", ", (*v)(1), ", ", i, ", ", indices[(*v)(0)].transpose(), ", ", indices[(*v)(1)].transpose(), ", ", indices[i].transpose());
                (*v)(1) = (int)i;
            }
        }
    }

    neighbor_list.clear();
    for (auto it = face_hash_ptr->cbegin(); it != face_hash_ptr->cend(); it++) {
        Eigen::Vector2i elements = it->value.template head<2>();
        ZIRAN_ASSERT(elements(0) != -1);
        if (elements(1) != -1)
            neighbor_list.push_back(elements);
    }

    if (sort) {
        for (size_t i = 0; i < neighbor_list.size(); i++) {
            std::sort(neighbor_list[i].data(), neighbor_list[i].data() + neighbor_list[i].size());
        }
    }

    // set flag recording that neighbor_list has been created
    is_neighbor_list_created = true;
}

template <int dim>
void NQuadMesh<dim>::constructListOfFaces(StdVector<BIV>& face_list) const
{
    HashTable<BIV, BIV> face_hash;
    size_t number_elements = indices.size();
    Faces faces;

    for (size_t i = 0; i < number_elements; i++) {
        getFaces(indices[i], faces);
        for (size_t j = 0; j < faces.size(); j++) {
            BIV sorted = faces[j];
            std::sort(sorted.data(), sorted.data() + sorted.size());
            face_hash.insert(sorted, faces[j]);
        }
    }
    face_list.clear();
    face_list.reserve(face_hash.size());
    for (auto it = face_hash.begin(); it != face_hash.end(); ++it)
        face_list.emplace_back(it->value);
}

template <int dim>
void NQuadMesh<dim>::findIndexBounds(int& min, int& max) const
{
    min = std::numeric_limits<int>::max();
    max = std::numeric_limits<int>::lowest();
    for (const IV& element : indices) {
        int local_min = element.minCoeff();
        int local_max = element.maxCoeff();
        if (local_min < min)
            min = local_min;
        if (local_max > max)
            max = local_max;
    }
}

template <int dim>
void NQuadMesh<dim>::findVertexToMesh()
{
    ZIRAN_ASSERT(dim == 2, "only dim = 2 implemented");
    int max = 0, min = 0;
    findIndexBounds(min, max);
    vertex_to_mesh.resize(max + 1);
    for (size_t i = 0; i < vertex_to_mesh.size(); ++i) {
        for (size_t j = 0; j < 4; ++j)
            vertex_to_mesh[i][j] = -1;
    }

    for (size_t i = 0; i < indices.size(); ++i) {
        const IV& id = indices[i];
        vertex_to_mesh[id[0]][2] = i;
        vertex_to_mesh[id[1]][3] = i;
        vertex_to_mesh[id[2]][0] = i;
        vertex_to_mesh[id[3]][1] = i;
    }
}

template <int dim>
void NQuadMesh<dim>::findCubicBSplineInfluencedParticles()
{
    ZIRAN_ASSERT(dim == 2, "only dim = 2 implemented");
    cubicBSplineInfluencedParticles.resize(indices.size());
    for (size_t i = 0; i < cubicBSplineInfluencedParticles.size(); ++i) {
        for (size_t j = 0; j < 16; ++j)
            cubicBSplineInfluencedParticles[i][j] = -1;
    }
    findVertexToMesh();
    for (size_t i = 0; i < indices.size(); ++i) {
        cubicBSplineInfluencedParticles[i][0] = (vertex_to_mesh[indices[i][0]][0] == -1) ? -1 : indices[vertex_to_mesh[indices[i][0]][0]][0];
        cubicBSplineInfluencedParticles[i][1] = (vertex_to_mesh[indices[i][0]][0] == -1) ? -1 : indices[vertex_to_mesh[indices[i][0]][0]][1];
        cubicBSplineInfluencedParticles[i][2] = (vertex_to_mesh[indices[i][1]][1] == -1) ? -1 : indices[vertex_to_mesh[indices[i][1]][1]][0];
        cubicBSplineInfluencedParticles[i][3] = (vertex_to_mesh[indices[i][1]][1] == -1) ? -1 : indices[vertex_to_mesh[indices[i][1]][1]][1];
        cubicBSplineInfluencedParticles[i][4] = (vertex_to_mesh[indices[i][0]][0] == -1) ? -1 : indices[vertex_to_mesh[indices[i][0]][0]][3];
        cubicBSplineInfluencedParticles[i][5] = indices[i][0];
        cubicBSplineInfluencedParticles[i][6] = indices[i][1];
        cubicBSplineInfluencedParticles[i][7] = (vertex_to_mesh[indices[i][1]][1] == -1) ? -1 : indices[vertex_to_mesh[indices[i][1]][1]][2];
        cubicBSplineInfluencedParticles[i][8] = (vertex_to_mesh[indices[i][3]][3] == -1) ? -1 : indices[vertex_to_mesh[indices[i][3]][3]][0];
        cubicBSplineInfluencedParticles[i][9] = indices[i][3];
        cubicBSplineInfluencedParticles[i][10] = indices[i][2];
        cubicBSplineInfluencedParticles[i][11] = (vertex_to_mesh[indices[i][2]][2] == -1) ? -1 : indices[vertex_to_mesh[indices[i][2]][2]][1];
        cubicBSplineInfluencedParticles[i][12] = (vertex_to_mesh[indices[i][3]][3] == -1) ? -1 : indices[vertex_to_mesh[indices[i][3]][3]][3];
        cubicBSplineInfluencedParticles[i][13] = (vertex_to_mesh[indices[i][3]][3] == -1) ? -1 : indices[vertex_to_mesh[indices[i][3]][3]][2];
        cubicBSplineInfluencedParticles[i][14] = (vertex_to_mesh[indices[i][2]][2] == -1) ? -1 : indices[vertex_to_mesh[indices[i][2]][2]][3];
        cubicBSplineInfluencedParticles[i][15] = (vertex_to_mesh[indices[i][2]][2] == -1) ? -1 : indices[vertex_to_mesh[indices[i][2]][2]][2];
    }
}

template class NQuadMesh<1>;
template class NQuadMesh<2>;
template class NQuadMesh<3>;
} // namespace ZIRAN
