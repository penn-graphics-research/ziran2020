#ifndef VTK_IO_H
#define VTK_IO_H

#include <Ziran/CS/Util/Debug.h>
#include <Ziran/Math/Geometry/SimplexMesh.h>
#include <fstream>

namespace ZIRAN {

template <class T>
void readTetmeshVtk(std::istream& in, StdVector<Vector<T, 3>>& X, StdVector<Vector<int, 4>>& indices)
{
    auto initial_X_size = X.size();
    auto initial_indices_size = indices.size();

    std::string line;
    Vector<T, 3> position;

    bool reading_points = false;
    bool reading_tets = false;
    size_t n_points = 0;
    size_t n_tets = 0;

    while (std::getline(in, line)) {
        std::stringstream ss(line);
        if (line.size() == (size_t)(0)) {
        }
        else if (line.substr(0, 6) == "POINTS") {
            reading_points = true;
            reading_tets = false;
            ss.ignore(128, ' '); // ignore "POINTS"
            ss >> n_points;
        }
        else if (line.substr(0, 5) == "CELLS") {
            reading_points = false;
            reading_tets = true;
            ss.ignore(128, ' '); // ignore "CELLS"
            ss >> n_tets;
        }
        else if (line.substr(0, 10) == "CELL_TYPES") {
            reading_points = false;
            reading_tets = false;
        }
        else if (reading_points) {
            for (size_t i = 0; i < 3; i++)
                ss >> position(i);
            X.emplace_back(position);
        }
        else if (reading_tets) {
            ss.ignore(128, ' '); // ignore "4"
            Vector<int, 4> tet;
            for (size_t i = 0; i < 4; i++)
                ss >> tet(i);
            indices.emplace_back(tet);
        }
    }
    ZIRAN_ASSERT(n_points == X.size() - initial_X_size, "vtk read X count doesn't match.");
    ZIRAN_ASSERT((size_t)n_tets == indices.size() - initial_indices_size, "vtk read element count doesn't match.");
}

template <class T>
void readTetmeshVtk(const std::string vtk_file, StdVector<Vector<T, 3>>& X, StdVector<Vector<int, 4>>& indices)
{
    std::ifstream fs;
    fs.open(vtk_file);
    ZIRAN_ASSERT(fs, "could not open ", vtk_file);
    readTetmeshVtk(fs, X, indices);
    fs.close();
}

template <class T>
void readTetmeshVtk(const std::string vtk_file, StdVector<Vector<T, 2>>& X, StdVector<Vector<int, 3>>& indices)
{
    ZIRAN_ASSERT(false, "input has wrong dimensions");
}

template <class T>
void readTetMeshTetWild(std::istream& in, StdVector<Vector<T, 3>>& X, StdVector<Vector<int, 4>>& indices, StdVector<Vector<int, 3>>* faces = nullptr)
{
    auto initial_X_size = X.size();
    auto initial_indices_size = indices.size();

    std::string line;
    Vector<T, 3> position;
    Vector<int, 4> tet;
    Vector<int, 3> face;

    bool reading_points = false;
    bool reading_faces = false;
    bool reading_tets = false;
    size_t n_points = 0;
    size_t n_faces = 0;
    size_t n_tets = 0;

    while (std::getline(in, line)) {
        std::stringstream ss(line);
        if (line.size() == (size_t)(0)) {
            // skip empty line
        }
        else if (line[0] == '#') {
            // skip comment line
        }
        else if (line.substr(0, 3) == "End") {
            break;
        }
        else if (line.substr(0, 20) == "MeshVersionFormatted") {
            ss.ignore(128, ' ');
            int mesh_ver_formatted;
            ss >> mesh_ver_formatted;
            assert(mesh_ver_formatted == 1);
        }
        else if (line.substr(0, 9) == "Dimension") {
            ss.ignore(128, ' ');
            int dimension;
            ss >> dimension;
            assert(dimension == 3);
        }
        else if (line.substr(0, 8) == "Vertices") {
            in >> n_points;
            reading_points = true;
            reading_faces = false;
            reading_tets = false;
        }
        else if (line.substr(0, 9) == "Triangles") {
            in >> n_faces;
            reading_points = false;
            reading_faces = true && (faces != nullptr);
            reading_tets = false;
        }
        else if (line.substr(0, 10) == "Tetrahedra") {
            in >> n_tets;
            reading_points = false;
            reading_faces = false;
            reading_tets = true;
        }
        else if (reading_points) {
            for (size_t i = 0; i < 3; i++)
                ss >> position[i];
            X.emplace_back(position);
            int end_mark;
            ss >> end_mark;
            assert(end_mark == -1 || end_mark == 0);
        }
        else if (reading_faces) {
            for (size_t i = 0; i < 3; i++)
                ss >> face[i];
            face.array() -= 1;
            faces->emplace_back(face);
            int end_mark;
            ss >> end_mark;
            assert(end_mark == -1 || end_mark == 0);
        }
        else if (reading_tets) {
            for (size_t i = 0; i < 4; i++)
                ss >> tet[i];
            tet.array() -= 1;
            indices.emplace_back(tet);
            int end_mark;
            ss >> end_mark;
            assert(end_mark == -1 || end_mark == 0);
        }
    }

    ZIRAN_ASSERT(n_points == X.size() - initial_X_size, "mesh read X count doesn't match.");
    ZIRAN_ASSERT((size_t)n_tets == indices.size() - initial_indices_size, "mesh read element count doesn't match.");
}

template <class T>
void readTetMeshTetWild(const std::string mesh_file, StdVector<Vector<T, 3>>& X, StdVector<Vector<int, 4>>& indices)
{
    std::string ext = mesh_file.substr(mesh_file.find_last_of('.') + 1);
    ZIRAN_ASSERT(ext == "mesh");

    std::ifstream fs;
    fs.open(mesh_file);
    ZIRAN_ASSERT(fs, "could not open ", mesh_file);
    readTetMeshTetWild(fs, X, indices);
    fs.close();
}

template <class T>
void readTetMeshTetWild(const std::string mesh_file, StdVector<Vector<T, 2>>& X, StdVector<Vector<int, 3>>& indices)
{
    ZIRAN_ASSERT(false, "input has wrong dimensions");
}

template <class T>
void writeTetmeshVtk(std::ostream& os, const StdVector<Vector<T, 3>>& X, const StdVector<Vector<int, 4>>& indices)
{

    ZIRAN_ASSERT(X.size() != 0, "The X array for writing tetmesh vtk is empty.");
    ZIRAN_ASSERT(indices.size() != (size_t)0, "The tet mesh data structure for writing tetmesh vtk is empty.");

    os << "# vtk DataFile Version 2.0\n";
    os << "Unstructured Grid\n";
    os << "ASCII\n";
    os << "DATASET UNSTRUCTURED_GRID\n";

    os << "POINTS " << X.size() << " ";
    if (std::is_same<T, float>::value)
        os << "float\n";
    else
        os << "double\n";

    for (size_t i = 0; i < X.size(); i++) {
        os << X[i](0) << " " << X[i](1) << " " << X[i](2) << "\n";
    }

    os << std::endl;

    os << "CELLS " << indices.size() << " " << 5 * indices.size() << "\n";
    for (auto m : indices) {
        os << 4 << " " << m(0) << " " << m(1) << " " << m(2) << " " << m(3) << "\n";
    }
    os << std::endl;

    os << "CELL_TYPES " << indices.size() << "\n";
    for (size_t i = 0; i < indices.size(); i++) {
        os << 10 << std::endl;
    }
}

template <class T>
void writeTetmeshVtk(const std::string& filename, const StdVector<Vector<T, 3>>& X, StdVector<Vector<int, 4>>& indices)
{
    std::ofstream fs;
    fs.open(filename);
    ZIRAN_ASSERT(fs, "could not open ", filename);
    writeTetmeshVtk(fs, X, indices);
    fs.close();
}
} // namespace ZIRAN

#endif
