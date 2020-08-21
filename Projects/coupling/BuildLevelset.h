#pragma once

#include "../splitting/UMGrid.h"
#include <MPM/MpmGrid.h>
#include <MPM/MpmSimulationBase.h>
#include <Ziran/CS/DataStructure/SpatialHash.h>
#include <Ziran/Math/Geometry/CollisionObject.h>

namespace ZIRAN {

template <class T, int dim>
class SplittingSimulation;

template <class T, int dim>
class BuildLevelset {
public:
    BuildLevelset() = default;
    ~BuildLevelset() = default;

    using TV = Vector<T, dim>;
    using IV = Vector<int, dim>;
    using TM = Matrix<T, dim, dim>;
    using Vec = Vector<T, Eigen::Dynamic>;
    using TStack = Vector<T, Eigen::Dynamic>;
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;

    using ULL = unsigned long long;
    T dx;
    UMGrid<T, dim> cell_phi;
    UMGrid<int, dim> cell_sign;
    UMGrid<TV, dim> boundary_phi;

    MpmGrid<T, dim, 1> grid;
    bool first_run = true;

    int xmin, xmax, ymin, ymax;

    T lerp(T v0, T v1, T c)
    {
        return (1 - c) * v0 + c * v1;
    }

    T bilerp(const T& v00, const T& v10,
        const T& v01, const T& v11,
        T cx, T cy)
    {
        return lerp(lerp(v00, v10, cx),
            lerp(v01, v11, cx),
            cy);
    }

    T trilerp(const T& v000, const T& v100,
        const T& v010, const T& v110,
        const T& v001, const T& v101,
        const T& v011, const T& v111,
        T fx, T fy, T fz)
    {
        return lerp(bilerp(v000, v100, v010, v110, fx, fy),
            bilerp(v001, v101, v011, v111, fx, fy),
            fz);
    }

    void findInterface()
    {
        cell_sign.clear();
        T LARGE_VALUE = 100;
        //TODO: MAYBE SEPERATE AND PARALLIZE
        cell_phi.iterateGrid([&](int x, int y, T value) {
            if (cell_phi(x, y) > 0) {
                cell_sign(x, y) = 1;
                cell_phi(x, y) *= (T)cell_sign(x, y);
            }
            else {
                cell_sign(x, y) = -1;
                cell_phi(x, y) = LARGE_VALUE;
            }
        });
    }

    T findSmallValueX(int i, int j)
    {
        if (!cell_phi.find(i - 1, j) && cell_phi.find(i + 1, j))
            return cell_phi(i + 1, j);
        else if (!cell_phi.find(i + 1, j) && cell_phi.find(i - 1, j))
            return cell_phi(i - 1, j);
        else if (!cell_phi.find(i + 1, j) && !cell_phi.find(i - 1, j))
            return cell_phi(i, j);
        else
            return std::min(cell_phi(i - 1, j), cell_phi(i + 1, j));
    }

    T findSmallValueY(int i, int j)
    {
        if (!cell_phi.find(i, j - 1) && cell_phi.find(i, j + 1))
            return cell_phi(i, j + 1);
        else if (!cell_phi.find(i, j + 1) && cell_phi.find(i, j - 1))
            return cell_phi(i, j - 1);
        else if (!cell_phi.find(i, j + 1) && !cell_phi.find(i, j - 1))
            return cell_phi(i, j);
        else
            return std::min(cell_phi(i, j - 1), cell_phi(i, j + 1));
    }

    T calculateUbar(float a, float b, T h)
    {
        if (std::abs(a - b) >= h)
            return std::min(a, b) + h;
        else
            return 0.5f * (a + b + std::sqrt(2 * h * h - (a - b) * (a - b)));
    }

    void fastSweep()
    {
        T ubar;
        findInterface();
        for (int c = 0; c <= 5; c++) {
            for (int i = xmin; i <= xmax; i++) {
                for (int j = ymin; j <= ymax; j++) {
                    if (cell_sign.find(i, j) && cell_sign(i, j) < 0) {
                        float a = findSmallValueX(i, j);
                        float b = findSmallValueY(i, j);
                        ubar = calculateUbar(a, b, dx);
                        cell_phi(i, j) = std::min(cell_phi(i, j), ubar);
                    }
                }
            }

            for (int i = xmax; i >= xmin; i--) {
                for (int j = ymin; j <= ymax; j++) {
                    if (cell_sign.find(i, j) && cell_sign(i, j) < 0) {
                        float a = findSmallValueX(i, j);
                        float b = findSmallValueY(i, j);
                        ubar = calculateUbar(a, b, dx);
                        cell_phi(i, j) = std::min(cell_phi(i, j), ubar);
                    }
                }
            }

            for (int i = xmax; i >= xmin; i--) {
                for (int j = ymax; j >= ymin; j--) {
                    if (cell_sign.find(i, j) && cell_sign(i, j) < 0) {
                        float a = findSmallValueX(i, j);
                        float b = findSmallValueY(i, j);
                        ubar = calculateUbar(a, b, dx);
                        cell_phi(i, j) = std::min(cell_phi(i, j), ubar);
                    }
                }
            }

            for (int i = xmin; i <= xmax; i++) {
                for (int j = ymax; j >= ymin; j--) {
                    if (cell_sign.find(i, j) && cell_sign(i, j) < 0) {
                        float a = findSmallValueX(i, j);
                        float b = findSmallValueY(i, j);
                        ubar = calculateUbar(a, b, dx);
                        cell_phi(i, j) = std::min(cell_phi(i, j), ubar);
                    }
                }
            }
        }

        cell_phi.iterateGrid([&](int x, int y, int kind) {
            cell_phi(x, y) *= (T)cell_sign(x, y);
        });
    }

    bool queryInside(TV& particleX, T& phi, TV& normal, T isocontour)
    {
        BSplineWeights<T, dim, 1> spline(particleX, dx);
        int x = spline.base_node[0];
        int y = spline.base_node[1];

        if (!cell_phi.find(x, y) || !cell_phi.find(x + 1, y) || !cell_phi.find(x + 1, y + 1) || !cell_phi.find(x, y + 1)) {
            return false;
        }
        else {
            ZIRAN_ASSERT(cell_phi.find(x, y));
            ZIRAN_ASSERT(cell_phi.find(x + 1, y));
            ZIRAN_ASSERT(cell_phi.find(x + 1, y + 1));
            ZIRAN_ASSERT(cell_phi.find(x, y + 1));
            T phi00 = cell_phi(x, y);
            T phi10 = cell_phi(x + 1, y);
            T phi11 = cell_phi(x + 1, y + 1);
            T phi01 = cell_phi(x, y + 1);

            T px = std::abs(particleX[0] / dx - (T)x);
            T py = std::abs(particleX[1] / dx - (T)y);
            phi = bilerp(phi00, phi10, phi01, phi11, px, py);

            T dphidx = 0;
            T dphidy = 0;
            grid.iterateArena(spline, [&](IV node, T w, TV dw) {
                dphidx += cell_phi[node] * dw[0];
                dphidy += cell_phi[node] * dw[1];
            });

            normal = TV(dphidx, dphidy);
            normal = normal.normalized();
            return phi < isocontour;
        }
    }

    void build(const StdVector<TV>& solid_particles, T build_dx, T blob_radius, T erosion)
    {
        ZIRAN_TIMER();
        cell_phi.clear();
        dx = build_dx;
        if (solid_particles.size() > 0) {
            for (int i = 0; i < solid_particles.size(); i++) {
                if constexpr (dim == 2) {
                    TV Xp = solid_particles[i];
                    BSplineWeights<T, dim, 1> spline(Xp, dx);
                    if (i == 0) {
                        xmin = spline.base_node[0];
                        xmax = spline.base_node[0] + 1;
                        ymin = spline.base_node[1];
                        ymax = spline.base_node[1] + 1;
                    }
                    for (int x = spline.base_node[0] - 20; x <= spline.base_node[0] + 20; ++x)
                        for (int y = spline.base_node[1] - 20; y <= spline.base_node[1] + 20; ++y) {
                            if (!cell_phi.find(x, y)) cell_phi(x, y) = std::numeric_limits<T>::max();
                            TV cell_pos = TV(x, y) * dx;
                            TV dist_vec = cell_pos - Xp;
                            T dist = dist_vec.norm() - blob_radius;
                            cell_phi(x, y) = std::min(cell_phi(x, y), dist);

                            xmin = std::min(xmin, x);
                            xmax = std::max(xmax, x);
                            ymin = std::min(ymin, y);
                            ymax = std::max(ymax, y);
                        }
                }
                else {
                    // TODO: 3D construct solid particles level-set
                }
            }
            fastSweep();
            cell_phi.iterateGrid([&](int x, int y, int kind) {
                cell_phi(x, y) += erosion;
            });
        }
    }

    void marchingSquare(StdVector<TV>& points, StdVector<Vector<int, 2>>& segments)
    {
        points.clear();
        segments.clear();
        int count = 0;
        cell_phi.iterateGrid([&](int x, int y, T value) {
            int idx_a = -1;
            int idx_b = -1;
            int idx_c = -1;
            int idx_d = -1;
            if (cell_phi.find(x + 1, y) && cell_phi.find(x, y + 1) && cell_phi.find(x + 1, y + 1)) {
                if (cell_phi(x, y) * cell_phi(x + 1, y) < 0) {
                    TV shift = TV(std::abs(cell_phi(x, y) / (cell_phi(x, y) - cell_phi(x + 1, y))) * dx, 0);
                    TV pos = TV(x, y) * dx + shift;
                    points.push_back(pos);
                    idx_a = count;
                    count += 1;
                }
                if (cell_phi(x + 1, y) * cell_phi(x + 1, y + 1) < 0) {
                    TV shift = TV(0, std::abs(cell_phi(x + 1, y) / (cell_phi(x + 1, y) - cell_phi(x + 1, y + 1))) * dx);
                    TV pos = TV(x + 1, y) * dx + shift;
                    points.push_back(pos);
                    idx_b = count;
                    count += 1;
                }
                if (cell_phi(x, y + 1) * cell_phi(x + 1, y + 1) < 0) {
                    TV shift = TV(std::abs(cell_phi(x, y + 1) / (cell_phi(x, y + 1) - cell_phi(x + 1, y + 1))) * dx, 0);
                    TV pos = TV(x, y + 1) * dx + shift;
                    points.push_back(pos);
                    idx_c = count;
                    count += 1;
                }
                if (cell_phi(x, y) * cell_phi(x, y + 1) < 0) {
                    TV shift = TV(0, std::abs(cell_phi(x, y) / (cell_phi(x, y) - cell_phi(x, y + 1))) * dx);
                    TV pos = TV(x, y) * dx + shift;
                    points.push_back(pos);
                    idx_d = count;
                    count += 1;
                }
            }
            if (idx_a != -1 && idx_b != -1) segments.push_back(Vector<int, 2>(idx_a, idx_b));
            if (idx_b != -1 && idx_c != -1) segments.push_back(Vector<int, 2>(idx_b, idx_c));
            if (idx_c != -1 && idx_d != -1) segments.push_back(Vector<int, 2>(idx_c, idx_d));
            if (idx_d != -1 && idx_a != -1) segments.push_back(Vector<int, 2>(idx_d, idx_a));
            if (idx_a != -1 && idx_c != -1) segments.push_back(Vector<int, 2>(idx_a, idx_c));
            if (idx_b != -1 && idx_d != -1) segments.push_back(Vector<int, 2>(idx_b, idx_d));
        });
    }
};

} // namespace ZIRAN