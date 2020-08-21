#ifndef BULK_VISCOSITY_MPM_FORCE_HELPER_H
#define BULK_VISCOSITY_MPM_FORCE_HELPER_H

#include <Ziran/Math/Geometry/Particles.h>
#include <MPM/Force/MpmForceBase.h>

namespace ZIRAN {

template <class T, int dim>
class BulkViscosity {
public:
    /**
       b1: linear bulk damping coefficient
       b2: quadratic bulk damping coefficient
       Rho: original density (R)
       Le: element characteristic length (dx)
       E: youngs modulus, for computing dilational wave speed c=sqrt(E/rho)
    */
    T b;

    BulkViscosity() {}

    BulkViscosity(const T b_in)
        : b(b_in)
    {
    }

    // Maybe I don't need read/write because the data are trivial.
    void write(std::ostream& out) const
    {
        writeEntry(out, b);
    }
    static BulkViscosity<T, dim> read(std::istream& in)
    {
        BulkViscosity<T, dim> target;
        target.b = readEntry<T>(in);
        return target;
    }

    static const char* name()
    {
        return "BulkViscosity";
    }
};

/**
   This is the bulk viscosity damping model that Abaqus uses:
   http://50.16.225.63/v2016/books/gsk/default.htm?startat=ch09s05.html
   Note:
   1. It is supposed to work with EXPLICIT time integration. It won't do anything for
      implicit.
   2. It will apply forces on particles with 'F', 'element_measure' and 'BulkViscosity'
 */
template <class T, int dim>
class BulkViscosityMpmForceHelper : public MpmForceHelperBase<T, dim> {
public:
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;
    typedef BulkViscosity<T, dim> Bulk;

    Particles<T, dim>& particles; // contains Xn and Vn

    DataArray<Bulk>& bulk_dummy_reference;
    DataArray<T>& element_measure_dummy_reference;

    // TODO: when adding the force helper (in MpmSimulation), make sure to do
    // something like particles.add(helper.bulk_name(), range, TM::Identity())
    // so that the bulk attribute exists on particles.

    StdVector<T> div_vn; // volumetric strain rate

    BulkViscosityMpmForceHelper(Particles<T, dim>& particles)
        : particles(particles)

        // assert these names exist
        , bulk_dummy_reference(particles.get(bulk_name()))
        , element_measure_dummy_reference(particles.get(element_measure_name()))
    {
    }

    virtual ~BulkViscosityMpmForceHelper() {}

    bool needGradVn() override;

    void getGradVn(const StdVector<TM>& gradVn) override;

    // add stress to V^0 tau = V^n sigma
    void updateState(const DisjointRanges& subrange, StdVector<TM>& vtau, TVStack& fp) override;
    void updateStateHelper(const BulkViscosity<T, dim>& bulk, const T element_measure, const T& J, const T rate, TM& vtau);

    inline static AttributeName<Bulk> bulk_name()
    {
        return AttributeName<Bulk>(Bulk::name());
    }

    inline static AttributeName<T> element_measure_name()
    {
        return AttributeName<T>("element measure");
    }

    // "F" is a possible strain measures for computing J
    inline static AttributeName<TM> F_name()
    {
        return AttributeName<TM>("F");
    }

    // "cotangent" is a possible strain measure for computing J
    inline static AttributeName<TM> cotangent_name()
    {
        return AttributeName<TM>("cotangent");
    }

    // "J" is a possible strain measure for computing J
    inline static AttributeName<T> J_name()
    {
        return AttributeName<T>("J");
    }
};
} // namespace ZIRAN

#endif
