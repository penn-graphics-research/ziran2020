#ifndef PARALLEL_PINNING_TARGET_H
#define PARALLEL_PINNING_TARGET_H
#include <Ziran/CS/Util/Forward.h>
#include <Ziran/CS/Util/BinaryIO.h>

namespace ZIRAN {

template <class T, int dim>
class ParallelPinningTarget {
public:
    using TV = Vector<T, dim>;
    T k, d; // stiffness and damping coefficient
    // TV x, v; // target position and target velocity
    StdVector<TV> x, v;

    ParallelPinningTarget()
    {
        k = 0;
        d = 0;
    }

    ParallelPinningTarget(const T k_in, const T d_in, const StdVector<TV>& x_in, const StdVector<TV>& v_in)
    {
        k = k_in;
        d = d_in;
        x = x_in;
        v = v_in;
    }

    void write(std::ostream& out) const
    {
        writeEntry(out, k);
        writeEntry(out, d);
        writeSTDVector(out, x);
        writeSTDVector(out, v);
    }

    static ParallelPinningTarget<T, dim> read(std::istream& in)
    {
        ParallelPinningTarget<T, dim> target;
        target.k = readEntry<T>(in);
        target.d = readEntry<T>(in);
        readSTDVector<TV>(in, target.x);
        readSTDVector<TV>(in, target.v);
        return target;
    }

    static const char* name()
    {
        return "ParallelPinningTarget";
    }
};

template <class T, int dim>
struct ParallelPinningTargetScratch {
    using TV = Vector<T, dim>;
    T A;
    TV b;

    ParallelPinningTargetScratch()
        : A(0)
        , b(TV::Zero())
    {
    }

    static const char* name()
    {
        return "ParallelPinningTargetScratch";
    }
};

template <class T, int dim>
struct RW<ParallelPinningTargetScratch<T, dim>> {
    using Tag = NoWriteTag<ParallelPinningTargetScratch<T, dim>>;
};

template <class T, int dim>
struct RW<ParallelPinningTarget<T, dim>> {
    using Tag = CustomTypeTag<ParallelPinningTarget<T, dim>>;
};
} // namespace ZIRAN

#endif
