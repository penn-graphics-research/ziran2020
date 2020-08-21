#ifndef PINNING_TARGET_H
#define PINNING_TARGET_H
#include <Ziran/CS/Util/Forward.h>
#include <Ziran/CS/Util/BinaryIO.h>

namespace ZIRAN {

template <class T, int dim>
class PinningTarget {
public:
    using TV = Vector<T, dim>;
    T k, d; // stiffness and damping coefficient
    TV x, v; // target position and target velocity
    TV X;

    PinningTarget()
    {
        k = 0;
        d = 0;
        x = TV::Zero();
        v = TV::Zero();
        X = x;
    }

    PinningTarget(const T k_in, const T d_in, const TV& x_in, const TV& v_in)
    {
        k = k_in;
        d = d_in;
        x = x_in;
        v = v_in;
        X = x;
    }

    void write(std::ostream& out) const
    {
        writeEntry(out, k);
        writeEntry(out, d);
        writeEntry(out, x);
        writeEntry(out, v);
        writeEntry(out, X);
    }

    static PinningTarget<T, dim> read(std::istream& in)
    {
        PinningTarget<T, dim> target;
        target.k = readEntry<T>(in);
        target.d = readEntry<T>(in);
        target.x = readEntry<TV>(in);
        target.v = readEntry<TV>(in);
        target.X = readEntry<TV>(in);
        return target;
    }

    static const char* name()
    {
        return "PinningTarget";
    }
};

template <class T, int dim>
struct PinningTargetScratch {
    using TV = Vector<T, dim>;
    T A;
    TV b;

    PinningTargetScratch()
        : A(0)
        , b(TV::Zero())
    {
    }

    static const char* name()
    {
        return "PinningTargetScratch";
    }
};

template <class T, int dim>
struct RW<PinningTargetScratch<T, dim>> {
    using Tag = NoWriteTag<PinningTargetScratch<T, dim>>;
};

template <class T, int dim>
struct RW<PinningTarget<T, dim>> {
    using Tag = CustomTypeTag<PinningTarget<T, dim>>;
};
} // namespace ZIRAN

#endif
