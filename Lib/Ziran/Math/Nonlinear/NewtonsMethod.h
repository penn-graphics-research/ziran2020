#ifndef NEWTONS_METHOD_H
#define NEWTONS_METHOD_H
#include <Ziran/CS/Util/ErrorContext.h>
#include <Ziran/CS/Util/Meta.h>
#include <Ziran/CS/Util/Timer.h>
#include <iostream>

namespace ZIRAN {
/**
  Newton's Method

  Templatized on TOBJ the objective function
  which should implement the following functions
  T computeGradient(TV& x)
  TV& computeStep()  // computes the next step based on the previous x passed into computeResidual
  and typedefs
  TV and T
*/
template <class Objective>
class NewtonsMethod {
    using Vec = typename Objective::NewtonVector;
    using T = typename Objective::Scalar;

    Objective& objective;

public:
    Vec step_direction;
    Vec residual;

    T tolerance;
    int max_iterations;
    T linear_solve_tolerance_scale;
    bool linesearch;

    NewtonsMethod(Objective& objective, const T tolerance = (T)1e-6, const int max_iterations = 5, const bool use_linesearch = false)
        : objective(objective)
        , tolerance(tolerance)
        , max_iterations(max_iterations)
        , linear_solve_tolerance_scale((T)1)
        , linesearch(use_linesearch)
    {
    }

    bool solve(Vec& x, const bool verbose = false)
    {
        ZIRAN_TIMER();
        step_direction.resizeLike(x);
        residual.resizeLike(x);

        objective.updateState(x);
        T E0 = T(0.0); // totalEnergy of last iteration, for line search only
        Vec x0; // x of last iteration, for line search only
        if (linesearch) {
            E0 = objective.totalEnergy();
            x0 = x;
        }

        for (int it = 0; it < max_iterations; it++) {
            objective.computeResidual(residual);
            T residual_norm = objective.computeNorm(residual);
            ZIRAN_VERB_IF(verbose, "Newton iter ", it, "; (designated norm) residual = ", residual_norm);
            ZIRAN_CONTEXT("Newton step", it);
            ZIRAN_CONTEXT(residual_norm);
            if (residual_norm < tolerance) {
                ZIRAN_VERB_IF(true, "Newton terminates at ", it, "; (designated norm) residual = ", residual_norm);
                return true;
            }
            else {
                ZIRAN_VERB_IF(true, "Newton residual = ", residual_norm);
            }
            T linear_solve_relative_tolerance = std::min((T)0.5, linear_solve_tolerance_scale * std::sqrt(std::max(residual_norm, tolerance)));
            objective.computeStep(step_direction, residual, linear_solve_relative_tolerance);

            if (linesearch) {
                T stepSize = T(1.0), E;
                do {
                    x = x0 + stepSize * step_direction;
                    objective.updateState(x);
                    E = objective.totalEnergy();
                    stepSize /= T(2.0);
                } while (E > E0);
                ZIRAN_INFO("Newton it ", it, " stepSize = ", stepSize * T(2.0));
                E0 = E;
                x0 = x;
            }
            else {
                x += step_direction;
                objective.updateState(x);
            }
        }
        // max iterations reached
        //NOTE: previously then x and state are not matched
        // now they are matched, will this be a problem?
        return false;
    }
};
} // namespace ZIRAN
#endif
