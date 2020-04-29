import jax.numpy as np
from jax import value_and_grad
from jax.experimental.optimizers import make_schedule
from jax.tree_util import pytree

from typing import Union

from optimisation.utils import _log_after_epoch, vec_max, vec_abs, vec_neg, vec_scale, vec_add_prefactor, vec_conj, \
    vec_sum, vec_dot, vec_copy
from utils.logging import Logger


# readable typing
Array = np.ndarray
Scalar = Union[float, complex, Array]


# FIXME revamp interface, doc

def minimise(cost_function: callable,
             initial_guess: pytree,
             options: dict,
             logger: Logger,
             cost_print_name: str,
             cost_repr: callable):

    step_size = options['step_size']
    step_size = make_schedule(step_size)
    max_iter = options['max_iter']  # maximum number of optimisation iterations
    max_eval = options['max_eval']  # maximum number of function evaluations
    tolerance_grad = options['tolerance_grad']
    tolerance_change = options['tolerance_change']
    line_search_fn = options['line_search_fn']
    history_size = options['history_size']
    wolfe_c1 = options['wolfe_c1']
    wolfe_c2 = options['wolfe_c1']
    max_iter_ls = options['max_iter_ls']

    # function that evaluates cost and the complex gradient ∂* and keeps track of the number of evaluations

    def evaluate_cost(_x, num_evals):
        _cost, _grad = value_and_grad(cost_function)(_x)
        return _cost, vec_conj(_grad), num_evals + 1

    # initial evaluation
    current_evals = 0
    initial_cost, grad, current_evals = evaluate_cost(initial_guess, current_evals)
    _log_after_epoch(0, initial_cost, cost_print_name, cost_repr, logger, last_cost=None)

    # already optimal?
    opt_cond = vec_max(vec_abs(grad)) <= tolerance_grad
    if opt_cond:
        logger.log('The initial guess is already optimal.')
        return initial_guess, initial_cost

    # initialise variables for the loop
    x = initial_guess  # current best guess for the minimum
    last_x = None
    last_grad = None
    cost = initial_cost
    n_iter = 0
    y_history = []
    s_history = []
    rho_history = []
    gamma = 1.  # scale of diagonal Hessian approximation

    # optimise for up to max_iter iterations
    while n_iter < max_iter:
        n_iter += 1

        # --------------------------------
        #    compute descend direction
        # --------------------------------
        if n_iter == 1:
            # no curvature information yet, just descend in gradient direction
            direction = vec_neg(grad)
        else:
            # L-BFGS update
            y = vec_add_prefactor(grad, -1, last_grad)
            s = vec_add_prefactor(x, -1, last_x)
            rho_inv = np.real(vec_dot(vec_conj(y), s))
            if rho_inv > 1e-10:  # else skip the update
                if len(y_history) == history_size:  # shift memory by one
                    y_history.pop(0)
                    s_history.pop(0)
                    rho_history.pop(0)

                y_history.append(y)
                s_history.append(s)
                rho_history.append(1. / rho_inv)

                # scale of diagonal Hessian approximation
                gamma = rho_inv / np.real(vec_dot(vec_conj(y), y))

            # Compute product of approximate inverse Hessian with gradient ("two-loop recursion")
            current_history_size = len(y_history)
            alpha = [np.nan] * current_history_size
            direction = vec_neg(grad)
            for i in range(current_history_size - 1, -1, -1):  # iterate over history newest to oldest
                alpha[i] = rho_history[i] * np.real(vec_dot(vec_conj(s_history[i]), direction))
                direction = vec_add_prefactor(direction, -alpha[i], y_history[i])

            direction = vec_scale(direction, gamma)

            for i in range(current_history_size):  # oldest to newest
                beta_i = rho_history[i] * np.real(vec_dot(vec_conj(y_history[i]), direction))
                direction = vec_add_prefactor(direction, alpha[i] - beta_i, s_history[i])

        # save grad for computing the next y
        last_grad = vec_copy(grad)
        last_cost = cost

        # --------------------------------
        #       compute step length
        # --------------------------------
        # reset initial guess
        t = min(1., 1. / vec_sum(vec_abs(grad))) * step_size(n_iter) if n_iter == 1 else step_size(n_iter)
        # directional derivative
        gtd = np.real(vec_dot(vec_conj(grad), direction))

        # check for significant change
        if gtd > - tolerance_change:
            break

        # optional line search
        if line_search_fn in ['strong_wolfe', 'strong wolfe', 'Strong Wolfe']:
            def cost_fun_line(_t: float, num_evals: int):
                # evaluates the cost function and its gradient at a point that is a step t in direction from x
                _x = vec_add_prefactor(x, t, direction)
                _cost, _grad, num_evals = evaluate_cost(_x, num_evals)
                return _cost, _grad, num_evals

            # cost_new, grad_new, t, func_evals
            cost, grad, t, ls_func_evals = _strong_wolfe(cost_fun_line, t, direction, cost, grad, gtd, wolfe_c1,
                                                         wolfe_c2, tolerance_change, max_iter_ls)
            current_evals += ls_func_evals
            last_x = vec_copy(x)
            x = vec_add_prefactor(x, t, direction)
            opt_cond = vec_max(vec_abs(grad)) <= tolerance_grad
        elif line_search_fn is None:
            # simply move with fixed step size
            last_x = vec_copy(x)
            x = vec_add_prefactor(x, t, direction)
            cost, grad, current_evals = evaluate_cost(x, current_evals)
            opt_cond = vec_max(vec_abs(grad)) <= tolerance_grad
        else:
            raise RuntimeError(f'Line search function {line_search_fn} is not supported')

        _log_after_epoch(n_iter, cost, cost_print_name, cost_repr, logger, last_cost)

        # --------------------------------
        #       check conditions
        # --------------------------------
        # optimality reached
        if opt_cond:
            logger.log(f'The gradient vanishes up to numerical tolerance -> Optimum found')
            break

        # allotted resources exceeded
        if n_iter == max_iter:
            logger.warn(f'Maximum number of iterations reached before optimisation has converged. aborting.')
            break
        if current_evals >= max_eval:
            logger.warn(f'Maximum number of function evaluations exceeded before optimisation has converged. aborting.')
            break

        # lack of progress
        if abs(t) * vec_max(vec_abs(direction)) <= tolerance_change:
            logger.log(f'Lack of progress in parameter space. aborting.')
            break
        if abs(cost - last_cost) < tolerance_change:
            logger.log(f'Lack of progress in cost. aborting.')
            break

    return x, cost


def _strong_wolfe(cost_fun_line: callable,
                  t: float,
                  direction: pytree,
                  cost_init: float,
                  grad_init: pytree,
                  gtd_init: complex,
                  c1: float,  # default 1e-4
                  c2: float,  # default 0.9
                  tolerance_change: float,  # default 1e-9
                  max_ls: int):  # default 25
    """
    Performs a line search to find a step length that fulfils the Wolfe conditions

    Parameters
    ----------
    cost_fun_line
        function that evaluates the cost function on a line
        Parameters: t (step length), n_evals
        Returns: cost (value of cost function), grad (cost gradient), n_evals
    t
        the initial guess for the step length
    direction
        the search direction from L-BFGS
    cost_init
        the value of the cost function at the initial point
    grad_init
        the gradient of the cost function at the initial point
    gtd_init
        the directional derivative of the cost function at the initial point
    c1
        The Armijo parameter. Armijo condition is f(z + t*d) <= f(z) + c1 * t * gtd(z)
    c2
        The Wolfe parameter. (2nd) Wolfe condition is gtd(z + t*d) >= c2 * gtd(z)
    tolerance_change
        value changes below this tolerance are considered to be zero
    max_ls
        maximum number of line-search iterations

    Returns
    -------
    cost_new
        The cost function at the new point
    grad_new
        The gradient of the cost function at the new point
    t
        The step length required to reach the new point, that satisfies the wolfe conditions
    func_evals
        The number of times the cost function was evaluated
    """

    d_norm = vec_max(vec_abs(direction))

    # evaluate at initially proposed step
    func_evals = 0
    cost_new, grad_new, func_evals = cost_fun_line(t, func_evals)
    gtd_new = np.real(vec_dot(vec_conj(grad_new), direction))

    # ------------------------
    #      bracket phase
    # ------------------------
    # bracket an interval containing a point that satisfies the Wolfe criteria
    t_last, cost_last, grad_last, gtd_last = 0, cost_init, vec_copy(grad_init), gtd_init
    bracket, bracket_cost, bracket_grad, bracket_gtd = [], [], [], []
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # check conditions
        if cost_new > (cost_init + c1 * t * gtd_init) or (ls_iter > 1 and cost_new >= cost_last):
            bracket = [t_last, t]
            bracket_cost = [cost_last, cost_new]
            bracket_grad = [grad_last, grad_new]
            bracket_gtd = [gtd_last, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd_init:
            bracket = [t]
            bracket_cost = [cost_new]
            bracket_grad = [grad_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_last, t]
            bracket_cost = [cost_last, cost_new]
            bracket_grad = [grad_last, grad_new]
            bracket_gtd = [gtd_last, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_last)
        max_step = 10 * t
        tmp = t
        t = _cubic_interpolate(t_last, cost_last, gtd_last, t, cost_new, gtd_new, bounds=(min_step, max_step))

        # next step
        t_last = tmp
        cost_last = cost_new
        grad_last = grad_new
        gtd_last = gtd_new
        cost_new, grad_new, func_evals = cost_fun_line(t, func_evals)
        gtd_new = np.real(vec_dot(vec_conj(grad_new), direction))
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_cost = [cost_init, cost_new]
        bracket_grad = [grad_init, grad_new]

    # ------------------------
    #       zoom phase
    # ------------------------
    # We either have a point satisfying the criteria or a bracket around it
    # Refine bracket until a point that satisfies criteria is found
    insufficient_progress = False
    low_pos, high_pos = (0, 1) if bracket_cost[0] < bracket_cost[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        # compute new trial value
        t = _cubic_interpolate(bracket[0], bracket_cost[0], bracket_gtd[0],
                               bracket[1], bracket_cost[1], bracket_gtd[1])

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   - we have made insufficient progress in the last step, or
        #   - `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insufficient_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 10% away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insufficient_progress = False
            else:
                insufficient_progress = True
        else:
            insufficient_progress = False

        # evaluate at new point
        cost_new, grad_new, func_evals = cost_fun_line(t, func_evals)
        gtd_new = np.real(vec_dot(vec_conj(grad_new), direction))
        ls_iter += 1

        if cost_new > (cost_init + c1 * t * gtd_init) or cost_new >= bracket_cost[low_pos]:
            # Armijo condition not satisfied or not lower than low_pos
            bracket[high_pos] = t
            bracket_cost[high_pos] = cost_new
            bracket_grad[high_pos] = grad_new
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_cost[0] <= bracket_cost[1] else (1, 0)
        else:
            # Armijo condition satisfied and lower than low_pos
            if abs(gtd_new) <= -c2 * gtd_init:
                # Wolfe condition satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                bracket[high_pos] = bracket[low_pos]
                bracket_cost[high_pos] = bracket_cost[low_pos]
                bracket_grad[high_pos] = bracket_grad[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            bracket[low_pos] = t
            bracket_cost[low_pos] = cost_new
            bracket_grad[low_pos] = grad_new
            bracket_gtd[low_pos] = gtd_new

        # line-search bracket is small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

    # return stuff
    t = bracket[low_pos]
    cost_new = bracket_cost[low_pos]
    grad_new = bracket_grad[low_pos]
    return cost_new, grad_new, t, func_evals


def _cubic_interpolate(x1: float, f1: float, g1: float, x2: float, f2: float, g2: float, bounds=None) -> float:
    """
    returns the minimiser of the cubic polynomial that matches a function f(x) in two points in value and derivative

    Parameters
    ----------
    x1
        the first point
    f1
        f(x1)
    g1
        f'(x1)
    x2
        the second point
    f2
        f(x2)
    g2
        f'(x2)
    bounds
        bounds that restrict the output, defaults to ( min(x1,x2), max(x1,x2) )

    Returns
    -------

    """
    if bounds is not None:
        x_min, x_max = bounds
    else:
        x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)

    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_squared = d1 ** 2 - g1 * g2
    if d2_squared >= 0:
        d2 = np.sqrt(d2_squared)
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, x_min), x_max)
    else:
        return (x_min + x_max) / 2
