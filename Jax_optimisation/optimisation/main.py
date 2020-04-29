import jax.numpy as np
from jax import value_and_grad
from jax.experimental.optimizers import sgd
from jax.tree_util import tree_flatten, pytree

from typing import Optional, Tuple

from optimisation.utils import _log_after_epoch, pytree_to_dtype, default_cost_repr
from utils.logging import Logger
from optimisation import lbfgs, utils


def maximise(cost_function: callable,
             initial_guess: pytree,
             algorithm: Optional[str] = 'L-BFGS',
             complex_args: Optional[bool] = True,
             double_precision: Optional[bool] = True,
             options: Optional[dict] = None,
             logger: Optional[Logger] = Logger(None, True, prefix='[optimisation]'),
             cost_print_name: Optional[str] = 'Cost',
             cost_repr: Optional[callable] = default_cost_repr):
    """
        Calculates the inputs that maximise a real-valued cost function.
        The cost function can be a function of arbitrarily many real or complex numbers or arrays and must return a
        single real value.
        The cost function must be traceable by jax.
        The cost function must have a single argument which can be an arbitrary pytree of arrays or numbers.
        A pytree is a tree structure of python containers (tuples, lists, ...).
        Every container in the tree contains numbers, arrays or nested containers.

        Parameters
        ----------
        double_precision
        cost_function
            The function to be maximised.
            cost_function(inputs: pytree) -> float
        initial_guess
            The initial best guess. Must be a pytree that can be passed to `cost_function`
        algorithm
            A keyword for the algorithm to be used. Currently supported:
             - "sgd" (Stochastic Gradient Descend)
             - "L-BFGS" (Limited Memory BFGS, quasi-Newton algorithm with optional line search)
            L-BFGS is recommended for applications in physics.
            Line search is recommended unless the cost_function is *extremely* costly to evaluate.
            But even then, line search with low `max_iter_ls` might give best results
        complex_args
            If complex_args == True: Algorithm considers cost_function as a function of complex arguments
                and returns its complex maximiser
            If complex_args == False: Algorithm considers cost_function as a function of real arguments
                and returns its real maximiser
        options
            Either a dictionary of optimisation parameters or `None` (use defaults)
            Options are algorithm-specific. Consider `_parse_options` for documentation.
            Not all values need to be specified, for missing keys, the default value is used.
        logger
            A Logger instance that receives information about the optimisation. Default: Log to console but not to file
        cost_print_name
            A name for the value of the cost_function to be used for logging. e.g. "energy"
        cost_repr
            Converts a value of the cost function to a printable representation.
            Mainly used so that maximise prints values of the correct cost function, not its negative

        Returns
        -------
        x_optim
            The optimal input that maximise the cost_function. A pytree that can be passed to `cost_function`
        cost_optim
            The maximal value of the cost function. i.e. `cost_function(x_optim)`

        """

    def _cost_function(x):
        return - cost_function(x)

    def _cost_repr(cost):
        return cost_repr(-cost)

    logger.log(f'Starting Maximisation of {cost_print_name}.', timestamp='short')
    logger.log(f'Optimisation parameters: '
               f'algorithm = {algorithm}\n'
               f'complex_args = {complex_args}\n'
               f'double_precision = {double_precision}\n'
               f'options = {options}', file_only=True)
    x_optim, cost_optim = _minimise(_cost_function, initial_guess, algorithm, complex_args,
                                    double_precision, options, logger, cost_print_name, _cost_repr)

    return x_optim, -cost_optim


def minimise(cost_function: callable,
             initial_guess: pytree,
             algorithm: Optional[str] = 'L-BFGS',
             complex_args: Optional[bool] = True,
             double_precision: Optional[bool] = True,
             options: Optional[dict] = None,
             logger: Optional[Logger] = Logger(None, True, prefix='[optimisation]'),
             cost_print_name: Optional[str] = 'Cost',
             cost_repr: Optional[callable] = default_cost_repr) -> pytree:
    """
    Calculates the inputs that minimise a real-valued cost function.
    The cost function can be a function of arbitrarily many real or complex numbers or arrays and must return a
    single real value.
    The cost function must be traceable by jax.
    The cost function must have a single argument which can be an arbitrary pytree of arrays or numbers.
    A pytree is a tree structure of python containers (tuples, lists, ...).
    Every container in the tree contains numbers, arrays or nested containers.

    Parameters
    ----------
    double_precision
    cost_function
        The function to be minimised.
        cost_function(inputs: pytree) -> float
    initial_guess
        The initial best guess. Must be a pytree that can be passed to `cost_function`
    algorithm
        A keyword for the algorithm to be used. Currently supported:
         - "sgd" (Stochastic Gradient Descend)
         - "L-BFGS" (Limited Memory BFGS, quasi-Newton algorithm with optional line search)
        L-BFGS is recommended for applications in physics.
        Line search is recommended unless the cost_function is *extremely* costly to evaluate.
        But even then, line search with low `max_iter_ls` might give best results
    complex_args
        If complex_args == True: Algorithm considers cost_function as a function of complex arguments
            and returns its complex minimiser
        If complex_args == False: Algorithm considers cost_function as a function of real arguments
            and returns its real minimiser
    options
        Either a dictionary of optimisation parameters or `None` (use defaults)
        Options are algorithm-specific. Consider `_parse_options` for documentation.
        Not all values need to be specified, for missing keys, the default value is used.
    logger
        A Logger instance that receives information about the optimisation. Default: Log to console but not to file
    cost_print_name
        A name for the value of the cost_function to be used for logging. e.g. "energy"
    cost_repr
        Converts a value of the cost function to a printable representation.
        Mainly used so that maximise prints values of the correct cost function, not its negative

    Returns
    -------
    x_optim
        The optimal input that minimises the cost_function. A pytree that can be passed to `cost_function`
    cost_optim
        The minimal value of the cost function. i.e. `cost_function(x_optim)`

    """

    logger.log(f'Starting Minimisation of {cost_print_name}.', timestamp='short')
    logger.log(f'Optimisation parameters: '
               f'algorithm = {algorithm}\n'
               f'complex_args = {complex_args}\n'
               f'double_precision = {double_precision}\n'
               f'options = {options}', file_only=True)
    return _minimise(cost_function, initial_guess, algorithm, complex_args,
                     double_precision, options, logger, cost_print_name, cost_repr)


def _minimise(cost_function, initial_guess, algorithm, complex_args,
              double_precision, options, logger, cost_print_name, cost_repr):
    """
    Helper function used by both minimise and maximise

    """
    options = _parse_options(options, algorithm)

    if double_precision:
        dtype = np.complex128 if complex_args else np.float64
    else:
        dtype = np.complex64 if complex_args else np.float32
    initial_guess = pytree_to_dtype(initial_guess, dtype)

    if algorithm in ['sgd', 'SGD', 'stochastic gradient descend']:
        step_size = options['step_size']
        return _jax_fun_triple(cost_function, initial_guess, sgd(step_size), options, logger, cost_print_name,
                               cost_repr)
    elif algorithm in ['lbfgs', 'LBFGS', 'L-BFGS']:
        return lbfgs.minimise(cost_function, initial_guess, options, logger, cost_print_name, cost_repr)
    else:
        raise ValueError(f'Algorithm Keyword "{algorithm}" is not a valid keyword or the algorithm is not implemented')


def _jax_fun_triple(cost_function: callable,
                    initial_value,
                    fun_triple: Tuple[callable, callable, callable],
                    options: dict,
                    logger: Logger,
                    cost_print_name: str,
                    cost_repr: callable):

    tolerance_change = options['tolerance_change']
    tolerance_grad = options['tolerance_grad']

    init_fun, update_fun, get_params = fun_triple

    def opt_cond(_grads: pytree) -> bool:
        flat_grads, _ = tree_flatten(_grads)
        return utils.vec_sum(utils.vec_abs(flat_grads)) < tolerance_grad

    def _sufficient_change(x_new: pytree, x_old: pytree) -> bool:
        flat_new, _ = tree_flatten(x_new)
        flat_old, _ = tree_flatten(x_old)
        return utils.vec_max(
            utils.vec_add_prefactor(flat_new, -1, flat_old)) >= tolerance_change

    opt_state = init_fun(initial_value)
    last_cost = 0
    for i in range(options['n_epochs']):
        x = get_params(opt_state)
        cost, grads = value_and_grad(cost_function)(x)

        if opt_cond(grads):
            logger.log(f'The gradient vanishes up to numerical tolerance -> Optimum found')
            break

        if abs(cost - last_cost) < tolerance_change:
            logger.warn(f'Lack of progress in cost. aborting.')
            break

        opt_state = update_fun(i, grads, opt_state)

        if not _sufficient_change(get_params(opt_state), x):
            logger.warn(f'Lack of progress in parameter space. aborting.')
            break

        last_cost = cost
        _log_after_epoch(i, cost, cost_print_name, cost_repr, logger, last_cost)

    return get_params(opt_state), last_cost


def _parse_options(options: dict, algorithm: str) -> dict:
    # Default Values

    # --------------
    #      SGD
    # --------------
    if algorithm in ['sgd', 'SGD', 'stochastic gradient descend']:
        defaults = {'n_epochs': 200,  # total number of updates (=total number of calls of cost_function)
                    'step_size': 1,  # step_size, i.e. x_new = x_old - step_size * grad, can be float or schedule
                                     # (see jax.experimental.optimizers)
                    'tolerance_grad': 1e-5,  # Numerical tolerance for gradients. If max(abs(grad)) < tolerance_grad,
                                             # the gradient is considered vanishing and a minimum is found
                    'tolerance_change': 1e-9,  # Numerical tolerance for change in cost or variables.
                                               # Changes below this tolerance are considered negligible
                    }
    # --------------
    #    L-BFGS
    # --------------
    elif algorithm in ['lbfgs', 'LBFGS', 'L-BFGS']:
        defaults = {'step_size': 1.,  # step_size, i.e. x_new = x_old - step_size * grad, can be float or schedule
                                      # (see jax.experimental.optimizers)
                    'max_iter': 200,  # Maximum number of iterations of the L-BFGS update
                    'max_eval': 'default',  # Maximum number of evaluations of the cost function.
                                            # Default: 1.25 * max_iter
                    'tolerance_grad': 1e-5,  # Numerical tolerance for gradients. If abs(grad) < tolerance_grad,
                                             # the gradient is considered vanishing and a minimum is found
                    'tolerance_change': 1e-9,  # Numerical tolerance for change in cost or variables.
                                               # Changes below this tolerance are considered negligible
                    'line_search_fn': 'strong_wolfe',  # Either 'strong_wolfe' or None
                    'history_size': 100,  # Number of old steps and gradients that are stored to approximate the Hessian
                                          # Need to store 2 * history_size objects of form of `initial_guess` in memory.
                    'wolfe_c1': 1e-4,  # The Parameter in the Armijo condition
                    'wolfe_c2': 0.9,  # The Parameter in the (2nd) Wolfe condition
                    'max_iter_ls': 25,  # The maximum number of iterations in each line search
                    }
    else:
        raise RuntimeError(f'Algorithm keyword {algorithm} was not recognised in _parse_options, even though it passed '
                           f'the input check of minimise. This should not happen')

    # compare and stitch options with defaults
    if options is None:
        options = defaults
    for key in options:
        if key not in defaults:
            raise ValueError(f'"{key}" is not a valid option for algorithm {algorithm}')
    for key in defaults:
        if key not in options:
            options[key] = defaults[key]

    # For some algorithms, defaults depend on other options:
    if algorithm in ['lbfgs', 'LBFGS', 'L-BFGS']:
        if options['max_eval'] == 'default':
            options['max_eval'] = round(1.25 * options['max_iter'])

    return options
