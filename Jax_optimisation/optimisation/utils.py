import functools
from typing import Optional, Union

from jax import numpy as np, tree_map, tree_multimap
from jax.tree_util import tree_flatten, tree_unflatten
from jaxlib import pytree

from utils.logging import Logger


# readable typing
Array = np.ndarray
Scalar = Union[float, complex, Array]


def pytree_optimiser(optimiser_fun):
    """
    A decorator that promotes an optimiser function operating on cost-functions that take a list of arrays as input
    to an optimiser function operating on cost-functions that take an arbitrary pytree as input
    """

    @functools.wraps(optimiser_fun)
    def optimiser_fun_tree(cost_fun_tree, initial_tree, *args, **kwargs):
        initial_flat, tree = tree_flatten(initial_tree)

        def cost_fun_flat(x_flat):
            x_tree = tree_unflatten(tree, x_flat)
            return cost_fun_tree(x_tree)

        result_flat, cost = optimiser_fun(cost_fun_flat, initial_flat, *args, **kwargs)
        result_tree = tree_unflatten(tree, result_flat)
        return result_tree, cost

    return optimiser_fun_tree


def _log_after_epoch(epoch_num: int,
                     cost: float,
                     cost_print_name: str,
                     cost_repr: callable,
                     logger: Logger,
                     last_cost: Optional[float] = None):
    msg = f'Epoch {epoch_num:03d},  {cost_print_name}={cost_repr(cost)}'
    if last_cost is not None:
        msg += f',  {cost_print_name} change={cost_repr(cost - last_cost)}'
    logger.log(msg, timestamp='short')


def pytree_to_dtype(data: pytree, dtype: np.dtype) -> pytree:
    flat, tree = tree_flatten(data)
    flat = [np.asarray(arr, dtype=dtype) for arr in flat]
    return tree_unflatten(tree, flat)


def default_cost_repr(cost: Scalar) -> str:
    return f'{cost:.12f}'


# --------------------
#     vector ops
# --------------------
# treat a pytrees like vectors


def vec_max(x: pytree) -> Scalar:
    return np.max(tree_flatten(tree_map(np.max, x))[0])


def vec_abs(x: pytree) -> pytree:
    return tree_map(np.abs, x)


def vec_neg(x: pytree) -> pytree:
    return vec_scale(x, -1.)


def vec_add(x: pytree, y: pytree) -> pytree:
    return tree_multimap(lambda arr1, arr2: arr1 + arr2, x, y)


def vec_scale(x: pytree, a: Scalar) -> pytree:
    return tree_map(lambda arr: a * arr, x)


def vec_add_prefactor(x: pytree, a: Scalar, y: pytree) -> pytree:
    # x + a * y
    return vec_add(x, vec_scale(y, a))


def vec_conj(x: pytree) -> pytree:
    return tree_map(np.conj, x)


def vec_real(x: pytree) -> pytree:
    return tree_map(np.real, x)


def vec_sum(x: pytree) -> Scalar:
    return np.sum(tree_flatten(tree_map(np.sum, x))[0])


def vec_mul_elementwise(x: pytree, y: pytree) -> pytree:
    return tree_multimap(lambda arr1, arr2: arr1 * arr2, x, y)


def vec_dot(x: pytree, y: pytree) -> Scalar:
    return vec_sum(vec_mul_elementwise(x, y))


def vec_copy(x: pytree) -> pytree:
    return tree_map(lambda arr: arr.copy(), x)
