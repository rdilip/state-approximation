"""
During the moses move, we decrease the bond dimension of the split wavefunction
at each step. In the final step, we want to have bond dimension 1 (0 entanglement
entropy). Some procedures are better than others. This module has different functions
to get specific schedules. It should interface cleanly with state_approximation.
"""

import numpy as np
import scipy
from scipy.optimize import curve_fit
from inspect import signature

def get_schedule(bond_dimension, depth, mode, final_bond_dimension = 1):
    """
    Given the maximum bond dimension of the starting wavefunction, returns
    a list of length depth containing the maximum bond dimension of the final
    wavefunction.
    """
    if mode not in ['half', 'linear']:
        raise ValueError("Predefined modes are 'half' or 'linear'")
    schedule = [1 for i in range(depth)]
    if depth == 1:
        return schedule
    if mode == 'half':
        schedule[0] = largest_power_of_two(bond_dimension)
        schedule[0] = int(0.95* bond_dimension)
        for i in range(depth-1):
            next_eta = int(schedule[i] / 2)
            if next_eta == 0:
                next_eta = 1
            schedule[i+1] = next_eta
        schedule[-1] = 1
        if schedule[-1] != 1:
            warnings.warn("Does not reach a product state.")
        if schedule[-2] == 1:
            warnings.warn("Reached a product state in fewer iterations than depth.")
        return schedule
    if mode == 'linear':
        return np.array(np.linspace(int(0.95*bond_dimension), 1, depth), dtype=np.int)

def process_function(fn_name):
    """
    List of pre-written functions. 
    Parameters
    ----------
    fn_name : str
        Can be one of the following values:
            * exponential
            * linear
            * quadratic
    Returns
    -------
    fn : function   
        Corresponding function
    """
    if fn_name == 'exponential':
        return (lambda x, a, b: np.exp(-x*a) + b
    elif fn_name == 'linear':
        return lambda x, a, b: a*x + b
    elif fn_name == 'quadratic':
        # No horizontal shift because we know what 0 should be
        return lambda x, a, b: -a*x*x + b

def interpolate(Si, Sf, depth, fn, initial_guess=None):
    """ 
    Gives an entropy scaling.
    Parameters
    ----------
    Si : np.float
        Initial entropy
    Sf : np.float
        Final entropy (usually 0)
    depth : int
        number of layers in circuit
    fn : function
        Entropy scaling function
    initial_guess : list
        List of parameters for the initial guess. 
    Returns
    -------
    Slist : list
        List of floats corresponding to entanglement entropy at each step.
    """
    if initial_guess is None:
        num_params = len(signature(fn).parameters) - 1
        initial_guess = [1.] * num_params
    if type(fn) is str:
        fn = process_function(fn)
        
    popt, pcov = curve_fit(fn, [0, depth], [Si, Sf], p0=initial_guess)
    desired_ee = [fn(i, *popt) for i in range(1, depth+1)]
    return desired_ee

