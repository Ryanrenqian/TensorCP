#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:31:09 2019

@author: ryan
"""

import numpy as np
import warnings

import tensorly as tl
from tensorly.random import check_random_state
from tensorly.base import unfold
from tensorly.kruskal_tensor import kruskal_to_tensor
from tensorly.tenalg import khatri_rao

def normalize_factors(factors):
    # allocate variables for weights, and normalized factors
    rank = factors[0].shape[1]
    weights = tl.ones(rank, **tl.context(factors[0]))
    normalized_factors = []

    # normalize columns of factor matrices
    for factor in factors:
        scales = tl.norm(factor, axis=0)
        weights *= scales
        scales_non_zero = tl.where(scales==0, tl.ones(tl.shape(scales), **tl.context(factors[0])), scales)
        normalized_factors.append(factor/scales_non_zero)
    return normalized_factors, weights


def initialize_factors(tensor, rank, random_state=None, non_negative=False):

    rng = check_random_state(random_state)
    factors = [tl.tensor(rng.random_sample((tensor.shape[i], rank)), **tl.context(tensor)) for i in range(tl.ndim(tensor))]
    if non_negative:
        return [tl.abs(f) for f in factors]
    else:
        return factors


def cpwopt(tensor, rank,W=None, n_iter_max=100, tol=1e-8,orthogonalise=False, random_state=None, verbose=False, return_errors=False):
    if orthogonalise and not isinstance(orthogonalise, int):
        orthogonalise = n_iter_max

    factors = initialize_factors(tensor, rank, random_state=random_state)
    rec_errors = []
    norm_tensor = tl.norm(tensor*W, 2)
    for iteration in range(n_iter_max):
        if orthogonalise and iteration <= orthogonalise:
            factor = [tl.qr(factor)[0] for factor in factors]

        for mode in range(tl.ndim(tensor)):
            pseudo_inverse = tl.tensor(np.ones((rank, rank)), **tl.context(tensor))
            for i, factor in enumerate(factors):
                if i != mode:
                    pseudo_inverse = pseudo_inverse*tl.dot(tl.transpose(factor), factor)
            factor = tl.dot(unfold(tensor, mode), khatri_rao(factors, skip_matrix=mode))
            factor = tl.transpose(tl.solve(tl.transpose(pseudo_inverse), tl.transpose(factor)))
            factors[mode] = factor

        if tol:
            rec_error = tl.norm(W*(tensor - kruskal_to_tensor(factors)), 2)/norm_tensor
            rec_errors.append(rec_error)
            if iteration > 1:
                if verbose:
                    print('reconstruction error={}, variation={}.'.format(
                        rec_errors[-1], rec_errors[-2] - rec_errors[-1]))

                if tol and abs(rec_errors[-2] - rec_errors[-1]) < tol:
                    if verbose:
                        print('converged in {} iterations.'.format(iteration))
                    break
                    
    if return_errors:
        return factors, rec_errors
    else:
        return factors
    