# Author: Henry Heitmann

from collections import OrderedDict

import numpy as np
from numpy.random import PCG64, RandomState, SeedSequence

from fedml_api.distributed.utils.utils import get_shared_key


def add_models(m1, m2):
    result = OrderedDict()
    for key in m1.keys():
        assert m1[key].shape == m2[key].shape
        result[key] = np.add(m1[key], m2[key])
    return result


def subtract_models(m1, m2):
    result = OrderedDict()
    for key in m1.keys():
        assert m1[key].shape == m2[key].shape
        result[key] = np.subtract(m1[key], m2[key])
    return result


def divide_model(model, divisor):
    result = OrderedDict()
    for key in model.keys():
        result[key] = np.divide(model[key], divisor)
    return result


def generate_shared_mask(device_id, weights, shared_seeds, datatype='float32'):
    # init empty mask
    mask = OrderedDict()
    for key in weights.keys():
        mask[key] = np.zeros(weights[key].shape)

    for i, key in enumerate(shared_seeds):
        # generate a random mask for each seed
        rs = RandomState(PCG64(SeedSequence(shared_seeds[key])))
        temp_mask = OrderedDict()
        for k in mask.keys():
            # generate an array for each tensor
            if datatype == 'int8':
                temp_mask[k] = rs.randint(low=-16, high=16, size=(mask[k].shape), dtype='int8')
            if datatype == 'int16':
                temp_mask[k] = rs.randint(low=-3200, high=3200, size=(mask[k].shape), dtype='int16')
            if datatype == 'float32':
                temp_mask[k] = rs.random(size=(mask[k].shape))

        # add or subtract mask depending on device_id
        if key < device_id:
            mask = add_models(mask, temp_mask)
        elif key > device_id:
            mask = subtract_models(mask, temp_mask)
    return mask


def generate_private_mask(b_u, weights, datatype='float32'):
    int_bu = int.from_bytes(b_u, "big")
    rs = RandomState(PCG64(SeedSequence(int_bu)))
    mask = OrderedDict()
    for key in weights.keys():
        mask[key] = rs.random(size=weights[key].shape)
        if datatype == 'int8':
            mask[key] = rs.randint(low=-16, high=16, size=(weights[key].shape), dtype='int8')
        if datatype == 'int16':
            mask[key] = rs.randint(low=-3200, high=3200, size=(weights[key].shape), dtype='int16')
        if datatype == 'float32':
            mask[key] = rs.random(size=(weights[key].shape))
    return mask

def generate_private_mask_from_shape(b_u, shape, datatype='float32'):
    int_bu = int.from_bytes(b_u, "big")
    rs = RandomState(PCG64(SeedSequence(int_bu)))
    mask = OrderedDict()
    for key in shape.keys():
        mask[key] = rs.random(size=shape[key])
        if datatype == 'int8':
            mask[key] = rs.randint(low=-16, high=16, size=(shape[key]), dtype='int8')
        if datatype == 'int16':
            mask[key] = rs.randint(low=-3200, high=3200, size=(shape[key]), dtype='int16')
        if datatype == 'float32':
            mask[key] = rs.random(size=(shape[key]))
    return mask


def generate_shared_seeds(public_keys, s_sk):
    seeds = dict()
    for i, key in enumerate(public_keys):
        shared_key = get_shared_key(public_keys[key]["s_pk"], s_sk)
        int_seed = int.from_bytes(shared_key, "big")
        seeds[key] = int_seed
    return seeds


def model_masking(device_id, weights, b_u, s_sk, public_keys, datatype='float32'):
    p_u = generate_private_mask(b_u, weights, datatype)
    local_mask = get_mask(device_id, weights, s_sk, public_keys, datatype)
    local_mask = add_models(local_mask, p_u)
    result = add_models(weights, local_mask)
    return result


def get_mask(device_id, weights, s_sk, public_keys, datatype='float32'):
    seeds = generate_shared_seeds(public_keys, s_sk)
    return generate_shared_mask(device_id, weights, seeds, datatype)
