# Author: Henry Heitmann

import pickle
from collections import OrderedDict

import numpy as np
import torch
from Crypto.Cipher import AES
from Crypto.Hash import SHA256
from Crypto.Protocol.SecretSharing import Shamir
from Crypto.Random import get_random_bytes
from diffiehellman import DiffieHellman


def generate_key_pair():
    dh = DiffieHellman(key_bits=64)
    private_key = get_random_bytes(16)
    dh.set_private_key(private_key)
    pk, sk = dh.get_public_key(), dh.get_private_key()
    if len(sk) != 16:
        return generate_key_pair()
    return pk, sk


def get_shared_key(pk, sk):
    dh = DiffieHellman(key_bits=64)
    dh.set_private_key(sk)
    return dh.generate_shared_key(pk)


def generate_shares(key, t, n):
    return Shamir.split(t, n, key, ssss=False)


def reconstruct_secret(shares):
    return Shamir.combine(shares, ssss=False)


def encrypt_message(key, message):
    byte_message = pickle.dumps(message)

    h = SHA256.new()
    h.update(key)
    hashed_key = h.digest()

    cipher = AES.new(hashed_key, AES.MODE_EAX, nonce=key)
    return cipher.encrypt(byte_message)


def decrypt_message(key, ciphertext):
    h = SHA256.new()
    h.update(key)
    hashed_key = h.digest()

    cipher = AES.new(hashed_key, AES.MODE_EAX, nonce=key)
    return pickle.loads(cipher.decrypt(ciphertext))


def transform_tensor_to_array(model_params):
    model_array = OrderedDict()
    for k in model_params.keys():
        tmp = np.array(model_params[k])
        model_array[k] = tmp
    return model_array


def transform_array_to_tensor(model_array):
    model_tensor = OrderedDict()
    for k in model_array.keys():
        tmp = np.array(model_array[k])
        tmp = torch.from_numpy(tmp)
        model_tensor[k] = tmp
    return model_tensor


def convert_float32_to_int8(weights):
    quantized = OrderedDict()
    for key in weights.keys():
        values = weights[key] * 100
        quantized[key] = np.array(values, dtype='int8')
    return quantized


def convert_int8_to_float32(weights):
    unquantized = OrderedDict()
    for key in weights.keys():
        unquantized[key] = np.array(weights[key], dtype='float32') / 100
    return unquantized


def convert_float32_to_int16(weights):
    quantized = OrderedDict()
    for key in weights.keys():
        values = weights[key] * 10000
        quantized[key] = np.array(values, dtype='int16')
    return quantized


def convert_int16_to_float32(weights):
    unquantized = OrderedDict()
    for key in weights.keys():
        unquantized[key] = np.array(weights[key], dtype='float32') / 10000
    return unquantized


def get_model_shape(model):
    shape = OrderedDict()
    for key in model.keys():
        shape[key] = model[key].shape
    return shape
