"""Built-in regularizers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from keras import backend as K
from keras.utils.generic_utils import serialize_keras_object
from keras.utils.generic_utils import deserialize_keras_object


class Regularizer(object):
    """Regularizer base class.
    """

    def __call__(self, x):
        return 0.

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class L1L2(Regularizer):
    """Regularizer for L1 and L2 regularization.
    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0., l2=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)

    def __call__(self, x):
        regularization = 0.
        if self.l1:
            regularization += K.sum(self.l1 * K.abs(x))
        if self.l2:
            regularization += K.sum(self.l2 * K.square(x))
        return regularization

    def get_config(self):
        return {'l1': float(self.l1),
                'l2': float(self.l2)}

class LP(Regularizer):
    """Regularizer for L1 and L2 regularization.
    # Arguments
        lp: Float; LP regularization factor.
        p : Float; LP regularization power.
    """

    def __init__(self, lp=0.01, p=1.):
        self.lp = K.cast_to_floatx(lp)
        self.p = K.cast_to_floatx(p)

    def __call__(self, x):
        regularization = 0.
        regularization += K.pow(K.sum(self.lp * K.pow(x,self.p)),1/self.p)
        return regularization

    def get_config(self):
        return {'lp': float(self.lp),
                'p': float(self.p)}


# Aliases.


def l1(l=0.01):
    return L1L2(l1=l)


def l2(l=0.01):
    return L1L2(l2=l)


def l1_l2(l1=0.01, l2=0.01):
    return L1L2(l1=l1, l2=l2)

def lp(lp=0.01,p=0.1):
    return LP(lp=0.01,p=0.1)

def serialize(regularizer):
    return serialize_keras_object(regularizer)


def deserialize(config, custom_objects=None):
    return deserialize_keras_object(config,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='regularizer')


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, six.string_types):
        config = {'class_name': str(identifier), 'config': {}}
        return deserialize(config)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret regularizer identifier: ' +
                         str(identifier))
