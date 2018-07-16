#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py

General utilities and functionalities
"""

import diger_fd.logging_utils.dgrlogging as dgrlog


class Borg():
    """Borg class, used for subclassing.

    Objects of subclasses will be shared across each subclass AS WELL AS across
    all subclasses.

    Attributes
    ----------
    __shared_state : dict
        The shared state of the borg
    """

    __shared_state = {}

    def __init__(self):
        self.__dict__ = self.__shared_state


class SingletonMeta(type):
    """Singleton metaclass, used for creating singletons

    Objects of this type of class will be singletons, but not across their
    subclasses.

    Attributes
    ----------
    _instances :
        Dictionary for bookkeeping of what has been instantiated
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

    def _drop(self):
        """Drop the instances (for testing purposes)"""
        self._instances = {}


class IdProvider(dgrlog.LogMixin, metaclass=SingletonMeta):
    """Provides ids for model elements

    Attributes
    ----------
    _models : dict
        Container for increasing model ids
    """

    def __init__(self):
        self._models = {}
        super().__init__(__name__)

    def new_model(self, mdl_name):
        if mdl_name in self._models.keys():
            raise KeyError("Model name {} already registered in IdProvider".
                           format(mdl_name))
        'Initialize model ID'
        self._models[mdl_name] = 0
        self._logger.debug('Registered new model:{}'.format(mdl_name))

    def req_id(self, mdl_name):
        if mdl_name not in self._models.keys():
            raise KeyError("Model name {} not registered in IdProvider".
                           format(mdl_name))
        self._models[mdl_name] += 1
        self._logger.debug('Allocated new ID:{} for model:{}'.
                           format(self._models[mdl_name], mdl_name))
        return self._models[mdl_name]

    def clear(self):
        """Reset the IdProvider. Deleting objects will not work, because
        the class is a Singleton"""
        self._models = {}
        self._logger.debug('Cleared ID registry')


def to_list(arg):
    """If arg is not iterable, wrap it into a list"""
    if not hasattr(arg, '__iter__'):
        return [arg]
    else:
        return arg


# =============================================================================
# Errors and Warnings
# =============================================================================
class DgrException(Exception):
    """Base class for exceptions in DiGeR-FD"""


class DgrError(DgrException):
    """Base error for the whole package"""


class DgrWarning(Warning):
    """Base warning for the whole package"""
