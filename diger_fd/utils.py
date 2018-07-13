#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py

General utilities and functionalities
"""

import logging


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


class IdProvider(metaclass=SingletonMeta):
    """Provides ids for model elements

    Attributes
    ----------
    _models : dict
        Container for increasing model ids
    """

    def __init__(self):
        self._models = {}

    def new_model(self, mdl_name):
        if mdl_name in self._models.keys():
            raise KeyError("Model name {} already registered in IdProvider".
                           format(mdl_name))
        'Initialize model ID'
        self._models[mdl_name] = 0

    def req_id(self, mdl_name):
        if mdl_name not in self._models.keys():
            raise KeyError("Model name {} not registered in IdProvider".
                           format(mdl_name))
        self._models[mdl_name] += 1
        return self._models[mdl_name]

    def clear(self):
        """Reset the IdProvider. Deleting objects will not work, because
        the class is a Singleton"""
        self._models = {}


class DgrException(Exception):
    """Base exception for the whole package"""

    def __init__(self, message):
        super(DgrException, self).__init__(message)
        self.message = message
