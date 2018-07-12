#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py

General utilities and functionalities
"""


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


class Singleton():
    """Singleton class, used for subclassing.

    Objects of subclasses will be shared across each subclass, but not across
    all subclasses.

    Attributes
    ----------
    _instance :
        The shared instance of the singleton
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance


class IdProvider(Singleton):
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


class DgrException(Exception):
    """Base exception for the whole package"""

    def __init__(self, message):
        super(DgrException, self).__init__(message)
        self.message = message
