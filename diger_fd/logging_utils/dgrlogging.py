#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
logging.py: Custom logging utilities
"""

import os
import logging.config


config_path = os.path.join(os.path.dirname(__file__), 'logging_config.ini')
logging.config.fileConfig(config_path)


class LogMixin():
    def __init__(self, prefix):
        self._logger = logging.getLogger('.'.join([prefix, self.__class__.__name__]))
