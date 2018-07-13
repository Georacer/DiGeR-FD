#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graph_backend.py: Module providing graph-related functionality

All graphs supported by this module are bipartite graphs.
"""

import diger_fd.logging_utils.dgrlogging as dgrlog
import diger_fd.utils as utils

import networkx as nx
from enum import Enum
import collections


class NodeType(Enum):
    """Enumeration of the two bipartite graph sets, for use with the bipartite
    functionalitites of netkworkx"""
    EQUATION = 0
    VARIABLE = 1


class BadGraphError(utils.DgrException):
    """Graph is not available for use """

    def __init__(self, message):
        super().__init__(message)
        self.message = message


GraphMetadataEntry = collections.namedtuple('GraphMetadataEntry',
                                            'is_initialized parent')


class GraphBackend(dgrlog.LogMixin, metaclass=utils.SingletonMeta):
    """Class to handle all graphs and provide the graph API"""

    def __init__(self):
        super().__init__(__name__)  # Initialize logger
        self.graphs = dict()
        self.graphs_metadata = dict()

    def clear(self):
        """Reset the graph_backend. Deleting objects will not work, because
        the class is a Singleton"""
        self.graphs = dict()
        self._logger.debug('Cleared graph_backend')

    def allocate_graph(self, mdl_name=None):
        """Create a new graph"""
        if mdl_name is None:
            self._logger.warning('Tried to register graph with no model name')
            raise ValueError('All graphs must have a corresponding model name')

        if mdl_name in self.graphs.keys():
            self._logger.warning('Graph {} already registered'.format(mdl_name))
            raise KeyError('Graphs must have unique names')

        self.graphs[mdl_name] = GraphBipartite(mdl_name)
        self.graphs_metadata[mdl_name] = GraphMetadataEntry(
                is_initialized=False,
                parent=mdl_name)

    def add_equations(self, mdl_name, equ_ids):

    def add_variables(self, mdl_name, var_ids):

    def add_edges(self, mdl_name, edge_ids, edge_weights=None):

    def check_initialized(self, mdl_name=None):
        if not self.graphs_metadata[mdl_name].is_initialized:
            raise BadGraphError('Graph not initialized yet')


class GraphBipartite(nx.Digraph):
    """Bipartite graph class"""
    def __init__(self, mdl_name):
        super().__init__(name=mdl_name)
