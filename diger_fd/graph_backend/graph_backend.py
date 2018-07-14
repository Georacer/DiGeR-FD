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


# =============================================================================
# Enumerations
# =============================================================================
class NodeType(Enum):
    """Enumeration of the two bipartite graph sets, for use with the bipartite
    functionalitites of netkworkx"""
    EQUATION = 0
    VARIABLE = 1


# =============================================================================
# Errors and Warnings
# =============================================================================
class GraphInterfaceError(utils.DgrError):
    """Wrong use of interface error"""


class BadGraphError(utils.DgrError):
    """Graph is not available for use """

    def __init__(self, message):
        super().__init__(message)
        self.message = message


# =============================================================================
# Custom structures
# =============================================================================
class GraphMetadataEntry(collections.UserDict):
    """Metadata record for each graph entry"""
    def __init__(self, init_dict):
        if init_dict is not None:
            super().__init__(init_dict)
        else:
            super().__init__()
            self.data['is_initialized'] = False
            self.data['parent'] = None


# =============================================================================
# Main body
# =============================================================================
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
        self.graphs_metadata.clear()
        self._logger.debug('Cleared graph_backend')

    def __contains__(self, mdl_name):
        if mdl_name in self.graphs.keys():
            return True
        else:
            return False

    def __missing__(self, mdl_name):
        raise KeyError('Graph {} is not registered'.format(mdl_name))

    def __getitem__(self, mdl_name):
        """Query for a graph with key k"""
        return self.graphs[mdl_name]

    def __setitem__(self, mdl_name, value):
        raise GraphInterfaceError('Graphs cannot be set by assignment')

    def __delitem__(self, mdl_name):
        del self.graphs[mdl_name]
        del self.graphs_metadata[mdl_name]

    def allocate_graph(self, mdl_name=None):
        """Create a new graph"""
        if mdl_name is None:
            self._logger.warning('Tried to register graph with no model name')
            raise ValueError('All graphs must have a corresponding model name')

        if mdl_name in self.graphs.keys():
            self._logger.warning('Graph {} already registered'.format(mdl_name))
            raise KeyError('Graphs must have unique names')

        self.graphs[mdl_name] = BipartiteGraph(mdl_name)
        self.graphs_metadata[mdl_name] = GraphMetadataEntry({
                'is_initialized':False,
                'parent':mdl_name
                })

    def set_initialized(self, mdl_name):
        """Mark a graph as initalized and available for use"""
        self.graphs_metadata[mdl_name]['is_initialized'] = True


class BipartiteGraph(nx.DiGraph, dgrlog.LogMixin):
    """Bipartite graph class"""
    def __init__(self, mdl_name):
        super().__init__(name=mdl_name)

    # Add methods
    def add_equations(self, equ_ids):
        """Add equations to the graph"""
        self.add_nodes_from([equ_ids], biparite=NodeType.EQUATION)

    def add_variables(self, var_ids):
        """Add variables to the graph"""
        self.add_nodes_from([var_ids], biparite=NodeType.VARIABLE)

    def add_edges(self, edges_iterator):
        """Add edges to the graph
        edges_iterator : contains items (equ_id, var_id, edge_id, weight=None)
        """
        for edge in edges_iterator:
            if len(edge)==3:
                equ_id, var_id, edge_id = edge
                edge_weight = None
            elif len(edge)==4:
                equ_id, var_id, edge_id, edge_weight = edge
            else:
                raise GraphInterfaceError('Edges are specified by at least (equ_id, var_id, edge_id)')
            if edge_weight is None:
                edge_weight = 1
            self.add_edge(equ_id, var_id, weight=edge_weight, id=edge_id)

    # Delete methods
    def del_equations(self, equ_ids):
        """Delete equations from graph"""
        self.remove_nodes_from([equ_ids])

    def del_variables(self, var_ids):
        """Delete variables from graph"""
        self.remove_nodes_from([var_ids])

    def del_edges(self, edge_ids):
        edges = self._get_edge_pairs(edge_ids)
        self.remove_edges_from(edges)

    # Get methods
    def get_edges(self, node_ids):
        """
        Get edge_ids of input nodes.
        Returns a tuple of tuples for each node_id
        """
        answer = []
        for node_id in node_ids:
            edge_list = []
            for _, _, d in self.edges(nbunch=node_id, data=True):
                edge_list.append(d[id])
            answer.append(tuple(edge_list))
        return tuple(answer)

    def _get_edge_pairs(self, edge_ids):
        """Get the (n1, n2) pairs of edges for the requested edge_ids"""
        answer = []
        for n1, n2, edge_id in self.edges.data('id'):
            if edge_id in edge_ids:
                answer.append((n1, n2))
        return tuple(answer)

    def get_neighbours(self, node_ids):
        """Get the neighbours respecting directionality"""
        answer = []
        for node_id in node_ids:
            answer.append(tuple(self.neighbors(node_id)))
        return tuple(answer)

    def get_neighbours_undir(self, node_ids):
        """Get the neighbours regardless of directionality"""
        answer = []
        for node_id in node_ids:
            neighbors = (set(self.successors(node_id))
                        + set(self.predecessors(node_id)) )
            answer.append(tuple(neighbors))
        return tuple(answer)

    # Set methods
    def set_e2v(self, edge_ids):
        """Direct an edge pair from equations to variables"""
        edges = self._get_edge_pairs(edge_ids)
        for n1, n2 in edges:
            if n1 in self.variables:
                self.remove_edge(n1, n2)

    def set_v2e(self, edge_ids):
        """Direct an edge pair from variables to equations"""
        edges = self._get_edge_pairs(edge_ids)
        for n1, n2 in edges:
            if n1 in self.equations:
                self.remove_edge(n1, n2)

    def set_edge_weight(self, edge_ids, weights):
        edges = self._get_edge_pairs(edge_ids)
        for n1, n2, weight in zip(edges, weights):
            self.edges[n1, n2]['weight'] = weight

    @property
    def numEqs(self):
        """Return the number of equations in the graph"""
        return len()

    @property
    def equations(self):
        """Return the equation ids as a tuple"""
        return tuple(n for n, d in self.nodes(data=True)
                if d['bipartite']==NodeType.EQUATION)

    @property
    def variables(self):
        """Return the variable ids as a tuple"""
        return tuple(n for n, d in self.nodes(data=True)
                if d['bipartite']==NodeType.VARIABLE)

    def check_initialized(self):
        if not self.graphs_metadata[mdl_name]['is_initialized']:
            raise BadGraphError('Graph not initialized yet')