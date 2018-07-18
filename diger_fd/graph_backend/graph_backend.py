#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graph_backend.py: Module providing graph-related functionality

All graphs supported by this module are bipartite graphs.
"""

import diger_fd.logging_utils.dgrlogging as dgrlog
import diger_fd.utils as utils

import networkx as nx
from enum import IntEnum
import collections


# =============================================================================
# Enumerations
# =============================================================================
class NodeType(IntEnum):
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
                'is_initialized': False,
                'parent': mdl_name
                })

    def set_initialized(self, mdl_name):
        """Mark a graph as initalized and available for use"""
        self.graphs_metadata[mdl_name]['is_initialized'] = True

    def check_initialized(self, mdl_name):
        if not self.graphs_metadata[mdl_name]['is_initialized']:
            raise BadGraphError('Graph not initialized yet')


class BipartiteGraph(nx.DiGraph, dgrlog.LogMixin):
    """Bipartite graph class"""
    def __init__(self, mdl_name):
        super().__init__(name=mdl_name)
        # Remap networkx edges. .edges is meant to be a public API
        self.nx_edges = super().edges
        self.matchings = []  # Container for matchings corresponding to this graph

    # Add methods
    def add_equations(self, equ_ids):
        """Add equations to the graph"""
        self.add_nodes_from(equ_ids, bipartite=NodeType.EQUATION)

    def add_variables(self, var_ids):
        """Add variables to the graph"""
        self.add_nodes_from(var_ids, bipartite=NodeType.VARIABLE)

    def add_edges(self, edges_iterator):
        """Add edges to the graph
        edges_iterator : contains items (equ_id, var_id, edge_id, weight=None)
        """
        for edge in edges_iterator:
            if len(edge) == 3:
                equ_id, var_id, edge_id = edge
                edge_weight = None
            elif len(edge) == 4:
                equ_id, var_id, edge_id, edge_weight = edge
            else:
                raise GraphInterfaceError('Edges are specified by at least (equ_id, var_id, edge_id)')
            if edge_weight is None:
                edge_weight = 1
            self.add_edge(equ_id, var_id, weight=edge_weight, id=edge_id)

    # Delete methods
    def del_equations(self, equ_ids):
        """Delete equations from graph"""
        equ_ids = utils.to_list(equ_ids)
        self.remove_nodes_from(equ_ids)

    def del_variables(self, var_ids):
        """Delete variables from graph"""
        var_ids = utils.to_list(var_ids)
        self.remove_nodes_from([var_ids])

    def del_edges(self, edge_ids):
        edge_ids = utils.to_list(edge_ids)
        edges = self.get_edge_pairs(edge_ids)
        self.remove_edges_from(edges)

    # Get methods
    def _get_edges(self, node_ids):
        """
        Get edge_ids of input nodes.
        Returns a tuple of tuples for each node_id
        """
        node_ids = utils.to_list(node_ids)
        answer = []
        for node_id in node_ids:
            edge_list = []
            for _, _, d in self.nx_edges(nbunch=node_id, data=True):
                edge_list.append(d[id])
            answer.append(tuple(edge_list))
        return tuple(answer)

    def get_edge_pairs(self, edge_ids):
        """Get the (n1, n2) pairs of edges for the requested edge_ids
        Retuns a tuple of tuples with max 2 tuples"""
        edge_ids = utils.to_list(edge_ids)
        answer = []
        for n1, n2, edge_id in self.nx_edges.data('id'):
            if edge_id in edge_ids:
                answer.append((n1, n2))
        return tuple(answer)

    def get_equations(self, ids):
        """Get equations related to a variable or edge
        Returns a tuple of tuples"""
        ids = utils.to_list(ids)
        answer = []
        for e_id in ids:
            if e_id in self.variables:
                answer.append(tuple(self.get_neighbours_undir(e_id)))
            elif e_id in self.edges:
                u, v = self.get_edge_pairs(e_id)[0]  # Get the first of the max 2 pairs
                if u in self.equations:
                    answer.append((u,))
                else:
                    answer.append((v,))
            elif e_id in self.equations:
                self._logger.warning('Requested get_equations from an equation')
                answer.append((e_id,))

    def get_variables(self, ids):
        """Get variables related to an equation or edge"""
        ids = utils.to_list(ids)
        answer = []
        for e_id in ids:
            if e_id in self.equations:
                answer.append(tuple(self.get_neighbours_undir(e_id)))
            elif e_id in self.edges:
                u, v = self.get_edge_pairs(e_id)[0]  # Get the first of the max 2 pairs
                if u in self.variables:
                    answer.append((u,))
                else:
                    answer.append((v,))
            elif e_id in self.variables:
                self._logger.warning('Requested get_variables from a variable')
                answer.append((e_id,))

    def get_edges(self, ids):
        """Get the edges related to provided elements"""
        ids = utils.to_list(ids)
        answer = []
        for e_id in ids:
            print('Examining id {}'.format(e_id))
            if (e_id in self.equations) or (e_id in self.variables):
                local_edges = []
                neighbours = self.get_neighbours_undir(e_id)[0]
                print('Its neighbours are {}'.format(neighbours))
                for n in neighbours:
                    if self.has_edge(n, e_id):
                        local_edges.append(self.get_edge_ids(((n, e_id),))[0])
                    elif self.has_edge(e_id, n):
                        local_edges.append(self.get_edge_ids(((e_id, n),))[0])
                    else:
                        raise KeyError('No edge found between {} and {}'.format(n, e_id))
                answer.append(tuple(local_edges))
            elif e_id in self.edges:
                self._logger.warning('Requested get_edges from an edge')
                answer.append((e_id,))
        return tuple(answer)

    def get_edge_list(self):
        raise NotImplementedError
        # TODO: Implement this

    def get_neighbours(self, node_ids):
        """Get the neighbours respecting directionality"""
        node_ids = utils.to_list(node_ids)
        answer = []
        for node_id in node_ids:
            answer.append(tuple(self.neighbors(node_id)))
        return tuple(answer)

    def get_neighbours_undir(self, node_ids):
        """Get the neighbours regardless of directionality"""
        node_ids = utils.to_list(node_ids)
        answer = []
        for node_id in node_ids:
            neighbors = (set(self.successors(node_id))
                         | set(self.predecessors(node_id)))
            answer.append(tuple(neighbors))
        return tuple(answer)

    def get_ancestor_equations(self):
        raise NotImplementedError()

    def get_node_property(self, node_ids=None, key_str=None):
        """Return a tuple of the item value for the requested nodes"""
        if node_ids is None:
            node_ids = self.nodes
        node_ids = utils.to_list(node_ids)

        if key_str is None:
            raise ValueError('Property string must be provided')

        return tuple(d[key_str] for n, d in self.nodes(data=True)
                     if n in node_ids)

    def get_edge_property(self, edge_ids=None, key_str=None):
        """Return a tuple of the item value for the requested nodes"""
        if edge_ids is None:
            edge_ids = self.edges
        edge_ids = utils.to_list(edge_ids)

        if key_str is None:
            raise ValueError('Property string must be provided')

        return tuple(d[key_str] for _, _, d in self.nx_edges.data('key_str'))

    def get_edge_ids(self, edge_tuples):
        """Return a tuple of the ids of the provided edge pairs"""
        # TODO: Check if input is tuple of tuples
        answer = []
        for pair in edge_tuples:
            answer.append(self.nx_edges[pair[0], pair[1]]['id'])
        return tuple(answer)

    def get_edge_ids_undir(self, edge_tuples):
        """Return a tuple of the ids of the provided edge pairs
        Disregard directionality"""
        answer = []
        for pair in edge_tuples:
            try:
                answer.append(self.nx_edges[pair[0], pair[1]]['id'])
            except KeyError:
                try:
                    answer.append(self.nx_edges[pair[1], pair[0]]['id'])
                except KeyError:
                    raise KeyError('Node pair {} does not exist in graph {}'.
                                   format(pair, self['mdl_name']))
        return tuple(answer)

    # Set methods
    def set_e2v(self, edge_ids):
        """Direct an edge pair from equations to variables"""
        edge_ids = utils.to_list(edge_ids)
        edges = self.get_edge_pairs(edge_ids)
        for n1, n2 in edges:
            if n1 in self.variables:
                self.remove_edge(n1, n2)

    def set_v2e(self, edge_ids):
        """Direct an edge pair from variables to equations"""
        edge_ids = utils.to_list(edge_ids)
        edges = self.get_edge_pairs(edge_ids)
        for n1, n2 in edges:
            if n1 in self.equations:
                self.remove_edge(n1, n2)

    def set_edge_weight(self, edge_ids, weights):
        edge_ids = utils.to_list(edge_ids)
        edges = self.get_edge_pairs(edge_ids)
        for n1, n2, weight in zip(edges, weights):
            self.edges[n1, n2]['weight'] = weight


class Matching():
    """Implementation of a matching edge set"""

    def __init__(self, graph):
        self.graph = graph
        self.edges = set()  # Edges are represented by their ID
        self.covered_vertices = set()
        self.cost = None

    def add_edge(self, e_id):
        if self.validate_edge(e_id):
            self.edges.add(e_id)
            u, v = self.graph.get_edge_pairs(e_id)
            self.covered_vertices.add(u)
            self.covered_vertices.add(v)
        else:
            raise GraphInterfaceError('Tried to add invalide edge to matching')

    def validate_edge(self, e_id):
        u, v = self.graph.get_edge_pairs(e_id)
        if (u in self.covered_vertices) or (v in self.covered_vertices):
            return False
        else:
            return True

# =============================================================================
#     Other methods
# =============================================================================

    def has_cycles(self):
        raise NotImplementedError

# =============================================================================
#     Custom properties
# =============================================================================

    @property
    def num_eqs(self):
        """Return the number of equations in the graph"""
        return len(self.equations)

    @property
    def num_vars(self):
        """Return the number of variables in the graph"""
        return len(self.variables)

    @property
    def num_edges(self):
        """Return the number of edges in the graph
        Edges with the same ID (two directions) are not counted twice
        """
        return len(set(self.edges))

    # TODO: Add caching functionality
    @property
    def equations(self):
        """Return the equation ids as a tuple"""
        return tuple(n for n, d in self.nodes(data=True)
                     if d['bipartite'] == NodeType.EQUATION)

    @property
    def variables(self):
        """Return the variable ids as a tuple"""
        return tuple(n for n, d in self.nodes(data=True)
                     if d['bipartite'] == NodeType.VARIABLE)

    @property
    def edges(self):
        """Return the edge ids as a tuple"""
        return tuple(edge_id for _, _, edge_id in self.nx_edges.data('id'))

