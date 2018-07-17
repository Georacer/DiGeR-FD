#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
structural_engine.py: Provides Structural Analysis functionality
"""

import diger_fd.logging_utils.dgrlogging as dgrlog
import diger_fd.utils as utils


# =============================================================================
# Enumerations
# =============================================================================


# =============================================================================
# Errors and Warnings
# =============================================================================
class StructuralInterfaceError(utils.DgrError):
    """Wrong use of interface error"""


# =============================================================================
# Custom structures
# =============================================================================

class ModelProperties():
    mdl = 'model'

    def __init__(self, **kwargs):
        self.valid_properties = utils.filter_dunder(dir(self.__class__))
        self.active_keywords = []
        for key, value in kwargs.items():
            if key in self.valid_properties:
                self.__setattr__(key, value)
                self.active_keywords.append(key)

MdlProps = ModelProperties

class GraphElementProperties(ModelProperties):
    e_id = 'element_id'
    is_matched = 'is_matched'

    def __init__(self, **kwargs):
        self.valid_properties = utils.filter_dunder(dir(self.__class__))
        self.active_keywords = []
        for key, value in kwargs.items():
            if key in self.valid_properties:
                self.__setattr__(key, value)
                self.active_keywords.append(key)

GrElProps = GraphElementProperties

class EdgeProperties(GraphElementProperties):
    is_derivative = 'is_derivative'
    is_integral = 'is_integral'
    is_non_solvable = 'is_non_solvable'

    def __init__(self, **kwargs):
        self._valid_properties = utils.filter_dunder(dir(self.__class__))
        self.active_keywords = []
        for key, value in kwargs.items():
            if key in self.valid_properties:
                self.__setattr__(key, value)
                self.active_keywords.append(key)

EdgeProps = EdgeProperties

class VertexProperties(GraphElementProperties):
    alias = 'alias'
    descr = 'description'
    matched_to = 'matched_to'

    def __init__(self, **kwargs):
        self.valid_properties = utils.filter_dunder(dir(self.__class__))
        self.active_keywords = []
        for key, value in kwargs.items():
            if key in self.valid_properties:
                self.__setattr__(key, value)
                self.active_keywords.append(key)

VertProps = VertexProperties

class EquationProperties(VertexProperties):
    is_static = 'is_static'
    is_dynamic = 'is_dynamic'
    is_non_linear = 'is_non_linear'
    is_residual_generator = 'is_residual_generator'
    is_faultable = 'is_faultable'
    expression_structural = 'expression_structural'
    expression_analytic = 'expression_analytic'
    subsystem = 'subsystem'

    def __init__(self, **kwargs):
        self.valid_properties = utils.filter_dunder(dir(self.__class__))
        self.active_keywords = []
        for key, value in kwargs.items():
            if key in self.valid_properties:
                self.__setattr__(key, value)
                self.active_keywords.append(key)

EquProps = EquationProperties

class VariableProperties(VertexProperties):
    is_known = 'is_known'
    is_measured = 'is_measured'
    is_input = 'is_input'
    is_output = 'is_output'
    is_residual = 'is_residual'
    is_vector = 'is_vector'
    is_fault = 'is_fault'
    is_parameter = 'is_parameter'

    def __init__(self, **kwargs):
        self.valid_properties = utils.filter_dunder(dir(self.__class__))
        self.active_keywords = []
        for key, value in kwargs.items():
            if key in self.valid_properties:
                self.__setattr__(key, value)
                self.active_keywords.append(key)

VarProps = VariableProperties

# =============================================================================
# Main body
# =============================================================================

class SimpleModelQuery():
    def __init__(self, mdl):
        self.mdl = mdl
        self.operation = 'get'

    def __call__(self):
        db = StructuralEngine().db
        return db.parse_simple_query(self)

    def __getitem__(self, key):
        return SimpleElementQuery(self.mdl, key)

class SimpleElementQuery(SimpleModelQuery):
    def __init__(self, mdl, e_id):
        super().__init__(mdl)
        self.e_id = e_id

    def __call__(self):
        db = StructuralEngine().db
        return db.parse_simple_query(self)

    def __getitem__(self, key):
        return SimplePropertyQuery(mdl, e_id, prop):


class SimplePropertyQuery(SimpleElementQuery):
    def __init__(self, mdl, e_id, prop):
        super().__init__(mdl, e_id)
        self.prop = prop

    def __call__(self):
        db = StructuralEngine().db
        return db.parse_simple_query(self)


class DBQuery():
    """A query to the structural engine database"""

    def __init__(self, db_operation=None, modifier=None):
        valid_operations = ('get', 'set')
        valid_modifiers = (None, 'or', 'and')

        if db_operation is None:
            db_operation = 'get'
        elif db_operation not in valid_operations:
            raise StructuralInterfaceError()
        self.db_operation = db_operation

        if modifier not in valid_modifiers:
            raise StructuralInterfaceError()
        self.modifier = modifier


class ModelQuery(ModelProperties, DBQuery):
    """Model accessor object to help db calls"""
    def __init__(self, *, db_operation=None, modifier=None, **kwargs):
        DBQuery.__init__(self, db_operation, modifier)
        ModelProperties.__init__(self, **kwargs)


class GraphElementQuery(GraphElementProperties, DBQuery):
    """Model accessor object to help db calls"""
    def __init__(self, *, db_operation=None, modifier=None, **kwargs):
        DBQuery.__init__(self, db_operation, modifier)
        GraphElementProperties.__init__(self, **kwargs)


class VertexQuery(DBQuery, VertexProperties):
    """Model accessor object to help db calls"""
    def __init__(self, *, db_operation=None, modifier=None, **kwargs):
        DBQuery.__init__(self, db_operation, modifier)
        VertexProperties.__init__(self, **kwargs)


class EquationQuery(DBQuery, EquationProperties):
    """Model accessor object to help db calls"""
    def __init__(self, *, db_operation=None, modifier=None, **kwargs):
        DBQuery.__init__(self, db_operation, modifier)
        EquationProperties.__init__(self, **kwargs)


class VariableQuery(DBQuery, VariableProperties):
    """Model accessor object to help db calls"""
    def __init__(self, *, db_operation=None, modifier=None, **kwargs):
        DBQuery.__init__(self, db_operation, modifier)
        VariableProperties.__init__(self, **kwargs)


class EdgeQuery(DBQuery, EdgeProperties):
    """Model accessor object to help db calls"""
    def __init__(self, *, db_operation=None, modifier=None, **kwargs):
        DBQuery.__init__(self, db_operation, modifier)
        EdgeProperties.__init__(self, **kwargs)



class StructuralEngine((dgrlog.LogMixin, metaclass=utils.SingletonMeta):
    """Class to provide Structural Analysis API"""

    def __init__(self):
        super().__init__(__name__)  # Initialize logger
        self.db = DatabaseAdapter()

    def clear(self):
        """Reset the database"""

    def __contains__(self, mdl):

    def __missing__(self, mdl):

    def __getitem__(self, mdl):
        """Query for a model"""
        return SimpleModelQuery(db = self.db, mdl=self.mdl)

    def __setitem__(self, mdl, value):
        raise StructuralInterfaceError('Models cannot be set by assignment')

    def __delitem__(self, mdl):
        """Order the deletion of a model"""
        self.db.delete(ModelQuery(mdl_name))

    def allocate_model(self, mdl=None):
        """Create a new model"""
        # Adapt this from GB to SE
        pass

    def set_initialized(self, mdl_name):
        """Mark a graph as initalized and available for use"""
        self.graphs_metadata[mdl_name]['is_initialized'] = True


    def check_initialized(self, mdl_name):
        # Adapt this from GB to SE
        if not self.graphs_metadata[mdl_name]['is_initialized']:
            raise BadGraphError('Graph not initialized yet')

    def access():
        raise NotImplementedError

    def get_elements():
        raise NotImplementedError

    def set_elements():
        raise NotImplementedError

    def get_equations_with():
        raise NotImplementedError

    def isMatchable():
        raise NotImplementedError



class DatabaseAdapter():
    """Adapter for the database holding the structural characteristics of each
    model"""

    def __init__(self, mdl_name):
        self.db =  # What shall we use?

    def query_get(self, q):
        raise NotImplementedError

    def query_set(self, q):
        raise NotImplementedError

    # Add methods
    def add_equations(self, equ_ids):
        """Add equations to the graph"""
        # Adapt this from GB to SE
        self.add_nodes_from(equ_ids, bipartite=NodeType.EQUATION)

    def add_variables(self, var_ids):
        """Add variables to the graph"""
        # Adapt this from GB to SE
        self.add_nodes_from(var_ids, bipartite=NodeType.VARIABLE)

    def add_edges(self, edges_iterator):
        """Add edges to the graph
        edges_iterator : contains items (equ_id, var_id, edge_id, weight=None)
        """
        # Adapt this from GB to SE
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
        # Adapt this from GB to SE
        self.remove_nodes_from([equ_ids])

    def del_variables(self, var_ids):
        """Delete variables from graph"""
        # Adapt this from GB to SE
        self.remove_nodes_from([var_ids])

    def del_edges(self, edge_ids):
        # Adapt this from GB to SE
        edges = self._get_edge_pairs(edge_ids)
        self.remove_edges_from(edges)

    # Get methods
    def get_edges(self, node_ids):
        """
        Get edge_ids of input nodes.
        Returns a tuple of tuples for each node_id
        """
        # Adapt this from GB to SE
        answer = []
        for node_id in node_ids:
            edge_list = []
            for _, _, d in self.nx_edges(nbunch=node_id, data=True):
                edge_list.append(d[id])
            answer.append(tuple(edge_list))
        return tuple(answer)

    def _get_edge_pairs(self, edge_ids):
        """Get the (n1, n2) pairs of edges for the requested edge_ids"""
        # Adapt this from GB to SE
        answer = []
        for n1, n2, edge_id in self.nx_edges.data('id'):
            if edge_id in edge_ids:
                answer.append((n1, n2))
        return tuple(answer)

    def get_neighbours(self, node_ids):
        """Get the neighbours respecting directionality"""
        # Adapt this from GB to SE
        answer = []
        for node_id in node_ids:
            answer.append(tuple(self.neighbors(node_id)))
        return tuple(answer)

    def get_neighbours_undir(self, node_ids):
        """Get the neighbours regardless of directionality"""
        # Adapt this from GB to SE
        answer = []
        for node_id in node_ids:
            neighbors = (set(self.successors(node_id))
                        + set(self.predecessors(node_id)) )
            answer.append(tuple(neighbors))
        return tuple(answer)

    # Set methods
    def set_e2v(self, edge_ids):
        """Direct an edge pair from equations to variables"""
        # Adapt this from GB to SE
        edges = self._get_edge_pairs(edge_ids)
        for n1, n2 in edges:
            if n1 in self.variables:
                self.remove_edge(n1, n2)

    def set_v2e(self, edge_ids):
        """Direct an edge pair from variables to equations"""
        # Adapt this from GB to SE
        edges = self._get_edge_pairs(edge_ids)
        for n1, n2 in edges:
            if n1 in self.equations:
                self.remove_edge(n1, n2)

    def set_edge_weight(self, edge_ids, weights):
        edges = self._get_edge_pairs(edge_ids)
        for n1, n2, weight in zip(edges, weights):
            self.edges[n1, n2]['weight'] = weight

    @property
    def num_eqs(self):
        """Return the number of equations in the graph"""
        # Adapt this from GB to SE
        return len(self.equations)

    @property
    def num_vars(self):
        """Return the number of variables in the graph"""
        # Adapt this from GB to SE
        return len(self.variables)

    @property
    def num_edges(self):
        """Return the number of edges in the graph
        Edges with the same ID (two directions) are not counted twice
        """
        # Adapt this from GB to SE
        return len(set(self.edges))

    @property
    def equations(self):
        """Return the equation ids as a tuple"""
        # Adapt this from GB to SE
        return tuple(n for n, d in self.nodes(data=True)
                     if d['bipartite'] == NodeType.EQUATION)

    @property
    def variables(self):
        """Return the variable ids as a tuple"""
        # Adapt this from GB to SE
        return tuple(n for n, d in self.nodes(data=True)
                     if d['bipartite'] == NodeType.VARIABLE)

    @property
    def edges(self):
        """Return the edge ids as a tuple"""
        # Adapt this from GB to SE
        return tuple(edge_id for _, _, edge_id in self.nx_edges.data('id'))
