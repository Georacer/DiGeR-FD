#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
structural_engine.py: Provides Structural Analysis functionality
"""

import diger_fd.logging_utils.dgrlogging as dgrlog
import diger_fd.utils as utils

from enum import Enum
import sqlite3 as sql
import collections


# =============================================================================
# Enumerations
# =============================================================================
class StructuralType(Enum):
    MODEL = 1
    ELEMENT = 2
    EDGE = 3
    VERTEX = 4
    EQUATION = 5
    VARIABLE = 6


# =============================================================================
# Errors and Warnings
# =============================================================================
class StructuralInterfaceError(utils.DgrError):
    """Wrong use of interface error"""


# =============================================================================
# Custom structures
# =============================================================================
ModelIdPair = collections.namedtuple('ModelIdPair', 'model id')


# =============================================================================
#     Structural Types Properties
# =============================================================================
class ModelProperties():
    valid_properties = ['model']

    def __init__(self, **kwargs):
        self.active_keywords = []
        for key, value in kwargs.items():
            if key in self.valid_properties:
                self.__setattr__(key, value)
                self.active_keywords.append(key)
            else:
                raise StructuralInterfaceError('{} is not a valid property for {}'
                                               .format(key, self.__class__.__name__))

    def to_dict(self):
        answer = dict()
        answer.fromkeys(self.active_keywords)
        for key in answer.keys():
            answer[key] = self.__getattribute__(key)
        return answer

    def __iter__(self):
        return self.iterkeys()

    def iterkeys(self):
        return self.to_dict().iterkeys()

    def itervalues(self):
        return self.to_dict().itervalues()

    def iteritems(self):
        return self.to_dict().iteritems()

    def __getitem__(self, key):
        if key not in self.valid_properties:
                raise StructuralInterfaceError('{} is not a valid property for {}'
                                               .format(key, self.__class__.__name__))
        if key in self.iterkeys():
            return self.to_dict()[key]
        else:
            raise KeyError('{} property is not set'.format(key))



MdlProps = ModelProperties


class GraphElementProperties(ModelProperties):
    valid_properties = ModelProperties.valid_properties\
                        + ['element_id', 'is_matched']

    def __init__(self, **kwargs):
        self.active_keywords = []
        for key, value in kwargs.items():
            if key in self.valid_properties:
                self.__setattr__(key, value)
                self.active_keywords.append(key)


GrElProps = GraphElementProperties


class EdgeProperties(GraphElementProperties):
    valid_properties = GraphElementProperties.valid_properties\
                        + ['is_derivative', 'is_integral', 'is_non_solvable']

    def __init__(self, **kwargs):
        self.active_keywords = []
        for key, value in kwargs.items():
            if key in self.valid_properties:
                self.__setattr__(key, value)
                self.active_keywords.append(key)


EdgeProps = EdgeProperties


class VertexProperties(GraphElementProperties):
    valid_properties = GraphElementProperties.valid_properties\
                        + ['alias', 'description']

    def __init__(self, **kwargs):
        self.active_keywords = []
        for key, value in kwargs.items():
            if key in self.valid_properties:
                self.__setattr__(key, value)
                self.active_keywords.append(key)


VertProps = VertexProperties


class EquationProperties(VertexProperties):
    valid_properties = VertexProperties.valid_properties\
                        + ['is_static',
                           'is_dynamic',
                           'is_non_linear',
                           'is_residual_generator',
                           'is_faultable',
                           'expression_structural',
                           'expression_analytic',
                           'subsystem']

    def __init__(self, **kwargs):
        self.active_keywords = []
        for key, value in kwargs.items():
            if key in self.valid_properties:
                self.__setattr__(key, value)
                self.active_keywords.append(key)


EquProps = EquationProperties


class VariableProperties(VertexProperties):
    valid_properties = VertexProperties.valid_properties\
                        + ['is_known',
                           'is_measured',
                           'is_input',
                           'is_output',
                           'is_residual',
                           'is_vector',
                           'is_fault',
                           'is_parameter']

    def __init__(self, **kwargs):
        self.active_keywords = []
        for key, value in kwargs.items():
            if key in self.valid_properties:
                self.__setattr__(key, value)
                self.active_keywords.append(key)


VarProps = VariableProperties


# =============================================================================
# Main body
# =============================================================================

# =============================================================================
#     Simple Queries
# =============================================================================
class SimpleModelQuery():
    def __init__(self, mdl):
        self.model = mdl
        self.operation = 'get'

    def __call__(self):
        query = parse_simple_query(self)
        StructuralEngine.call_query(query)

    def __getitem__(self, key):
        if isinstance(key, int):
            return SimpleElementQuery(self.model, e_id=key)
        elif isinstance(key, str):
            return SimpleElementQuery(self.model, alias=key)
        else:
            raise StructuralInterfaceError('Key can be either integer id or string alias')


class SimpleElementQuery(SimpleModelQuery):
    def __init__(self, mdl, e_id=None, alias=None):
        super().__init__(mdl)
        self.e_id = e_id
        self.alias = alias

    def __call__(self):
        query = parse_simple_query(self)
        StructuralEngine.call_query(query)

    def __getitem__(self, key):
        return SimplePropertyQuery(self.mdl, self.e_id, self.alias, self.prop)


class SimplePropertyQuery(SimpleElementQuery):
    def __init__(self, mdl, e_id, alias, prop):
        super().__init__(mdl, e_id, alias)
        self.prop = prop

    def __call__(self):
        query = parse_simple_query(self)
        StructuralEngine.call_query(query)

    def __setitem__(self, key, value):
        self.operation = 'get'
        self.value = value
        return parse_simple_query(self)


def parse_simple_query(query):
    """Convert a SimpleQuery to a proper Query"""

    if not isinstance(query, SimpleModelQuery):
        raise TypeError("Only SimpleQueries are accepted")

    operation = query.operation

    db = StructuralEngine.db
    e_id = query.e_id
    alias = query.alias

    # If id is not given, look it up based on alias
    if e_id is not None:
        s_type = db.get_type(e_id)
    else:
        try:
            e_id = db.lookup_alias(query.model, alias)
        except KeyError as e:
            raise e

    if isinstance(query, SimplePropertyQuery):
        if s_type is StructuralType.EDGE:
            full_query = EdgeQuery(db_operation=operation,
                                   modifier=None,
                                   mdl=query.model,
                                   e_id=e_id,
                                   **{query.prop: query.value})
        elif s_type is StructuralType.EQUATION:
            full_query = EquationQuery(db_operation=operation,
                                       modifier=None,
                                       mdl=query.model,
                                       e_id=e_id,
                                       **{query.prop: query.value})
        elif s_type is StructuralType.VARIABLE:
            full_query = VariableQuery(db_operation=operation,
                                       modifier=None,
                                       mdl=query.model,
                                       e_id=e_id,
                                       **{query.prop: query.value})

    elif isinstance(query, SimpleElementQuery):
        if operation is not 'get':
            raise StructuralInterfaceError('Simple queries can only read structural elements')

        full_query = GraphElementQuery(db_operation=operation,
                                       modifier=None,
                                       mdl=query.model,
                                       e_id=e_id)

    elif isinstance(query, ModelQuery):
        if operation is not 'get':
            raise StructuralInterfaceError('Simple queries can only read models')

        full_query = ModelQuery(db_operation=operation,
                                modifier=None,
                                mdl=query.model)

    return full_query


# =============================================================================
#     Full Queries
# =============================================================================
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


# =============================================================================
#     Core Structures
# =============================================================================
class StructuralEngine(dgrlog.LogMixin, metaclass=utils.SingletonMeta):
    """Class to provide Structural Analysis API"""

    def __init__(self):
        super().__init__(__name__)  # Initialize logger
        self.db = DatabaseAdapter()

    def parse_query(self, query):
        """Parse a *Query and call the corresponding action"""
        pass

    def add_edge(self, edge_properties):
        pass

    def add_equation(self, equation_properties):
        pass

    def add_variable(self, variable_properties):
        pass

    def edit_edge(self, edge_properties, mode=None):
        pass

    def edit_equation(self, equation_properties, mode=None):
        pass

    def edit_variable(self, variable_properties, mode=None):
        pass

    def get_edges(self, edge_properties):
        pass

    def get_equations(self, equation_properties):
        pass

    def get_variables(self, variable_properties):
        pass

    def del_edges(self, model_id_pairs):
        pass

    def del_equations(self, model_id_pairs):
        pass

    def del_variables(self, model_id_pairs):
        pass

    def is_edge(self, model_id_pair):
        pass

    def is_equation(self, model_id_pair):
        pass

    def is_variable(self, model_id_pair):
        pass

#    def clear(self):
#        """Reset the database"""
#
#    def __contains__(self, mdl):
#
#    def __missing__(self, mdl):
#
    def __getitem__(self, mdl):
        """Query for a model"""
        return SimpleModelQuery(db = self.db, mdl=self.mdl)
#
#    def __setitem__(self, mdl, value):
#        raise StructuralInterfaceError('Models cannot be set by assignment')
#
#    def __delitem__(self, mdl):
#        """Order the deletion of a model"""
#        self.db.delete(ModelQuery(mdl_name))
#
#    def allocate_model(self, mdl=None):
#        """Create a new model"""
#        # Adapt this from GB to SE
#        pass
#
#    def set_initialized(self, mdl_name):
#        """Mark a graph as initalized and available for use"""
#        self.graphs_metadata[mdl_name]['is_initialized'] = True
#
#
#    def check_initialized(self, mdl_name):
#        # Adapt this from GB to SE
#        if not self.graphs_metadata[mdl_name]['is_initialized']:
#            raise BadGraphError('Graph not initialized yet')


def dbwrap(func):
    """Database wrapper for use as a decorator"""
    def new_func(db_adapter, *args, **kwargs):
        cursor = db_adapter.get_cursor()
        try:
            cursor.execute("BEGIN")
            retval = func(db_adapter, cursor, *args, **kwargs)
            cursor.execute("COMMIT")
        except Exception as e:
            cursor.exectute("ROLLBACK")
            raise e
        finally:
            cursor.close()

        return retval


class DatabaseAdapter():
    """Adapter for the database holding the structural characteristics of each
    model"""

    def __init__(self):
        self.db_open()
        self.db_init()

    def db_open(self):
        self.db = sql.connect(':memory:')

    @dbwrap
    def db_init(self, cursor):
        """Initialize the structural elements databases"""
        cursor.execute(self.build_equations_table_string())
        cursor.execute(self.build_variables_table_string())
        cursor.execute(self.build_edges_table_string())

    def __del__(self):
        self.db_close()

    def db_close(self):
        self.db.close()

    def get_cursor(self):
        return self.db.cursor()

    def query_parse(self, query):
        if query.db_operation is 'get':
            self.query_get(query)
        elif query.db_operation is 'set':
            self.query_set(query)

    @dbwrap
    def query_set(self, cursor, query):
        cursor.execute(self.build_set_string(query), tuple(query.itervalues))

    @dbwrap
    def query_get(self, cursor, query):
        cursor.execute(self.build_get_string(query))
        return cursor.fetchall()

    def build_equations_table_string(self):
        columns = {'element_id': 'element_id INTEGER NOT NULL',
                   'model': 'model TEXT NOT NULL',
                   'is_matched': 'is_matched INTEGER',
                   'alias': 'alias TEXT KEY',
                   'descr': 'description TEXT',
                   'is_static': 'is_static INTEGER',
                   'is_dynamic': 'is_dynamic INTEGER',
                   'is_non_linear': 'is_non_linear INTEGER',
                   'is_residual_generator': 'is_residual_generator INTEGER',
                   'is_faultable': 'is_faultable INTEGER',
                   'expression_structural': 'expression_structural TEXT',
                   'expression_analytic': 'expression_analytic TEXT',
                   'subsystem': 'subsystem TEXT',
                   }
        # Check if all column names are valid
        for col_name in columns.keys():
            if col_name not in EquationProperties().valid_properties:
                raise KeyError('Invalid column specified in Equations table')

        equations_string = 'CREATE TABLE equations('
                            + ', '.join(column_def for column_def in columns.values())
                            + ', CONSTRAINT CON_MDL_ID PRIMARY KEY (model, element_id)'
                            + ')'
        return equations_string

    def build_variables_table_string(self):
        columns = {'element_id': 'element_id INTEGER NOT NULL',
                   'model': 'model TEXT NOT NULL',
                   'is_matched': 'is_matched INTEGER',
                   'alias': 'alias TEXT KEY',
                   'descr': 'description TEXT',
                   'is_known': 'is_known INTEGER'
                   'is_measured': 'is_measured INTEGER'
                   'is_input': 'is_input INTEGER'
                   'is_output': 'is_output INTEGER'
                   'is_residual': 'is_residual INTEGER'
                   'is_vector': 'is_vector INTEGER'
                   'is_fault': 'is_fault INTEGER'
                   'is_parameter': 'is_parameter INTEGER'
                   }
        # Check if all column names are valid
        for col_name in columns.keys():
            if col_name not in VariableProperties().valid_properties:
                raise KeyError('Invalid column specified in Equations table')

        variables_string = 'CREATE TABLE variables('
                            + ', '.join(column_def for column_def in columns.values())
                            + ', CONSTRAINT CON_MDL_ID PRIMARY KEY (model, element_id)'
                            + ')'
        return variables_string

    def build_edges_table_string(self):
        columns = {'element_id': 'element_id INTEGER NOT NULL',
                   'model': 'model TEXT NOT NULL',
                   'is_matched': 'is_matched INTEGER',
                   'is_derivative': 'is_derivative INTEGER'
                   'is_integral': 'is_integral INTEGER'
                   'is_non_solvable': 'is_non_solvable INTEGER'
                   }
        # Check if all column names are valid
        for col_name in columns.keys():
            if col_name not in EdgeProperties().valid_properties:
                raise KeyError('Invalid column specified in Equations table')

        edges_string = 'CREATE TABLE edges('
                            + ', '.join(column_def for column_def in columns.values())
                            + ', CONSTRAINT CON_MDL_ID PRIMARY KEY (model, element_id)'
                            + ')'
        return edges_string

    def build_edges_set_string(self, query):
        return self.build_set_string(query)

    def build_equations_set_string(self, query):
        return self.build_set_string(query)

    def build_variables_set_string(self, query):
        return self.build_set_string(query)

    def build_edges_get_string(self, query):
        return self.build_get_string(query)

    def build_equations_get_string(self, query):
        return self.build_get_string(query)

    def build_variables_get_string(self, query):
        return self.build_get_string(query)

    def build_set_string(self, query):
        if isinstance(query, EdgeQuery):
            table_name = 'edges'
        elif isinstance(query, EquationQuery):
            table_name = 'equations'
        elif isinstance(query, VariableQuery):
            table_name = 'variables'

        keys = query.to_dict().keys()
        values = query.to_dict().values()

        query_string = 'INSERT INTO {}('.format(table_name)
                        + ', '.join(keys)
                        + ') VALUES('
                        + ', '.join(len(values)*['?'])
                        + ')'
        return query_string

    def build_get_string(self, query):
        if isinstance(query, EdgeQuery):
            table_name = 'edges'
            all_properties = EdgeProperties.valid_properties
        elif isinstance(query, EquationQuery):
            table_name = 'equations'
            all_properties = EquationProperties.valid_properties
        elif isinstance(query, VariableQuery):
            table_name = 'variables'
            all_properties = VariableProperties.valid_properties

        keys = query.to_dict().keys()
        values = query.to_dict().values()

        if 'model' not in keys:
            raise StructuralInterfaceError('Please specify model')

        if 'element_id' in keys:  # User queries for the whole element row
            query_string = 'SELECT ALL FROM {} '.format(table_name)
                            + 'WHERE (model={}, element_id={}'.format(
                                    query['model'], query['element_id'])
        else:
            query_string = 'SELECT '
                            + ' '.join(keys)
                            + ' FROM {}'.format(table_name)
        return query_string


#    # Add methods
#    def add_equations(self, equ_ids):
#        """Add equations to the graph"""
#        # Adapt this from GB to SE
#        self.add_nodes_from(equ_ids, bipartite=NodeType.EQUATION)
#
#    def add_variables(self, var_ids):
#        """Add variables to the graph"""
#        # Adapt this from GB to SE
#        self.add_nodes_from(var_ids, bipartite=NodeType.VARIABLE)
#
#    def add_edges(self, edges_iterator):
#        """Add edges to the graph
#        edges_iterator : contains items (equ_id, var_id, edge_id, weight=None)
#        """
#        # Adapt this from GB to SE
#        for edge in edges_iterator:
#            if len(edge) == 3:
#                equ_id, var_id, edge_id = edge
#                edge_weight = None
#            elif len(edge) == 4:
#                equ_id, var_id, edge_id, edge_weight = edge
#            else:
#                raise GraphInterfaceError('Edges are specified by at least (equ_id, var_id, edge_id)')
#            if edge_weight is None:
#                edge_weight = 1
#            self.add_edge(equ_id, var_id, weight=edge_weight, id=edge_id)
#
#    # Delete methods
#    def del_equations(self, equ_ids):
#        """Delete equations from graph"""
#        # Adapt this from GB to SE
#        self.remove_nodes_from([equ_ids])
#
#    def del_variables(self, var_ids):
#        """Delete variables from graph"""
#        # Adapt this from GB to SE
#        self.remove_nodes_from([var_ids])
#
#    def del_edges(self, edge_ids):
#        # Adapt this from GB to SE
#        edges = self._get_edge_pairs(edge_ids)
#        self.remove_edges_from(edges)
#
#    # Get methods
#    def get_edges(self, node_ids):
#        """
#        Get edge_ids of input nodes.
#        Returns a tuple of tuples for each node_id
#        """
#        # Adapt this from GB to SE
#        answer = []
#        for node_id in node_ids:
#            edge_list = []
#            for _, _, d in self.nx_edges(nbunch=node_id, data=True):
#                edge_list.append(d[id])
#            answer.append(tuple(edge_list))
#        return tuple(answer)
#
#    def _get_edge_pairs(self, edge_ids):
#        """Get the (n1, n2) pairs of edges for the requested edge_ids"""
#        # Adapt this from GB to SE
#        answer = []
#        for n1, n2, edge_id in self.nx_edges.data('id'):
#            if edge_id in edge_ids:
#                answer.append((n1, n2))
#        return tuple(answer)
#
#    def get_neighbours(self, node_ids):
#        """Get the neighbours respecting directionality"""
#        # Adapt this from GB to SE
#        answer = []
#        for node_id in node_ids:
#            answer.append(tuple(self.neighbors(node_id)))
#        return tuple(answer)
#
#    def get_neighbours_undir(self, node_ids):
#        """Get the neighbours regardless of directionality"""
#        # Adapt this from GB to SE
#        answer = []
#        for node_id in node_ids:
#            neighbors = (set(self.successors(node_id))
#                        + set(self.predecessors(node_id)) )
#            answer.append(tuple(neighbors))
#        return tuple(answer)
#
#    # Set methods
#    def set_e2v(self, edge_ids):
#        """Direct an edge pair from equations to variables"""
#        # Adapt this from GB to SE
#        edges = self._get_edge_pairs(edge_ids)
#        for n1, n2 in edges:
#            if n1 in self.variables:
#                self.remove_edge(n1, n2)
#
#    def set_v2e(self, edge_ids):
#        """Direct an edge pair from variables to equations"""
#        # Adapt this from GB to SE
#        edges = self._get_edge_pairs(edge_ids)
#        for n1, n2 in edges:
#            if n1 in self.equations:
#                self.remove_edge(n1, n2)
#
#    def set_edge_weight(self, edge_ids, weights):
#        edges = self._get_edge_pairs(edge_ids)
#        for n1, n2, weight in zip(edges, weights):
#            self.edges[n1, n2]['weight'] = weight
#
#    @property
#    def num_eqs(self):
#        """Return the number of equations in the graph"""
#        # Adapt this from GB to SE
#        return len(self.equations)
#
#    @property
#    def num_vars(self):
#        """Return the number of variables in the graph"""
#        # Adapt this from GB to SE
#        return len(self.variables)
#
#    @property
#    def num_edges(self):
#        """Return the number of edges in the graph
#        Edges with the same ID (two directions) are not counted twice
#        """
#        # Adapt this from GB to SE
#        return len(set(self.edges))
#
#    @property
#    def equations(self):
#        """Return the equation ids as a tuple"""
#        # Adapt this from GB to SE
#        return tuple(n for n, d in self.nodes(data=True)
#                     if d['bipartite'] == NodeType.EQUATION)
#
#    @property
#    def variables(self):
#        """Return the variable ids as a tuple"""
#        # Adapt this from GB to SE
#        return tuple(n for n, d in self.nodes(data=True)
#                     if d['bipartite'] == NodeType.VARIABLE)
#
#    @property
#    def edges(self):
#        """Return the edge ids as a tuple"""
#        # Adapt this from GB to SE
#        return tuple(edge_id for _, _, edge_id in self.nx_edges.data('id'))
