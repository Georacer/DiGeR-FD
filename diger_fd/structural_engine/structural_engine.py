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

    def get_active_properties(self):
        """Get a dict with the active properties
        except model and element_id"""
        answer = self.to_dict()
        for key in ['model', 'element_id']:
            del answer[key]

        return answer


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
        valid_operations = ('get', 'add', 'edit', 'del')
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
            cursor.execute("ROLLBACK")
            raise e
        finally:
            cursor.close()

        return retval
    return new_func


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
        elif query.db_operation is 'add':
            self.query_add(query)
        elif query.db_operation is 'edit':
            self.query_del(query)
        elif query.db_operation is 'del':
            self.query_del(query)

    @dbwrap
    def query_add(self, cursor, query):
        query_str, values = self.build_add_string(query)
        cursor.execute(query_str, values)

    @dbwrap
    def query_get(self, cursor, query):
        query_str, values = self.build_get_string(query)
        cursor.execute(query_str, values)
        return cursor.fetchall()

    @dbwrap
    def query_edit(self, cursor, query):
        query_str, values = self.build_edit_string(query)
        cursor.execute(query_str, values)

    @dbwrap
    def query_del(self, cursor, query):
        query_str, values = self.build_del_string(query)
        cursor.execute(query_str, values)

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

        equations_string = ('CREATE TABLE equations('
                            + ', '.join(column_def for column_def in columns.values())
                            + ', CONSTRAINT CON_MDL_ID PRIMARY KEY (model, element_id)'
                            ')')
        return equations_string

    def build_variables_table_string(self):
        columns = {'element_id': 'element_id INTEGER NOT NULL',
                   'model': 'model TEXT NOT NULL',
                   'is_matched': 'is_matched INTEGER',
                   'alias': 'alias TEXT KEY',
                   'descr': 'description TEXT',
                   'is_known': 'is_known INTEGER',
                   'is_measured': 'is_measured INTEGER',
                   'is_input': 'is_input INTEGER',
                   'is_output': 'is_output INTEGER',
                   'is_residual': 'is_residual INTEGER',
                   'is_vector': 'is_vector INTEGER',
                   'is_fault': 'is_fault INTEGER',
                   'is_parameter': 'is_parameter INTEGER'
                   }
        # Check if all column names are valid
        for col_name in columns.keys():
            if col_name not in VariableProperties().valid_properties:
                raise KeyError('Invalid column specified in Equations table')

        variables_string = ('CREATE TABLE variables('
                            + ', '.join(column_def for column_def in columns.values())
                            + ', CONSTRAINT CON_MDL_ID PRIMARY KEY (model, element_id)'
                            + ')')
        return variables_string

    def build_edges_table_string(self):
        columns = {'element_id': 'element_id INTEGER NOT NULL',
                   'model': 'model TEXT NOT NULL',
                   'is_matched': 'is_matched INTEGER',
                   'is_derivative': 'is_derivative INTEGER',
                   'is_integral': 'is_integral INTEGER',
                   'is_non_solvable': 'is_non_solvable INTEGER'
                   }
        # Check if all column names are valid
        for col_name in columns.keys():
            if col_name not in EdgeProperties().valid_properties:
                raise KeyError('Invalid column specified in Equations table')

        edges_string = ('CREATE TABLE edges('
                        + ', '.join(column_def for column_def in columns.values())
                        + ', CONSTRAINT CON_MDL_ID PRIMARY KEY (model, element_id)'
                        + ')')
        return edges_string

    def get_table_from_query(self, query):
        if isinstance(query, EdgeQuery):
            table_name = 'edges'
        elif isinstance(query, EquationQuery):
            table_name = 'equations'
        elif isinstance(query, VariableQuery):
            table_name = 'variables'

        return table_name

    def get_valid_properties_from_query(self, query):
        if isinstance(query, EdgeQuery):
            all_properties = EdgeProperties.valid_properties
        elif isinstance(query, EquationQuery):
            all_properties = EquationProperties.valid_properties
        elif isinstance(query, VariableQuery):
            all_properties = VariableProperties.valid_properties

        return all_properties

    def build_edges_add_string(self, query):
        return self.build_set_string(query)

    def build_equations_add_string(self, query):
        return self.build_set_string(query)

    def build_variables_add_string(self, query):
        return self.build_set_string(query)

    def build_edges_get_string(self, query):
        return self.build_get_string(query)

    def build_equations_get_string(self, query):
        return self.build_get_string(query)

    def build_variables_get_string(self, query):
        return self.build_get_string(query)

    def build_add_string(self, query):
        table_name = self.get_table_from_query(query)
        all_properties = query.to_dict()

        keys = all_properties.keys()
        values = all_properties.values()

        query_string = ('INSERT INTO {}('.format(table_name)
                        + ', '.join(keys)
                        + ') VALUES('
                        + ', '.join(len(values)*['?'])
                        + ')')
        return query_string, values

    def build_get_string(self, query):
        table_name = self.get_table_from_query(query)
        all_properties = query.get_active_properties()

        keys = all_properties.keys()
        values = all_properties.values()

        if 'model' not in keys:
            raise StructuralInterfaceError('Please specify model')

        if 'element_id' in keys:  # User queries for the whole element row
            query_string = ('SELECT ALL FROM {} '.format(table_name)
                            + 'WHERE (model = ?, element_id = ?')
            values = (query.model, query.element_id)
        else:
            query_string = ('SELECT '
                            + ' '.join(keys)
                            + ' FROM {}'.format(table_name))
            values = tuple()
        return query_string, values

    def build_edit_string(self, query):
        active_properties = query.get_active_properties()
        keys = active_properties.keys()
        values = active_properties.values()

        if not all([key in ['model', 'element_id'] for key in keys]):
            raise StructuralInterfaceError('Please specify model and element_id')

        table_name = self.get_table_from_query(query)

        query_string = ('UPDATE {} SET '.format(table_name)
                        + ' = ? '.join(keys)
                        + 'WHERE ( model = ?, element_id = ?)')
        values = tuple(values + [query.model, query.element_id])
        return query_string, values

    def build_del_string(self, query):
        table_name = self.get_table_from_query(query)

        query_string = ('DELETE FROM {} WHERE '.format(table_name)
                        + '(model = ?, element_id = ?)')
        values = (query.model, query.element_id)

        return query_string, values
