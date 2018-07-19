#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_structural_engine.py: Tests for the structural_engine.py module.

Run with pytest.
"""

import pytest

import diger_fd.structural_engine.structural_engine as se


@pytest.fixture(scope='function')
def use_database_adapter():
    dba = se.DatabaseAdapter()
    yield dba
    dba.db_close()

# =============================================================================
# DatabaseAdapter tests
# =============================================================================

class TestDbOpenClose():
    """Test whether the database can be opened, initialized and closed"""

    def test_db_open(self):
        self.dba = se.DatabaseAdapter()

    def test_get_cursor(self, use_database_adapter):
        dba = use_database_adapter
        cursor = dba.get_cursor()

    def test_db_initialized(self, use_database_adapter):
        dba = use_database_adapter
        # Test that all tables have been created
        cursor = dba.get_cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        reply = cursor.fetchall()
        tables_expected = [('equations',), ('variables',), ('edges',)]
        for table in tables_expected:
            if table not in reply:
                assert False

        # Test that all tables have the correct columns
        for table in tables_expected:
            table_type = table[0]
            print('Checking table {}'.format(table_type))
            property_constructor = {'equations': se.EquationProperties,
                                    'variables': se.VariableProperties,
                                    'edges': se.EdgeProperties}
            property_obj = property_constructor[table_type]

            cursor = dba.get_cursor()
            cursor.execute("SELECT * FROM {}".format(table_type))
            column_names = se.get_cursor_columns(cursor)
            print(column_names)

            expected_names = property_obj.valid_properties
            assert set(column_names) == set(expected_names)

    def test_db_close(self):
        self.test_db_open()
        self.dba.db_close


@pytest.mark.usefixtures('use_database_adapter')
class TestDbBasicOperations():
    """Test add, get, edit, delete operations"""

    def test_query_add_invalid(self, use_database_adapter):
        dba = use_database_adapter
        query = se.ModelQuery(db_operation='add', model='mdl1')
        with pytest.raises(se.StructuralInterfaceError):
            dba.query_parse(query)

        query = se.GraphElementQuery(db_operation='add', model='mdl1',
                                     element_id=1)
        with pytest.raises(se.StructuralInterfaceError):
            dba.query_parse(query)

        query = se.VertexQuery(db_operation='add', model='mdl1', element_id=1)
        with pytest.raises(se.StructuralInterfaceError):
            dba.query_parse(query)

    def test_query_add_equation(self, use_database_adapter):
        dba = use_database_adapter
        query = se.EquationQuery(db_operation='add', model='mdl1',
                                 element_id=1)
        dba.query_parse(query)