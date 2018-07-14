#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_graph_backend.py: Tests for the graph_backend.py module.

Run with pytest.
"""

import pytest
from collections import namedtuple

import diger_fd.graph_backend.graph_backend as graph_backend


@pytest.fixture(scope='function')
def init_graph_backend(request):
    print('Setting up new graph backend')
    request.cls.gb = graph_backend.GraphBackend()
    yield
    request.cls.gb.clear()


@pytest.fixture()
def g1(request):
    print('Creating data for g1')
    mdl_name = 'g1'
    equ_ids = (1, 6)
    var_ids = (2, 4)
    edges = ((1, 2, 3),
             (2, 1, 3),
             (1, 4, 5),
             (4, 1, 5),
             (6, 2, 7),
             (2, 6, 7),
             (6, 4, 8),  # This is a directed edge
             )
    answer = namedtuple('g1', 'mdl_name equ_ids var_ids edges')
    return answer._make([mdl_name, equ_ids, var_ids, edges])


@pytest.mark.usefixtures('init_graph_backend')
class TestBasic():
    """Tests to cover basic functionality"""

    def test_allocate_graphs(self):
        self.gb.allocate_graph('g1')
        assert('g1' in self.gb)
        self.gb.allocate_graph('g2')
        assert('g2' in self.gb)

    def test_mdl_name_conflict(self):
        self.gb.allocate_graph('g1')
        with pytest.raises(KeyError):
            self.gb.allocate_graph('g1')

    def test_create_single_graph(self, g1):
        self.gb.allocate_graph(g1.mdl_name)
        self.gb[g1.mdl_name].add_equations(g1.equ_ids)
        self.gb[g1.mdl_name].add_variables(g1.var_ids)
        self.gb[g1.mdl_name].add_edges(g1.edges)
