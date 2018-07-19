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
def g1():
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
    num_edges = 4  # 3 bidirectional and 1 single-directional
    answer = namedtuple('g1', 'mdl_name equ_ids var_ids edges num_edges')
    return answer._make([mdl_name, equ_ids, var_ids, edges, num_edges])


def add_g1(gb):
    """Add g1 graph to the passed graph_backend"""
    new_graph = g1()
    gb.allocate_graph(new_graph.mdl_name)
    gb[new_graph.mdl_name].add_equations(new_graph.equ_ids)
    gb[new_graph.mdl_name].add_variables(new_graph.var_ids)
    gb[new_graph.mdl_name].add_edges(new_graph.edges)


@pytest.mark.usefixtures('init_graph_backend')
class TestBasic():
    """Tests to cover basic functionality"""

    def test_singleton(self):
        add_g1(self.gb)
        gb2 = graph_backend.GraphBackend()
        assert(self.gb == gb2)
        assert('g1' in gb2)

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
        add_g1(self.gb)
        assert(self.gb[g1.mdl_name].num_eqs == len(g1.equ_ids))
        assert(set(self.gb[g1.mdl_name].equations) == set(g1.equ_ids))
        assert(self.gb[g1.mdl_name].num_vars == len(g1.var_ids))
        assert(set(self.gb[g1.mdl_name].variables) == set(g1.var_ids))
        assert(self.gb[g1.mdl_name].num_edges == g1.num_edges)
        assert(set(self.gb[g1.mdl_name].edges)
               == set(ids for _, _, ids in g1.edges))

    def test_clear(self):
        self.gb.allocate_graph('g1')
        self.gb.allocate_graph('g2')
        self.gb.clear()
        assert(len(self.gb.graphs.items()) == 0)
        assert(len(self.gb.graphs_metadata.items()) == 0)

    def test_get_edge_pairs(self):
        add_g1(self.gb)
        pairs_7 = self.gb['g1'].get_edge_pairs(7)
        assert((6, 2) in pairs_7)
        assert((2, 6) in pairs_7)
        pairs_8 = self.gb['g1'].get_edge_pairs(8)
        assert(((6, 4),) == pairs_8)

    def test_get_edge_ids(self):
        add_g1(self.gb)
        ids = self.gb['g1'].get_edge_ids(((6, 4),))
        assert(ids[0] == 8)

    def test_get_edges(self):
        add_g1(self.gb)
        edge_ids = self.gb['g1'].get_edges(6)[0]
        assert(sorted(edge_ids) == [7, 8])

@pytest.mark.usefixtures('init_graph_backend')
class TestBipartite():
    """Test bipartite properties of the graph, as implemented by networkx"""

    def test_enum(self):
        add_g1(self.gb)
        # Test equations
        equ_ids = self.gb['g1'].equations
        nodeset = self.gb['g1'].get_node_property(equ_ids, 'bipartite')
        for node_type in nodeset:
            assert(node_type == 0)
        # Test variables
        var_ids = self.gb['g1'].variables
        nodeset = self.gb['g1'].get_node_property(var_ids, 'bipartite')
        for node_type in nodeset:
            assert(node_type == 1)


@pytest.mark.usefixtures('init_graph_backend')
class TestErrors():
    """Test raising of errors"""

    def test_missing_graph(self):
        with pytest.raises(KeyError):
            self.gb['unobtainium']

    def test_assignment(self):
        with pytest.raises(graph_backend.GraphInterfaceError):
            self.gb['unsettable'] = {'dict': 'contents'}

    def test_uninitialized_graph(self):
        add_g1(self.gb)
        with pytest.raises(graph_backend.BadGraphError):
            self.gb.check_initialized('g1')
