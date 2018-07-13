#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_utils.py: Tests for the utils.py module.
"""

import unittest as ut

""" IdProvider tests """
class IdProviderTestCase(ut.TestCase):
    def setUp(self):
        import diger_fd.utils as utils
        self.idp1 = utils.IdProvider()
        self.idp2 = utils.IdProvider()

    def test_first_use(self):
        self.idp1.new_model('m1')
        id = self.idp1.req_id('m1')
        self.assertEqual(id, 1)

    def test_increments(self):
        self.idp1.new_model('m1')
        id = self.idp1.req_id('m1')
        id = self.idp1.req_id('m1')
        self.assertEqual(id, 2)
        id = self.idp1.req_id('m1')
        self.assertEqual(id, 3)

    def test_second_model(self):
        self.idp1.new_model('m1')
        self.idp1.new_model('m2')
        id = self.idp1.req_id('m2')
        self.assertEqual(id, 1)

    def test_singleton(self):
        self.assertEqual(self.idp1, self.idp2)
        self.idp1.new_model('m1')
        self.idp1.new_model('m2')
        self.idp1.req_id('m2')
        self.assertEqual(self.idp2._models, {'m1': 0, 'm2': 1})

    def tearDown(self):
        self.idp1.clear()
