# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 12:54:43 2016

@author: Chris
"""

def tagtest(root, num):
    test = []
    for kid in root.iterchildren():
        test.append(kid.tag)
        
    assert len(set(test))==num
    