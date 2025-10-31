#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Apr  4 15:32:33 2019

@author: Wojciech chacholski, 2019

Copyright Wojciech chacholski, 2019
This software is to be used only for activities related  with TDA group at KTH 
and TDA course SF2956 at KTH
"""
import numpy as np
inf = float("inf")



import stablerank.srank as sr

import scipy.stats as st

import itertools as it


def circle(c, r, s, error=0):
    t = np.random.uniform(high=2 * np.pi, size=s)
    y = np.sin(t) * r + c[1]
    x = np.cos(t) * r + c[0]
    sd = error * 0.635
    pdf = st.norm(loc=[0, 0], scale=(sd, sd))
    return pdf.rvs((s, 2)) + np.vstack([x, y]).transpose()


def disc(c, r, s, error=0):
    u = np.random.normal(0, 1, size=s)
    v = np.random.normal(0, 1, size=s)
    norm = (u * u + v * v)**0.5
    rad = r*np.random.rand(s)**0.5
    x = (rad * u / norm) + c[0]
    y = (rad * v / norm) + c[1]
    return np.vstack([x, y]).transpose()


def lp_circle(c, r, p, s, error=0):
    t = np.random.uniform(high=2 * np.pi, size=s)
    y = np.sin(t) * r * 1 / (((np.absolute(np.sin(t))) ** p + (np.absolute(np.cos(t))) ** p) ** (1 / p)) \
        + c[1]
    x = np.cos(t) * r * 1 / (((np.absolute(np.sin(t))) ** p + (np.absolute(np.cos(t))) ** p) ** (1 / p)) \
        + c[0]
    sd = error * 0.635
    pdf = st.norm(loc=[0, 0], scale=(sd, sd))
    return pdf.rvs((s, 2)) + np.vstack([x, y]).transpose()


def closed_path(vertices, s, error=0):
    v = np.asarray(vertices)
    number_v = len(v)
    l1 = np.linalg.norm(v[1:, :] - v[:-1, :], axis=1)
    _l = np.concatenate([l1, np.array([np.linalg.norm(v[0] - v[-1])])])
    accum_l = np.asarray(list(it.accumulate(_l)))
    t = np.random.uniform(high=accum_l[-1], size=s)
    points = np.empty([0, 2])
    for i in t:
        index = np.searchsorted(accum_l, i)
        coeff = (accum_l[index] - i) / (_l[index])
        if index == number_v - 1:
            points = np.vstack((points, (coeff * v[0] + (1 - coeff) * v[-1])))
        else:
            points = np.vstack((points, (coeff * v[index + 1] + (1 - coeff) * v[index])))
    sd = error * 0.635
    pdf = st.norm(loc=[0, 0], scale=(sd, sd))
    return pdf.rvs((s, 2)) + points


def open_path(vertices, s, error=0):
    _v = np.asarray(vertices)
    _l = np.linalg.norm(_v[1:, :] - _v[:-1, :], axis=1)
    accum_l = np.asarray(list(it.accumulate(_l)))
    t = np.random.uniform(high=accum_l[-1], size=s)
    points = np.empty([0, 2])
    for x in t:
        index = np.searchsorted(accum_l, x)
        coeff = (accum_l[index] - x) / (_l[index])
        points = np.vstack((points, (coeff * _v[index+1] + (1 - coeff) * _v[index])))
    sd = error * 0.635
    pdf = st.norm(loc=[0, 0], scale=(sd, sd))
    n = pdf.rvs((s, 2))
    return n + points


def normal_point(p, s, error=[[1, 0], [0, 1]]):
    return np.random.multivariate_normal(p, error, size=s)


def uniform_noise(x_min, x_max, y_min, y_max, s):
    x = (np.random.random(s) * (x_max - x_min)) + x_min
    y = (np.random.random(s) * (y_max - y_min)) + y_min
    return np.vstack((x, y)).transpose()
