#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-03-19 10:33:38
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-03-23 00:56:03
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/code/ITN/srmg/core/RiemannianLeft.py
Description: Modify here please
Init from https://github.com/yuanwei1989/plane-detection Author: Yuanwei Li (3 Oct 2018)
# Copyright (c) 2006-2017, Nina Milone, Bishesh Kanal, Benjamin Hou
# Copyright (c) 2006-2017, Imperial College of Science, Technology and Medicine 
# Produced at Biomedical Image Analysis Group
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

Statistics on Riemannian Manifolds and Groups
---------------------------------------------
This is a set of codes to compare the computing of the different types of means
on Lie groups. These codes can be used to reproduce the experiments illustrated in the
video developed for the MICCAI Educational challenge 2014, available at:
url of the video.
:Authors:
    `Nina Miolane <website>`
    `Bishesh Khanal <website>`
:Organization:
    Asclepios Team, INRIA Sophia Antipolis.
:Version: 
    2017.07.05
Requirements
------------
* `Numpy 1.11 <http://www.numpy.org>`_
Notes
-----
References
----------
(1) Defining a mean on Lie group.
    Nina Miolane. Medical Imaging. 2013. <hal-00938320>
'''
import numpy
import math

from srmg.common.group import *
from srmg.common.util import *

EPS = 1e-5


def riemExpL(a,f0,v):
    """
    start: TODO
    What the function does
    clearer function name ? 
    Inputs description:
    Outputs description:
    end:  TODO
    Riemannian exponential and logarithm from any point f0 (for left- and right-invariant metric)
    """
    f = grpCompose(f0, riemExpIdL(a, numpy.linalg.lstsq(jL(f0),v)[0]))
    return f


def riemExpIdL(a,v):
    """
    start: TODO
    What the function does
    clearer function name ? 
    Inputs description:
    Outputs description:
    end:  TODO
    Riemannian exponential and logarithm from Id (for left- and right-invariant metric)
    """
    v=grpReg(v);
    f = numpy.zeros(6)
    f[0:3] = v[0:3]
    f[3:6] = a * v[3:6];

    return f


def sigma2L(a,m,tabf,tabw):
    """
    start: TODO
    What the function does
    clearer function name ? 
    Inputs description:
    Outputs description:
    end:  TODO
    """
    siz = tabf.shape[0]

    if siz < 2:
        print('Error: Calculating variance requires at least 2 points')
        return 0

    s = 0

    for i in range(1,siz):
        s = s + tabw[i] * normA2L(a,m,riemLogL(a,m,tabf[i,:]));

    return s


def riemLogL(a,f0,f):
    """ 
    DESCRIPTION
    Attributes:
        a:             ?????
        f0:            ????
        f:             ????
    Return:
        v:             ?????
    """
    v=numpy.dot(jL(f0),riemLogIdL(a,grpCompose(grpInv(f0), f)))
    return v


def riemLogIdL(a,f):
    """ 
    DESCRIPTION
    Attributes:
        a:             ?????
        f:             ????
    Return:
        v:             ?????
    """
    v = numpy.zeros(6)
    if (a != 0):
        v[0:3] = f[0:3]
        v[3:6] = (1.0/numpy.float(a))*f[3:6];
    else:
        print('ERROR: alpha = 0')

    return v



def qL(a,f):
    """ 
    Left- and right- invariant inner product in the principal chart (propagation of Frobenius inner product)
    Attributes:
        a:             ?????
        f:             ????
    Return:
        g:             ?????
    """
    f = grpReg(f)
    g0 = numpy.zeros([6,6])
    g0[0:3,0:3] = numpy.eye(3)
    g0[3:6,3:6] = a * numpy.eye(3)
    g = numpy.dot(numpy.dot(numpy.linalg.inv(jL(f).T) , g0) , numpy.linalg.inv(jL(f)))

    return g


def jL(f):
    """ 
    Differentials of the left and right translations for SO(3) in the principal chart
    Attributes:
        r:             ?????
    Return:
        Jl:            ?????
    """
    #f = makeColVector(f,6); # unnecessary if 1D
    f = grpReg(f);
    Jl = numpy.zeros([6,6])
    Jl[0:3,0:3]=jRotL(f[0:3]);
    Jl[3:6,3:6]=rotMat(f[0:3]);

    return Jl


def normA2L(a,f,v): 
    """ 
    This function calculates the normalised left 
    Attributes:
        a:             ?????
        f:             ?????
        v:             ?????
    Return:
        n:             normalised vector
    """
    v=grpReg(v);
    n=numpy.dot(numpy.dot(v.T,qL(a,f)),v); 

    return n


def frechetL(a,tabf,tabw):
    """ 
    This function computes the frechet-L mean
    Attributes:
        img:            The fixed image that will be transformed (simpleitk type)
        a:              ?????
        tabf:           SE3 data points (Nx6 vector)
        tabw:           data point weights (Nx1 vector)
    Return:
        m:              The mean
    """
    siz = tabf.shape[0]

    if siz < 2:
        print('Error: Calculating mean requires at least 2 points')
    
    m = tabf[0,:]

    # Iteration 0
    mbis=m;
    # print 'mbisL=' + str(mbis)
    aux=numpy.zeros(6);
    for i in range (0,siz):
        aux=aux+tabw[i]*riemLogL(a,mbis,tabf[i,:]);
    m=riemExpL(a,mbis,aux);

    # Iteration 1 until converges
    while  (normA2L(a,mbis,riemLogL(a,mbis,m))>EPS*sigma2L(a,mbis,tabf,tabw)):
        mbis=m;
        # print 'mbisL=' + str(mbis)
        aux=numpy.zeros(6);
        for i in range (0,siz):
            aux=aux+tabw[i]*riemLogL(a,mbis,tabf[i,:]);
        m=riemExpL(a,mbis,aux);

    return m
