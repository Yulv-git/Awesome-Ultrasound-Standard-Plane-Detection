# -*- coding: utf-8 -*-
# RiemannianRight.py

# Copyright (c) 2006-2017, Nina Milone, Bishesh Kanal, Benjamin Hou
# Copyright (c) 2006-2017, Imperial College of Science, Technology and Medicine 
# Produced at Biomedical Image Analysis Group
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
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

"""
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
"""


import numpy
import math

from srmg.common.group import *
from srmg.common.util import *

EPS = 1e-5


def riemExpR(a,f0,v):
    """
    start: TODO
    What the function does
    clearer function name ? 
    Inputs description:
    Outputs description:
    end:  TODO
    
    Riemannian exponential and logarithm from any point f0 (for left- and right-invariant metric)
    """

    f = grpCompose((riemExpIdR(a, numpy.linalg.lstsq(jR(f0),v)[0])), f0)
    return f


def riemExpIdR(a,v):
    """
    start: TODO
    What the function does
    clearer function name ? 
    Inputs description:
    Outputs description:
    end:  TODO
    
    Riemannian exponential and logarithm from Id (for left- and right-invariant metric)
    """

    v=grpReg(-v);
    f = numpy.zeros(6)
    f[0:3] = v[0:3]
    f[3:6] = a * v[3:6]
    f = grpInv(f)
    return f


def sigma2R(a,m,tabf,tabw):
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
        print 'Error: Calculating variance requires at least 2 points'
        return 0

    s = 0

    for i in range(0,siz):
        s = s + tabw[i] * normA2R(a,m,riemLogR(a,m,tabf[i,:]));

    return s


def riemLogR(a,f0,f):
    """ 
    DESCRIPTION
    Attributes:
        a:             ?????
        f0:            ????
        f:             ????
    Return:
        v:             ?????
    """

    v=numpy.dot(jR(f0),riemLogIdR(a,grpCompose(f,grpInv(f0))))
    return v


def riemLogIdR(a,f):
    """ 
    DESCRIPTION
    Attributes:
        a:             ?????
        f:             ????
    Return:
        v:             ?????
    """

    v = numpy.zeros(6)
    v[0:3] = f[0:3]
    v[3:6] = numpy.dot(rotMat(-f[0:3]),f[3:6]);

    return v



def qR(a,f):
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
    g = numpy.dot(numpy.dot(numpy.linalg.inv(jR(f).T) , g0) , numpy.linalg.inv(jR(f)))

    return g


def jR(f):
    """ 
    Differentials of the left and right translations for SO(3) in the principal chart
    Attributes:
        r:             ?????
    Return:
        Jl:            ?????
    """

    #f = makeColVector(f,6); # unnecessary if 1D
    f = grpReg(f);
    Jr = numpy.zeros([6,6])
    Jr[0:3,0:3] = jRotR(f[0:3]);
    Jr[3:6,0:3] = -skew(f[3:6]);
    Jr[3:6,3:6] = numpy.eye(3);

    return Jr


def normA2R(a,f,v): 
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
    n=numpy.dot(numpy.dot(v.T,qR(a,f)),v); 

    return n




def frechetR(a,tabf,tabw):
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
        print 'Error: Calculating mean requires at least 2 points'
    
    m = tabf[0,:]

    # Iteration 0
    mbis=m;
    print 'mbisR=' + str(mbis)
    aux=numpy.zeros(6);
    for i in range (0,siz):
        aux=aux+tabw[i]*riemLogR(a,mbis,tabf[i,:]);
    m=riemExpR(a,mbis,aux);

    # Iteration 1 until converges
    while  (normA2R(a,mbis,riemLogR(a,mbis,m))>EPS*sigma2R(a,mbis,tabf,tabw)):
        mbis=m;
        print 'mbisR=' + str(mbis)
        aux=numpy.zeros(6);
        for i in range (0,siz):
            aux=aux+tabw[i]*riemLogR(a,mbis,tabf[i,:]);
        m=riemExpR(a,mbis,aux);

    return m
