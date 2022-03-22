#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-03-19 10:33:38
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-03-23 01:00:36
FilePath: /Awesome-Ultrasound-Standard-Plane-Detection/src/ITN/srmg/core/ExponentialBarycenter.py
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

from srmg.common.util import *

EPS = 1e-5


def sigma2(m,tabr,tabw):
    """
    start: TODO
    What the function does
    clearer function name ? 
    Inputs description:
    Outputs description:
    end:  TODO

    Computations of the variance in the 3 geometric frameworks: Group, L and R
    """

    siz = tabr.shape[0]

    if siz < 2:
        print('Error: Calculating sigma requires at least 2 points')

    s=0;

    for i in range(0,siz):
        s = s + tabw[i]*numpy.linalg.norm(logRotL(m,tabr[i,:]))**2;

    return s


def expRotL(r,a):
    """
    start: TODO
    What the function does
    clearer function name ? 
    Inputs description:
    Outputs description:
    end:  TODO
    Group exponential and logarithm from any point f (first for SO(3))
    #Group geodesics are left- and right- invariant
    """
    return rotVect(numpy.dot(rotMat(r),rotMat(numpy.dot(numpy.linalg.inv(jRotL(r)),a))))
    #R*Exp(DL(R^-1)*a)


def logRotL(r,rr):
    """
    start: TODO
    What the function does
    clearer function name ? 
    Inputs description:
    Outputs description:
    end:  TODO
    """
    return numpy.dot(jRotL(r),rotVect(numpy.dot(rotMat(-r),rotMat(rr))));


def rotMean(tabr,tabw):    
    """
    start: TODO
    What the function does
    clearer function name ? 
    Inputs description:
    Outputs description:
    end:  TODO
    Preliminaries: computation of the mean for rotations
    """

    siz = tabr.shape[0]

    if siz < 2:
        print('Error: Calculating mean requires at least 2 points')
    
    m = tabr[0,:]
    #mbis=mt et m=m(t+1)
    
    aux = numpy.zeros(6);

    mbis=m;
    aux = numpy.zeros(3);
    for i in range(0,siz):
        aux=aux+tabw[i]*logRotL(mbis,tabr[i,:])

    m = expRotL(mbis,aux);

    # BRACKETS!! in while
    
    while (numpy.dot(numpy.linalg.norm(logRotL(mbis,m)),numpy.linalg.norm(logRotL(mbis,m))) > EPS*sigma2(mbis,tabr,tabw)):
        mbis = m
        aux = numpy.zeros(3)
        for i in range(0,siz):
            aux=aux+tabw[i]*logRotL(mbis,tabr[i,:])

        m = expRotL(mbis,aux)

    return m


def matDeExp(v):
    """
    start: TODO
    What the function does
    clearer function name ? 
    Inputs description:
    Outputs description:
    end:  TODO
    Group exponential barycenter on SE(3):
     - the previous mean for the rotation part,
     - closed form from article of Xavier Pennec "Exponential Barycenters of...",(2012)
    """
    v = regRot(v);
    r = v[0:3];
    theta = numpy.linalg.norm(r);
    Sr = skew(r);
    if (theta==0): 
        M = numpy.eye(3)
    elif (theta<EPS): 
        M = numpy.eye(3) + (1.0/6.0-theta**3/120.0)*numpy.dot(Sr,Sr)+(1.0/2.0-theta**2/24.0)*Sr;
    else:
        M = numpy.eye(3) + theta**(-2)*(1.0-numpy.sin(theta)/theta)*numpy.dot(Sr,Sr)+theta**(-2)*(1.0-numpy.cos(theta))*Sr;

    return M



def expBar(tabf,tabw):
    """
    start: TODO
    What the function does
    clearer function name ? 
    Inputs description:
    Outputs description:
    end:  TODO
    """
    # tabf: SE3 data points, Nx6 array
    # tabw: data point weights, Nx1 array

    siz = tabf.shape[0]

    if siz < 2:
        print('Error: Calculating mean requires at least 2 points')
    
    tabr=tabf[:,0:3]
    tabt=tabf[:,3:6]

    rmean=rotMean(tabr,tabw);
    
    # Partie translation, p34 de expbar
    M = numpy.zeros([3,3])
    t = numpy.zeros(3)
    
    for i in range(0,siz):
        Maux = numpy.linalg.inv(matDeExp(rotVect(numpy.dot(rotMat(rmean),rotMat(-tabr[i,:])))));
        M=M+tabw[i]*Maux;
        t=t+tabw[i]*numpy.dot(numpy.dot(Maux,rotMat(-tabr[i,:])),tabt[i,:]);
    
    m = numpy.zeros(6)
    m[0:3]=rmean;
    m[3:6]=numpy.linalg.lstsq(M,t)[0]

    return m
