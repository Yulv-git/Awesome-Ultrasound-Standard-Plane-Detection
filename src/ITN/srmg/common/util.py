# -*- coding: utf-8 -*-
# util.py

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

EPS = 1e-5


def rotVect(R):
    """
    Take a rotation matrix and convert it into a vector form.
    Input: 
    R: Rotation matrix of size 3x3
    Output: 
    r: vector with three elements. Norm of r is the rotation angle about an
    axis which is the vector r itself.
    """

    M = numpy.dot(R,R.T) - numpy.eye(3)
    if (numpy.trace(M)>1e-20): # do we have numeric precision less than 1e-12?
        R=getClosestRotMat(R)

    c = (numpy.trace(R)-1.0)/2.0

    if (c > 1):
        c=1

    if (c < -1):
        c=-1

    theta = numpy.arccos(c);
    
    if (theta<EPS):
        fact = 0.5 * (1.0 + theta**2 / 6.0)
        Sr = fact * ( R - R.T)
        r = numpy.array([Sr[2,1], Sr[0,2], Sr[1,0]]).T;
    elif abs(theta-math.pi)<EPS:
        print 'attention r'    # a remplir ?
    else:
        fact = 0.5 * theta / numpy.sin(theta)
        Sr = fact * (R-R.T)
        r=numpy.array([Sr[2,1], Sr[0,2], Sr[1,0]]).T;

    return r


def getClosestRotMat(M):
    """ 
    Computation of the closest rotation matrix R of a given matrix M 
    (avoids computational errors.)    
    Attributes:
        M:             rotation matrix
    Return:
        R:             rotation matrix
    """

    u , s , v = numpy.linalg.svd(M)
    R = numpy.dot(u,v)
    s = numpy.eye(3) * s

    if (numpy.linalg.det(R)<0):
        s[0,0] = 1
        s[1,1] = 1
        s[2,2] = -1
        R=numpy.dot(numpy.dot(u,s),v)

    return R


def rotMat(r):
    """ 
    Converts rotation vector r to rotation matrix
    Attributes:
        r:             rotation vector
    Return:
        R:             rotation matrix
    """

    r=regRot(r)
    theta=numpy.linalg.norm(r)
    Sr=skew(r)
    if (theta<EPS):     # if theta is small use Taylor expansion.
        s = 1.0 - ((theta**2)/6.0) # to avoid numerical problems.
        k = 1.0 / 2.0 - theta**2
        R = numpy.eye(3) + s * Sr + k * Sr**2
    else:
        R = numpy.eye(3) + (numpy.sin(theta)/theta)*Sr + ((1-numpy.cos(theta))/(theta**2))*(numpy.dot(Sr,Sr))
    
    return R


def jRotL(r):
    """ 
    Differentials of the left and right translations for SO(3) in the principal chart
    Attributes:
        r:             ?????
    Return:
        Jl:            ?????
    """

    r=regRot(r)
    theta=numpy.linalg.norm(r)
    if theta<EPS:
        phi=1.0-(theta**2)/12.0
        w=1.0/12.0+theta**2/720.0
    
    elif (numpy.abs((theta-math.pi))<EPS):
        phi=theta*(math.pi-theta)/4.0
        w=(1.0-phi)/theta**2
    
    else:
        phi=(theta/2.0)/(numpy.tan(theta/2.0))
        w=(1.0-phi)/theta**2
    
    Jl=phi*numpy.eye(3) + (w*(numpy.outer(r,r))) + skew(r)/2.0

    return Jl

def jRotR(r):
    """ 
    Differentials of the left and right translations for SO(3) in the principal chart
    Attributes:
        r:             ?????
    Return:
        Jl:            ?????
    """

    r=regRot(r)
    theta=numpy.linalg.norm(r)
    if theta<EPS:
        phi=1.0-(theta**2)/12.0
        w=1.0/12.0+theta**2/720.0
    
    elif (numpy.abs((theta-math.pi))<EPS):
        phi=theta*(math.pi-theta)/4.0
        w=(1.0-phi)/theta**2
    
    else:
        phi=(theta/2.0)/(numpy.tan(theta/2.0))
        w=(1.0-phi)/theta**2
    
    Jr=phi*numpy.eye(3) + (w*(numpy.outer(r,r))) - skew(r)/2.0

    return Jr


def skew(r):
    """ 
    Calculates the Skew matrix
    Attributes:
        r:             vector
    Return:
        S:             Skew symmetric matrix
    """

    S = numpy.array( [   [0,   -r[2],  r[1]  ],
                        [   r[2],     0, -r[0]  ],
                        [  -r[1],  r[0],    0]  ])

    return S


def regRot(r):
    """ 
    This function limits the angle of rotation between 0 to 2*pi
    Attributes:
        r:             a rotation vector
    Return:
        u:             a normalized rotation vector
    """

    phi = numpy.linalg.norm(r)
    u = r
    if (phi != 0):
        k0=numpy.double(numpy.floor( (phi/(2.0*math.pi)) + (1.0/2.0)) )
        u=(phi-2.0*math.pi*k0)*r/phi

    return u


def unifRnd():
    """ 
    This function limits the angle of rotation between 0 to 2*pi
    Attributes:
        None
    Return:
        f:             a random normalized SE3 vector
    """    

    f = numpy.zeros(6)
    f[0:3] = regRot( numpy.random.rand(3) * 2 - 1 )     # rotation
    f[3:6] = numpy.random.rand(3) * 2 - 1               # translation

    return f
