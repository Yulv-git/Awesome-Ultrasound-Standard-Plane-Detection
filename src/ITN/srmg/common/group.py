# -*- coding: utf-8 -*-
# group.py

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


from srmg.common.util import *



def grpInv(f):
    """
    Compute the inverse of the group elemnt in SE(3).
    Inputs:
    f: an element in SE(3) with 6 elements: [r t]
    Outupts:
    fInv: inverse of f, with 6 elements.
    end:  TODO
    Group inversion for SE(3)
    """

    f = grpReg(f)
    r = f[0:3]
    t = f[3:6]
    fInv = numpy.zeros(6)
    fInv[0:3] = -r
    fInv[3:6] = numpy.dot(rotMat(-r),(-t))

    return fInv



def grpCompose(f1,f2):
    """
    compose two elements of group SE(3).
    f1.f2 = [R1*R2, (R1*t2)+t1]
    where, R1,R2 are rotation matrices and t1,t2 are translation vectors.
    But note that the function expects f1 to contain rotation vector instead
    of rotation matrix.
    Inputs:
    f1 -> Element in SE(3) represented as: [r t] where r is a rotation vector
    and t as a translation vector.
    f2 -> Another element in SE(3)
    Outputs:
    f -> Composition of f1 and f2. i.e. f1.f2
    Attributes:
        f1:            ??????
        f1:            ??????
    Return:
        f:             ??????
    """

    R1 = getClosestRotMat(rotMat(f1[0:3])); t1 = f1[3:6];
    # R1 = rotMat(f1[0:3]); t1 = f1[3:6];
    R2 = getClosestRotMat(rotMat(f2[0:3])); t2 = f2[3:6];
    # R2 = rotMat(f2[0:3]); t2 = f2[3:6];
    f = numpy.zeros(6);
    f[0:3] = rotVect(numpy.dot(R1,R2));
    f[3:6] = numpy.dot(R1,t2) + t1;

    return f




def grpReg(f):
    """ 
    Regularize the input element of the group SE(3).
    This extracts the rotation vector r from the input [r t] and uses
    regRot to normalize the rotation and limit it to 0 to 2*pi.
    Attributes:
        f:             An element in SE(3) represented as [r t]. i.e. it has 6 elements.
    Return:
        ff:            An element in SE(3) but with it's rotation normalized.
    """

    ff = numpy.zeros(6)
    ff[0:3]=regRot(f[0:3])
    ff[3:6]=f[3:6]

    return ff


def grpExpId(v):
    """ 
    NOTE
    Attributes:
        v:             An element in SE(3) represented as [r t]. i.e. it has 6 elements.
    Return:
        f:             An element in SE(3) but with it's rotation normalized.
    """

    v = grpReg(v)
    r = v[0:3]
    dt = v[3:6]
    theta = numpy.linalg.norm(r)
    
    f = numpy.zeros(6)
    f[0:3] = r

    Sr = skew(r)
    
    if (theta==0): 
        f[3:6] = dt
    elif (theta<EPS): 
        f[3:6] = dt + (1.0/6.0-theta**3 / 120.0) * numpy.dot(numpy.dot(Sr,Sr),dt) \
                    + (1.0/2.0-theta**2 / 24.0) * numpy.dot(Sr,dt)
    else:
        f[3:6] = dt + theta**(-2)*(1.0-numpy.sin(theta)/theta)*numpy.dot(numpy.dot(Sr,Sr),dt) \
                    + theta**(-2)*(1.0-numpy.cos(theta))*numpy.dot(Sr,dt)

    return f



def grpExp(f,v):
    """ 
    NOTE
    Attributes:
        f:             ??????
        v:             ??????
    Return:
        ff:            ??????
    """

    v=grpReg(v)
    f=grpReg(f)
    ff=grpReg(grpCompose(f, grpExpId(numpy.linalg.lstsq(jL(f),v)[0])))

    return ff







def grpLogId(f):
    """ 
    NOTE
    Attributes:
        f:             ??????
    Return:
        v:             ??????
    """

    f = grpReg(f)
    r = f[0:3]
    t = f[3:6]
    theta = numpy.linalg.norm(r)
    
    v = numpy.zeros(6)
    v[0:3] = r
    
    Sr = skew(r)

    if (theta==0): 
        v[3:6] = t
    elif (theta < EPS):
        v[3:6] = t - 1.0/2.0*numpy.dot(Sr,t)+(1.0/2.0-theta**2/90.0)*numpy.dot(numpy.dot(Sr,Sr),t);
    else:
        fact=0.5*theta*numpy.sin(theta)/(1.0-numpy.cos(theta))
        v[3:6] = t-1.0/2.0*numpy.dot(Sr,t)+(1.0-fact)/theta**2*numpy.dot(numpy.dot(Sr,Sr),t);
    
    return v



def grpLog(f,ff):
    """ 
    NOTE
    Attributes:
        f:             ??????
        ff:            ??????
    Return:
        v:             ??????
    """

    ff = grpReg(ff);
    f = grpReg(f);
    test1 = jL(f);
    test2 = grpLogId(grpCompose(grpInv(f),ff));  
    v = numpy.dot(test1,test2);

    return v 
