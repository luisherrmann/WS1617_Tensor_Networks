# -*- coding: utf-8 -*-
"""
@author: Luis Herrmann

----------------------------------------------------------------------------------------------------------------------------------------------

Important notes:

MPS/MPOs are implemented as lists of numpy arrays, where we use the tensor convention: [WSEN]
The methods referred to as RSEM and LSIM in my Bachelor thesis are covered by analysis12,analysis18 and analysis16,analysis17 resepectively.

Help functions implement simple algorithms that are used repeatedly throughout the code

Core functions (mostly <...MPS/MPO>) implement algebraic operations on MPS/MPOs and are introduced as "Main functions"

Analysis functions <analysis...> are used to test different hypothesis by running simulations and to visualize the results

----------------------------------------------------------------------------------------------------------------------------------------------

"""

import gc
import sys
import time
import pickle
import itertools
import numpy as np
import numpy.random as rand
import numpy.linalg as la
import math

import pylab as pl
import matplotlib as mpl
import plotly.offline as py
import plotly.graph_objs as go
from scipy.optimize import curve_fit as fit
#import LPTN_class_V08 as tenmod
from timeit import default_timer as timer
#from multimethod import multimethod

""" ###############  Simple help functions  ################ """

def remap(m,n,nmax):
    if n < nmax:
        return nmax * m + n
    else:
        raise Exception('Second argument out of bounds')
        
def remap_(digt,dim):
    if(not(len(dim)+1 == len(digt))):
        raise Exception("You must pass exactly one more digit that dimensions")
    c = digt[0]
    for i in range(len(dim)):
        c = c*dim[i] + digt[i+1]
    return(c)
        
def gcd(*numbers):
    from fractions import gcd
    return reduce(gcd, numbers)
    
def lcm(numbers):
    if(numbers == []):
        raise Exception('List must contain at least one number')
    else:
        a = numbers[0]
        for b in numbers[1:]:
            a = np.abs(a*b) // gcd(a,b)
    return(a)
    
def dectobased(num,d):
    num_d = []
    q = num
    while(not(q == 0)):
        num_d.insert(0,q%d)
        q = q // d
    return(num_d)

""" #################### Main functions #################### """     
    
    
def mapMattoVec(M,d,N):
    """ Expects a matrix with dimensions d^N x d^N of the type A^(r1r2...rN)_(s1s2...sN) and reshapes it to a d^(2N) vector of the type A_(r1,s1)(r2,s2)...(rN,sN) """
    if(not(M.shape[0] == M.shape[1])):
        raise Exception('The matrix passed needs to be square!')
    if(not(M.shape[0] == d**N)):
        raise Exception('The shape of the matrix should be [' + str(d**N) + ',' + str(d**N) + '], but is [' + str(M.shape[0]) + ',' + str(M.shape[1]) + ']')
    v = np.transpose(np.reshape(M,[d]*(2*N)),list(itertools.chain(*[[i,i+N] for i in range(N)]))).flatten()
    return(v)
    
def mapVectoMat(v,d,N):
    """ Expects a matrix with dimensions d^N x d^N of the type A^(r1r2...rN)_(s1s2...sN) and reshapes it to a d^(2N) vector of the type A_(r1,s1)(r2,s2)...(rN,sN) """
    if(not(len(v) == d**(2*N))):
        raise Exception('The length of the vector should be ' + str(d**(2*N)) + ', but is ' + str(len(v)))
    M = np.reshape(np.transpose(np.reshape(v,[d]*(2*N)),list(range(0,2*N,2)) + list(range(1,2*N,2))),[d**N,d**N])
    return(M)


def sumMPS(MPS1,MPS2):
    if(len(MPS1) != len(MPS2)):
        raise Exception('MPSs have to be of same length')
    else:
        MPS3 = []
        for i in range(len(MPS1)):
            if(not(len(MPS1[i][0,:,0]) == len(MPS2[i][0,:,0]))):
                raise Exception('The ' + str(i) + '-th tensors do not have the same physical dimension')
            else:
                m1,m2 = MPS1[i].shape[0],MPS2[i].shape[0]
                n1,n2 = MPS1[i].shape[2],MPS1[i].shape[2]
                d = MPS1[i].shape[1]
                if(i == 0):
                    MPS3.append(np.zeros([1,d,n1*n2]))
                    for j in range(d):
                        MPS3[i][:,j,0:n1] = MPS1[i][:,j,:]
                        MPS3[i][:,j,n1:(n1+n2)] = MPS2[i][:,j,:]
                elif(i == len(MPS1)-1):
                    MPS3.append(np.zeros([m1*m2,d,1]))
                    for j in range(d):
                        MPS3[i][0:m1,j,:] = MPS1[i][:,j,:]
                        MPS3[i][m1:(m1+m2),j,:] = MPS2[i][:,j,:]
                else:
                    MPS3.append(np.zeros([m1+m2,d,n1+n2]))
                    for j in range(d):
                        MPS3[i][0:m1,j,0:n1] = MPS1[i][:,j,:]
                        MPS3[i][m1:(m1+m2),j,n1:(n1+n2)] = MPS2[i][:,j,:]
    return(MPS3)
    
def sumMPO(MPO1,MPO2):
    if(len(MPO1) != len(MPO2)):
        raise Exception('MPOs have to be of same length')
    else:
        MPO3 = []
        for i in range(len(MPO1)):
            if(not(MPO1[i].shape[1] == MPO2[i].shape[1] and MPO1[i].shape[3] == MPO2[i].shape[3])):
                raise Exception('The ' + str(i) + '-th tensors do not have the same physical dimensions')
            else:
                m1,m2 = MPO1[i].shape[0],MPO2[i].shape[0]
                n1,n2 = MPO1[i].shape[2],MPO2[i].shape[2]
                d = MPO1[i].shape[1]
                if(i == 0):
                    MPO3.append(np.zeros([1,d,n1+n2,d],dtype=complex))
                    for j in range(d):
                        for k in range(d):
                            MPO3[i][:,j,0:n1,k] = MPO1[i][:,j,:,k]
                            MPO3[i][:,j,n1:(n1+n2),k] = MPO2[i][:,j,:,k]
                elif(i == len(MPO1)-1):
                    MPO3.append(np.zeros([m1+m2,d,1,d],dtype=complex))
                    for j in range(d):
                        for k in range(d):
                            MPO3[i][0:m1,j,:,k] = MPO1[i][:,j,:,k]
                            MPO3[i][m1:(m1+m2),j,:,k] = MPO2[i][:,j,:,k]
                else:
                    MPO3.append(np.zeros([m1+m2,d,n1+n2,d],dtype=complex))
                    for j in range(d):
                        for k in range(d):
                            MPO3[i][0:m1,j,0:n1,k] = MPO1[i][:,j,:,k]
                            MPO3[i][m1:(m1+m2),j,n1:(n1+n2),k] = MPO2[i][:,j,:,k]
    return(MPO3)
    
def contractMPS(MPS,direction=0):
    """ Expects a list of tensors of the sort [WSE] and computes their product (for fixed S indices)
        Parameter direction specifies direction of contraction:
            0: left->right
            1: right->left
    """
    if(direction==0):
        M = MPS[0]
        for i in range(1,len(MPS)):
            D1,d1,D2,d2 = M.shape[0],M.shape[1],MPS[i].shape[2],MPS[i].shape[1]
            M = np.reshape(np.tensordot(M,MPS[i],[[2],[0]]),[D1,d1*d2,D2])
            
    elif(direction==1):
        M = MPS[-1]
        for i in range(len(MPS)-2,-1,-1):
            D1,d1,D2,d2 = MPS[i].shape[0],MPS[i].shape[1],M.shape[2],M.shape[1]
            M = np.reshape(np.tensordot(MPS[i],M,[[2],[0]]),[D1,d1*d2,D2])
    else:
        raise Exception('Direction has to be 0 (left->right) or 1 (left<-right)')
    return(M)    
    
def contractMPO_(MPO,n,plot=False,ret_norm=False):
    """ Calculates the contracted MPO subset with indices (0,1,...,n) x (0,1,...,n) """
    d = MPO[-1].shape[1]
    q = []
    num = n
    for i in range(len(MPO)-1,-1,-1):
        q.insert(0,num%d)
        num = num // d
    
    if(q[0] == 0):
        M = MPO[0][:,:1,:,:1]
        red = True
    else:
        M = MPO[0]
        red = False
    for i in range(1,len(MPO)):
        Dw1,ds1,De1,dn1 = M.shape
        M = M.transpose([0,1,3,2]).reshape([Dw1*ds1*dn1,De1])        
        
        if(q[i] == 0 and red):
            Dw2,ds2,De2,dn2 = MPO[i][:,:1,:,:1].shape        
            M_ = MPO[i][:,:1,:,:1].reshape([Dw2,ds2*De2*dn2])
        else:
            Dw2,ds2,De2,dn2 = MPO[i].shape        
            M_ = MPO[i].reshape([Dw2,ds2*De2*dn2])
            red = False
            
        M = np.dot(M,M_).reshape([Dw1,ds1,dn1,ds2,De2,dn2]).transpose([0,1,3,4,2,5]).reshape([Dw1,ds1*ds2,De2,dn1*dn2])
        
    norm = np.mean([M[:,i,:,i] for i in range(M.shape[1])])
    #norm = max([abs(M[:,i,:,i]) for i in range(M.shape[1])])
    #M = M/norm
    if(plot):
        zvals = np.reshape(np.array(list(map(np.abs,M[:,:n+1,:,:n+1]))),[n+1,n+1])
        cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                                   ['blue','white','red'],
                                                   256)
        img2 = mpl.pyplot.imshow(zvals,interpolation='nearest',
                            cmap = cmap2,
                            origin='lower')
        mpl.pyplot.colorbar(img2,cmap=cmap2)
        #mpl.pyplot.show()
        mpl.pyplot.draw()
        mpl.pyplot.savefig('matrix.pdf')
    if(ret_norm):
        return(M[0,:n+1,0,:n+1],norm)
    else:
        return(M[0,:n+1,0,:n+1])
    
def contractMPO(MPO,direction=0):
    """ Expects a list of tensors of the sort [WSEN] and computes their product (for fixed S,N indices)
        Parameter direction specifies direction of contraction:
            0: left->right
            1: right->left
    """
    
    if(direction==0):
        M = MPO[0]
        for i in range(1,len(MPO)):
            D1,ds1,dn1,D2,ds2,dn2 = M.shape[0],M.shape[1],M.shape[3],MPO[i].shape[2],MPO[i].shape[1],MPO[i].shape[3]
            M = np.reshape(np.transpose(np.tensordot(M,MPO[i],[[2],[0]]),[0,1,3,4,2,5]),[D1,ds1*ds2,D2,dn1*dn2])
            
    elif(direction==1):
        M = MPO[-1]
        for i in range(len(MPO)-2,-1,-1):
            D1,ds1,dn1,D2,ds2,dn2 = MPO[i].shape[0],MPO[i].shape[1],MPO[i].shape[3],M.shape[2],M.shape[1],M.shape[3]
            M = np.reshape(np.transpose(np.tensordot(MPO[i],M,[[2],[0]]),[0,1,3,4,2,5]),[D1,ds1*ds2,D2,dn1*dn2])
    else:
        raise Exception('Direction has to be 0 (left->right) or 1 (left<-right)')
    return(M)
    
def contractMPOEntries(MPO,indexlist,direction=0):
    """ Contracts MPO for given entries """
    south,north = indexlist
    M = MPO[0][:,south[0],:,north[0]]
    for i in range(1,len(MPO)):
        M = np.dot(M,MPO[i][:,south[i],:,north[i]])
    return(M)

def scaleMPO(a,MPO,pos=0):
    """ Scales MPO with factor a. If parameter pos is passed, tensor to be multiplied with the factor can be specified."""
    MPO_ = list(MPO)
    MPO_[pos] = a * MPO[pos]
    return(MPO_)
    
def prodMPO(MPO1,MPO2,tr_sites=[]):
    """ Sanity checks: """
    if(not(len(MPO1) == len(MPO2))):
        raise Exception('MPOs have to be of same length')
    problemSites = [i for i in range(len(MPO1)) if not(MPO1[i].shape[1] == MPO2[i].shape[3])]
    if(len(problemSites) > 0):
        raise Exception("Tensors have incompatible dimensions at sites " + str(problemSites))
    
    trs = list(set(tr_sites))
    if(any(t >= len(MPO1) for t in trs) or not(len(trs) <= len(MPO1))):
        raise Exception("The sites marked for tracing do not match the dimensions of the passed MPO")
        
    """ Begin contraction over N,S-indices: """
    MPO3 = []
    for i in range(len(MPO1)):
        dn1,Dw1,De1 = MPO1,MPO1[i].shape[0],MPO1[i].shape[2]
        ds2,Dw2,De2 = MPO2[i].shape[1],MPO2[i].shape[0],MPO2[i].shape[2]
        if(i in trs or (i - len(MPO1) in trs)):
            M = np.reshape(np.transpose(np.tensordot(MPO1[i],MPO2[i],[[1,3],[3,1]]),[0,2,1,3]),[Dw1,Dw2,1,De1,De2,1])
        else:
            M = np.transpose(np.tensordot(MPO1[i],MPO2[i],[[1],[3]]),[0,3,4,1,5,2])
        M = np.reshape(M,[Dw1*Dw2,M.shape[2],De1*De2,M.shape[5]])
        MPO3.append(M)
    return(MPO3)
    
def identityMPO(MPO,expand=False):
    """Contructs a matching identity MPO for a given MPO"""
    idMPO = []
    for i in range(len(MPO)):
        if(expand):
            M = np.zeros(MPO[i].shape,dtype=complex)
            D,d = M.shape[0],M.shape[1]
        else:
            D,d = 1,MPO[i].shape[1]
            M = np.zeros([D,d,D,d],dtype=complex)
        if(i == 0 or i == len(MPO)-1):
            for j in range(d):
                M[0,j,0,j] = 1.0
        else:
            for j in range(d):
                M[:,j,:,j] = np.identity(D,dtype=complex)
        idMPO.append(M)
    return(idMPO)
    
def adjMPO(MPO):
    MPO_ = []
    for i in range(len(MPO)):
        M = np.zeros([MPO[i].shape[0],MPO[i].shape[3],MPO[i].shape[2],MPO[i].shape[1]],dtype=complex)
        for j in range(MPO[i].shape[0]):
            for k in range(MPO[i].shape[2]):
                M[j,:,k,:] = np.transpose(np.conjugate(MPO[i][j,:,k,:]))
        MPO_.append(M)
    return(MPO_)
        
def traceMPO(MPO,tr_sites=None,contract=True):
    if(tr_sites == None):
        tr_sites = range(0,len(MPO))
    else:
        tr_sites = list(set(tr_sites))

    """ Sanity Checks: """    
    if(any(t >= len(MPO) for t in tr_sites) or not(len(tr_sites) <= len(MPO))):
        raise Exception("The sites marked for tracing do not match the dimensions of the passed MPO")
        
    for i in tr_sites:
        D1,D2,d = MPO[i].shape[0],MPO[i].shape[2],MPO[i].shape[1]
        M = np.zeros([D1,1,D2,1],dtype=complex)
        for j in range(D1):
            for k in range(D2):
                M[j,0,k,0] = sum([MPO[i][j,l,k,l] for l in range(d)])
        MPO[i] = M
        
    if(contract):
        return(contractMPO(MPO))
    else:
        return(MPO)

""" Deprecated: Slower than new implementation"""
def normMPO_(MPO):
    """ Due to numerical instability, for very small traces, the result can sometimes be negative, such that we have to take the absolute value. """
    return(np.sqrt(abs(contractMPO(prodMPO(MPO,adjMPO(MPO),tr_sites = range(len(MPO))))))[0][0][0][0])
    
def normMPO(MPO):
    """ w1,s1,e1,n1 x w2,s2,e2,n2 -> w1,e1,w2,e2 -> w1,w2,e1,e2"""
    adjMPO(MPO[:1])[0].shape
    M = np.tensordot(MPO[0],adjMPO(MPO[:1])[0],[[1,3],[3,1]]).transpose([0,2,1,3])
    for i in range(1,len(MPO)):
        #dw1,ds1,de1,dn1 = MPO[i].shape
        """ w1,w2,e1,e2 x w3,s3,e3,n3 -> w1,w2,e2,s3,e3,n3 """
        M = np.tensordot(M,MPO[i],[[2],[0]])
        """ w1,w2,e2,s3,e3,n3 x w4,s4,e4,n4 -> w1,w2,e3,e4"""
        M = np.tensordot(M,adjMPO(MPO[i:i+1])[0],[[2,5,3],[0,1,3]])
    return(np.sqrt(abs(M[0,0,0,0])))

def canonizeMPS(MPO,direction=0,normalize=False):
    """Takes an MPS as input and returns left-canonical (direction=0) or right-canonical (direction=1) MPS."""
    MPO_ = []
    
    if(direction == 0):
        M = MPO[0]
        for i in range(len(MPO)):
            d,D1,D2 = M.shape[1],M.shape[0],M.shape[2]
            M_re = np.reshape(M,[D1*d,D2])
            A,S,V = la.svd(M_re)
            """ Cap A, remap and append """
            A_re = np.reshape(A[:,0:len(S)],[D1,d,len(S)])
            if(i == len(MPO)-1):
                """ Remaining S,V are norm of MPO """
                if(not(normalize)):
                    A_re = A_re * S[0] * V[0]
            else:
                M = np.tensordot(np.dot(np.identity(len(S))*S,V[0:len(S),:]),MPO[i+1],[[1],[0]])
            MPO_.append(A_re)
        
    elif(direction == 1):
        M = MPO[-1]
        for i in range(len(MPO)-1,-1,-1):
            d,D1,D2 = M.shape[1],M.shape[0],M.shape[2]
            M_re = np.reshape(M,[D1,d*D2])
            U,S,B = la.svd(M_re)
            """ Cap B, remap and append """
            B_re = np.reshape(B[0:len(S),:],[len(S),d,D2])
            if(i == 0):
                """ Remaining U,S are norm of MPO """
                if(not(normalize)):
                    B_re = B_re * S[0] * U[0]
            else:
                M = np.tensordot(MPO[i-1],np.dot(U[:,0:len(S)],np.identity(len(S))*S),[[2],[0]])
            MPO_.insert(0,B_re)
    else:
        raise Exception('Direction has to be 0 (left-normalized) or 1 (right-normalized)')
            
    return(MPO_)
    
def canonizeMPO(MPO,normalize=False,direction=0):
    MPO_ = []
    d_dimensions = []
    for i in range(len(MPO)):
        D1,d1,D2,d2 = MPO[i].shape[0],MPO[i].shape[1],MPO[i].shape[2],MPO[i].shape[3]
        d_dimensions.append([d1,d2])
        M = np.reshape(np.transpose(MPO[i],[0,1,3,2]),[D1,d1*d2,D2])
        MPO_.append(M)
    
    MPO_ = canonizeMPS(MPO_,direction,normalize)
    #print(map(lambda x: x.shape,MPO_))
    for i in range(len(MPO)):
        d1,d2 = d_dimensions[i]
        D1,D2 = MPO_[i].shape[0],MPO_[i].shape[2]
        MPO_[i] = np.transpose(np.reshape(MPO_[i],[D1,d1,d2,D2]),[0,1,3,2])
    return(MPO_)
    
def decomposeTens(M,d):
    D1,D2 = M.shape[0],M.shape[2]
    M_ = np.reshape(np.transpose(np.reshape(M,[D1,d,d,D2,d,d]),[0,1,4,3,2,5]),[D1*d**2,D2*d**2])
    U,S,V = la.svd(M_,full_matrices=False)
    A1 = np.transpose(np.reshape(U,[D1,d,d,len(S)]),[0,1,3,2])
    for j in range(len(S)):
        V[j,:] = S[j] * V[j,:]
    A2 = np.transpose(np.reshape(V,[len(S),D2,d,d]),[0,2,1,3])
    return([A1,A2])
    

def locexpMPS(MPO):
    """ Expands all tensors composing the MPO, such that for fixed S index, one obtains square matrices of the same dimension."""
    N = len(MPO)-2
    d = len(MPO[0][0,:,0])
    """ Find maximum bond dimension dim and expand all tensors to square matrices in C^dimxdim (for fixed S index) """
    dim = max([max([len(M[:,0,0]) for M in MPO[1:N+1]]),max([len(M[0,0,:]) for M in MPO[1:N+1]])])
    """ Expand left and right boundary tensor: """
    L = np.zeros([1,d,dim], dtype=complex)
    R = np.zeros([dim,d,1], dtype=complex)
    for i in range(d):
        L[:,i,0:MPO[0].shape[2]] = MPO[0][:,i,:]
        R[0:MPO[-1].shape[0],i,:] = MPO[-1][:,i,:]
    MPO[0] = L
    MPO[-1] = R
    """ Expand all other tensors using scheme: MPO[i] = Ai = (Ai)_(j,k) in C^(mxn)
                              i
                              v
        |a11 a22 .. a1n  0 .. 0 .. 0|
          :   :      :   :    :    :  
        |am1 am2 .. amn  0 .. 0 .. 0|
        | 0   0  ..  0   0 .. 0 .. 0|
          :   :      :   : .. :    :
        | 0   0  ..  0   0 .. 1 .. 0| < i
          :   :      :   : .. :    :
        | 0   0  ..  0   0 .. 0 .. 1|
    """
    for i in range(1,N+1):
        d = len(MPO[i][0,:,0])
        M_ = np.zeros([dim,d,dim], dtype=complex)
        for j in range(d):
            leftD = len(MPO[i][:,0,0])
            rightD = len(MPO[i][0,0,:])
            M_[0:leftD,j,0:rightD] = MPO[i][:,j,:]
            for k in range(min(leftD,rightD),dim):
                M_[k,j,k] = 1
        MPO[i] = M_
        
def locexpMPO(MPO):
    """ Expands all tensors composing the MPO, such that for fixed S,N indices, one obtains square matrices of the same dimension."""
    N = len(MPO)-2
    d = MPO[0].shape[1]
    """ Find maximum bond dimension dim and expand all tensors to square matrices in C^dimxdim (for fixed S index) """
    dim = max([max([M.shape[0] for M in MPO[1:N+1]]),max([M.shape[2] for M in MPO[1:N+1]])])
    """ Expand left and right boundary tensor: """
    L = np.zeros([1,d,dim,d], dtype=complex)
    R = np.zeros([dim,d,1,d], dtype=complex)
    for i in range(d):
        for j in range(d):
            L[:,i,0:MPO[0].shape[2],j] = MPO[0][:,i,:,j]
            R[0:MPO[-1].shape[0],i,:,j] = MPO[-1][:,i,:,j]
    MPO[0] = L
    MPO[-1] = R
    for i in range(1,N+1):
        d = MPO[i].shape[1]
        M_ = np.zeros([dim,d,dim,d], dtype=complex)
        for j in range(d):
            for k in range(d):
                leftD,rightD = MPO[i].shape[0],MPO[i].shape[2]
                M_[0:leftD,j,0:rightD,k] = MPO[i][:,j,:,k]
                for l in range(min(leftD,rightD),dim):
                    M_[l,j,l,k] = 1
        MPO[i] = M_

def SVDVec(X,d,N,expand=False,direction=0,returnsval=False,tol=None,trunc_mode=0,drop=0,full_matrices=True):
    """ Description:
        ----------------------------------------------------------------------------------------------------------------------------------------------
        Decomposes a vector of dimension d^N of the kind X_(r1,r2,...rN) to an MPO of length N (excluding left and right boundary tensor) 
        through repeated reshaping and singular value decompositions.
        
        Parameters:
        ----------------------------------------------------------------------------------------------------------------------------------------------
        direction:      Specifies the direction of the iterated SVDs, where:
                            0: left->right
                            1: right->left
        tol:            If specified, singular values smaller than tol will be dropped, yielding a reduced SVD.
        trunc_mode:     Specifies mode for truncation of singular values:
                            0: Singular value is set to 0. This prevents asymmetric tensor dimensions and shape mismatches
                               during MPO elongation.
                            1: Rows/columns of matrices corresponding a truncated singular value are deleted.
        drop:           If specified, the last [drop] singular values will be dropped, yielding a reduced SVD.
    """
    MPO = []
    if(returnsval):
        svals = []
        
    if(direction == 0):
        """ First SVD yields left boundary tensor """    
        Y = np.reshape(X,[d,len(X)//d])
        U,S,V = la.svd(Y,full_matrices=full_matrices)
        
        """ Drop last values """
        if(drop > 0):
            S = S[:-drop]
        """ Drop values smaller than tolerance"""
        if tol is not None:
            """ This works because the singular values are ordered in descending order:"""
            if(trunc_mode == 0):
                while S[-1]<tol and len(S)>1:
                    S = S[:-1]
            else:
                for i in range(len(S)):
                    if(S[i]<tol):
                        S[i] = 0
                
        MPO.append(np.reshape(U[:,:len(S)],[1,U.shape[0],len(S)]))
        if(returnsval):
            svals.append(S.tolist())
        for i in range(len(S)):
            V[i,:] = S[i] * V[i,:]
        V = V[:len(S),:]
        
        """ Repeat SVD N times """
        for i in range(1,N+1):
            Y = np.reshape(V,[len(V[:,0])*d,len(V[0,:])//d])
            U,S,V = la.svd(Y)
            if(drop > 0):
                S = S[:-drop]
            if tol is not None:
                if(trunc_mode == 0):
                    while S[-1]<tol and len(S)>1:
                        S = S[:-1]
                else:
                    for i in range(len(S)):
                        if(S[i]<tol):
                            S[i] = 0
                            
            A = np.reshape(U[:,:len(S)],[len(U[:,0])//d,d,len(S)])
            MPO.append(A)
            if(returnsval):
                svals.append(S.tolist())
            for j in range(len(S)):
                V[j,:] = S[j] * V[j,:]
            V = V[:len(S),:]
        
        MPO.append(np.reshape(V,[V.shape[0],V.shape[1],1]))
        
    elif(direction == 1):
        Y = np.reshape(X,[len(X)//d,d])
        U,S,V = la.svd(Y,full_matrices=full_matrices)
        if(drop > 0):
            S = S[:-drop]
        if tol is not None:
            if(trunc_mode == 0):
                while S[-1]<tol and len(S)>1:
                    S = S[:-1]
            else:
                for i in range(len(S)):
                    if(S[i]<tol):
                        S[i] = 0
                
        MPO.insert(0,np.reshape(V[:len(S),:],[len(S),V.shape[1],1]))
        if(returnsval):
            svals.append(S.tolist())
        for i in range(len(S)):
            U[:,i] = U[:,i] * S[i]
        U = U[:,:len(S)]
        
        """ Repeat SVD N times """
        for i in range(1,N+1):
            Y = np.reshape(U,[len(U[:,0])//d,len(U[0,:])*d])
            U,S,V = la.svd(Y)
            if(drop > 0):
                S = S[:-drop]
            if tol is not None:
                if(trunc_mode == 0):
                    while S[-1]<tol and len(S)>1:
                        S = S[:-1]
                else:
                    for i in range(len(S)):
                        if(S[i]<tol):
                            S[i] = 0
                    
            B = np.reshape(V[:len(S),:],[len(S),d,len(V[0,:])//d])
            MPO.insert(0,B)
            if(returnsval):
                svals.append(S.tolist())
            for j in range(len(S)):
                U[:,j] = U[:,j] * S[j]
            U = U[:,:len(S)]
        
        MPO.insert(0,np.reshape(U,[1,U.shape[0],U.shape[1]]))

    else:
        raise Exception('Direction has to be 0 (left->right) or 1 (left<-right)')

    """ The expand-parameter allows for expansion of the obtained tensors, delivering an MPO with equidimensional tensors """
    if(expand):
        locexpMPS(MPO)
    if(returnsval):
        return(MPO,svals)
    else:
        return(MPO)

def SVDMat(X,d,N,expand=False,direction=0,returnsval=False,tol=None,trunc_mode=0,drop=0):
    """ Decomposes a matrix X of dimension d^N x d^N of the kind A^(r1r2...rN)_(s1s2...sN) to an MPO of length N
        (excluding left and right boundary tensor) through repeated reshaping and singular value decomposition.    
    """
    
    """ Matrix X of the type X^(r1r2...rN)_(s1s2...sN) is remapped to to vector of type X_(r1s1)(r2s2)...(rNsN)"""
    Y = mapMattoVec(X,d,N+2)
    """ Expand-parameter allows for expansion of the obtained tensors, delivering an MPO with equidimensional tensors """
    if(returnsval):
        MPO,svals = SVDVec(Y,d**2,N,expand,direction,True,tol=tol,trunc_mode=trunc_mode,drop=drop)
        #MPO,svals = SVDVec(Y,d**2,N,expand,direction,True,tol=tol)

    else:
        MPO = SVDVec(Y,d**2,N,expand,direction,False,tol=tol,trunc_mode=trunc_mode,drop=drop)
        #MPO = SVDVec(Y,d**2,N,expand,direction,False,tol=tol)

    """ Reshape [WSE] tensors of MPO with S dimension d^2 to [WSEN] tensors """
    for i in range(0,N+2):
        #np.reshape(MPO[i],[MPO[i].shape[0],d,MPO[i].shape[2],d])
        MPO[i] = np.transpose(np.reshape(MPO[i],[MPO[i].shape[0],d,d,MPO[i].shape[2]]),[0,1,3,2])
    if(returnsval):
        return(MPO,svals)
    else:
        return(MPO)

def solveMPO(X,MPO,MPO_,tr_sites=[]):
    """ MPO_ must have an empty list at the position of the tensor that needs to be solved"""    
    pos = [i for i in range(len(MPO_)) if(len(MPO_[i]) == 0)]
    if(not(len(pos))) == 1:
        raise Exception("The second MPO_ must have EXACTLY ONE empty list occurence")
    else:
        pos = pos[0]
    trs = list(set(tr_sites))
    
    L = contractMPO(prodMPO(MPO[:pos],MPO_[:pos],tr_sites = list(filter(lambda x:x<pos and x >= 0,trs))))
    R = contractMPO(prodMPO(MPO[pos+1:],MPO_[pos+1:],tr_sites = list(map(lambda x:x-pos-1,filter(lambda x:x>pos,trs))) +  list(filter(lambda x:x<0,trs))))
    A = MPO[pos]
    #print([x.shape for x in L if len(x)>0])
    #print([x.shape for x in R if len(x)>0])
    
    d1,d2,d3,D1,D2 = L.shape[1],A.shape[1],R.shape[1],A.shape[0],A.shape[2]
    D1_,D2_ = MPO_[pos-1].shape[2],MPO_[pos+1].shape[0]
    """ Reshape X to vector """
    X_ = np.reshape(X,[X.shape[0]**2])
    """ Index structure: s1,e1,e1',n1 x w2,s2,e2,n2 -tensordot->  s1,e1',n1,s2,e2,n2 """
    M = np.tensordot(np.reshape(L,[L.shape[1],D1,L.shape[2]//D1,L.shape[3]]),A,[[1],[0]])
    """ Index structure: s1,e1',n1,s2,e2,n2 x w3,w3',s3,n3 -tensordot-> s1,e1',n1,s2,n2,w3',s3,n3 """
    M = np.tensordot(M,np.reshape(R,[D2,R.shape[0]//D2,R.shape[1],R.shape[3]]),[[4],[0]])
    """ Index structure: s1,e1',n1,s2,n2,w3',s3,n3 x s4,n4 -tensordot-> s1,e1',n1,s2,n2,w3',s3,n3,s4,n4 """
    M = np.tensordot(M,np.identity(d2),[[],[]])
    """ Transpose and reshape M to have index structure: (s1,s4,s3,n1,n2,n3);(e1',n4,w3',s2) """
    M = np.reshape(np.transpose(M,[2,4,7,0,8,6,1,9,5,3]),[d1**2*d2**2*d3**2,D1_*D2_*d2**2])
    #print(M.shape,X_.shape)
    
    Y = la.lstsq(M,X_)[0]
    Ai = np.reshape(Y,[D1_,d2,D2_,d2])
    
    return(Ai)
    
def solveMPO2(X,MPO_left,MPO_right,A,tr_sites=[]):
    trs = list(set(tr_sites))
    pos = len(MPO_left)+1
    
    L = traceMPO(MPO_left,contract=True,tr_sites = list(filter(lambda x:x<pos and x >= 0,trs)))
    R = traceMPO(MPO_right,contract=True,tr_sites = list(map(lambda x:x-pos-1,filter(lambda x:x>pos,trs))) +  list(filter(lambda x:x<0,trs)))
    
    d1,d2,d3,D1,D2 = L.shape[1],A.shape[1],R.shape[1],A.shape[0],A.shape[2]
    D1_,D2_ = L.shape[2]//A.shape[0],R.shape[0]//A.shape[2]
    """ Reshape X to vector """
    X_ = np.reshape(X,[X.shape[0]**2])
    """ Index structure: s1,e1,e1',n1 x w2,s2,e2,n2 -tensordot->  s1,e1',n1,s2,e2,n2 """
    M = np.tensordot(np.reshape(L,[L.shape[1],D1,L.shape[2]//D1,L.shape[3]]),A,[[1],[0]])
    """ Index structure: s1,e1',n1,s2,e2,n2 x w3,w3',s3,n3 -tensordot-> s1,e1',n1,s2,n2,w3',s3,n3 """
    M = np.tensordot(M,np.reshape(R,[D2,R.shape[0]//D2,R.shape[1],R.shape[3]]),[[4],[0]])
    """ Index structure: s1,e1',n1,s2,n2,w3',s3,n3 x s4,n4 -tensordot-> s1,e1',n1,s2,n2,w3',s3,n3,s4,n4 """
    M = np.tensordot(M,np.identity(d2),[[],[]])
    """ Transpose and reshape M to have index structure: (s1,s4,s3,n1,n2,n3);(e1',n4,w3',s2) """
    M = np.reshape(np.transpose(M,[2,4,7,0,8,6,1,9,5,3]),[d1**2*d2**2*d3**2,D1_*D2_*d2**2])
    
    Y = la.lstsq(M,X_)[0]
    Ai = np.reshape(Y,[D1_,d2,D2_,d2])
    
    return(Ai)

def createSet(d,D,direction=0,normalize=False,imaginary=True):
    #print(l.shape == np.zeros([1,1,1,1]).shape)
    l = rand.random([1,d,D,d]) 
    A = rand.random([D,d,D,d])
    r = rand.random([D,d,1,d])
    if(imaginary):
        l = l + 1j * rand.random([1,d,D,d])
        A = A + 1j * rand.random([D,d,D,d])
        r = r + 1j * rand.random([D,d,1,d])
    if(normalize):
        l,A,r = canonizeMPO([l,A,r],direction,normalize)
    return(A,l,r)
    
def createSetXY(d,D,imaginary=True):
    full_rank = False
    while(not(full_rank)):
        Y,Z = rand.random([d**2,d**2]), rand.random([d,d,d,d])
        if(imaginary):
            Y,Z = Y + 1j*rand.random([d**2,d**2]), Z + 1j*rand.random([d,d,d,d])
        Y_l,Y_r = SVDMat(Y,d,0,expand=False,direction=1,drop=d**2-D)
        if(d**2 > D):
            if(la.matrix_rank(contractMPO([Y_l,Y_r])[0,:,0,:]) == 4):
                full_rank = True
            else:
                print("Matrix Y does not have full rank.")
        else:
            full_rank = True
    DY = Y_r.shape[0]
    #Z_l,Z_r = SVDMat(np.reshape(Z,[4,4]),2,0,expand=False,direction=1)
    #DZ = Z_r.shape[0]
    """ w1,s1,e1,n1 x s2,s2',n2,n2' -> w1,s1,e1,s2,n2,n2' -> w1,(s1,s2),e1,(n2,n2')"""
    l = np.reshape(np.transpose(np.tensordot(Y_l,Z,[[3],[1]]),[0,1,3,2,4,5]),[1,4,DY,4])
    """ w1,s1,e1,n1 x s2,s2',n2,n2' -> w1,s1,e1,s2',n2,n2' -> w1,(s1,s2'),e1,(n2,n2')"""
    r = np.reshape(np.transpose(np.tensordot(Y_r,Z,[[3],[0]]),[0,1,3,2,4,5]),[DY,4,1,4])
    """ w1,s1,e1,n1 x s2,s2',n2,n2' -> w1,s1,e1,s2',n2,n2'"""
    A = np.tensordot(Y_r,Z,[[3],[0]])
    """ w1,s1,e1,s2',n2,n2' x w3,s3,e3,n3 -> w1,s1,e1,n2,n2',w3,s3,e3 -> (w1,w3),(s1,s3),(e1,e3),(n2,n2')"""
    A = np.reshape(np.transpose(np.tensordot(A,Y_l,[[3],[3]]),[0,5,1,6,2,7,3,4]),[DY,4,DY,4])
    return(A,l,r)
    
def createSetYXY(d,D):
    full_rank = False
    while(not(full_rank)):
        Y,Z = rand.random([d,d,d,d]),rand.random([d**2,d**2])
        Z_l,Z_r = SVDMat(Z,d,0,expand=False,direction=1,drop=d**2-D)
        if(d**2 > D):
            if(la.matrix_rank(contractMPO([Z_l,Z_r])[0,:,0,:]) == 4):
                full_rank = True
            else:
                print("Matrix Y does not have full rank.")
        else:
            full_rank = True
    DZ = Z_r.shape[0]
    #Z_l,Z_r = SVDMat(np.reshape(Z,[4,4]),2,0,expand=False,direction=1)
    #DZ = Z_r.shape[0]
    """ s1,s1',n1,n1' x w2,s2,e2,n2 -> s1,s1',n1,w2,e2,n2 """
    l = np.tensordot(Y,Z_l,[[3],[1]])
    """ s1,s1',n1,w2,e2,n2 x s3,s3',n3,n3' -> s1,s1',w2,e2,n3,n3' -> w2,(s1,s1'),e2,(n3,n3') """
    l = np.reshape(np.transpose(np.tensordot(l,Y,[[2,5],[0,1]]),[2,0,1,3,4,5]),[1,d**2,DZ,d**2])
    """ s1,s1',n1,n1' x ´w2,s2,e2,n2 -> s1,s1',n1',w2,e2,n2"""
    r = np.tensordot(Y,Z_r,[[2],[1]])
    """ s1,s1',n1',w2,e2,n2 x s3,s3',n3,n3' -> s1,s1',w2,e2,n3,n3' -> w2,(s1,s1'),e2,(n3,n3')"""
    r = np.reshape(np.transpose(np.tensordot(r,Y,[[2,5],[1,0]]),[2,0,1,3,4,5]),[DZ,d**2,1,d**2])
    """ s1,s1',n1,n1' x w2,s2,e2,n2 -> s1,s1',n1',w2,e2,n2"""
    A = np.tensordot(Y,Z_r,[[2],[1]])
    """ s1,s1',n1',w2,e2,n2 x w3,s3,e3,n3 -> s1,s1',w2,e2,n2,w3,e3,n3"""
    A = np.tensordot(A,Z_l,[[2],[1]])
    """ s1,s1',w2,e2,n2,w3,e3,n3 x s4,s4',n4,n4' -> s1,s1',w2,e2,w3,e3,n4,n4' -> (w2,w3),(s1,s1'),(e2,e3),(n4,n4') """
    A = np.reshape(np.transpose(np.tensordot(A,Y,[[4,7],[0,1]]),[2,4,0,1,3,5,6,7]),[DZ,d**2,DZ,d**2])
    return(A,l,r)

""" Infinity matrix norm """
def norm1(M):
    ldim = len(M[:,0])
    lsums = np.zeros(ldim)
    for i in range(ldim):
        lsums[i] = sum(map(abs,M[i,:]))
    return(max(lsums))

""" Spectral norm / 2-norm """
def norm2(M):
    N = np.dot(M,np.transpose(np.conjugate(M)))
    eigval = la.eigvalsh(N)
    return(np.sqrt(max(eigval)))

""" Frobenius norm"""
def norm3(M):
    ldim = len(M[:,0])
    rdim = len(M[0,:])
    sum = 0
    for i in range(ldim):
        for j in range(rdim):
            sum = sum + (M[i,j].real)**2 + (M[i,j].imag)**2
    return(np.sqrt(sum))

def fitmodel(x,a,b):
    return(a+b*x)
    
def delta(i,j):
    if i==j:
        return 1
    else:
        return 0    
    

""" #################### Deprecated functions #################### """
 
"""Deprecated: Use contractMPS with [WSE] tensors instead"""
def computeAppVec(param1,param2):
    """ Expects a tensor A of the sort [WSE], a tensor of the sort [SE] or [WS] and computes the respective product v_r1 * A_r2 or A_r1 * v_r2
        for fixed S indices, yielding a [SE] or [WS] tensor."""
    if(len(param1.shape)==3):
        A = param1
        v = param2
        slen1 = len(A[0,:,0])
        slen2 = len(v[0,:])
        m = len(A[:,0,0])
        w = np.zeros([m,slen1*slen2],dtype=complex)
        for j in range(slen1):
            for k in range(slen2):
                w[:,remap(j,k,slen2)] = np.dot(A[:,j,:],v[:,k])
    elif(len(param1.shape)==2):
        v = param1
        A = param2
        slen1 = len(v[:,0])
        slen2 = len(A[0,:,0])
        m = len(A[:,0,0])
        w = np.zeros([slen1*slen2,m],dtype=complex)
        for j in range(slen1):
            for k in range(slen2):
                w[remap(j,k,slen2),:] = np.dot(v[j,:],A[:,k,:])
    else:
        raise Exception('The tensors passed have too many legs!')
    return(w)

"""Deprecated: Use contractMPS with [WSEN] tensors instead"""
def computeAppMat(param1,param2):
    """ Expects a tensor A of the sort [WSEN], a tensor of the sort [SWN] or [ESN] and computes the respective product v_r1 * A_r2 or A_r1 * v_r2
        for fixed S indices, yielding a [SWN] or [ESN] tensor."""
    if(len(param1.shape) == 4):
        A = param1
        v = param2
    elif(len(param1.shape) == 3):
        v = param1
        A = param2
    d1 = len(A[0,:,0,0])
    D1 = len(A[:,0,0,0])
    D2 = len(A[0,0,:,0])
    A = np.reshape(A,[D1,d1**2,D2])
    d2 = len(v[:,0,0])
    D3 = len(v[0,:,0])
    if(len(param1.shape) == 4):
        v = np.reshape(v,[D3,d2**2])
        w = np.reshape(computeAppVec(A,v),[d1*d2,D3,d1*d2])
    elif(len(param1.shape) == 3):
        v = np.reshape(v,[d2**2,D3])
        w = np.reshape(computeAppVec(v,A),[d1*d2,D3,d1*d2])
    return(w)

"""Deprecated: Use contractMPS with [WSE] tensors instead"""
def computeXVec(Mlist,L,R):
    """ Expects a list of [WSE] tensors, as well as a left [SW] and right [ES] boundary tensor and computes their product,
        yielding a [S] vector """
    for i in range(len(Mlist)-1,-1,-1):
        alen = len(Mlist[i][:,0,0])
        slen1 = len(Mlist[i][0,:,0])
        slen2 = len(R[0,:])
        R_ = np.zeros([alen,slen1*slen2], dtype=complex)
        for k in range(slen1):
            for l in range(slen2):
                R_[:,remap(k,l,slen2)] = np.dot(Mlist[i][:,k,:],R[:,l])
        R = R_
    
    m = len(L[:,0])
    n = len(R[0,:])
    X = np.zeros([m*n], dtype=complex)
    for i in range(m):
        for j in range(n):
            X[remap(i,j,n)] = np.dot(L[i,:],R[:,j])
    return(X)
    
"""Deprecated: Use contractMPO with [WSEN] tensors instead"""
def computeXMat(Mlist,L,R,d=None):
    """ Expects a list of [WSEN] tensors, as well as a left [SWN] and right [ESN] boundary tensor and computes their product,
        yielding a [NS] matrix 
        ATTENTION: If another tensor has alredy been applied to l or r, you MUST specify the physical dimension d for correct remapping!"""
    if(d == None):
        d = len(L[0,0,:])
    L = np.reshape(L,[len(L[0,0,:])**2,len(L[0,:,0])])
    R = np.reshape(R,[len(R[0,:,0]),len(R[:,0,0])**2])
    for i in range(len(Mlist)):
        Dlen1 = len(Mlist[i][:,0,0,0])
        Dlen2 = len(Mlist[i][0,0,:,0])
        dlen = len(Mlist[i][0,:,0,0])
        Mlist[i] = np.reshape(Mlist[i],[Dlen1,dlen**2,Dlen2])
    X = computeXVec(Mlist,L,R)
    return(mapVectoMat(X,d,int(round(math.log(len(X),d**2)))))



""" #################### Analysis functions #################### """


    
def analysis1(drange,Drange,Nrange,samples,normed,timestamp=True,interactive=False,plot=False):
    """ Following steps are iterated for physical dimensions d, bond dimensions D and lattice lengths N in specified ranges:
    
    i)   Generate random A,l,r
    ii)  Compute TI matrix X as: X^(s0s1...sNs(N+1))_(t0t1...tNt(N+1)) = l^s0_t0 A^s1_t1 ... A^sN_tN l^s(N+1)_t(N+1)
             |   |   |         |   |
             l - A - A - ... - A - r
             |   |   |         |   |
    iii) Compute inverse Xi of X
    iv)  Invert A,l,r for all W,E-indices respectively and compute fake TI inverse Xi_:
             |    |    |           |   |
             li - Ai - Ai - ... - Ai - ri
             |    |    |           |   |
    v)   Compute norm-difference ||XXi - XXi_||
    """
    
    """ Error check """    
    for d in drange:
        if d <= 1:
            raise Exception('d has to be greater than 1!')
    """ Actual analysis """
    if(normed):
        plotData = np.zeros([len(drange),len(Drange),len(Nrange),samples,2])
    else:
        plotData = np.zeros([len(drange),len(Drange),len(Nrange),samples,1])
    for d in range(len(drange)): #d is physical dimension
        for D in range(len(Drange)): #D is bond dimension
            #ndiflist = []
            #if(normed):
            #    ndiflistNormed = []
            for s in range(samples):
                singular = True
                while(singular):
                    try:
                        A,l,r = createSet(drange[d],Drange[D])
                        li = np.zeros(l.shape, dtype=complex)
                        Ai = np.zeros(A.shape, dtype=complex)
                        ri = np.zeros(r.shape, dtype=complex)
                        for i in range(Drange[D]):
                            li[0,:,i,:] = la.inv(l[0,:,i,:])
                            ri[i,:,0,:] = la.inv(r[i,:,0,:])
                            for j in range(Drange[D]):
                                Ai[i,:,j,:] = la.inv(A[i,:,j,:])
                        for N in range(len(Nrange)):
                            X = [l] + [A]*Nrange[N] + [r]
                            if(normed):
                                X = scaleMPO(1/normMPO(X),X)
                            Xi_ = [li] + [Ai]*Nrange[N] + [ri]
                            id_ = scaleMPO(-1,prodMPO(X,Xi_))
                            plotData[d,D,N,s,0] = normMPO(sumMPO(identityMPO(id_),id_))
                            #ndiflist.append(norm3(np.dot(Xi,X) - np.dot(Xi_,X)))
                            if(normed):
                                Xi_ = scaleMPO(-1/normMPO(Xi_),Xi_)
                                id_ = prodMPO(X,Xi_)
                                #ndiflistNormed.append(norm3(np.dot(Xi,X) - np.dot(Xi_,X)))
                                plotData[d,D,N,s,1] = normMPO(sumMPO(identityMPO(id_),id_))#/drange[d]**Nrange[N]
                            #E = np.identity(len(X[:,]))
                            #check = norm3(E-np.dot(X,Xi))
                        singular = False
                    except np.linalg.linalg.LinAlgError as e:
                        if 'Singular matrix' in e.message:
                            singular = True
                        else:
                            raise
    if(interactive):
        plotBool = input('Analysis complete. You can plot the data now or save the data and plot it later.\n Plot now? y/n\t')
        if(plotBool == 'y'):
            plot = True
        else:
            plot = False
    if(plot):
        analysis1Plot([plotData,drange,Drange,Nrange,samples],filenamePrefix='analysis1')
    else:
        filename = 'analysis1d'+str(drange[0])+str(drange[-1])+'D'+str(Drange[0])+'-'+str(Drange[-1])+'N'+str(Nrange[0])+'-'+str(Nrange[-1])+'samples'+str(samples)
        if(timestamp):        
            filename = filename + str(int(time.time()))
        filename = filename + '.data'
        print('Saving to ' + filename)
        pickle.dump([plotData,drange,Drange,Nrange,samples],open(filename,'wb'))
        
def analysis1Plot(data,filenamePrefix='analysis1'):
    """ Can be called either with a string, which will be interpreted as a filename containing the data to be plotted, 
        or with a np.array - object with dimensions [len(Drange),len(Nrange),samples] """
    if(type(data) == str):
        print('Reading...')
        plotData,drange,Drange,Nrange,samples = pickle.load(open(data))
        if(len(plotData[:,0,0,0,0]) != len(drange)):
            raise Exception('The physical dimension range does not match the plot data!')
        if(len(plotData[0,:,0,0,0]) != len(Drange)):
            raise Exception('The bond dimension range does not match the plot data!')
        if(len(plotData[0,0,:,0,0]) != len(Nrange)):
            raise Exception('The lattice length range does not match the plot data!')
        if(len(plotData[0,0,0,:,0]) != samples):
            raise Exception('The sample size does not match the plot data!')
        print('Plot data read. Plotting...')
    else:
        plotData,drange,Drange,Nrange,samples = data
    """ Fit results with linear model (on logarithmic scale) """
    if(len(plotData[0,0,0,0,:]) == 1):
        normed = False
    elif(len(plotData[0,0,0,0,:]) == 2):
        normed = True
    else:
        raise Exception('Something appears to be wrong with your data!')
    for d in range(len(drange)):
        for D in range(len(Drange)):
            #params1, cov1 = fit(fitmodel,Nrange*samples,map(lambda x: math.log(x),ndiflist))
            #if(normed):
            #    params2, cov2 = fit(fitmodel,Nrange*samples,map(lambda x: math.log(x),ndiflistNormed))
            pl.figure()
            pl.xlabel(r'Lattice length $N$')
            pl.xlim([Nrange[0],Nrange[-1]+1])
            if(normed):
                pl.ylim([10**(math.floor(math.log(min([min(plotData[d,D,:,:,0].flatten()),min(plotData[d,D,:,:,1].flatten())]),10))-1),10**(math.ceil(math.log(max([max(plotData[d,D,:,:,0].flatten()),max(plotData[d,D,:,:,1].flatten())]),10))+1)])
            else:
                pl.ylim([10**(math.floor(math.log(min(plotData[d,D,:,:].flatten()),10))-1),10**(math.ceil(math.log(max(plotData[d,D,:,:].flatten()),10))+1)])
            """if(normed):
                pl.ylim([10**(math.floor(math.log(min([min(ndiflist),min(ndiflistNormed)]),10))-1),10**(math.ceil(math.log(max([max(ndiflist),max(ndiflistNormed)]),10))+1)])
            else:
                pl.ylim([10**(math.floor(math.log(min(ndiflist),10))-1),10**(math.ceil(math.log(max(ndiflist),10))+1)])"""
            pl.xticks(range(Nrange[0]-1,Nrange[-1]+1))
            pl.ylabel(r'$\Vert E - X X^{-1} \Vert$')
            pl.yscale('log', nonposy='clip')
            pl.title(r'Simulated $X^{-1}$ deviation as a function of lattice length $N$')
            pl.grid(b=True, which='major', color='black', linestyle='--')
            pl.grid(b=True, which='minor', color='orange', linestyle='--')
            y = []
            yerror = np.zeros([2,len(Nrange)])
            for N in range(len(Nrange)):
                """y.append(np.mean(ndiflist[N::len(Nrange)]))
                yerror.append(np.std(ndiflist[N::len(Nrange)]))"""
                y.append(np.mean(plotData[d,D,N,:,0]))
                yerror[:,N] = [0,np.std(plotData[d,D,N,:,0])]
            #pl.plot(Nrange,map(lambda x: math.exp(fitmodel(x,params1[0],params1[1])), Nrange),'-r')
            if(not(normed)):
                pl.errorbar(Nrange,y,yerr=yerror,fmt='ro',label=r'',capthick=2)
            else:
                pl.errorbar(Nrange,y,yerr=yerror,fmt='ro',label=r'$X^{-1}$ not normed',capthick=2)
                yNormed = []
                yerrorNormed = np.zeros([2,len(Nrange)])
                for N in range(len(Nrange)):
                    """yNormed.append(np.mean(ndiflistNormed[N::samples]))
                    yerrorNormed.append(np.std(ndiflistNormed[N::samples]))"""
                    yNormed.append(np.mean(plotData[d,D,N,:,1]))
                    yerrorNormed[:,N] = [0,np.std(plotData[d,D,N,:,1])]
                pl.errorbar(Nrange,yNormed,yerr=yerrorNormed,fmt='bo',label=r'$X^{-1}$ normed')
                #pl.plot(Nrange,map(lambda x: math.exp(fitmodel(x,params2[0],params2[1])), Nrange),'-b')
                pl.legend(loc='best')
            pl.savefig(filenamePrefix+'_d'+str(drange[d])+'D'+str(Drange[D])+'N'+str(Nrange[0])+'-'+str(Nrange[-1])+'s'+str(samples)+'.pdf')
            

def analysis2(drange,Drange,Nrange,samples,normed=True,timestamp=True,interactive=False,plot=False):
    """ Following steps are iterated for physical dimensions d, bond dimensions D and lattice lengths N+2 in specified ranges:
    
    i)   Generate random A,l,r
    ii)  Compute TI matrix X as: X^(s0s1...sNs(N+1))_(t0t1...tNt(N+1)) = l^s0_t0 A^s1_t1 ... A^sN_tN l^s(N+1)_t(N+1)
             |   |   |         |   |
             l - A - A - ... - A - r
             |   |   |         |   |
    iii) Compute inverse Xi of X
    iv)  Decompose Xi through repeated singular value decomposition and reshaping into an MPO of lattice length N:
             |    |    |          |      |
             M0 - M1 - M2 - ... - MN - M(N+1)
             |    |    |          |      |
    v)   Select center tensor B (two for even lattice length N), invert B,M0 and M(N+1) for all W,E-indices and compute fake TI inverse X_:
             |    |    |          |       |
            M0i - Bi - Bi - ... - Bi - M(N+1)i
             |    |    |          |       | 
    vi)  Compute norm-difference ||XiX - XiX_||
    """    
    
    """ Error check """    
    for d in drange:
        if d <= 1:
            raise Exception('d has to be greater than 1!')
    """ Actual analysis """
    if(normed):
        plotData = np.zeros([len(drange),len(Drange),len(Nrange),samples,2])
    else:
        plotData = np.zeros([len(drange),len(Drange),len(Nrange),samples,1])
    for d in range(len(drange)): #d is physical dimension
        for D in range(len(Drange)): #D is bond dimension
            #ndiflist = []
            #if(normed):
            #    ndiflistNormed = []
            for s in range(samples):
                singular = True
                while(singular):
                    try:
                        A,l,r = createSet(drange[d],Drange[D])
                        for N in range(len(Nrange)):
                            X = contractMPO([l] + [A]*Nrange[N] + [r])
                            #if(normed):
                            #    X = X / norm3(X)
                            Xi = la.inv(X)
                            """ Decompose Xi into new MPO """
                            XiMPO = SVDMat(Xi,drange[d],Nrange[N],True)
                            B = contractMPO(XiMPO[(Nrange[N]//2) + (Nrange[N]%2):(Nrange[N]//2) + (Nrange[N]%2) + ((Nrange[N]+1)%2) + 1])
                            Bi = np.zeros(B.shape, dtype=complex)
                            li = np.zeros(XiMPO[0].shape, dtype=complex)
                            ri = np.zeros(XiMPO[-1].shape, dtype=complex)
                            for i in range(Drange[D]):
                                li[:,i,:] = la.inv(l[:,i,:])
                                ri[:,i,:] = la.inv(r[:,i,:])
                                for j in range(Drange[D]):
                                    Bi[i,:,j,:] = la.inv(B[i,:,j,:])
                            X_ = contractMPO([li] + [Bi]*(Nrange[N]//(2 - Nrange[N]%2)) + [ri])
                            #ndiflist.append(norm3(np.dot(Xi,X) - np.dot(Xi,X_)))
                            plotData[d,D,N,s,0] = norm3(np.identity(len(X[0,:]), dtype=complex) - np.dot(Xi,X_))
                            #ndiflist.append(norm3(np.identity(len(X[0,:]), dtype=complex) - np.dot(Xi,X_)))
                            if(normed):
                                #X = X / norm3(X)
                                X_ = X_ / norm3(X_)
                                plotData[d,D,N,s,1] = norm3(np.identity(len(X[0,:]), dtype=complex) - np.dot(Xi,X_))
                                #ndiflistNormed.append(norm3(np.dot(Xi,X) - np.dot(Xi,X_)))
                        singular = False
                    except np.linalg.linalg.LinAlgError as e:
                        if 'Singular matrix' in e.message:
                            singular = True
                        else:
                            raise
    if(interactive):
        plotBool = input('Analysis complete. You can plot the data now or save the data and plot it later.\n Plot now? y/n\t')
        if(plotBool == 'y'):
            plot = True
        else:
            plot = False
    if(plot):
        analysis1Plot([plotData,drange,Drange,Nrange,samples],'analysis2')
    else:
        filename = 'analysis2d'+str(drange[0])+str(drange[-1])+'D'+str(Drange[0])+'-'+str(Drange[-1])+'N'+str(Nrange[0])+'-'+str(Nrange[-1])+'samples'+str(samples)
        if(timestamp):        
            filename = filename + str(int(time.time()))
        filename = filename + '.data'
        print('Saving to ' + filename)
        pickle.dump([plotData,drange,Drange,Nrange,samples],open(filename,'wb'))
   

def analysis3(d,D,N,n,samples,normed=True,timestamp=True,interactive=False,plot=False):
    """ Following steps are iterated for inversion block lengths n in {1,3,...,N+1} or {2,4,...,N+1}
        (depending on whether N is even or odd):
    
    i)   Generate random A in C^(d x d)_(D x D),l in C^(dxd)_D, r in C^(dxd)_D
    ii)  Compute TI matrix X as: X^(s0s1...sNs(N+1))_(t0t1...tNt(N+1)) = l^s0_t0 A^s1_t1 ... A^sN_tN l^s(N+1)_t(N+1)
             |   |   |         |   |
             l - A - A - ... - A - r
             |   |   |         |   |
    iii) Compute inverse Xi of X
    iv)  Take a block of length n:
             |   |         |         |         |   |
             l - A - ... - A - ... - A - ... - A - r
             |   |         |         |         |   |
                           \____ ____/
                                v nxA = [Block]
                           
         ... and invert all components of the MPO locally and compute their product, obtaining a fake inverse Xi_:
             |    |          || .... ||         |    |
             li - Ai - ... - [ Blocki ] - ... - Ai - ri
             |    |          || .... ||         |    |
    v)   Compute norm-difference ||XXi - XXi_||
    """
    if(n>N):
        proceedBool = input('The inversion block size you specified is larger than the maximum lattice length N. Proceed with n = N? y/n\t')
        if(proceedBool == 'y' or proceedBool == 'Y'):
            n = N
        else:
            raise Exception('Analysis aborted!')
    nrange = range(1,n+1)
    AlrSets = []
    for s in range(samples):
        A,l,r = createSet(d,D)
        AlrSets.append([A,l,r])
    if(normed):
        plotData = [[[[[] for s in range(samples)] for N_ in range(n_,N+1,2)] for a in range(2)] for n_ in range(1,n+1)]
    else:
        plotData = [[[[[] for s in range(samples)] for N_ in range(n_,N+1,2)] for a in range(1)] for n_ in range(1,n+1)]
    for s in range(samples):
        singular = True
        while(singular):
            try:
                A,l,r = AlrSets[s]
                for n_ in range(len(nrange)):
                    Nrange = range(nrange[n_],N+1,2)
                    for N_ in range(len(Nrange)):
                        X = [l] + [A]*Nrange[N_] + [r]
                        if(normed):
                            X = scaleMPO(1/normMPO(X),X)
                        Ai = np.zeros(A.shape, dtype=complex)
                        li = np.zeros(l.shape, dtype=complex)
                        ri = np.zeros(r.shape, dtype=complex)
                        block = contractMPO([A]*nrange[n_])
                        blocki = np.zeros(block.shape, dtype=complex)
                        for i in range(D):
                            li[0,:,i,:] = la.inv(l[0,:,i,:])
                            ri[i,:,0,:] = la.inv(r[i,:,0,:])
                            for j in range(D):
                                blocki[i,:,j,:] = la.inv(block[i,:,j,:])
                                Ai[i,:,j,:] = la.inv(A[i,:,j,:])
                        normXi_ = normMPO([li] + ([Ai]*((Nrange[N_]-nrange[n_])//2) + [blocki] + [Ai]*((Nrange[N_]-nrange[n_])//2)) + [ri])
                        id_ = prodMPO([l] + ([A]*((Nrange[N_]-nrange[n_])//2)) + [contractMPO([A]*nrange[n_])] + ([A]*((Nrange[N_]-nrange[n_])//2)) + [r],[li] + ([Ai]*((Nrange[N_]-nrange[n_])//2)) + [blocki] + ([Ai]*((Nrange[N_]-nrange[n_])//2)) + [ri])
                        plotData[n_][0][N_][s] = normMPO(sumMPO(identityMPO(id_),id_))
                        if(normed):
                            id_ = prodMPO([l] + ([A]*((Nrange[N_]-nrange[n_])//2)) + [contractMPO([A]*nrange[n_])] + ([A]*((Nrange[N_]-nrange[n_])//2)) + [r],scaleMPO(-1/normXi_,[li] + ([Ai]*((Nrange[N_]-nrange[n_])//2)) + [blocki] + ([Ai]*((Nrange[N_]-nrange[n_])//2)) + [ri]))
                            plotData[n_][1][N_][s] = normMPO(sumMPO(identityMPO(id_),id_))
                singular = False
            except np.linalg.linalg.LinAlgError as e:
                if 'Singular matrix' in e.message:
                    singular = True
                else:
                    raise
    if(interactive):
        plotBool = input('Analysis complete. You can plot the data now or save the data and plot it later.\n Plot now? y/n\t')
        if(plotBool == 'y'):
            plot = True
        else:
            plot = False
    if(plot):
        analysis3Plot([plotData,d,D,N,n,samples],'analysis3')
    else:
        filename = 'analysis3d'+str(d)+'D'+str(D)+'N'+str(N)+'samples'+str(samples)
        if(timestamp):        
            filename = filename + str(int(time.time()))
        filename = filename + '.data'
        print('Saving to ' + filename)
        pickle.dump([plotData,d,D,N,n,samples],open(filename,'wb'))
    
def analysis3Plot(data,filenamePrefix='analysis3'):
    """ Retrieve plot data """
    if(type(data) == str):
        print('Reading...')
        plotData,d,D,N,n,samples = pickle.load(open(data))
    else:
        plotData,d,D,N,n,samples = data
    if(len(plotData[0]) == 1):
        normed = False
    elif(len(plotData[0]) == 2):
        normed = True
    else:
        raise Exception('Something appears to be wrong with your data!')
    
    """ Plot obtained norm differences for different block inversion lengths n """  
    nrange = range(1,n+1)
    merged = plotData
    for i in range(3):
        merged = list(itertools.chain(*merged))
    pl.figure()
    pl.xlabel(r'Lattice length $N$')
    pl.xlim([0,N+1])
    pl.xticks(range(0,N+1))
    pl.ylabel(r'$\Vert E - X X^{-1} \Vert$')
    pl.ylim([10**(math.floor(math.log(min(merged),10))-1),10**(math.ceil(math.log(max(merged),10))+1)])
    pl.yscale('log',nonposy='clip')
    pl.title(r'Simulated $X^{-1}$ deviation for different inversion block sizes')
    pl.grid(b=True, which='major', color='black', linestyle='--')
    pl.grid(b=True, which='minor', color='orange', linestyle='--')
    for n_ in range(len(nrange)):
        Nrange = range(nrange[n_],N+1,2)
        y = []
        yerror = np.zeros([2,len(Nrange)])
        for N_ in range(len(Nrange)):
            y.append(np.mean(plotData[n_][0][N_]))
            yerror[:,N_] = [0,np.std(plotData[n_][0][N_])]
        pl.errorbar(Nrange,y,yerr=yerror,fmt='o',label='n=' + str(nrange[n_]))
    pl.legend(loc='best')
    if(normed):
        pl.figure()
        pl.xlabel(r'Lattice length $N$')
        pl.xlim([0,N+1])
        pl.xticks(range(0,N+1))
        pl.ylabel(r'$\Vert E - X X^{-1} \Vert$')
        pl.ylim([10**(math.floor(math.log(min(merged),10))-1),10**(math.ceil(math.log(max(merged),10))+1)])
        pl.yscale('log',nonposy='clip')
        pl.title(r'Simulated $X^{-1}$ deviation for different inversion block sizes')
        pl.grid(b=True, which='major', color='black', linestyle='--')
        pl.grid(b=True, which='minor', color='orange', linestyle='--')
        for n_ in range(len(nrange)):
            Nrange = range(nrange[n_],N+1,2)
            yNormed = []
            yerrorNormed = np.zeros([2,len(Nrange)])
            for N_ in range(len(Nrange)):
                yNormed.append(np.mean(plotData[n_][1][N_]))
                yerrorNormed[:,N_] = [0,np.std(plotData[n_][1][N_])]
            pl.errorbar(Nrange,yNormed,yerr=yerrorNormed,fmt='o',label='n=' + str(nrange[n_]))
        pl.legend(loc='best')
        pl.savefig(filenamePrefix+'norm_error_blockInversion_d'+str(d)+'D'+str(D)+'N'+str(N)+'Normed.pdf')
    else:
        pl.savefig(filenamePrefix+'norm_error_blockInversion_d'+str(d)+'D'+str(D)+'N'+str(N)+'.pdf')


def analysis4(d,D,N,samples,normed=True,timestamp=True,interactive=False,plot=False):
    """ Following steps are iterated for block lengths n in {1,3,...,N+1} or {2,4,...,N+1}
        (depending on whether N is even or odd):
        
    i)   Generate random A in C^(d x d)_(D x D),l in C^(dxd)_D, r in C^(dxd)_D
    ii)  Compute TI matrix X as: X^(s0s1...sNs(N+1))_(t0t1...tNt(N+1)) = l^s0_t0 A^s1_t1 ... A^sN_tN l^s(N+1)_t(N+1)
             |   |   |         |   |
             l - A - A - ... - A - r
             |   |   |         |   |
    iii) Compute inverse Xi of X
    iv)  Decompose Xi through repeated singular value decomposition and reshaping into an MPO of lattice length N:
             |    |    |          |      |
             M0 - M1 - M2 - ... - MN - M(N+1)
             |    |    |          |      |
    v)   Take a block of length n:
             |    |          |          |            |      |
             M0 - M1 - ... - Mi - ... - Mi+n - ... - MN - M(N+1)
             |    |          |          |            |      |
                             \____ _____/
                                  v nxA = [Block]
                                  
         ... and invert all components of the MPO locally and compute their product, obtaining a fake inverse X_:
             |     |           || .... ||         |     |
             M0i - M1i - ... - [ Blocki ] - ... - MNi - M(N+1)i
             |     |           || .... ||         |     |
    vi)  Compute norm-difference ||XiX - XiX_||
    """
    nrange = range(2-(N%2),N+1,2)
    if(normed):
        plotData = np.zeros([len(nrange),samples,2])
    else:
        plotData = np.zeros([len(nrange),samples,1])
    for s in range(samples):
        singular = True
        while(singular):
            try:
                A,l,r = createSet(d,D)
                X = contractMPO([l] + N*[A] + [r])
                if(normed):
                    X = X / norm3(X)
                Xi = la.inv(X)
                XiMPO = SVDMat(Xi,d,N,False)
                L = XiMPO[0]
                R = XiMPO[-1]
                Li = np.zeros(L.shape, dtype=complex)
                Ri = np.zeros(R.shape, dtype=complex)
                for i in range(len(Li[0,:,0])):
                        Li[:,i,:] = la.inv(L[:,i,:])
                        Ri[:,i,:] = la.inv(R[:,i,:])
                for n in range(len(nrange)):
                    block = contractMPO(XiMPO[(N-n)//2 + 1:(N+n)//2 + 1])
                    blocki = np.zeros(block.shape, dtype=complex)
                    for j in range(len(block[:,0,0,0])):
                        for k in range(len(block[0,0,:,0])):
                            blocki[j,:,k,:] = la.inv(block[j,:,k,:])
                    XiMPOiLeft = [None]*((N-nrange[n])//2)
                    XiMPOiRight = [None]*((N-nrange[n])//2)
                    """ Invert all tensors to the left of [Block] """
                    for j in range(0,(N-nrange[n])//2):
                        for k in range(len(XiMPO[j+1][:,0,0,0])):
                            for l in range(len(XiMPO[j+1][0,0,:,0])):
                                XiMPOiLeft[j] = np.zeros(XiMPO[j+1].shape, dtype=complex)
                                XiMPOiLeft[j][k,:,l,:] = la.inv(XiMPO[j+1][k,:,l,:])
                    """ Invert all tensors to the right of [Block] """
                    for j in range(0,(N-n)//2):
                        for k in range(len(XiMPO[j+1+(N+nrange[n])//2][:,0,0,0])):
                            for l in range(len(XiMPO[j+1+(N+nrange[n])//2][0,0,:,0])):
                                XiMPOiRight[j] = np.zeros(XiMPO[j+1+(N+n)//2].shape, dtype=complex)
                                XiMPOiRight[j][k,:,l,:] = la.inv(XiMPO[j+1+(N+nrange[n])//2][k,:,l,:])
                    X_ = contractMPO([Li] + (XiMPOiLeft + [blocki] + XiMPOiRight) + [Ri])
                    plotData[n,s,0] = norm3(np.dot(X,Xi) - np.dot(X_,Xi))
                    #ndiflist.append(norm3(np.dot(X,Xi) - np.dot(X_,Xi)))
                    if(normed):
                        #X = X / norm3(X)
                        X_ = X_ / norm3(X_)
                        plotData[n,s,1] = norm3(np.dot(X,Xi) - np.dot(X_,Xi))
                        #ndiflistNormed.append(norm3(np.dot(X,Xi) - np.dot(X_,Xi)))
                singular = False
            except np.linalg.linalg.LinAlgError as e:
                if 'Singular matrix' in e.message:
                    singular = True
                else:
                    raise
    if(interactive):
        plotBool = input('Analysis complete. You can plot the data now or save the data and plot it later.\n Plot now? y/n\t')
    if(plotBool == 'y'):
        plot = True
    else:
        plot = False
    if(plot):
        analysis3Plot([plotData,d,D,N,samples,nrange],'analysis3')
    else:
        filename = 'analysis4d'+str(d)+'D'+str(D)+'N'+str(N)+'samples'+str(samples)
        if(timestamp):        
            filename = filename + str(int(time.time()))
        filename = filename + '.data'
        print('Saving to ' + filename)
        pickle.dump([plotData,d,D,N,samples,nrange],open(filename,'wb'))
    
def analysis5(d,D,N,samples,normed):
    """ Similar to analysis3, but skips step v) as well as plotting and returns a list of the products XXi_ instead."""
    #ndiflist = []
    #nrange = []
    XXlist = []
    for s in range(samples):
        singular = True
        while(singular):
            try:
                A,l,r = createSet(d,D)
                X = contractMPO([l] + N*[A] + [r])
                if(normed):
                    X = X / norm3(X)
                Ai = np.zeros(A.shape, dtype=complex)
                li = np.zeros(l.shape, dtype=complex)
                ri = np.zeros(r.shape, dtype=complex)
                Xi = la.inv(X)
                for n in range(2-(N%2),N+1,2):
                    block = contractMPO([A]*n)
                    blocki = np.zeros(block.shape, dtype=complex)
                    for i in range(D):
                        li[:,i,:] = la.inv(l[:,i,:])
                        ri[:,i,:] = la.inv(r[:,i,:])
                        for j in range(D):
                            blocki[i,:,j,:] = la.inv(block[i,:,j,:])
                            Ai[i,:,j,:] = la.inv(A[i,:,j,:])
                    Xi_ = contractMPO([li] + ([Ai]*((N-n)//2) + [blocki] + [Ai]*((N-n)//2)) + [ri])
                    if(normed):
                        # X = X / norm3(X)
                        Xi_ = Xi_ / norm3(Xi_)
                    XXlist.append(np.dot(X,Xi_))
                    #ndiflist.append(norm3(np.dot(X,Xi) - np.dot(X,Xi_)))
                singular = False
                #ndiflist.append(norm3(np.dot(X,Xi) - np.dot(X,Xi)))
                #nrange.extend(range(2-(N%2),(N+3),2))
            except np.linalg.linalg.LinAlgError as e:
                if 'Singular matrix' in e.message:
                    singular = True
                else:
                    raise
    return(XXlist)


def analysis6Old(d,D,Nrange,samples,normed):
    """ Following steps are iterated for different lattice lengths N:
    i)   Generates random [WSEN]-tensor A, as well as [ESN]- and [WSN]-boundary conditions l,l' and r,r' respectively.
    ii)  Constructs TI MPOs X(l,r) and X_(l',r') of lattice length N:
            |   |   |         |   |          |    |   |         |   |
            l - A - A - ... - A - r          l' - A - A - ... - A - r'
            |   |   |         |   |          |    |   |         |   |
    iii) Computes inverse Xi of X(l,r)
    iv)  Computes norm difference ||Xi(l,r)X(l,r) - Xi(l,r)X_(l',r')||
    """
    ndiflist = []
    ndiflistNormed = []
    nrange = []
    for s in range(samples):
        A,l,r = createSet(d,D)
        l_ = rand.random([1,d,D,d]) + 1j * rand.random([1,d,D,d])
        r_ = rand.random([D,d,1,d]) + 1j * rand.random([D,d,1,d])
        for N in Nrange:
            singular = True
            while(singular):
                try:
                    X = contractMPO([l] + [A]*N + [r])
                    #X  = X / norm3(X)
                    X_ = contractMPO([l_] + [A]*N + [r_])
                    Xi = la.inv(X)
                    singular = False
                    ndiflist.append(norm3(np.dot(Xi,X) - np.dot(Xi,X_)))
                    if(normed):
                        X = X / norm3(X)
                        X_ = X_ / norm3(X_)
                        ndiflistNormed.append(norm3(np.dot(Xi,X) - np.dot(Xi,X_)))
                except np.linalg.linalg.LinAlgError as e:
                    if 'Singular matrix' in e.message:
                        singular = True
                    else:
                        raise
        nrange.extend(Nrange)
    pl.figure()
    pl.xlabel(r'Lattice length $N$')
    pl.xlim([0,Nrange[-1]+1])
    pl.xticks(range(Nrange[0]-1,Nrange[-1]+1))
    pl.ylabel(r'$\Vert X^{-1}(l,r)X(l,r) - X^{-1}(l,r)X(l\',r\') \Vert$')
    pl.semilogy()
    pl.title(r'Dependence on boundary conditions l,r for different lattice lengths $N$')
    pl.grid(b=True, which='major', color='black', linestyle='--')
    pl.grid(b=True, which='minor', color='orange', linestyle='--')
    pl.plot(nrange,ndiflist,'ro')
    if(normed):
        pl.plot(nrange,ndiflistNormed,'bo')
        pl.savefig('analysis6_d'+str(d)+'D'+str(D)+'N'+str(Nrange[0])+'-'+str(Nrange[-1])+'Normed.pdf')
    else:
        pl.savefig('analysis6_d'+str(d)+'D'+str(D)+'N'+str(Nrange[0])+'-'+str(Nrange[-1])+'.pdf')


def analysis6(d,Drange,Nrange,samples,timestamp = True,interactive=False,plot=False):
    """ Following steps are iterated for different lattice lengths N and bond dimensions D:
    i)   Generates random [WSEN]-tensor A, as well as [ESN]- and [WSN]-boundary conditions l,l' and r,r' respectively.
    ii)  Constructs TI MPOs X(l,r) and X_(l',r') of lattice length N:
            |   |   |         |   |          |    |   |         |   |
            l - A - A - ... - A - r          l' - A - A - ... - A - r'
            |   |   |         |   |          |    |   |         |   |
    iii) Computes inverse Xi of X(l,r)
    iv)  Computes norm difference ||Xi(l,r)X(l,r) - Xi(l,r)X_(l',r')||

    The analysis can contemplate two norming techniques: In the first case, only X is normed.
    In the second case, X and Xi_ are normed after computing Xi
    """
    plotData = np.zeros([len(Drange),len(Nrange),samples,2])
    for i in range(len(Drange)):
        for s in range(samples):
            A,l,r = createSet(d,Drange[i])
            l_ = rand.random([1,d,Drange[i],d]) + 1j * rand.random([1,d,Drange[i],d])
            r_ = rand.random([Drange[i],d,1,d]) + 1j * rand.random([Drange[i],d,1,d])
            for j in range(len(Nrange)):
                singular = True
                while(singular):
                    try:
                        X = [l] + [A]*Nrange[j] + [r]
                        Xnormed = scaleMPO(1/normMPO(X),X)
                        X_ = [l_] + [A]*Nrange[j] + [r_]
                        Xnormed_ = scaleMPO(-1/normMPO(X_),X_)
                        """ Norming posterior to inversion (post-norming) """
                        id_ = prodMPO(X,scaleMPO(-1,X_))
                        plotData[i,j,s,0] = normMPO(sumMPO(identityMPO(id_),id_))
                        """ Norming previous to inversion (pre-norming) """
                        #Xi = la.inv(Xnormed)
                        #plotData[i,j,s,1] = norm3(np.dot(Xi,X) - np.dot(Xi,X_))
                        id_ = prodMPO(Xnormed,Xnormed_)
                        plotData[i,j,s,1] = normMPO(sumMPO(identityMPO(id_),id_))
                        singular = False
                    except np.linalg.linalg.LinAlgError as e:
                        if 'Singular matrix' in e.message:
                            singular = True
                        else:
                            raise
    if(interactive):
        plotBool = input('Analysis complete. You can plot the data now or save the data and plot it later.\n Plot now? y/n\t')
        if(plotBool == 'y'):
            plot = True
        else:
            plot = False
    if(plot):
        analysis6Plot([plotData,d,Drange,Nrange,samples])
    else:
        filename = 'analysis6_d'+str(d)+'D'+str(Drange[0])+'-'+str(Drange[-1])+'N'+str(Nrange[0])+'-'+str(Nrange[-1])
        if(timestamp):        
            filename = filename + str(int(time.time()))
        filename = filename + '.data'
        print('Saving to ' + filename)
        pickle.dump([plotData,d,Drange,Nrange,samples],open(filename,'wb'))
    
def analysis6Plot(data):
    """ Can be called either with a string, which will be interpreted as a filename containing the data to be plotted, 
        or with a np.array - object with dimensions [len(Drange),len(Nrange),samples] """
    if(type(data) == str):
        print('Reading...')
        plotData,d,Drange,Nrange,samples = pickle.load(open(data))
        if(len(plotData[:,0,0]) != len(Drange)):
            raise Exception('The bond dimension ranges does not match the plot data!')
        if(len(plotData[0,:,0]) != len(Nrange)):
            raise Exception('The lattice length range does not match the plot data!')
        if(len(plotData[0,0,:]) != samples):
            raise Exception('The sample size does not match the plot data!')
        print('Plot data read. Plotting...')
    else:
        plotData,d,Drange,Nrange,samples = data
    for normingtype in range(2):       
        """ Prepare plot """
        pl.figure()
        pl.xlabel(r'Lattice length $N$')
        pl.xlim([0,Nrange[-1]+1])
        pl.ylim([10**(math.floor(math.log(min(plotData[:,:,:,normingtype].flatten()),10))-1),10**math.ceil(math.log(max(plotData[:,:,:,normingtype].flatten()),10))])
        pl.xticks(range(Nrange[0]-1,Nrange[-1]+1))
        pl.ylabel(r'$\Vert X^{-1}(l,r)X(l,r) - X^{-1}(l,r)X(l \',r \') \Vert$')
        pl.yscale('log', nonposy='clip')
        pl.title(r'Dependence on boundary conditions l,r for different lattice lengths $N$')
        pl.grid(b=True, which='major', color='black', linestyle='--')
        pl.grid(b=True, which='minor', color='orange', linestyle='--')
        for i in range(len(Drange)):
            y = []
            yerror = np.zeros([2,len(Nrange)])
            for j in range(len(Nrange)):
                    y.append(np.mean(plotData[i,j,:,normingtype]))
                    yerror[:,j] = [0,np.std(plotData[i,j,:,normingtype])]
            pl.errorbar(Nrange,y,yerr=yerror,fmt='o',label='D='+str(Drange[i]),capthick=2)
        pl.legend(loc='best')
        if(normingtype == 0):
            pl.savefig('analysis6_d'+str(d)+'D'+str(Drange[0])+'-'+str(Drange[-1])+'N'+str(Nrange[0])+'-'+str(Nrange[-1])+'postnormed.pdf')
        else:
            pl.savefig('analysis6_d'+str(d)+'D'+str(Drange[0])+'-'+str(Drange[-1])+'N'+str(Nrange[0])+'-'+str(Nrange[-1])+'prenormed.pdf')

#TODO: Change to incorporate both normingtypes 
def analysis7(d,Drange,blockrange,Nrange,samples,timestamp = True,interactive=False,plot=False):
    """
    i)   Generate random A in C^(d x d)_(D x D),l in C^(dxd)_D, r in C^(dxd)_D
    ii)  Compute TI matrix X as: X^(s0s1...sNs(N+1))_(t0t1...tNt(N+1)) = l^s0_t0 A^s1_t1 ... A^sN_tN l^s(N+1)_t(N+1)
             |   |   |         |   |
             l - A - A - ... - A - r
             |   |   |         |   |
    iii) Compute inverse Xi of X
    iv)  Create a block of length n:
               |         |
             - A - ... - A -
               |         |
               \____ ____/
                    v nxA = -[Block]-
                           
         ... invert all components of the MPO locally and compute a TIMPO as product of [Blocki], obtaining a fake inverse Xi_:
             |    || .... ||         || .... ||         || .... ||   |
             li - [ Blocki ] - ... - [ Blocki ] - ... - [ Blocki ] - ri
             |    || .... ||         || .... ||         || .... ||   |
    v)   Compute norm-difference ||XiX - XiX_||
    """
    if(blockrange[-1] >= Nrange[-1]//2):
        continueBool = input('The block range and lattice length range you have selected allows for only two data points for the last block size. Do you want to proceed anyway? y/n\t')
        if(continueBool != 'y'):
            raise Exception('Analysis cancelled!')
    """ Create list with A,l,r for each sample and dimension D """
    AlrSets = []
    for D in Drange:
        AlrSetD = []
        for s in range(samples):
            A,l,r = createSet(d,D)
            AlrSetD.append([A,l,r])
        AlrSets.append(AlrSetD)
    """ Iterate over all blockrange sizes """
    plotData = np.zeros([len(blockrange),len(Drange),len(Nrange),samples,2])
    for n in range(len(blockrange)):
        for i in range(len(Drange)):
            for N in range(len(Nrange)):
                singular = True
                while(singular):
                    try:
                        if(Nrange[N]%blockrange[n] == 0):
                            for s in range(samples):
                                A,l,r = AlrSets[i][s]
                                block = contractMPO([A]*blockrange[n])
                                #X = [l] + [A]*Nrange[N] + [r]
                                #X = X / norm3(X)
                                li = np.zeros(l.shape, dtype=complex)
                                ri = np.zeros(r.shape, dtype=complex)
                                blocki = np.zeros(block.shape, dtype=complex)
                                for j in range(Drange[i]):
                                    li[0,:,j,:] = la.inv(l[0,:,j,:])
                                    ri[j,:,0,:] = la.inv(r[j,:,0,:])
                                    for k in range(Drange[i]):
                                        blocki[j,:,k,:] = la.inv(block[j,:,k,:])
                                X = [l] + [block]*(Nrange[N]//blockrange[n]) + [r]
                                Xi_ = [li] + [blocki]*(Nrange[N]//blockrange[n]) + [ri]
                                id_ = prodMPO(X,scaleMPO(-1,Xi_))
                                plotData[n,i,N,s,0] = normMPO(sumMPO(identityMPO(id_),id_))
                                id_ = scaleMPO(1/normMPO(Xi_),id_)
                                plotData[n,i,N,s,1] = normMPO(sumMPO(identityMPO(id_),id_))
                        singular = False
                    except np.linalg.linalg.LinAlgError as e:
                            if 'Singular matrix' in e.message:
                                A,l,r = createSet(d,Drange[i])
                                AlrSets[i][s] = [A,l,r]
                                singular = True
                            else:
                                raise
    if(interactive):
        plotBool = input('Analysis complete. You can plot the data now or save the data and plot it later.\n Plot now? y/n\t')
        if(plotBool == 'y'):
            plot = True
        else:
            plot = False
    if(plot):
        analysis7Plot([plotData,d,Drange,blockrange,Nrange,samples])
    else:
        filename = 'analysis7d'+str(d)+'D'+str(Drange[0])+'-'+str(Drange[-1])+'blocksize'+str(blockrange[0])+'-'+str(blockrange[-1])+'N'+str(Nrange[0])+'-'+str(Nrange[-1])+'samples'+str(samples)
        if(timestamp):        
            filename = filename + str(int(time.time()))
        filename = filename + '.data'
        print('Saving to ' + filename)
        pickle.dump([plotData,d,Drange,blockrange,Nrange,samples],open(filename,'wb'))

def analysis7Plot(data):
    """ Can be called either with a string, which will be interpreted as a filename containing the data to be plotted, 
        or with a np.array - object with dimensions [len(Drange),len(Nrange),samples] """
    if(type(data) == str):
        print('Reading...')
        plotData,d,Drange,blockrange,Nrange,samples = pickle.load(open(data))
        if(len(plotData[:,0,0,0,0]) != len(blockrange)):
            raise Exception('The block ranges does not match the plot data!')
        if(len(plotData[0,:,0,0,0]) != len(Drange)):
            raise Exception('The bond dimension range does not match the plot data!')
        if(len(plotData[0,0,:,0,0]) != len(Nrange)):
            raise Exception('The lattice length range does not match the plot data!')
        if(len(plotData[0,0,0,:,0]) != samples):
            raise Exception('The sample size does not match the plot data!')
        print('Plot data read. Plotting...')
    else:
        plotData,d,Drange,blockrange,Nrange,samples = data
    for normingtype in range(2):
        for n in range(len(blockrange)):
            """ Prepare plot """
            pl.figure()
            pl.xlabel(r'Lattice length $N$')
            pl.xlim([0,Nrange[-1]+1])
            pl.xticks(range(Nrange[0]-1,Nrange[-1]+1))
            pl.ylabel(r'$\Vert E - X X^{-1} \Vert$')
            pl.yscale('log', nonposy='clip')
            pl.title(r'Simulated $X^{-1}$ deviation dependence on lattice length $N$ and bond dimension $D$ for inversion block size '+str(blockrange[n]))
            pl.grid(b=True, which='major', color='black', linestyle='--')
            pl.grid(b=True, which='minor', color='orange', linestyle='--')
            for D in range(len(Drange)):
                y = []
                yerror = np.zeros([2,len(Nrange)])
                for N in range(len(Nrange)):
                    y.append(np.mean(plotData[n,D,N,:,normingtype]))
                    yerror[:,N] = [0,np.std(plotData[n,D,N,:,normingtype])]
                pl.errorbar(Nrange,y,yerr=yerror,fmt='o',label='D='+str(Drange[D]),capthick=2)
            pl.legend(loc='best')
            if(normingtype == 0):
                pl.savefig('analysis7_d'+str(d)+'D'+str(Drange[0])+'-'+str(Drange[-1])+'blocksize'+str(blockrange[n])+'N'+str(Nrange[0])+'-'+str(Nrange[-1])+'.pdf')
            else:
                pl.savefig('analysis7_d'+str(d)+'D'+str(Drange[0])+'-'+str(Drange[-1])+'blocksize'+str(blockrange[n])+'N'+str(Nrange[0])+'-'+str(Nrange[-1])+'Xi_normed.pdf')

def analysis8(drange,Drange,N,samples,timestamp = True,interactive=False,plot=False):
    """
    The following steps are iterated for 
        -different physical dimensions d
        -different bond dimensions D
        -different block lengths n, where n < N//2.
    i)   Generate random A in C^(d x d)_(D x D),l in C^(dxd)_D, r in C^(dxd)_D
    ii)  Compute TI matrix X as: X^(s0s1...sNs(N+1))_(t0t1...tNt(N+1)) = l^s0_t0 A^s1_t1 ... A^sN_tN l^s(N+1)_t(N+1)
             |   |   |         |   |
             l - A - A - ... - A - r
             |   |   |         |   |
    iii) Compute inverse Xi of X
    iv)  Create left and right blocks of length n:
             |   |   |         |            |         |   |   |
             l - A - A - ... - A -        - A - ... - A - A - r
             |   |   |         |            |         |   |   |
             \________ ________/            \________ ________/
                      v n tensors                   v n tensors
                        || ... |                      || ... |
                        [Blockl]                      [Blockr]
                        || ... |                      || ... |
                        
        ... invert all components of the tensor network locally and collapse to MPO, obtaining a fake inverse Xi_
              || ...  |   |          |    || ...  |
        Xi_ = [Blockli] - Ai - ... - Ai - [Blockri]
              || ...  |   |          |    || ...  |
        
    v)   Compute norm-difference ||XXi - XXi_||"""
    nrange = range(0,N//2)
    for d in drange:
        if d <= 1:
            raise Exception('d has to be greater than 1!')
    plotData = np.zeros([len(drange),len(Drange),len(nrange),samples])
    """ Actual analysis """
    for d in range(len(drange)):
        for D in range(len(Drange)):
            for s in range(samples):
                singular = True
                while(singular):
                    try:
                        A,l,r = createSet(drange[d],Drange[D])
                        for n in range(len(nrange)):
                            if(n == 0):
                                 blockl = l
                                 blockr = r
                            else:
                                blockl = contractMPO([l] + contractMPO([A]*nrange[n]))
                                blockr = contractMPO([A]*nrange[n] + [r])
                            """ Invert all blocks """
                            Ai = np.zeros(A.shape,dtype=complex)
                            blockli = np.zeros(blockl.shape, dtype=complex)
                            blockri = np.zeros(blockr.shape, dtype=complex)
                            for i in range(Drange[D]):
                                blockli[0,:,i,:] = la.inv(blockl[0,:,i,:])
                                blockri[i,:,0,:] = la.inv(blockr[i,:,0,:])
                                for j in range(Drange[D]):
                                    Ai[i,:,j,:] = la.inv(A[i,:,j,:])
                            X = [blockl] + [A]*(N-2*nrange[n]) + [blockr]
                            X = scaleMPO(1/normMPO(X),X)
                            Xi_ = [blockli] + [Ai]*(N-2*nrange[n]) + [blockri]
                            Xi_ = scaleMPO(-1/normMPO(Xi_),Xi_)
                            id_ = prodMPO(X,Xi_)
                            plotData[d,D,n,s] = normMPO(sumMPO(identityMPO(id_) + id_))
                        singular = False
                    except np.linalg.linalg.LinAlgError as e:
                            if 'Singular matrix' in e.message:
                                singular = True
                            else:
                                raise
    if(interactive):
        plotBool = input('Analysis complete. You can plot the data now or save the data and plot it later.\n Plot now? y/n\t')
        if(plotBool == 'y'):
            plot = True
        else:
            plot = False
    if(plot):
        analysis8Plot([plotData,drange,Drange,N,samples])
    else:
        filename = 'analysis8d'+str(drange[0])+'-'+str(drange[-1])+'D'+str(Drange[0])+'-'+str(Drange[-1])+'N'+str(N)+'samples'+str(samples)
        if(timestamp):        
            filename = filename + str(int(time.time()))
        filename = filename + '.data'
        print('Saving to ' + filename)
        pickle.dump([plotData,drange,Drange,N,samples],open(filename,'wb'))
        
def analysis8Plot(data):
    """ Can be called either with a string, which will be interpreted as a filename containing the data to be plotted, 
        or with a np.array - object with dimensions [len(Drange),len(Nrange),samples] """
    if(type(data) == str):
        print('Reading...')
        plotData,drange,Drange,N,samples = pickle.load(open(data))
        if(len(plotData[:,0,0]) != len(drange)):
            raise Exception('The physical dimension range does not match the plot data!')
        if(len(plotData[0,:,0]) != len(Drange)):
            raise Exception('The bond dimension range does not match the plot data!')
        if(len(plotData[0,0,:]) != samples):
            raise Exception('The sample size does not match the plot data!')
        print('Plot data read. Plotting...')
    else:
        plotData,drange,Drange,N,samples = data
    """ Fit results with linear model (on logarithmic scale) """
    nrange = range(0,N//2)
    for d in range(len(drange)):
        for D in range(len(Drange)):
            #params1, cov1 = fit(fitmodel,Nrange*samples,map(lambda x: math.log(x),ndiflist))
            pl.figure()
            pl.xlabel(r'Inversion block size $n$')
            pl.xlim([nrange[0],nrange[-1]+1])
            #if(normed):
            #    pl.ylim([10**(math.floor(math.log(min([min(plotData[d,D,:,:,0].flatten()),min(plotData[d,D,:,:,1].flatten())]),10))-1),10**(math.ceil(math.log(max([max(plotData[d,D,:,:,0].flatten()),max(plotData[d,D,:,:,1].flatten())]),10))+1)])
            #else:
            pl.ylim([10**(math.floor(math.log(min(plotData[d,D,:,:].flatten()),10))-1),10**(math.ceil(math.log(max(plotData[d,D,:,:].flatten()),10))+1)])
            pl.xticks(range(nrange[0]-1,nrange[-1]+1))
            pl.ylabel(r'$\Vert E - X X^{-1} \Vert$')
            pl.yscale('log', nonposy='clip')
            pl.title(r'Simulated $X^{-1}$ deviation as a function of left and right inversion block size $n$')
            pl.grid(b=True, which='major', color='black', linestyle='--')
            pl.grid(b=True, which='minor', color='orange', linestyle='--')
            y = []
            yerror = np.zeros([2,len(nrange)])
            for n in range(len(nrange)):
                y.append(np.mean(plotData[d,D,n,:]))
                yerror[:,n] = [0,np.std(plotData[d,D,n,:])]
            #pl.plot(Nrange,map(lambda x: math.exp(fitmodel(x,params1[0],params1[1])), Nrange),'-r')
            pl.errorbar(nrange,y,yerr=yerror,fmt='o',label=r'',capthick=2)
        pl.savefig('analysis8_d'+str(d)+'D'+str(D)+'s'+str(samples)+'.pdf')
            
#TODO: Change to incorporate both normingtypes 
def analysis9(drange,Drange,Nrange,samples,analysis10 = False,timestamp = True,interactive=False,plot=False):
    """
    i)   Generate random A in C^(d x d)_(D x D),l in C^(dxd)_D, r in C^(dxd)_D
    ii)  Compute TI matrix X as: X^(s0s1...sNs(N+1))_(t0t1...tNt(N+1)) = l^s0_t0 A^s1_t1 ... A^sN_tN l^s(N+1)_t(N+1)
             |   |   |         |   |
             l - A - A - ... - A - r
             |   |   |         |   |
    iii) Compute inverse Xi of X
    iv)  Decompose X through repeated singular value decomposition and reshaping into an MPO of lattice length N:
             |    |    |          |      |
             M0 - M1 - M2 - ... - MN - M(N+1)
             |    |    |          |      |
    v)   Take a block of length n:
             |    |          |          |            |      |
             M0 - M1 - ... - Mi - ... - Mi+n - ... - MN - M(N+1)
             |    |          |          |            |      |
                             \____ _____/
                                  v nxA = [Block]
                           
         ... invert [Block] locally, obtaining [Blocki] and expand so as to yield an MPO Xi_
             |    || .... ||         || .... ||         || .... ||   |
             li - [ Blocki ] - ... - [ Blocki ] - ... - [ Blocki ] - ri
             |    || .... ||         || .... ||         || .... ||   |
    v)   Compute norm-difference ||XXi - XXi_||
    
    Remark: Analysis 10 is identical to analysis9 except for the fact that no X is generated for a succeeding SVD-decomposition.
            Instead, an MPO with dimensions emulating those of an SVD-decomposed MPO is generated to avoid a costly SVD-decomposition.
    """
    AlrSets = []
    for d in drange:
        AlrSetd = []
        for D in Drange:
            AlrSetD = []
            for s in range(samples):
                A,l,r = createSet(d,D)
                AlrSetD.append([A,l,r])
            AlrSetd.append(AlrSetD)
        AlrSets.append(AlrSetd)
    plotData = [[[[] for N in range(len(Nrange))] for D in range(len(Drange))] for d in range(len(drange))]
    """ Iterate over all blockrange sizes """
    for d in range(len(drange)):
        for D in range(len(Drange)):
            singular = True
            while(singular):
                try:
                    for N in range(len(Nrange)):
                        nrange = range(1,Nrange[N] + 1)
                        if(not(nrange == [])):
                            nstable = [[] for n in range(len(nrange))]
                            for s in range(samples):
                                if(not(analysis10)):
                                    A,l,r = AlrSets[d][D][s]
                                    X = [l] + [A]*Nrange[N] + [r]
                                    X = contractMPO(scaleMPO(1/normMPO(X),X))[0,:,0,:]
                                    XMPO = SVDMat(X,drange[d],Nrange[N],False)
                                else:
                                    lpart = list(map(lambda x:rand.random([Drange[D]**(2*x),drange[d],Drange[D]**(2*(x+1)),drange[d]]) + 1j* rand.random([Drange[D]**(2*x),drange[d],Drange[D]**(2*(x+1)),drange[d]]),range(Nrange[N]//2 + 1)))
                                    mpart = [rand.random([Drange[D]**(2*(Nrange[N]//2 + 1)),drange[d],Drange[D]**(2*(Nrange[N]//2 + 1)),drange[d]]) + 1j* rand.random([Drange[D]**(2*(Nrange[N]//2 + 1)),drange[d],Drange[D]**(2*(Nrange[N]//2 + 1)),drange[d]])]*(Nrange[N]%2)
                                    rpart = list(map(lambda x: rand.random([Drange[D]**(2*(x+1)),drange[d],Drange[D]**(2*x),drange[d]]) + 1j* rand.random([Drange[D]**(2*(x+1)),drange[d],Drange[D]**(2*x),drange[d]]),range(Nrange[N]//2,-1,-1)))
                                    XMPO = lpart + mpart + rpart
                                L = XMPO[0]
                                Li = np.zeros(L.shape,dtype=complex)
                                R = XMPO[-1]
                                Ri = np.zeros(R.shape,dtype=complex)
                                for n in range(len(nrange)):
                                    if((Nrange[N]%nrange[n] == 0 or nrange[n] == 1) and (Nrange[N]%2 == nrange[n]%2)):
                                        pos1 = (Nrange[N]-nrange[n])//2 + 1
                                        pos2 = (Nrange[N]+nrange[n])//2 + 1
                                        block = contractMPO(XMPO[pos1:pos2])
                                        blocki = np.zeros(block.shape, dtype=complex)
                                        for i in range(Drange[D]):
                                            Li[0,:,i,:] = la.inv(L[0,:,i,:])
                                            Ri[i,:,0,:] = la.inv(R[i,:,0,:])
                                            for j in range(Drange[D]):
                                                blocki[i,:,j,:] = la.inv(block[i,:,j,:])
                                        Mlist = [Li,blocki,Ri]
                                        locexpMPO(Mlist)
                                        Xi_ = [Mlist[0]] + [Mlist[1]]*(Nrange[N]//nrange[n]) + [Mlist[2]]
                                        """ Partially contract X to allow for vertical contraction of the product of X, Xi_"""
                                        X = [L]
                                        for i in range(Nrange[N]//nrange[n]):
                                            X.append(contractMPO(XMPO[i+1:i+1+nrange[n]]))
                                        X.append(R)
                                        """Form product"""
                                        id_ = prodMPO(X,scaleMPO(-1,Xi_))
                                        nstable[n].append(normMPO(sumMPO(identityMPO(id_),id_)))
                                plotData[d][D][N] = nstable
                                singular = False
                except np.linalg.linalg.LinAlgError as e:
                        if 'Singular matrix' in e.message:
                            singular = True
                            if(not(analysis10)):
                                A,l,r = createSet(drange[d],Drange[D])
                                AlrSets[d][D][s] = [A,l,r]
                        else:
                            raise
    if(interactive):
        plotBool = input('Analysis complete. You can plot the data now or save the data and plot it later.\n Plot now? y/n\t')
        if(plotBool == 'y'):
            plot = True
        else:
            plot = False
    if(plot):
        analysis9Plot([plotData,drange,Drange,Nrange,samples])
    else:
        filename = 'analysis9d'+str(drange[0])+'-'+str(drange[-1])+'D'+str(Drange[0])+'-'+str(Drange[-1])+'N'+str(Nrange[0])+'-'+str(Nrange[-1])+'samples'+str(samples)
        if(timestamp):        
            filename = filename + str(int(time.time()))
        filename = filename + '.data'
        print('Saving to ' + filename)
        pickle.dump([plotData,drange,Drange,Nrange,samples],open(filename,'wb'))

def analysis9Plot(data):
    """ Can be called either with a string, which will be interpreted as a filename containing the data to be plotted, 
        or with a np.array - object with dimensions [len(Drange),len(Nrange),samples] """
    if(type(data) == str):
        print('Reading...')
        plotData,drange,Drange,Nrange,samples = pickle.load(open(data))
        if(len(plotData[:][0][0][0]) != len(drange)):
            raise Exception('The physical dimension range does not match the plot data!')
        if(len(plotData[0][:][0][0]) != len(Drange)):
            raise Exception('The bond dimension range does not match the plot data!')
        if(len(plotData[0][0][:][0]) != len(Nrange)):
            raise Exception('The lattice length range does not match the plot data!')
        if(len(plotData[0][0][0][:]) != samples):
            raise Exception('The sample size does not match the plot data!')
        print('Plot data read. Plotting...')
    else:
        plotData,drange,Drange,Nrange,samples = data
    """ Fit results with linear model (on logarithmic scale) """
    for d in range(len(drange)):
        #params1, cov1 = fit(fitmodel,Nrange*samples,map(lambda x: math.log(x),ndiflist))
        pl.figure()
        pl.xlabel(r'Inversion block size $n$')
        pl.xlim([Nrange[0],Nrange[-1]+1])
        #if(normed):
        #    pl.ylim([10**(math.floor(math.log(min([min(plotData[d,D,:,:,0].flatten()),min(plotData[d,D,:,:,1].flatten())]),10))-1),10**(math.ceil(math.log(max([max(plotData[d,D,:,:,0].flatten()),max(plotData[d,D,:,:,1].flatten())]),10))+1)])
        #else:
        merged = plotData
        for i in range(4):
            merged = list(itertools.chain(*merged))
        pl.ylim([10**(math.floor(math.log(min(merged),10))-1),10**(math.ceil(math.log(max(merged),10))+1)])
        pl.xticks(range(Nrange[0]-1,Nrange[-1]+1))
        pl.ylabel(r'$\Vert E - X X^{-1} \Vert$')
        pl.yscale('log', nonposy='clip')
        pl.title(r'Simulated $X^{-1}$ deviation as a function of inversion block size $n$ for d = ' + str(drange[d]))
        pl.grid(b=True, which='major', color='black', linestyle='--')
        pl.grid(b=True, which='minor', color='orange', linestyle='--')
        for D in range(len(Drange)):
            for N in range(len(Nrange)):
                y = []
                yerror = []
                nrange = range(1,Nrange[N]+1)
                nvals = []
                for n in range(len(nrange)):
                    if(not(plotData[d][D][N][n] == [])):    
                        y.append(np.mean(plotData[d][D][N][n]))
                        yerror.append(np.std(plotData[d][D][N][n]))
                        nvals.append(nrange[n])
                pl.errorbar(nvals,y,yerr=yerror,fmt='o',label='(D,N) = ('+str(Drange[D]) + ',' + str(Nrange[N]) + ')',capthick=2)
        pl.legend(loc='best')
        pl.savefig('analysis9_d'+str(drange[0])+'-'+str(drange[-1])+'D'+str(Drange[0])+'-'+str(Drange[-1])+'s'+str(samples)+'.pdf')

def analysis10(drange,Drange,Nrange,samples,timestamp = True,interactive=False,plot=False):
    analysis9(drange,Drange,Nrange,samples,True,timestamp,interactive,plot)
    
def analysis11(drange,Drange,N,block,samples,timestamp = True,interactive=False,plot=False):
    """
    i)   Generate random A in C^(d x d)_(D x D),l in C^(dxd)_D, r in C^(dxd)_D
    ii)  Compute TI matrix X as: X^(s0s1...sNs(N+1))_(t0t1...tNt(N+1)) = l^s0_t0 A^s1_t1 ... A^sN_tN l^s(N+1)_t(N+1)
         (Chain length corresponds to VAR[block])
             |   |   |         |   |
             l - A - A - ... - A - r
             |   |   |         |   |
                 \______ ______/
                        v block x A
                        
    iii)  Contract to matrix X and decompose through repeated SVD:
             |    |             |              |     |
             M1 - M2 - ... - M(N/2) - ... - M(N-1) - MN
             |    |             |              |     |   
             
    iv)   Calculate Matrix X_N over N sites with N>block as:
             |   |   |         |   |
             l - A - A - ... - A - r
             |   |   |         |   |
                 \______ ______/
                        v N x A
                        
          ...and the approximation X_N' as:
             |    |             |       |              |      |
             M1 - M2 - ... - M(N/2) - M(N/2) - ... - M(N-1) - MN
             |    |             |       |              |      |   
        
          ... and extend MPO size by one site inserting tensor M(N/2) in the middle of X_N' for every iteration step
          X_N -> X_(N+1)
    v)    Calculate norm differences ||X_N - X_N'||
    """
    if(block>N):
        raise Exception('N has to be greater than blocksize')
    Nrange = range(block+1,N+1)
    plotData = np.zeros([len(drange),len(Drange),len(Nrange),samples])
    count = 0
    for d in range(len(drange)):
        for D in range(len(Drange)):
            for s in range(samples):
                A,l,r = createSet(drange[d],Drange[D])
                X = contractMPO([l] + [A]*block + [r])[0,:,0,:]
                XMPO = SVDMat(X,drange[d],block,True)
                L = XMPO[0:len(XMPO)//2 - (1-len(XMPO)%2)]
                M = XMPO[len(XMPO)//2 - (1-len(XMPO)%2):len(XMPO)//2 + 1]
                R = XMPO[len(XMPO)//2 + 1:]
                for N_ in range(len(Nrange)):
                    XMPO_N = [l] + [A]*Nrange[N_] + [r]
                    XMPO_N_ = L + M*(N_+2) + R
                    plotData[d,D,N_,s] = normMPO(sumMPO(XMPO_N,scaleMPO(-1,XMPO_N_)))#/normMPO(XMPO_N)
                    count += 1
                    sys.stdout.write("\rProgress: "+str(count)+"/"+str(len(drange)*len(Drange)*(N-block)*samples)+"norms calculated.")
    if(interactive):
        plotBool = input('Analysis complete. You can plot the data now or save the data and plot it later.\n Plot now? y/n\t')
        if(plotBool == 'y'):
            plot = True
        else:
            plot = False
    if(plot):
        analysis11Plot([plotData,drange,Drange,N,block,samples])
    filename = 'analysis11d'+str(drange[0])+'-'+str(drange[-1])+'D'+str(Drange[0])+'-'+str(Drange[-1])+'N'+str(Nrange[0])+'-'+str(Nrange[-1])+'samples'+str(samples)
    if(timestamp):
        filename = filename + str(int(time.time()))
    filename = filename + '.data'
    print('Saving to ' + filename)
    pickle.dump([plotData,drange,Drange,N,block,samples],open(filename,'wb'))
        
def analysis11Plot(data,filenamePrefix='analysis11'):
    """ Can be called either with a string, which will be interpreted as a filename containing the data to be plotted, 
        or with a np.array - object with dimensions [len(Drange),len(Nrange),samples] """
    if(type(data) == str):
        print('Reading...')
        plotData,drange,Drange,N,block,samples = pickle.load(open(data))
        if(len(plotData[:][0]) != len(range(block,N+1))):
            raise Exception('The lattice lengths of the data do not match!')
        if(len(plotData[0][:]) != samples):
            raise Exception('The sample size does not match the plot data!')
        print('Plot data read. Plotting...')
    else:
        plotData,drange,Drange,N,block,samples = data
    Nrange = range(block+1,N+1)
    
        #params1, cov1 = fit(fitmodel,Nrange*samples,map(lambda x: math.log(x),ndiflist))
    pl.figure()
    pl.xlabel(r'Lattice length $N$')
    pl.xlim([Nrange[0]-1,Nrange[-1]+1])
    #if(normed):
    #    pl.ylim([10**(math.floor(math.log(min([min(plotData[d,D,:,:,0].flatten()),min(plotData[d,D,:,:,1].flatten())]),10))-1),10**(math.ceil(math.log(max([max(plotData[d,D,:,:,0].flatten()),max(plotData[d,D,:,:,1].flatten())]),10))+1)])
    #else:
    pl.ylim([10**(math.floor(math.log(min(list(filter(lambda x:x!=0,plotData.flatten()))),10))-1),10**(math.ceil(math.log(max(list(filter(lambda x:x!=0,plotData.flatten()))),10))+1)])
    pl.yticks([0,10^4])
    #pl.xticks(range(Nrange[0]-1,Nrange[-1]+1))
    #pl.ylabel(r'$\Vert X_N - \hat{X}_N \Vert$')
    pl.ylabel(r'$\Vert X_{N+j}\hat{X}^{-1}_{N+j} - E\Vert$')
    pl.yscale('log', nonposy='clip')
    pl.title(r'Error of approximated $X^{-1}$')
    pl.grid(b=True, which='major', color='black', linestyle='--')
    pl.grid(b=True, which='minor', color='orange', linestyle='--')
    for d in range(len(drange)):
        for D in range(len(Drange)):
            y = []
            yerror = np.zeros([2,len(Nrange)])
            for N in range(len(Nrange)):
                x = list(map(lambda x:x+remap(D,d,len(drange))/(len(drange)*len(Drange)+1),Nrange))
                y.append(np.mean(plotData[d,D,N,:]))
                yerror[:,N] = [np.std(list(filter(lambda x: x <= y[-1],plotData[d,D,N,:]))),np.std(list(filter(lambda x: x > y[-1],plotData[d,D,N,:])))]
            #pl.errorbar(Nrange,y,yerr=yerror,fmt='o',label='(D,d) = ('+str(drange[d]) + ',' + str(Drange[D]) + ')',capthick=2)
            pl.plot(x,y,'-o',label='(d,D) = ('+str(drange[d]) + ',' + str(Drange[D]) + ')')
    pl.legend(loc='best')
    pl.savefig(filenamePrefix + '_d'+str(drange[0])+'-'+str(drange[-1])+'D'+str(Drange[0])+'-'+str(Drange[-1])+'N'+str(block)+'-'+str(Nrange[0])+'-'+str(Nrange[-1])+'s'+str(samples)+'.pdf')


def analysis12(dDlist,N,block=1,samples=1,imaginary=True,direction=1,normalize=True,norm_mode=0,pre_norm=False,post_norm=False,seed=None,timestamp = True,interactive=False,plot=True):
    """ Description:
        ----------------------------------------------------------------------------------------------------------------------------------------------
        
        i)   Generate random A in C^(d x d)_(D x D),l in C^(dxd)_D, r in C^(dxd)_D
        ii)  Compute TI matrix X as: X^(s0s1...sNs(N+1))_(t0t1...tNt(N+1)) = l^s0_t0 A^s1_t1 ... A^sN_tN l^s(N+1)_t(N+1)
             (Chain length corresponds to VAR[block])
                       |   |   |         |   |
                 X_N = l - A - A - ... - A - r
                       |   |   |         |   |
                           \______ ______/
                                  v block N x A
                            
        iii)  Contract to matrix X, invert and decompose inverse Xi through repeated SVD:
                        |    |             |              |     |
                 Xi_N = M1 - M2 - ... - M(N/2) - ... - M(N-1) - MN
                        |    |             |              |     |   
                 
        iv)   Calculate Matrix X_(N+j) as:
                         |   |   |         |   |
                 X_(N+j) l - A - A - ... - A - r
                         |   |   |         |   |
                             \______ ______/
                                    v (N+j) x A
                            
              ...and the approximated inverse as:
                            |    |             |             |              |      |
                 Xi_(N+j) = M1 - M2 - ... - M(N/2) - ... - M(N/2) - ... - M(N-1) - MN
                            |    |             |             |              |      |   
                                               \______ ______/
                                                      v (j+1) x M(N/2)
        
        v)    Calculate norm differences ||X_(N+j)Xi_(N+j) - E||
        
        Parameters:
        ----------------------------------------------------------------------------------------------------------------------------------------------
        drange:         
        Drange: 
        N:              Natural number > 0 specifying the maximum number of sites (N+2) in the system.
        block:          Natural number >= 0 specifying the number of sites (block+3) for which the equation system is solved in every iteration step.        
        samples:        Number of different initial A,l,r samples to be generated.
        imaginary:      If set to False, will generate purely real-valued tensors A,l,r instead of complex-valued ones.
        direction:      Direction of iterated SVD decomposition of inverse MPO Xi.
        normalize:      If set to False, the tensors A,l,r will not be normalized.
        norm_mode:      Norm used for calculation of the approximation error
                            0: Frobenius norm      1: max norm
        pre_norm:       Use a local renormalization of the center tensor
        post_norm:      Use a posterior renormalization, dividing by the mean diagonal entries of a subset of the contracted MPO
        seed:           Provides and integer seed for the generation of random tensors
        timestamp:      Appends a timestamp to the name of the files generated.
        interactive:    If set to True, will ask user how to proceed with the data after the analysis has run.
        plot:           If set to False, will not draw plots and simply save the data retrieved.
    """
    for dD in dDlist:
        if(not(len(dD) == 2)):
            raise Exception("Tuples of the type (D,d) are expected.")
        if(not(isinstance(dD[0],int) and isinstance(dD[1],int)) or dD[0] < 1 or dD[1] < 1):
            raise Exception("D,d have to be integers greater than 0.")
    
    #if(block>N):
    #    raise Exception('N has to be greater than blocksize')
    Nrange = range(1,N+1)
    plotData = np.zeros([len(dDlist),len(Nrange),samples])
    count = 0
    #check = []
    for dD in range(len(dDlist)):
        d = dDlist[dD][0]
        D = dDlist[dD][1]
        if seed is not None:
            rand.seed(seed)
        for s in range(samples):
            A,l,r = createSet(d,D,direction=direction,normalize=normalize,imaginary=imaginary)
            XMPO = [l] + [A]*block + [r]
            """if(canonical):
                XMPO = canonizeMPO(XMPO,normed,direction=0)"""
            L = XMPO[0:len(XMPO)//2 - (1-len(XMPO)%2)]
            M = XMPO[len(XMPO)//2 - (1-len(XMPO)%2):len(XMPO)//2 + 1]
            R = XMPO[len(XMPO)//2 + 1:]
            
            X = contractMPO(XMPO)[0,:,0,:]
            Xi = la.inv(X)
            
            if(pre_norm):
                XiMPO = SVDMat(Xi,d,block,False,tol=1E-12)
            else:
                XiMPO = SVDMat(Xi,d,block,False)
            """if(canonical):
                XiMPO = canonizeMPO(XiMPO,direction=0)"""
            Li = XiMPO[0:len(XiMPO)//2 - (1-len(XiMPO)%2)]
            Mi = XiMPO[len(XiMPO)//2 - (1-len(XiMPO)%2):len(XiMPO)//2 + 1]
            Ri = XiMPO[len(XiMPO)//2 + 1:]
            #check = check + [L,M,R,Li,Mi,Ri]
            if(pre_norm):
                norm_adapt = []
                
                LL = [no(np.tensordot(m,mi,[[1,3],[3,1]]).reshape(m.shape[0]*mi.shape[0],m.shape[2]*mi.shape[2])/m.shape[1]) for m,mi in zip(L,Li)]
                MM = [no(np.tensordot(m,mi,[[1,3],[3,1]]).reshape(m.shape[0]*mi.shape[0],m.shape[2]*mi.shape[2])/m.shape[1]) for m,mi in zip(M,Mi)]
                RR = [no(np.tensordot(m,mi,[[1,3],[3,1]]).reshape(m.shape[0]*mi.shape[0],m.shape[2]*mi.shape[2])/m.shape[1]) for m,mi in zip(R,Ri)]
                RR[-1] = RR/np.prod(LL + MM + RR)
                
                #norm_adapt = [no(LL)]+[no(x) for x in MM]+[no(RR)]
                #norm_adapt[-1] = norm_adapt[-1]/np.prod(LL + MM + RR)
                print(len(L),len(Li),len(M),len(Mi),len(R),len(Ri))
                print("Adjustment factors: ",norm_adapt, np.prod(norm_adapt))
                Ri = [m/n for m,n in zip(Ri,RR)]
                Li = [m/n for m,n in zip(Li,LL)]
                Mi = [m/n for m,n in zip(Mi,MM)]
            
#                norm_adapt = []
#                LL = np.tensordot(L[0],Li[0],[[1,3],[3,1]]).reshape(L[0].shape[0]*Li[0].shape[0],L[0].shape[2]*Li[0].shape[2])/np.sqrt(L[0].shape[1]*L[0].shape[3])
#                MM = [np.tensordot(m,mi,[[1,3],[3,1]]).reshape(m.shape[0]*mi.shape[0],m.shape[2]*mi.shape[2])/np.sqrt(m.shape[1]*m.shape[3]) for m,mi in zip(M,Mi)]
#                RR = np.tensordot(R[0],Ri[0],[[1,3],[3,1]]).reshape(R[0].shape[0]*Ri[0].shape[0],R[0].shape[2]*Ri[0].shape[2])/np.sqrt(R[0].shape[1]*R[0].shape[3])
#                
#                norm_adapt = [no(LL)]+[no(x) for x in MM]+[no(RR)]
#                norm_adapt[-1] = norm_adapt[-1]/np.prod(norm_adapt)
#                print("Adjustment factors: ",norm_adapt, np.prod(norm_adapt))
#                Ri = [Ri[0]/norm_adapt[-1]]
#                Li = [Li[0]/norm_adapt[0]]
#                Mi = [m/n for m,n in zip(Mi,norm_adapt[1:-1])]
                #check = check + [Li,Mi,Ri]
            
            MPOddim = np.prod([x.shape[1] for x in L]) * np.prod([x.shape[1] for x in M]) * np.prod([x.shape[1] for x in R])
            Mddim = np.prod([x.shape[1] for x in M])
            for N_ in range(len(Nrange)):
                XMPO_N = L + M*(N_+1) + R
                XiMPO_N = Li + Mi*(N_+1) + Ri
                if(post_norm):
                    mat,fac = contractMPO_(prodMPO(XMPO_N,XiMPO_N),3,ret_norm=True)
                    XiMPO_N[0] = scaleMPO(1/fac,XiMPO_N[:1])[0]
                    if(MPOddim*Mddim <= 64): #This number sets until which point we keep increasing the number of entries to be considered
                        MPOddim = MPOddim * Mddim
                E_ = prodMPO(XMPO_N,XiMPO_N)
                if(norm_mode == 1):
                    plotData[dD,N_,s] = max(map(abs,contractMPO(sumMPO(identityMPO(E_),scaleMPO(-1,E_))).flatten()))
                else:
                    plotData[dD,N_,s] = normMPO(sumMPO(identityMPO(E_),scaleMPO(-1,E_)))
                
                count += 1
                sys.stdout.write("\rProgress: "+str(count)+"/"+str(len(dDlist)*len(Nrange)*samples)+"norms calculated.")
                gc.collect()

    filename = 'analysis12_N'+str(Nrange[0])+'-'+str(Nrange[-1])+'samples'+str(samples)
    if(interactive):
        plotBool = input('Analysis complete. You can plot the data now or save the data and plot it later.\n Plot now? y/n\t')
        if(plotBool == 'y'):
            plot = True
        else:
            plot = False
    if(plot):
        ax = pl.figure().gca()        
        pl.xlabel(r'Lattice length $N$')
        ax.xaxis.set_major_locator(pl.MaxNLocator(integer=True))
        
        pl.xlim([1+block,3+block+(2-block%2)*N])
        pl.ylim([10**(math.floor(math.log(min(list(filter(lambda x:x!=0,plotData.flatten()))),10))-1),10**(math.ceil(math.log(max(list(filter(lambda x:x!=0,plotData.flatten()))),10))+1)])
        #pl.xticks(range(Nrange[0]-1,Nrange[-1]+1))
        if(norm_mode == 0):
            pl.ylabel(r'$\Vert X_{N}\tilde{X}_{N} - I\Vert_F$')
        else:
            pl.ylabel(r'$\Vert X_{N}\tilde{X}_{N} - I\Vert_{max}$')
        pl.yscale('log', nonposy='clip')
        pl.title(r'Error of approximated $X^{-1}$ for $N_0$ = ' + str(block))
        pl.grid(b=True, which='major', color='black', linestyle='--')
        pl.grid(b=True, which='minor', color='orange', linestyle='--')
        for dD in range(len(dDlist)):
            d = dDlist[dD][0]
            D = dDlist[dD][1]
            y = []
            yerror = np.zeros([2,len(Nrange)])
            for N_ in range(len(Nrange)):
                x = list(map(lambda x:(2-block%2)*x+dD/len(dDlist)+2+block,range(len(Nrange))))
                y.append(np.mean(plotData[dD,N_,:]))
                if(samples>1):
                    yerror[:,N_] = [np.std(list(filter(lambda x: x <= y[-1],plotData[dD,N_,:]))),np.std(list(filter(lambda x: x > y[-1],plotData[dD,N_,:])))]
            if(samples > 1):
                pl.errorbar(x,y,yerr=yerror,fmt='-o',label='(D,d) = ('+str(d) + ',' + str(D) + ')',capthick=2)
            else:
                pl.plot(x,y,'-^',label='(d,D) = ('+str(d) + ',' + str(D) + ')')
        lgd = pl.legend(loc='center left',ncol=1,bbox_to_anchor=(1,0.5))
        pl.savefig(filename + '.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')
    if(timestamp):
        filename = filename + str(int(time.time()))
    filename = filename + '.data'
    print('Saving to ' + filename)
    #pickle.dump([plotData,dDlist,N,block,samples],open(filename,'wb'))
    #return(check)
    

def no(mat):
    sq = np.trace(mat.dot(mat.T.conj()))
    #sq = np.trace(mat.dot(mat.T))
    #return np.sqrt(sq)*mat[0,0]/np.abs(mat[0,0])
    print("Shape:",mat.shape)
    return np.sqrt(sq)*np.trace(mat)/np.abs(np.trace(mat))


def analysis13(drange,Drange,Nrange,sweep=False):
    """ Description:
        ----------------------------------------------------------------------------------------------------------------------------------------------
        Proceeds as in analysis singular values at contracted MPO center:
        
        i)      Generate random A in C^(d x d)_(D x D),l in C^(dxd)_D, r in C^(dxd)_D
        ii)     Compute TI matrix X as: X^(s0s1...sNs(N+1))_(t0t1...tNt(N+1)) = l^s0_t0 A^s1_t1 ... A^sN_tN l^s(N+1)_t(N+1)
                    
                          |   |   |         |   |
                    X_N = l - A - A - ... - A - r
                          |   |   |         |   |
                    
        iii)    Contract the MPO to a matrix X compute the inverse
        iv)     Decompose inverse matrix through repeated SVD:
                           |    |             |              |     |
                    Xi_N = M1 - M2 - ... - M(N/2) - ... - M(N-1) - MN
                           |    |             |              |     |
                    
        v)      Contract the product of X_N and Xi_N up to middle M(N/2) and apply a singular value decomposition, returning singular values:
                    
                    |    |             |            ||.._|
                    l -  A  - ... -    A   -        |    |-
                    |    |             |       ->   |  L |    -> SVD
                    M1 - M2 - ... - M(N/2) -        |_.._|-
                    |    |             |            ||   |
        
        Parameters:
        ----------------------------------------------------------------------------------------------------------------------------------------------
        drange:
        Drange:
        Nrange:         A range of natural number > 0 specifying the system sizes to be analyzed.
        sweep:          If True, will perform a subsequent right-normalization and left-normalization on the MPO.
    """
    #pl.xlim([Nrange[0]-1,Nrange[-1]+1])
    sValueList = [[[[] for d in range(len(drange))] for D in range(len(Drange))] for N in range(len(Nrange))]
    for d in range(len(drange)):
        for D in range(len(Drange)):
            A,l,r = createSet(drange[d],Drange[D])
            for N in range(len(Nrange)):
                XMPO = [l] + [A]*Nrange[N] + [r]
                X = contractMPO(XMPO)[0,:,0,:]
                Xi = la.inv(X)
                XiMPO = SVDMat(Xi,drange[d],Nrange[N],False)
                if(sweep):
                    XiMPO = canonizeMPO(canonizeMPO(XiMPO,diretion=0),direction=1)
                E_ = prodMPO(XMPO,XiMPO)
                check = contractMPO(prodMPO(XMPO,XiMPO))[0,:,0,:]
                E_left = contractMPO(E_[0:Nrange[N]//2 + 2])
                dd1,DD,dd2 = E_left.shape[1],E_left.shape[2],E_left.shape[3]
                M = np.zeros([dd1*dd2,DD],dtype=complex)
                for j in range(dd1):
                    for k in range(dd2):
                        M[remap(j,k,dd2),:] = E_left[0,j,:,k]
                U,S,V = la.svd(M)
                Ss = S.tolist()
                sValueList[N][D][d] = Ss
    
    for N in range(len(Nrange)):
        pl.figure()
        pl.yscale('log', nonposy='clip', basey=2)
        #pl.xlabel(r'Lattice length $N$')
        pl.ylabel(r'Singular value $\sigma$')
        pl.title(r'Singular values of left-contracted block for size ' + str(Nrange[N]))
        pl.grid(b=True, which='major', color='black', linestyle='--')
        pl.grid(b=True, which='minor', color='orange', linestyle='--')
        #pl.xticks(range(len(S)+1))
        for d in range(len(drange)):
            for D in range(len(Drange)):
                plot = sValueList[N][D][d]
                pl.plot(range(len(plot)),plot,'o',label='(D,d) = ('+ str(Drange[D])+','+str(drange[d])+')')
        pl.savefig('analysis13_N' + str(Nrange[N]) + '.pdf')
        pl.legend(loc='best')
        pl.show()
        
def analysis14(drange,Drange,Nrange,samples):
    """ Description:
        ----------------------------------------------------------------------------------------------------------------------------------------------
        i)      Generate random A in C^(d x d)_(D x D),l in C^(dxd)_D, r in C^(dxd)_D
        ii)     Compute TI matrix X as: X^(s0s1...sNs(N+1))_(t0t1...tNt(N+1)) = l^s0_t0 A^s1_t1 ... A^sN_tN l^s(N+1)_t(N+1)
                 (Chain length corresponds to VAR[block])
                           |   |   |         |   |
                     X_N = l - A - A - ... - A - r
                           |   |   |         |   |
                               \______ ______/
                                      v block N x A
        iii)    Contract MPO to matrix and calculate inverse matrix.
        iv)     Apply iterated SVD to Xi and return singular values at middle iteration.
        
        Parameters:
        ----------------------------------------------------------------------------------------------------------------------------------------------
        drange:
        Drange:
        Nrange:         A range of natural number > 0 specifying the system sizes to be analyzed.
        samples:        Number of different initial A,l,r samples to be generated.
    """
    sValueList = [[[[[] for s in range(samples)] for d in range(len(drange))] for D in range(len(Drange))] for N in range(len(Nrange))]
    for d in range(len(drange)):
        for D in range(len(Drange)):
            for s in range(samples):
                A,l,r = createSet(drange[d],Drange[D])
                for N in range(len(Nrange)):
                    XMPO = [l] + [A]*Nrange[N] + [r]
                    X = contractMPO(XMPO)[0,:,0,:]
                    Xi = la.inv(X)
                    XiMPO,svals = SVDMat(Xi,drange[d],Nrange[N],False,returnsval=True)
                    sValueList[N][D][d][s] = svals[Nrange[N]//2-(1-Nrange[N]%2)]
    
    pl.figure()
    pl.xlabel(r'System size $N$')
    pl.xlim([0,Nrange[-1]+1])
    pl.xticks(range(0,Nrange[-1]+1))
    #pl.xlabel(r'Lattice length $N$')
    pl.ylabel(r'# singular values')
    pl.ylim([0,drange[-1]**(Nrange[-1]//2 +2)+1])
    pl.title(r'Number of singular values at middle iteration')
    pl.grid(b=True, which='major', color='black', linestyle='--')
    pl.grid(b=True, which='minor', color='orange', linestyle='--')
    for D in range(len(Drange)):
        for d in range(len(drange)):
            y = [len(sValueList[N][D][d][s]) for N in range(len(Nrange)) for s in range(samples)]
            x = list(map(lambda x:x+remap(D,d,len(drange))/(len(drange)*len(Drange)+1),[Nrange[N] for N in range(len(Nrange)) for s in range(samples)]))
            pl.plot(x,y,'o',label='(D,d) = ('+ str(Drange[D])+','+str(drange[d])+')')
    pl.legend(loc='center left',ncol=1,bbox_to_anchor=(1,0.5))
    pl.savefig('analysis14_1.pdf')
    pl.show()
    
    pl.figure()
    pl.yscale('log', nonposy='clip', basey=2)
    pl.xlabel(r'System size $N$')
    pl.xlim([0,Nrange[-1]+1])
    pl.xticks(range(0,Nrange[-1]+1))
    #pl.xlabel(r'Lattice length $N$')
    pl.ylabel(r'singular value $\sigma$')
    pl.title(r'Singular values at middle iteration')
    pl.grid(b=True, which='major', color='black', linestyle='--')
    pl.grid(b=True, which='minor', color='orange', linestyle='--')
    for D in range(len(Drange)):
        for d in range(len(drange)):
            data = [list(itertools.chain(*sValueList[N][D][d])) for N in range(len(Nrange))]
            #x = list(itertools.chain(*[[Nrange[N]]*len(data[N]) for N in range(len(Nrange))]))
            x = list(map(lambda x:x+remap(D,d,len(drange))/(len(drange)*len(Drange)+1),list(itertools.chain(*[[Nrange[N]]*len(data[N]) for N in range(len(Nrange))]))))
            y = list(itertools.chain(*data))
            pl.plot(x,y,'o',label='(D,d) = ('+ str(Drange[D])+','+str(drange[d])+')')
    pl.legend(loc='center left',ncol=1,bbox_to_anchor=(1,0.5))
    pl.savefig('analysis14_2.pdf')
    pl.show()
    
def analysis15(d,D,N):
    """ Description:
        ----------------------------------------------------------------------------------------------------------------------------------------------        
        For X_2, start by computing the inverse Xi_2. Then construct X_(n+1) iteratively by adding A to X_n. Solve inverse MPO for respective Ai(n+1).
        
        Parameters:
        ----------------------------------------------------------------------------------------------------------------------------------------------
        
    """
    data = []
    A,L,R = createSet(d,D)
    X_N = contractMPO([L,R])[0,:,0,:]
    Li,Ri = SVDMat(la.inv(X_N),d,0,False)
    for N_ in range(1,N):
        X_NMPO = [L,A,R]
        Xi_NMPO = [Li,[],Ri]
        Ai = solveMPO(np.identity(d**(N_+2)),X_NMPO,Xi_NMPO)
        E_ = prodMPO([L,A,R],[Li,Ai,Ri])
        data.append(normMPO(sumMPO(E_,scaleMPO(-1,identityMPO(E_)))))
        if(N_%2 == 0):
            L = contractMPO([L,A])
            Li = contractMPO([Li,Ai])
        else:
            R = contractMPO([A,R])
            Ri = contractMPO([Ai,Ri])
    return(data)

    
def analysis16(dDlist,N,block=0,tr_mode=0,norm_mode=0,normalize=True,drop_unitary=False,tol=None,seed=None,upper_tri=False,layered=False):
    """ Description:
        ----------------------------------------------------------------------------------------------------------------------------------------------
        i)   Generate random A in C^(d x d)_(D x D),l in C^(dxd)_D, r in C^(dxd)_D
        ii)  Compute TI matrix X as: X^(s0s1...sNs(N+1))_(t0t1...tNt(N+1)) = l^s0_t0 A^s1_t1 ... A^sN_tN l^s(N+1)_t(N+1)
             (Chain length corresponds to VAR[block]+3)
                       |   |   |         |   |
                 X_N = l - A - A - ... - A - r
                       |   |   |         |   |
                           \______ ______/
                                  v block N x A
                            
        iii)  Contract to matrix X, invert and decompose inverse Xi through repeated SVD:
                        |    |             |              |     |
                 Xi_N = M1 - M2 - ... - M(N/2) - ... - M(N-1) - MN
                        |    |             |              |     |   
                 
        iv)   Calculate Matrix X_(N+j) as:
                         |   |   |         |   |
                 X_(N+j) l - A - A - ... - A - r
                         |   |   |         |   |
                             \______ ______/
                                    v (N+j) x A
                            
              ...and calculate new Ai in every step through resolution of the trace-reduced equation system.
              For block=1, we are solving this kind of system:
              
               __    __                      __        __    __       ___         ___
              |  |  |  |        |   |   |   |  |      |  |  |  |     |  _|_|...|_|_  |
              l -|- A -|- ... - A - A - A - A -|- ... A -|- r  |     | |           | |
              |  |  |  |        |   |   |   |  |      |  |  |  |  =  | |     I     | |
              li-|- * -|- ... - * - ? - * - * -|- ... * -|- ri |     | |_ __...__ _| |
              |__|  |__|        |   |   |   |__|      |__|  |__|     |___| |   | |___|
              
              
        
        v)    Calculate norm differences ||X_(N+j)Xi_(N+j) - E||
        
        Parameters:
        ----------------------------------------------------------------------------------------------------------------------------------------------
        Ddlist:         A list of tuples [[D1,d1],[D2,d2],...,[Dm,dm]] setting physical and bond dimensions of the constructed tensors A,l,r
        N:              Natural number > 0 specifying the maximum number of sites (N+2) in the system.
        block:          Natural number >= 0 specifying the number of sites (block+3) for which the equation system is solved in every iteration step.
        tr_mode:        Sets the way in which sites are traced.
                            0: (Default) Outward-to-inward tracing. Trace left-most/right-most untraced tensor (alternately) along physical legs.
                            1: Inward-to-outward tracing. Trace newly appended tensors right away. 
                            2: No tracing. Solve the whole equation system. Computation cost increases exponantially in system size.
        norm_mode:      Sets the way in which the norm is calculated
                            0: (Default) Calculate Frobenius norm
                            1: Caculate max norm
        normalize:      Start with normalized tensors l,A,r
        drop_unitary:   Perform an SVD Ai=USV on the solution matrix Ai of every iterations step. Discard unitary matrix closest to appending bond.
        tol:            Truncates singular values < tol appearing in the SVD of Ai. Parameter is ignored unless drop_unitary = True.
        seed:           Provides and integer seed for the generation of random tensors
        upper_tri:      Draws random tensors L,A,R such that their contraction is a upper triangular matrix
    """
    #check = []    
    
    for dD in dDlist:
        if(not(len(dD) == 2)):
            raise Exception("Tuples of the type (D,d) are expected.")
        if(not(isinstance(dD[0],int) and isinstance(dD[1],int)) or dD[0] < 1 or dD[1] < 1):
            raise Exception("D,d have to be integers greater than 0.")
    
    Nrange = range(0,N)
    plotData = np.zeros([len(dDlist),len(Nrange)])
    for dD in range(len(dDlist)):
        d = dDlist[dD][0]
        D = dDlist[dD][1]
        if(block%2 == 0):
            trs = []
        else:
            trs = [0]
            
        if seed is not None:
            rand.seed(seed)
        if(layered):
            A,L,R = createSetXY(d,D,imaginary=True)
        else:
            A,L,R = createSet(d,D,direction=1,normalize=normalize,imaginary=True)
        X_N = contractMPO([L] + [A]*block + [R])[0,:,0,:]
        if(layered):
            d = d**2
        Xi_NMPO = SVDMat(la.inv(X_N),d,block,direction=1,returnsval=False)
        X_NMPO = [L] + [A]*block + [R]
        L,R,Li,Ri = [L] + [A]*int(np.ceil(block/2)),[A]*int(block - np.ceil(block/2)) + [R],Xi_NMPO[0:int(np.ceil(block/2) + 1)],Xi_NMPO[int(np.ceil(block/2) + 1):]
        
        if(tr_mode == 0):
            """ Outward-to-inward tracing """
            L,R = prodMPO(L,Li),prodMPO(R,Ri)
            Ltr,Rtr = [],[]
            fidx = int(block-(block%2)+3)
            for N_ in range(len(Nrange)):
#                    X_NMPO = L + [A] + R
#                    Xi_NMPO = Li + [[]] + Ri
            
                Ai = solveMPO2(np.identity(d**fidx)*d**(block+N_+3-fidx),Ltr + L[len(Ltr):],R[:len(R)-len(Rtr)] + Rtr,A,tr_sites = trs)
                
                """ X_N -> X_(N+1) and Xi_N -> Xi_(N+1)"""
                if(N_%2 == 0):
                    L = L + prodMPO([A],[Ai])
                    if(block%2 == 0):
                        Ltr = Ltr + [traceMPO([L[int(np.ceil(N_/2))]])]
                    else:
                        Ltr = Ltr + [traceMPO([L[(-1)*(int(np.floor(N_/2))+1)]])]
                else:
                    R = prodMPO([A],[Ai]) + R
                    if(block%2 == 0):
                        Rtr = [traceMPO([R[(-1)*int(np.ceil(N_/2))]])] + Rtr
                    else:
                        Rtr = [traceMPO([R[int(np.floor(N_/2))+1]])] + Rtr

                gc.collect()
                #check.append(L+R)
                E_ = L + R
                if(norm_mode == 0):
                    norm = normMPO(sumMPO(E_,scaleMPO(-1,identityMPO(E_))))
                elif(norm_mode == 1):
                    norm = max(map(abs,contractMPO(sumMPO(E_,scaleMPO(-1,identityMPO(E_))))[0,:,0,:].flatten()))
                else:
                    raise Exception("Norm mode has to be 0 or 1!")
                
                if(drop_unitary):
                    D1,D2,d_ = Ai.shape[0],Ai.shape[2],Ai.shape[1]
                    if(N_%2 == 0):
                        U,S,V = la.svd(np.reshape(np.transpose(Ai,[0,1,3,2]),[D1*d_**2,D2]),full_matrices=False)
                        if tol is not None:
                            while S[-1]<tol and len(S)>1:
                                S = S[:-1]
                        Ai_ = np.transpose(np.reshape(np.dot(U[:,:len(S)],np.identity(len(S))*S),[D1,d_,d_,len(S)]),[0,1,3,2])
                        L = L[:-1] + prodMPO([A],[Ai_]) 
                    else:
                        U,S,V = la.svd(np.reshape(np.transpose(Ai,[0,1,3,2]),[D1,D2*d_**2]),full_matrices=False)
                        if tol is not None:
                            while S[-1]<tol and len(S)>1:
                                S = S[:-1]
                        Ai_ = np.transpose(np.reshape(np.dot(np.identity(len(S))*S,V[:len(S),:]),[len(S),d_,d_,D2]),[0,1,3,2])
                        R = prodMPO([A],[Ai_]) + R[1:]
                        
                gc.collect()
                sys.stdout.write("\rProgress: "+str(N_+1)+"/"+ str(len(Nrange)) +"norms calculated.")
                #norm = norm//drange[d]**(block+N_+3)
                plotData[dD,N_] = norm
        
        elif(tr_mode == 1):
            """ Inward-to-outward tracing """
            L_tr,R_tr = prodMPO(L,Li,tr_sites=trs),prodMPO(R,Ri)
            L,R = prodMPO(L,Li),prodMPO(R,Ri)
            
            for N_ in range(len(Nrange)):
                
                fidx = int(block-(block%2)+3)
                Ai = solveMPO2(np.identity(d**fidx),L_tr,R_tr,A)
                """ X_N -> X_(N+1) and Xi_N -> Xi_(N+1)"""
                if(N_%2 == 0):
                    L_tr = L_tr + prodMPO([A],[Ai],tr_sites=[0])
                    L = L + prodMPO([A],[Ai])
                else:
                    R_tr = prodMPO([A],[Ai],tr_sites=[0]) + R_tr
                    R = prodMPO([A],[Ai]) + R
                
                E_ = L + R
                plotData[dD,N_] = normMPO(sumMPO(E_,scaleMPO(-1,identityMPO(E_))))#//drange[d]**(block+N_+3)

        elif(tr_mode == 2):
            """ No tracing """
            L,R = prodMPO(L,Li),prodMPO(R,Ri)
            for N_ in range(len(Nrange)):
                Ai = solveMPO2(np.identity(d**(3+block+N_)),L,R,A)
                if(N_%2 == 0)                :
                    L  = L + prodMPO([A],[Ai])
                else:
                    R = prodMPO([A],[Ai]) + R
                E_ = L + R
                if(norm_mode==0):
                    plotData[dD,N_] = normMPO(sumMPO(E_,scaleMPO(-1,identityMPO(E_))))#//drange[d]**(block+N_+3)
                elif(norm_mode==1):
                    plotData[dD,N_] = max(contractMPO(sumMPO(E_,scaleMPO(-1,identityMPO(E_))))[0,:,0,:].flatten())
                else:
                    raise Exception("Norm mode has to be 0 or 1!")
                
        else:
            raise Exception('Please select from tracing modes 0,1,2')
                
    pl.figure()
    pl.yscale('log', nonposy='clip')
    #pl.xlabel(r'Lattice length $N$')
    if(norm_mode==0):
        pl.ylabel(r'$\Vert X_{N}\tilde{X}_{N} - I\Vert_F$')
    else:
        pl.ylabel(r'$\Vert X_{N}\tilde{X}_{N} - I\Vert_{max}$')
    lim = [min(list(filter(lambda x:x>0,plotData.flatten())))/10,max(list(filter(lambda x:x>0,plotData.flatten())))*10]
    pl.ylim(lim)
    pl.xlabel(r'System size N')
    pl.xlim([block+1,block+2+N])
    pl.title(r'Quality of approximation for initial block size ' + str(block))
    pl.grid(b=True, which='major', color='black', linestyle='--')
    pl.grid(b=True, which='minor', color='orange', linestyle='--')
    #pl.xticks(range(len(S)+1))
    for dD in range(len(dDlist)):
        d = dDlist[dD][0]
        D = dDlist[dD][1]
        y = plotData[dD,:]
        x = list(map(lambda x:x+2+block+dD/(len(dDlist)+1),range(len(Nrange))))
        pl.plot(x,y,'-^',label='(D,d) = ('+ str(D)+','+str(d)+')')
    lgd = pl.legend(loc='center left',ncol=1,bbox_to_anchor=(1,0.5))
    pl.savefig('analysis16_N' + str(N) + 'block' + str(block) + 'mode' + str(tr_mode) + '.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')
    pl.show()
    #return(check)
    
def analysis17(dDlist,N,block=0,norm_type=0,imaginary=True,drop_unitary=False,tol=None,seed=None,cap=True):
    """ Description:
        ----------------------------------------------------------------------------------------------------------------------------------------------
        Modified analysis16 that works after the same principle. However, the tensors are constructed form createSetXY or createSetYXY, leading to an
        (implicitly) double- or triple-layered MPO, which is in principle easy to invert.
        
        Parameters:
        ----------------------------------------------------------------------------------------------------------------------------------------------
        Ddlist:         A list of tuples [[D1,d1],[D2,d2],...,[Dm,dm]] setting physical and bond dimensions of the constructed tensors A,l,r
        N:              Natural number > 0 specifying the maximum number of sites (N+2) in the system.
        block:          Natural number >= 0 specifying the number of sites (block+3) for which the equation system is solved in every iteration step.
        norm_mode:      Sets the way in which the norm is calculated
                            0: (Default) Calculate Frobenius norm
                            1: Caculate max norm
        drop_unitary:   Perform an SVD Ai=USV on the solution matrix Ai of every iterations step. Discard unitary matrix closest to appending bond.
        tol:            Truncates singular values < tol appearing in the SVD of Ai. Parameter is ignored unless drop_unitary = True.
        seed:           Provides and integer seed for the generation of random tensors
    """
    
    for dD in dDlist:
        if(not(len(dD) == 2)):
            raise Exception("Tuples of the type (D,d) are expected.")
        if(not(isinstance(dD[0],int) and isinstance(dD[1],int)) or dD[0] < 1 or dD[1] < 1):
            raise Exception("D,d have to be integers greater than 0.")
            
    #check = []
    Nrange = range(1,N)
    plotData = np.zeros([len(Nrange),len(dDlist)])
    for dD in range(len(dDlist)):
        d = dDlist[dD][0]
        D = dDlist[dD][1]
        if(block%2 == 0):
            trs = []
        else:
            trs = [0]
        
        if seed is not None:
            rand.seed(seed)
        A,l,r = createSet(d,D,imaginary=imaginary)
        #A,l,r = createSetXY(d,D,imaginary=imaginary)
        
        X_N = contractMPO([l] + [A]*block + [r])[0,:,0,:]
        Xi_NMPO = SVDMat(la.inv(X_N),d,block,direction=1,returnsval=False)
        X_NMPO = [l] + [A]*block + [r]
        l,r,li,ri = [l] + [A]*int(np.ceil(block/2)),[A]*int(block - np.ceil(block/2)) + [r],Xi_NMPO[0:int(np.ceil(block/2) + 1)],Xi_NMPO[int(np.ceil(block/2) + 1):]
    
        """ Outward-to-inward tracing """
        L,R = prodMPO(l,li),prodMPO(r,ri)
        Ltr,Rtr = [],[]
        fidx = int(block-(block%2)+3)
        for N_ in range(len(Nrange)):
            #X_NMPO = L + [A] + R
            #Xi_NMPO = Li + [[]] + Ric[]
        
            #Ai = solveMPO(np.identity(d**fidx)*d**(block+N_+3-fidx),X_NMPO,Xi_NMPO,tr_sites = trs)
            Ai = solveMPO2(np.identity(d**(fidx))*d**((block+N_+3-fidx)),Ltr + L[len(Ltr):],R[:len(R)-len(Rtr)] + Rtr,A,tr_sites = trs)
            #Ai = solveMPO2(np.identity(d**(2*fidx))*d**(2*(block+N_+3-fidx)),Ltr + L[len(Ltr):],R[:len(R)-len(Rtr)] + Rtr,A,tr_sites = trs)
            if(cap):
                shape = Ai.shape
                Ai = Ai.flatten()
                for i in range(len(Ai)):
                    if(abs(Ai[i]) < 1e-8):
                        Ai[i] = 0
                Ai = np.reshape(Ai,shape)
                    
            """ X_N -> X_(N+1) and Xi_N -> Xi_(N+1)"""
            if(N_%2 == 0):
                L = L + prodMPO([A],[Ai])
                if(block%2 == 0):
                    Ltr = Ltr + [traceMPO([L[int(np.ceil(N_/2))]])]
                else:
                    Ltr = Ltr + [traceMPO([L[(-1)*(int(np.floor(N_/2))+1)]])]
            else:
                R = prodMPO([A],[Ai]) + R
                if(block%2 == 0):
                    Rtr = [traceMPO([R[(-1)*int(np.ceil(N_/2))]])] + Rtr
                else:
                    Rtr = [traceMPO([R[int(np.floor(N_/2))+1]])] + Rtr
                    
            gc.collect()
            E_ = canonizeMPO(L + R,normalize=True)
            #check.append(L+R)
            if(norm_type == 0):
                norm = normMPO(sumMPO(E_,scaleMPO(-1,identityMPO(E_))))
            elif(norm_type == 1):
                norm = max(map(abs,contractMPO(sumMPO(E_,scaleMPO(-1,identityMPO(E_))))[0,:,0,:].flatten()))
                
                
            if(drop_unitary):
                D1,D2,d_ = Ai.shape[0],Ai.shape[2],Ai.shape[1]
                if(N_%2 == 0):
                    U,S,V = la.svd(np.reshape(np.transpose(Ai,[0,1,3,2]),[D1*d_**2,D2]),full_matrices=False)
                    if tol is not None:
                        while S[-1]<tol and len(S)>1:
                            S = S[:-1]
                    Ai_ = np.transpose(np.reshape(np.dot(U[:,:len(S)],np.identity(len(S))*S),[D1,d_,d_,len(S)]),[0,1,3,2])
                    L = L[:-1] + prodMPO([A],[Ai_]) 
                else:
                    U,S,V = la.svd(np.reshape(np.transpose(Ai,[0,1,3,2]),[D1,D2*d_**2]),full_matrices=False)
                    if tol is not None:                
                        while S[-1]<tol and len(S)>1:
                            S = S[:-1]
                    Ai_ = np.transpose(np.reshape(np.dot(np.identity(len(S))*S,V[:len(S),:]),[len(S),d_,d_,D2]),[0,1,3,2])
                    R = prodMPO([A],[Ai_]) + R[1:]
                    
            gc.collect()
            sys.stdout.write("\rProgress: "+str(N_+1)+"/"+ str(len(Nrange)) +"norms calculated.")
            #norm = norm//drange[d]**(block+N_+3)
            plotData[N_,dD] = norm
        
    pl.figure()
    pl.yscale('log', nonposy='clip', basey=10)
    pl.xlabel(r'Iteration step $n$')
    pl.xlim([0,Nrange[-1]+1])
    pl.xticks(range(0,Nrange[-1]+1))
    pl.ylabel(r'$\Vert X_{N+j}\hat{X}^{-1}_{N+j} - E\Vert$')
    pl.title(r'Quality of approximation for initial block size ' + str(block))
    pl.grid(b=True, which='major', color='black', linestyle='--')
    pl.grid(b=True, which='minor', color='orange', linestyle='--')
    for dD in range(len(dDlist)):
        d = dDlist[dD][0]
        D = dDlist[dD][1]
        x = list(map(lambda x:x+2+block+dD/(len(dDlist)+1),range(len(Nrange))))
        pl.plot(x,plotData[:,dD],'-^',label='(d,D) = (' + str(d) + ',' + str(D) + ')')
    lgd = pl.legend(loc='center left',ncol=1,bbox_to_anchor=(1,0.5))
    pl.savefig('analysis17_N' + str(N) + '.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')
    pickle.dump(plotData,open('analysis17.data','wb'))
    #return(plotData)
    #return(check)
    
def analysis18(dDlist,N,block=1,imaginary=True):
    """ Description:
        ----------------------------------------------------------------------------------------------------------------------------------------------
        Modified analysis12 that works after the same principle. However, the tensors are constructed from createSetXY or createSetYXY, leading to an
        implicitly double- or triple-layered MPO, which is in principle easy to invert.
        
        Parameters:
        ----------------------------------------------------------------------------------------------------------------------------------------------
        Ddlist:         A list of tuples [[D1,d1],[D2,d2],...,[Dm,dm]] setting physical and bond dimensions of the constructed tensors A,l,r
        N:              Natural number > 0 specifying the maximum number of sites (N+2) in the system.
        block:          Natural number > 1 specifying the number of sites of the block tensor used for iterative elongation of the MPO.
    """
#    if(block>N or block==0):
#        raise Exception('N has to be greater than blocksize and greater than 0')
    for dD in dDlist:
        if(not(len(dD) == 2)):
            raise Exception("Tuples of the type (D,d) are expected.")
        if(not(isinstance(dD[0],int) and isinstance(dD[1],int)) or dD[0] < 1 or dD[1] < 1):
            raise Exception("D,d have to be integers greater than 0.")
      
    #check = []
    Nrange = range(block+1,N+1)
    plotData = np.zeros([len(dDlist),len(Nrange)])
    XMPOs = []
    XiMPOs = []
    for dD in range(len(dDlist)):
        d = dDlist[dD][0]
        D = dDlist[dD][1]
        A,l,r = createSetXY(d,D,imaginary=imaginary)
        print(l.shape,A.shape,r.shape)
        
        XMPO = [l] + [A]*block + [r]
        """if(canonical):
            XMPO = canonizeMPO(XMPO,normed,direction=0)"""
        L = XMPO[0:len(XMPO)//2 - (1-len(XMPO)%2)]
        M = XMPO[len(XMPO)//2 - (1-len(XMPO)%2):len(XMPO)//2 + 1]
        R = XMPO[len(XMPO)//2 + 1:]
        
        X = contractMPO(XMPO)[0,:,0,:]
        Xi = la.inv(X)
        
        XiMPO = SVDMat(Xi,d**2,block,False)
        """if(canonical):
            XiMPO = canonizeMPO(XiMPO,direction=0)"""
        Li = XiMPO[0:len(XiMPO)//2 - (1-len(XiMPO)%2)]
        Mi = XiMPO[len(XiMPO)//2 - (1-len(XiMPO)%2):len(XiMPO)//2 + 1]
        Ri = XiMPO[len(XiMPO)//2 + 1:]
        #check = [L,M,R,Li,Mi,Ri]
        
        for N_ in range(len(Nrange)):
            #XMPO_N = [l] + [A]*Nrange[N_] + [r]
            XMPO_N = L + M*(N_+1) + R
            XiMPO_N = Li + Mi*(N_+1) + Ri
            XMPOs.append(XMPO_N)
            XiMPOs.append(XiMPO_N)
            E_ = prodMPO(XMPO_N,XiMPO_N)
            plotData[dD,N_] = normMPO(sumMPO(identityMPO(E_),scaleMPO(-1,E_)))
            sys.stdout.write("\rProgress: "+str(N_+1)+"/"+ str(len(Nrange)) +"norms calculated.")
    
    pl.figure()
    pl.xlabel(r'Lattice length $N$')
    pl.xlim([Nrange[0]-1,Nrange[-1]+1])
    #if(normed):
    #    pl.ylim([10**(math.floor(math.log(min([min(plotData[d,D,:,:,0].flatten()),min(plotData[d,D,:,:,1].flatten())]),10))-1),10**(math.ceil(math.log(max([max(plotData[d,D,:,:,0].flatten()),max(plotData[d,D,:,:,1].flatten())]),10))+1)])
    #else:
    pl.ylim([10**(math.floor(math.log(min(list(filter(lambda x:x!=0,plotData.flatten()))),10))-1),10**(math.ceil(math.log(max(list(filter(lambda x:x!=0,plotData.flatten()))),10))+1)])
    #pl.xticks(range(Nrange[0]-1,Nrange[-1]+1))
    #pl.ylabel(r'$\Vert X_N - \hat{X}_N \Vert$')
    pl.ylabel(r'$\Vert X_{N}\tilde{X}_{N} - I\Vert$')
    pl.yscale('log', nonposy='clip')
    pl.title(r'Quality of approximation for initial block size ' + str(block))
    pl.grid(b=True, which='major', color='black', linestyle='--')
    pl.grid(b=True, which='minor', color='orange', linestyle='--')
    for dD in range(len(dDlist)):
        d = dDlist[dD][0]
        D = dDlist[dD][1]#
        x = list(map(lambda x:(2-block%2)*x+dD/len(dDlist)+3+block-(block%2),range(len(Nrange))))
        y = plotData[dD,:]
        pl.plot(x,y,'-^',label='(d,D) = ('+ str(d)+','+str(D)+')')
    
    lgd = pl.legend(loc='center left',ncol=1,bbox_to_anchor=(1,0.5))
    pl.savefig('analysis18_N' + str(N) + 'block' + str(block) + '.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')
    #return(check)
    
    
def analysis19(dDlist,N,ti=True,layered=False,plot_extr=True):
    """ Description:
        ----------------------------------------------------------------------------------------------------------------------------------------------
        Analysizes the relationship between singular values and system size of (TI-)MPOs.
        
        i)      Generate random A in C^(d x d)_(D x D),l in C^(dxd)_D, r in C^(dxd)_D
        ii)     Compose TI-MPO X_N as: X^(s0s1...sNs(N+1))_(t0t1...tNt(N+1)) = l^s0_t0 A^s1_t1 ... A^sN_tN l^s(N+1)_t(N+1)

                           |   |   |         |   |
                     X_N = l - A - A - ... - A - r
                           |   |   |         |   |
                           
        iii)    Contract MPO to matrix and perform SVD to calculate singular values.
        
        Parameters:
        ----------------------------------------------------------------------------------------------------------------------------------------------
        Ddlist:         A list of tuples [[D1,d1],[D2,d2],...,[Dm,dm]] setting physical and bond dimensions of the constructed tensors A,l,r     
        N:              Natural number > 0 specifying the maximum number of sites (N+2) in the system.
        ti:             Allows for construction of translationally-invariant or non-translationally invariant MPOs:
                            True: (Default) Append initially generated tensor A to MPO in every iteration step.
                            False: Generate new random tensor to be appended in very iteration step.
        plot_extr:      Plot only the minimal and maximal singular of X_N.       
    """
    for dD in dDlist:
        if(not(len(dD) == 2)):
            raise Exception("Tuples of the type (D,d) are expected.")
        if(not(isinstance(dD[0],int) and isinstance(dD[1],int)) or dD[0] < 1 or dD[1] < 1):
            raise Exception("D,d have to be integers greater than 0.")
            
    Nrange = range(0,N)
    plotData = [[[] for N in Nrange] for dD in dDlist]
    for dD in range(len(dDlist)):
        d = dDlist[dD][0]
        D = dDlist[dD][1]
        if(layered):
            A,l,r = createSetXY(d,D)
        else:
            A,l,r = createSet(d,D)
        
        L = [l]
        for N_ in Nrange:
            X = contractMPO(L + [r])[0,:,0,:]
            if(la.matrix_rank(X) < X.shape[0]):
                print("Non-invertible matrix for N=" + str(N_))
            U,S,V = la.svd(X)
            plotData[dD][N_] = S
            L = L + [A]
            gc.collect()
            sys.stdout.write("\rProgress: "+str(N_+1)+"/"+ str(len(Nrange)) +"steps complete.")
    
    """ First plot: """
    colormap = pl.cm.hsv
    colorlist = [colormap(i) for i in np.linspace(0, 0.9,num=len(dDlist))]        
    
    pl.figure()
    pl.gca().set_color_cycle(colorlist)
    pl.yscale('log', nonposy='clip', basey=10)
    pl.xlabel(r'System size $N$')
    pl.xlim([1,3+Nrange[-1]+1])
    pl.xticks(range(1,3+Nrange[-1]+1))
    pl.ylabel(r'singular value $\sigma$')
    pl.title(r'Singular values X at Nth iteration')
    pl.grid(b=True, which='major', color='black', linestyle='--')
    pl.grid(b=True, which='minor', color='orange', linestyle='--')
    
    for dD in range(len(dDlist)):
        data = plotData[dD]
        d = dDlist[dD][0]
        D = dDlist[dD][1]
        x = list(map(lambda x:x+2+dD/(len(dDlist)+1),list(itertools.chain(*[[Nrange[N]]*len(data[N]) for N in range(len(Nrange))]))))
        #x = list(itertools.chain(*[[Nrange[N]]*len(data[N]) for N in range(len(Nrange))]))
        y = list(itertools.chain(*data))
        pl.plot(x,y,'o',label='(d,D) = (' + str(d) + ',' + str(D) + ')')
    
    lgd = pl.legend(loc='center left',ncol=1,bbox_to_anchor=(1,0.5))
    pl.savefig('analysis20.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    if(plot_extr):
        pl.figure()
        pl.gca().set_color_cycle(list(itertools.chain(*zip(colorlist,colorlist))))
        pl.yscale('log', nonposy='clip', basey=2)
        pl.xlabel(r'System size $N$')
        pl.xlim([1,3+Nrange[-1]+1])
        pl.xticks(range(1,3+Nrange[-1]+1))
        #pl.xlabel(r'Lattice length $N$')
        pl.ylabel(r'singular value $\sigma$')
        if(ti):
            pl.title(r'Extremal Singular values of TI $X_N$')
        else:
            pl.title(r'Extremal Singular values of $X_N$')
        pl.grid(b=True, which='major', color='black', linestyle='--')
        pl.grid(b=True, which='minor', color='orange', linestyle='--')
        for dD in range(len(dDlist)):
            d = dDlist[dD][0]
            D = dDlist[dD][1]
            ymin = [plotData[dD][N_][-1] for N_ in range(len(Nrange))]
            ymax = [plotData[dD][N_][0] for N_ in range(len(Nrange))]
            x = list(map(lambda x:x+2+dD/(len(dDlist)+1),Nrange))
            pl.plot(x,ymin,'-o',label='(D,d) = ('+ str(D)+','+str(d)+')')
            pl.plot(x,ymax,'-o')#,label='(D,d) = ('+ str(Drange[D])+','+str(drange[d])+')')
        lgd = pl.legend(loc='center left',ncol=1,bbox_to_anchor=(1,0.5))
        pl.savefig('analysis20_extr.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')
    

""" analysis20 was merged with analysis19 """


def analysis21(dDlist,N,normalize=False,tol=1e-8):
    """ Description:
        ----------------------------------------------------------------------------------------------------------------------------------------------
        Analyzes relation between system size N and bond dimension D for inverse MPO of a TI-MPO
        
        i)      Generate random A in C^(d x d)_(D x D),l in C^(dxd)_D, r in C^(dxd)_D
        ii)     Compose TI-MPO X_N as: X^(s0s1...sNs(N+1))_(t0t1...tNt(N+1)) = l^s0_t0 A^s1_t1 ... A^sN_tN l^s(N+1)_t(N+1)

                           |   |   |         |   |
                     X_N = l - A - A - ... - A - r
                           |   |   |         |   |
        iii)    Contract to matrix X, invert and decompose inverse Xi through repeated SVD:
                            |    |             |              |     |
                     Xi_N = M1 - M2 - ... - M(N/2) - ... - M(N-1) - MN
                            |    |             |              |     |
        iv)     Find maximum bond dimensions of Xi's MPO decomposition
        
        Parameters:
        ----------------------------------------------------------------------------------------------------------------------------------------------
        Ddlist:         A list of tuples [[D1,d1],[D2,d2],...,[Dm,dm]] setting physical and bond dimensions of the constructed tensors A,l,r
        N:              Natural number > 0 specifying the maximum number of sites (N+2) in the system.
        normalize:      Draws normalized initial tensors l,A,r
        tol:            Truncates singular values smaller than 1e-8 in the SVD
    """ 
    for dD in dDlist:
        if(not(len(dD) == 2)):
            raise Exception("Tuples of the type (D,d) are expected.")
        if(not(isinstance(dD[0],int) and isinstance(dD[1],int)) or dD[0] < 1 or dD[1] < 1):
            raise Exception("D,d have to be integers greater than 0.")
    
    Nrange = range(0,N)
    plotData = np.zeros([len(dDlist),len(Nrange)])
    for dD in range(len(dDlist)):
        d = dDlist[dD][0]
        D = dDlist[dD][1]
        A,l,r = createSet(d,D,direction=1,normalize=normalize)
        L = [l]
        R = [r]
        for N_ in range(len(Nrange)):
            X = contractMPO(L + R)[0,:,0,:]
            Xi = la.inv(X)
            XiMPO = SVDMat(Xi,d,N_,tol=tol)
            maxD = max(map(lambda x: max(x.shape[0],x.shape[2]),XiMPO))
            plotData[dD,N_] = maxD
            L = L + [A]
            gc.collect()
        
    pl.figure()
    pl.yscale('log', nonposy='clip', basey=10)
    pl.xlabel(r'System size $N$')
    pl.xlim([Nrange[0]-1,Nrange[-1]+1])
    pl.xticks(range(0,Nrange[-1]+1))
    #pl.xlabel(r'Lattice length $N$')
    pl.ylabel(r'Bond dimension $D$')
    pl.title(r'Bond dimension D at Nth iteration')
    pl.grid(b=True, which='major', color='black', linestyle='--')
    pl.grid(b=True, which='minor', color='orange', linestyle='--')
    #return(plotData)
    
    #x = list(itertools.chain(*[[Nrange[N]]*len(data[N]) for N in range(len(Nrange))]))
    for dD in range(len(dDlist)):
        d = dDlist[dD][0]
        D = dDlist[dD][1]
        x = Nrange
        #x = list(itertools.chain(*[[Nrange[N]]*len(data[N]) for N in range(len(Nrange))]))
        y = plotData[dD,:]
        pl.plot(x,y,'o',label='(d,D) = (' + str(d) + ',' + str(D) + ')')
    
    pl.legend(loc='best')
    pl.savefig("analysis21_normalize" + str(int(normalize)) + ".pdf")
    pl.show()
    
def analysis22(dDlist,N):
    """ Description:
        ----------------------------------------------------------------------------------------------------------------------------------------------
        Analyzes how the max-norm and the Frobenius tipically scale with respect to each other for a TI-MPO X_N with increasing system size N.
        
        Parameters:
        ----------------------------------------------------------------------------------------------------------------------------------------------
        Ddlist:         A list of tuples [[D1,d1],[D2,d2],...,[Dm,dm]] setting physical and bond dimensions of the constructed tensors A,l,r
        N:              Natural number > 0 specifying the maximum number of sites (N+2) in the system.
    """
    for dD in dDlist:
        if(not(len(dD) == 2)):
            raise Exception("Tuples of the type (D,d) are expected.")
        if(not(isinstance(dD[0],int) and isinstance(dD[1],int)) or dD[0] < 1 or dD[1] < 1):
            raise Exception("D,d have to be integers greater than 0.")
    
    Nrange = range(0,N)
    plotData = np.zeros([len(dDlist),len(Nrange)])
    for dD in range(len(dDlist)):
        d = dDlist[dD][0]
        D = dDlist[dD][1]
        A,l,r = createSet(d,D,direction=1)
        L = [l]
        R = [r]
        for N_ in range(len(Nrange)):
            plotData[dD,N_] = normMPO(L+R) - max(map(abs,contractMPO(L + R)[0,:,0,:].flatten()))
            L = L + [A]
    
    pl.figure()
    pl.yscale('log', nonposy='clip', basey=10)
    pl.xlabel(r'System size $N$')
    pl.xlim([Nrange[0]-1,Nrange[-1]+1])
    pl.xticks(range(0,Nrange[-1]+1))
    #pl.xlabel(r'Lattice length $N$')
    pl.ylabel(r'Difference of max norm and Frobenius norm')
    pl.title(r'$\Vert X \Vert_F - \Vert X \Vert_\max$')
    pl.grid(b=True, which='major', color='black', linestyle='--')
    pl.grid(b=True, which='minor', color='orange', linestyle='--')
    #x = list(itertools.chain(*[[Nrange[N]]*len(data[N]) for N in range(len(Nrange))]))
    parameters = []
    for dD in range(len(dDlist)):
        d = dDlist[dD][0]
        D = dDlist[dD][1]
        x = Nrange
        y = plotData[dD,:]
        params, cov = fit(fitmodel,Nrange,list(map(lambda x: math.log(x),y)))
        parameters.append(params[1])
        pl.plot(Nrange,list(map(lambda x:math.exp(fitmodel(x,params[0],params[1])),Nrange)),'-')
        pl.plot(x,y,'o',label='(d,D) = (' + str(d) + ',' + str(D) + ')')
        
    pl.legend(loc='best')
    pl.savefig('analysis22.pdf')
    pl.show()
    return(parameters)

def analysis16_(dDlist,N,block=0,seed=None,sweep=None):
    """ 
        A less efficient implementation of analysis16
    """
    
    for dD in dDlist:
        if(not(len(dD) == 2)):
            raise Exception("Tuples of the type (D,d) are expected.")
        if(not(isinstance(dD[0],int) and isinstance(dD[1],int)) or dD[0] < 1 or dD[1] < 1):
            raise Exception("D,d have to be integers greater than 0.")
            
    #check = []
    Nrange = range(1,N)
    plotData = np.zeros([len(dDlist),len(Nrange)])
    for dD in range(len(dDlist)):
        d = dDlist[dD][0]
        D = dDlist[dD][1]
        if seed is not None:
            rand.seed(seed)
        A,L,R = createSet(d,D,direction=1,normalize=True)
        X_N = contractMPO([L] + [A]*block + [R])[0,:,0,:]
        Xi_NMPO = SVDMat(la.inv(X_N),d,block,False)
        X_NMPO = [L] + [A]*block + [R]
        L,R,Li,Ri = [L] + [A]*int(np.ceil(block/2)),[A]*int(block - np.ceil(block/2)) + [R],Xi_NMPO[0:int(np.ceil(block/2) + 1)],Xi_NMPO[int(np.ceil(block/2) + 1):]
        
        for N_ in range(len(Nrange)):
            X_NMPO = L + [A] + R
            Xi_NMPO = Li + [[]] + Ri
            
            trs = list(range(len(L)-int(block/2)-1)) + list(range(-1,-len(R)+int(block/2),-1))
            fidx = int(block-(block%2)+3)
            Ai = solveMPO(np.identity(d**fidx)*d**(block+N_+3-fidx),X_NMPO,Xi_NMPO,tr_sites = trs)
            
            """ X_N -> X_(N+1) and Xi_N -> Xi_(N+1)"""
            if(N_%2 == 0):
                R = [A] + R
                Ri = [Ai] + Ri
            else:
                L = L + [A]
                Li = Li + [Ai]
            """Sweep:"""
            if(not(sweep == None)):
                X_NMPO,Xi_NMPO = sweeper(L,Li,R,Ri,A,block)
                L,R = X_NMPO[:len(L)],X_NMPO[len(L):]
                Li,Ri = Xi_NMPO[:len(L)],Xi_NMPO[len(L):]
                for i in range(sweep):
                    X_NMPO,Xi_NMPO = sweeper(L,Li,R,Ri,A,block,direction=1)
                    L,R = X_NMPO[:-1],X_NMPO[-1:]
                    Li,Ri = Xi_NMPO[:-1],Xi_NMPO[-1:]
                    X_NMPO,Xi_NMPO = sweeper(L,Li,R,Ri,A,block,direction=0)
                    L,R = X_NMPO[:1],X_NMPO[1:]
                    Li,Ri = Xi_NMPO[:1],Xi_NMPO[1:]
                    
                L,R = X_NMPO[:len(L)],X_NMPO[len(L):]
                Li,Ri = Xi_NMPO[:len(L)],Xi_NMPO[len(L):]
            
            E_ = prodMPO(L+R,Li+Ri)
            gc.collect()
            plotData[dD,N_] = normMPO(sumMPO(E_,scaleMPO(-1,identityMPO(E_))))
            sys.stdout.write("\rProgress: "+str(N_+1)+"/"+ str(len(Nrange)) +"norms calculated.")
        
    pl.figure()
    pl.yscale('log', nonposy='clip')
    #pl.xlabel(r'Lattice length $N$')
    pl.ylabel(r'Approximation deviation $\Vert X\tilde{X}^{-1} - E \Vert$')
    lim = [min(list(filter(lambda x:x>0,plotData.flatten())))/10,max(list(filter(lambda x:x>0,plotData.flatten())))*10]
    pl.ylim(lim)
    pl.xlabel(r'Iteration step $n$')
    pl.xlim([0,math.ceil(N*1.25)])
    pl.title(r'Quality of approximation for initial block size ' + str(block))
    pl.grid(b=True, which='major', color='black', linestyle='--')
    pl.grid(b=True, which='minor', color='orange', linestyle='--')
    #pl.xticks(range(len(S)+1))
    for dD in range(len(dDlist)):
        d = dDlist[dD][0]
        D = dDlist[dD][1]
        y = plotData[dD,:]
        x = list(map(lambda x:x+dD/(len(dDlist)+1),range(len(y))))
        pl.plot(x,y,'-o',label='(D,d) = ('+ str(D)+','+str(d)+')')
    pl.legend(loc='best')
    pl.savefig('analysis16#_N' + str(N) + 'block' + str(block) + '.pdf')
    pl.show()
        
def sweeper(L,Li,R,Ri,A,block,direction=0):
    if(direction == 0):
        if(len(L)-1>int(block/2)):
            d = A.shape[1]
            fidx = int(block-(block%2)+3)
            X_NMPO = L + [A] + R
            Xi_NMPO = Li + [[]] + Ri
            
            trs = list(range(len(L)-int(block/2)-1)) + list(range(-1,-len(R)+int(block/2),-1))
            Ai = solveMPO(np.identity(d**fidx)*d**(len(X_NMPO)-fidx),X_NMPO,Xi_NMPO,tr_sites = trs)
            return(sweeper(L[:-1],Li[:-1],[A] + R,[Ai] + Ri,A,block))
        else:
            return(L + R,Li + Ri)
    else:
        if(len(R)-1>int(block/2)):
            d = A.shape[1]
            fidx = int(block-(block%2)+3)
            X_NMPO = L + [A] + R
            Xi_NMPO = Li + [[]] + Ri
            
            trs = list(range(len(L)-int(block/2)-1)) + list(range(-1,-len(R)+int(block/2),-1))
            Ai = solveMPO(np.identity(d**fidx)*d**(len(X_NMPO)-fidx),X_NMPO,Xi_NMPO,tr_sites = trs)
            return(sweeper(L + [A],Li + [Ai],R[1:],Ri[1:],A,block))
        else:
            return(L + R,Li + Ri)



def generator(D,d,block,direction=1,normalize=True,imaginary=True,seed=None):
    if seed is not None:
        rand.seed(seed)
    A,l,r = createSet(d,D,direction=direction,normalize=normalize,imaginary=imaginary)
    XMPO = [l] + [A]*block + [r]
    """if(canonical):
        XMPO = canonizeMPO(XMPO,normed,direction=0)"""
    L = XMPO[0:len(XMPO)//2 - (1-len(XMPO)%2)]
    M = XMPO[len(XMPO)//2 - (1-len(XMPO)%2):len(XMPO)//2 + 1]
    R = XMPO[len(XMPO)//2 + 1:]
    
    X = contractMPO(XMPO)[0,:,0,:]
    Xi = la.inv(X)
    
    XiMPO = SVDMat(Xi,d,block,False)#,tol=1E-12)
    """if(canonical):
        XiMPO = canonizeMPO(XiMPO,direction=0)"""
    Li = XiMPO[0:len(XiMPO)//2 - (1-len(XiMPO)%2)]
    Mi = XiMPO[len(XiMPO)//2 - (1-len(XiMPO)%2):len(XiMPO)//2 + 1]
    Ri = XiMPO[len(XiMPO)//2 + 1:]
    return([L,M,R,Li,Mi,Ri])

def plotEntries(c,n,maxcol):
    """n: Iteration step. Use like this:
    c = generator(D,d,block)
    plot(c,n)
    """
    contractMPO_(prodMPO(c[0],c[3]) + prodMPO(c[1],c[4])*n + prodMPO(c[2],c[5]),maxcol,plot=True)