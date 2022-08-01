#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 10:20:40 2022

@author: mvera
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def convex_envelope(x, fs):
    """Computes convex envelopes of M functions which share a common grid.
    x is an (N, D)-matrix corresponding to the grid in D-dimensional space and fs is an (M, N)-matrix.
    The i-th function is given by the pairs (x[0], fs[i, 0]), ..., (x[N-1], fs[i, N-1]).
    The envelopes are returned as a list of lists.
    The i-th list is of the form [j_1, j_2, ..., j_n] where (X[j_k], fs[i, j_k]) is a point in the envelope.
    
    Keyword arguments:
    x  -- A shape=(N,D) numpy array.
    fs -- A shape=(M,N) or shape=(N,) numpy array."""
    
    assert(len(fs.shape) <= 2)
    if len(fs.shape) == 1: fs = np.reshape(fs, (-1, fs.shape[0]))
    M, N = fs.shape
    
    assert(len(x.shape) <= 2)
    if len(x.shape) == 1: x = np.reshape(x, (-1, 1))
    assert(x.shape[0] == N)
    D = x.shape[1]
    
    fs_pad = np.empty((M, N+2))
    fs_pad[:, 1:-1], fs_pad[:, (0,-1)] = fs, np.max(fs) + 1.
    
    x_pad = np.empty((N+2, D))
    x_pad[1:-1, :], x_pad[0, :], x_pad[-1, :] = x, x[0, :], x[-1, :]
    
    results = []
    for i in range(M):
        epi = np.column_stack((x_pad, fs_pad[i, :]))
        hull = ConvexHull(epi)
        result = [v-1 for v in hull.vertices if 0 < v <= N]
        result.sort()
        results.append(result)
    return np.array(results[0])



# Epoch - LR - wc - bs - at - loss 
file_name = str(Path.home()) + '/TOA_causal/log.txt'
M= np.loadtxt(file_name)
best = np.argmin(M[:,5])
print(f'best results: {M[best,:]}')
M_benchmark = M[M[:,4]==0.0,:]
best_benchmark = np.argmin(M_benchmark[:,5])
print(f'best benchmark results: {M_benchmark[best_benchmark,:]}')
M_causal= M[M[:,4]!=0.0,:]
best_causal = np.argmin(M_causal[:,5])
print(f'best causal results: {M_causal[best_causal,:]}')

lr,wc,bs,tau =M_causal[best_causal,1:5]
idxs = (M_causal[:,1]==lr) & (M_causal[:,2]==wc) & (M_causal[:,3]==bs) & (M_causal[:,4]==tau)
plt.plot(M_causal[idxs,0],M_causal[idxs,5], label = 'Causal')
lr,wc,bs =M_benchmark[best_benchmark,1:4]
idxs = (M_benchmark[:,1]==lr) & (M_benchmark[:,2]==wc) & (M_benchmark[:,3]==bs) & (M_benchmark[:,4]==0.0)
plt.plot(M_benchmark[idxs,0],M_benchmark[idxs,5], label = 'Benchmark')
plt.axis([1,50,0,0.2])
plt.legend()
plt.show() 


alphas = np.unique(M[:,1])
weight_costs = np.unique(M[:,2])
batchsizes = np.unique(M[:,3])
thresholds = np.unique(M[:,4])

## Focalizo sobre el wc - Parece no ser necesario

for wc in weight_costs:
    M_new = M_causal[M_causal[:,2]==wc,:]
    best = np.argmin(M_new[:,5])
    lr,bs,tau = M_new[best,[1,3,4]]
    idxs = (M[:,1]==lr) & (M[:,2]==wc) & (M[:,3]==bs) & (M[:,4]==tau)
    x = M[idxs,0]
    y = M[idxs,5]
    envelopes = convex_envelope(x,y)    
    plt.plot(x[envelopes],y[envelopes], label = 'Causal with wc = '+str(wc))
    M_new = M_benchmark[M_benchmark[:,2]==wc,:]
    best = np.argmin(M_new[:,5])
    lr,bs = M_new[best,[1,3]]
    idxs = (M[:,1]==lr) & (M[:,2]==wc) & (M[:,3]==bs) & (M[:,4]==0.0)
    x = M[idxs,0]
    y = M[idxs,5]
    envelopes = convex_envelope(x,y)    
    plt.plot(x[envelopes],y[envelopes], label = 'Benchmark with wc = '+str(wc))
plt.axis([1,50,0.002,0.05])
plt.legend()
plt.show()   

## Focalizo sobre el bs - Siempre gana el mas chico
for bs in batchsizes:
    M_new = M_causal[M_causal[:,3]==bs,:]
    best = np.argmin(M_new[:,5])
    lr,wc,tau = M_new[best,[1,2,4]]
    idxs = (M[:,1]==lr) & (M[:,2]==wc) & (M[:,3]==bs) & (M[:,4]==tau)
    x = M[idxs,0]
    y = M[idxs,5]
    envelopes = convex_envelope(x,y)    
    plt.plot(x[envelopes],y[envelopes], label = 'Causal with bs = '+str(bs))
    M_new = M_benchmark[M_benchmark[:,3]==bs,:]
    best = np.argmin(M_new[:,5])
    lr,wc = M_new[best,[1,2]]
    idxs = (M[:,1]==lr) & (M[:,2]==wc) & (M[:,3]==bs) & (M[:,4]==0.0)
    x = M[idxs,0]
    y = M[idxs,5]
    envelopes = convex_envelope(x,y)    
    plt.plot(x[envelopes],y[envelopes], label = 'Benchmark with bs = '+str(bs))
plt.axis([1,50,0.002,0.05])
plt.legend()
plt.show() 

## Focalizo sobre el lr - El de 0.01 no anda; Mas chico que 0.0001 no tiene sentido en 50 epochs

for lr in alphas:
    M_new = M_causal[M_causal[:,1]==lr,:]
    best = np.argmin(M_new[:,5])
    wc,bs,tau = M_new[best,[2,3,4]]
    idxs = (M[:,1]==lr) & (M[:,2]==wc) & (M[:,3]==bs) & (M[:,4]==tau)
    x = M[idxs,0]
    y = M[idxs,5]
    envelopes = convex_envelope(x,y)    
    plt.plot(x[envelopes],y[envelopes], label = 'Causal with lr = '+str(lr))
    M_new = M_benchmark[M_benchmark[:,1]==lr,:]
    best = np.argmin(M_new[:,5])
    wc,bs = M_new[best,[2,3]]
    idxs = (M[:,1]==lr) & (M[:,2]==wc) & (M[:,3]==bs) & (M[:,4]==0.0)
    x = M[idxs,0]
    y = M[idxs,5]
    try: 
        envelopes = convex_envelope(x,y)    
        plt.plot(x[envelopes],y[envelopes], label = 'Benchmark with lr = '+str(lr))
    except:
        print(f'Imposible {lr},{wc},{bs}')
plt.axis([1,50,0.002,0.05])
plt.legend()
plt.show() 

## Focalizo en el umbral
for tau in thresholds:
    M_new = M[M[:,4]==tau,:]
    best = np.argmin(M_new[:,5])
    lr,wc,bs = M_new[best,[1,2,3]]
    idxs = (M[:,1]==lr) & (M[:,2]==wc) & (M[:,3]==bs) & (M[:,4]==tau)
    x = M[idxs,0]
    y = M[idxs,5]
    envelopes = convex_envelope(x,y)    
    plt.plot(x[envelopes],y[envelopes], label = 'Causal with tau = '+str(tau))
plt.axis([1,50,0.002,0.05])
plt.legend()
plt.show() 