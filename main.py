#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:09:13 2019
@author: ryan
"""
import os
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac 
import matplotlib.pyplot as plt 
import scipy.io as scio


def FindRank(X,max_rank=10,accuracy=1e-5,save_data=True):
    losses=[]
    factors=[]
    ranks=[]
    for i in range(max_rank):
        filename="Rank%s.data.npz"%i
        if os.path.exists(filename):
            factor,error=np.load(filename).values()
        else:
            factor,error=parafac(X,i+1,return_errors=True)
            if save_data:
                np.savez(filename,factor=factor,error=error)
        losses.append(error)
        factors.append(factor)
        ranks.append(i)
        if error[-1]<accuracy:
            break
    return losses,factors,ranks

def Plot_Single_Rank(errors,rank):
    plt.figure(0)
    plt.plot(np.arange(len(errors[rank-1])),errors[rank-1])
    plt.title("Errors of Iteration in rank %d"%rank)
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.savefig("single_plot_rank%d.jpg"%rank)

def Plot_Ranks(errors,ranks):
    plt.figure(1)
    losses=[i[-1] for i in errors]
    plt.plot(ranks,losses)
    plt.title("Errors of in each Rank")
    plt.xlabel("Ranks")
    plt.ylabel("error")
    plt.savefig("ranks_plot.jpg")

def main():
    X=scio.loadmat("Real/Pingsdata.mat")
    data=np.array(X['data'])
    errores,factors,ranks=FindRank(data,max_rank=20,save_data=True)
    Plot_Single_Rank(errores,20)
    Plot_Ranks(errores,ranks)
    
if __name__ =="__main__":
    main()
