# -*- coding: utf-8 -*-
"""
@author: Valeria Fascianelli, 2023
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import re
import seaborn as sns
import scipy
import pickle


if __name__ == "__main__": 
    
    colors = ['orange', 'blue', 'green', 'red']    
    
    ### read data: set the main path where the data are stored
    main_path = '/neural_RNN/'
    
    # Open the file in binary read mode and load the data into a variable
    file_path = main_path+'decoding_results.pkl'
    with open(file_path, 'rb') as file:
        decoding_results = pickle.load(file)
    file_path = main_path+'ccgp_results.pkl'
    with open(file_path, 'rb') as file:
        ccgp_results = pickle.load(file)
    file_path = main_path+'decoding_input.pkl'
    with open(file_path, 'rb') as file:
        decoding_input = pickle.load(file)    
    file_path = main_path+'ccgp_input.pkl'
    with open(file_path, 'rb') as file:
        ccgp_input = pickle.load(file)  
    file_path = main_path+'rt_cond.pkl'
    with open(file_path, 'rb') as file:
        rt_cond = pickle.load(file)      
    file_path = main_path+'perf_cond.pkl'
    with open(file_path, 'rb') as file:
        perf_cond = pickle.load(file)    
    file_path = main_path+'id_nets.pkl'
    with open(file_path, 'rb') as file:
        id_nets = pickle.load(file)  
    file_path = main_path+'max_epoch.pkl'
    with open(file_path, 'rb') as file:
        max_epoch = pickle.load(file)   
    file_path = main_path+'max_epoch_normalize.pkl'
    with open(file_path, 'rb') as file:
        max_epoch_normalize = pickle.load(file)  
    file_path = main_path+'time.pkl'
    with open(file_path, 'rb') as file:
        time = pickle.load(file)      
        
        
    
    decoding_results = np.squeeze(decoding_results)
    ccgp_results     = np.squeeze(ccgp_results)
    
    decoding_input = np.squeeze(decoding_input)
    ccgp_input     = np.squeeze(ccgp_input)   
       
         
    iSeed=90
    rt_cond_compr=[]
    perf_cond_compr = []
    rt_cond_compr_std = []
    for j, kk in enumerate(range(0,8,2)):
        rt_cond_compr.append((rt_cond[iSeed,0,0,kk]+rt_cond[iSeed,0,0,kk+1])/2) #mean
        perf_cond_compr.append((perf_cond[iSeed,0,kk]+perf_cond[iSeed,0,kk+1])/2)
    
    
    fig_vis, axs = plt.subplots(nrows=2,ncols=2,dpi=500,figsize=(4,12))
    plt.suptitle('NET1: Figure 7A-B')
    
    
    axs[0,0].plot(time,decoding_results[iSeed,:,0], color=colors[0],lw=2.5)
    axs[0,0].plot(time,decoding_results[iSeed,:,1], color=colors[1],lw=2.5)
    axs[0,0].plot(time,decoding_results[iSeed,:,2], color=colors[2],lw=2.5)
    axs[0,0].plot(time,decoding_results[iSeed,:,3], color=colors[3],lw=2.5)
    axs[0,0].set_ylabel('Decoding')
    axs[0,0].tick_params(axis='x', labelsize=5)
    axs[0,0].tick_params(axis='y', labelsize=5)
    
    
    axs[0,1].plot(time,ccgp_results[iSeed,:,0], color=colors[0],lw=2.5)
    axs[0,1].plot(time,ccgp_results[iSeed,:,1], color=colors[1],lw=2.5)
    axs[0,1].plot(time,ccgp_results[iSeed,:,2], color=colors[2],lw=2.5)
    axs[0,1].plot(time,ccgp_results[iSeed,:,3], color=colors[3],lw=2.5)
    axs[0,1].set_ylabel('CCGP')
    axs[0,1].tick_params(axis='x', labelsize=5)
    axs[0,1].tick_params(axis='y', labelsize=5)
    
    axs[1,0].plot(range(4),rt_cond_compr,marker='o')
    axs[1,0].set_xticks(range(4),fontsize=5)
    axs[1,0].set_xticklabels(['YS','PS','VR','HR'],fontsize=5)
    axs[1,0].set_ylabel('Reaction time')
    axs[1,0].tick_params(axis='x', labelsize=5)
    axs[1,0].tick_params(axis='y', labelsize=5)
    
    axs[1,1].plot(range(4),perf_cond_compr,'o')
    axs[1,1].set_xticks(range(4),fontsize=5)
    axs[1,1].set_xticklabels(['YS','PS','VR','HR'],fontsize=5)
    axs[1,1].set_ylabel('Performance')
    axs[1,1].tick_params(axis='x', labelsize=5)
    axs[1,1].tick_params(axis='y', labelsize=5)
    axs[1,1].set_ylim([0,1])
    
    fig_vis.tight_layout()
              
        
    
    iSeed=72    
    rt_cond_compr=[]
    perf_cond_compr = []
    for j, kk in enumerate(range(0,8,2)):
        rt_cond_compr.append((rt_cond[iSeed,0,0,kk]+rt_cond[iSeed,0,0,kk+1])/2) #mean
        perf_cond_compr.append((perf_cond[iSeed,0,kk]+perf_cond[iSeed,0,kk+1])/2)
    
    
    fig_vis, axs = plt.subplots(nrows=2,ncols=2,dpi=500,figsize=(4,12))
    plt.suptitle('NET 2: Figure 7C-D')
    axs[0,0].plot(time,decoding_results[iSeed,:,0], color=colors[0],lw=2.5)
    axs[0,0].plot(time,decoding_results[iSeed,:,1], color=colors[1],lw=2.5)
    axs[0,0].plot(time,decoding_results[iSeed,:,2], color=colors[2],lw=2.5)
    axs[0,0].plot(time,decoding_results[iSeed,:,3], color=colors[3],lw=2.5)
    axs[0,0].set_ylabel('Decoding')
    axs[0,0].tick_params(axis='x', labelsize=5)
    axs[0,0].tick_params(axis='y', labelsize=5)
    
  
    axs[0,1].plot(time,ccgp_results[iSeed,:,0], color=colors[0],lw=2.5)
    axs[0,1].plot(time,ccgp_results[iSeed,:,1], color=colors[1],lw=2.5)
    axs[0,1].plot(time,ccgp_results[iSeed,:,2], color=colors[2],lw=2.5)
    axs[0,1].plot(time,ccgp_results[iSeed,:,3], color=colors[3],lw=2.5)
    axs[0,1].set_ylabel('CCGP')
    axs[0,1].tick_params(axis='x', labelsize=5)
    axs[0,1].tick_params(axis='y', labelsize=5)
    
  
    axs[1,0].plot(range(4),rt_cond_compr,'*')
    axs[1,0].set_xticks(range(4),fontsize=5)
    axs[1,0].set_xticklabels(['YS','PS','VR','HR'],fontsize=5)
    axs[1,0].set_ylabel('Reaction time')
    axs[1,0].tick_params(axis='x', labelsize=5)
    axs[1,0].tick_params(axis='y', labelsize=5)
    
    axs[1,1].plot(range(4),perf_cond_compr,'*')
    axs[1,1].set_xticks(range(4),fontsize=5)
    axs[1,1].set_xticklabels(['YS','PS','VR','HR'],fontsize=5)
    axs[1,1].set_ylabel('Performance')
    axs[1,1].tick_params(axis='x', labelsize=5)
    axs[1,1].tick_params(axis='y', labelsize=5)
    axs[1,1].set_ylim([0,1])
    
    fig_vis.tight_layout()
        
    ##### correlation of all networks
    all_nets_data = np.nan*np.ones((np.size(decoding_results,axis=0),4))
    all_nets_input = np.nan*np.ones((np.size(decoding_results,axis=0),4))
    all_seeds = np.arange(0,100)
    _,all_axes = plt.subplots(nrows=2,ncols=4,figsize=(4,2),dpi=300)
    
    bin_edges = [0,15]
    shape_info= []
    overall_perf = []
    amount_train = []
    neural_data = []
    rt_data = []
    id_rt = []
    name_net = []
    cmap_val = []
    shape_decoding = []
    shape_ccgp = []
    strategy_decoding = []
    strategy_ccgp = []
   
    bad_nets = [95,51,73,41,19,75,55,35,31,92] + [18,46,21,17,58,29,47,60,0,48,63,79]
    for iseed in all_seeds:
        
        if np.isin(iseed, bad_nets):
            continue
            
        rt_cond_compr=[]
        perf_cond_compr = []
        for j, kk in enumerate(range(0,8,2)):
            rt1 = []
            rt_cond_compr.append((rt_cond[iseed,0,0,kk]+rt_cond[iseed,0,0,kk+1])/2) #mean
            perf_cond_compr.append((perf_cond[iseed,0,kk]+perf_cond[iseed,0,kk+1])/2)
                  
        ##### plot all the networks in a single scatter plot
        delta_shape_rt = np.abs(np.mean(rt_cond_compr[0:2]) - np.mean(rt_cond_compr[2:4]))
        delta_shape_pf = np.abs(np.mean(perf_cond_compr[0:2]) - np.mean(perf_cond_compr[2:4]))
    
        delta_strategy_rt = np.abs((rt_cond_compr[0]+rt_cond_compr[2])/2 - (rt_cond_compr[1]+rt_cond_compr[3])/2)
     
        delta_strategy_pf = np.abs((perf_cond_compr[0]+perf_cond_compr[2])/2 - (perf_cond_compr[1]+perf_cond_compr[3])/2)
    
        delta_net_rt = delta_shape_rt-delta_strategy_rt
        delta_net_pf = delta_shape_pf-delta_strategy_pf
    
        ### recurrent 
        shape_decoding.append(np.mean(decoding_results[iseed,bin_edges[0]:bin_edges[1],0],axis=0))
        strategy_decoding.append(np.mean(decoding_results[iseed,bin_edges[0]:bin_edges[1],1],axis=0))
        delta_net_decoding = shape_decoding[-1]-strategy_decoding[-1]
        
        shape_ccgp.append(np.mean(ccgp_results[iseed,bin_edges[0]:bin_edges[1],0],axis=0))
        strategy_ccgp.append(np.mean(ccgp_results[iseed,bin_edges[0]:bin_edges[1],1],axis=0))
        delta_net_ccgp = shape_ccgp[-1]-strategy_ccgp[-1]
        
        ## plot all decoding for each nets along time
        plt.suptitle('Supplementary Figure 6')
        all_axes[0,0].plot(time, decoding_results[iseed,:,0], color='orange') # decoding shape
        all_axes[1,0].plot(time, ccgp_results[iseed,:,0],  color='orange') # ccgp shape
        all_axes[0,0].set_title('SHAPE')
        all_axes[0,0].set_ylabel('decoding')
        all_axes[1,0].set_ylabel('ccgp')
                
        all_axes[0,1].plot(time, decoding_results[iseed,:,1], color='blue') # decoding strategy
        all_axes[1,1].plot(time, ccgp_results[iseed,:,1], color='blue') # ccgp strategy
        all_axes[0,1].set_title('STRATEGY')
        
        all_axes[0,2].plot(time, decoding_results[iseed,:,2], color='green') # decoding previous
        all_axes[1,2].plot(time, ccgp_results[iseed,:,2], color='green') # ccgp previous
        all_axes[0,2].set_title('PREVIOUS') 
        
        all_axes[0,3].plot(time, decoding_results[iseed,:,3], color='red') # decoding current
        all_axes[1,3].plot(time, ccgp_results[iseed,:,3], color='red') # ccgp current
        all_axes[0,3].set_title('CURRENT')
        
        index = np.where(np.array(id_nets)==iseed+1)[0][0]       
        
        amount_train.append(max_epoch[index])
        neural_data.append(delta_net_decoding)
        rt_data.append(delta_net_rt)
        name_net.append(str(iseed))
        if np.abs(delta_net_rt) <0.5:
            id_rt.append(0)           
        elif delta_net_rt < 0.5:
            id_rt.append(-1)            
        elif delta_net_rt > 0.5:
            id_rt.append(1)                
    
    
    plt.figure()
        
    cmap = plt.get_cmap('viridis', 3)
    ## scatter plot for Figure 6
    scatter = plt.scatter(amount_train, neural_data, c=id_rt, s=100, cmap=cmap) 
    plt.suptitle('Figure 6B')
    plt.xscale('log')
    cbar = plt.colorbar(scatter)
 
    ## scatter plot Supplementary Figure 5
    fig,new_axes = plt.subplots(nrows=1,ncols=2, figsize=(20,10))
    plt.suptitle('Supplemetary Figure 5')
    new_axes[0].set_xscale('log')
    new_axes[0].scatter(amount_train, shape_ccgp, s=100, color='orange')
    new_axes[0].set_ylabel('CCGP shape')    
    new_axes[1].set_xscale('log')
    new_axes[1].scatter(amount_train, strategy_ccgp, s=100, color='blue')
    new_axes[1].set_ylabel('CCGP strategy')
    
    
  
    
   
    
    
    
    
    
    
