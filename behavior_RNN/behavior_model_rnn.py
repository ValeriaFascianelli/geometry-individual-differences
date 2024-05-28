# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 14:34:18 2023

@author: valer
"""

import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm 
import random 


def get_linear_regression_rnn(nSubSamples,trials_id,rt_cond,indep_variables):
    weights = np.nan*np.ones((nSubSamples,7))
    score = np.nan*np.ones((nSubSamples))
    
    ################ linear regression
    for iSubsample in range(nSubSamples):
        for iCond  in range(8):                    
            
            nTrials  = len(trials_id[iCond])
            trials   = (np.array(random.sample(range(nTrials), 1000))).astype(int)
            y_app    = np.array(trials_id[iCond])[trials]
            rt_app   = np.array(rt_cond[iCond])[trials]
            x_app    = np.tile(indep_variables[iCond,:],(1000,1))
            
            if iCond == 0:
                x = x_app
                y = y_app
                rt = rt_app
            else:
                x = np.vstack((x,x_app))
                y = np.concatenate((y,y_app))
                rt = np.concatenate((rt,rt_app))
                               
        poly = PolynomialFeatures(interaction_only=True)
        data_int = poly.fit_transform(x)
        mod = sm.OLS(rt, data_int)
        res = mod.fit()
        res.summary()
        weights[iSubsample,:] = np.abs(res.params)/np.sum(np.abs(res.params))
        score[iSubsample]     = res.rsquared
         
    return weights, score    


def get_complete_correct_trials(comp, samples):
    
    ## correct trials
    correct_ID= []
    for i in range(0,samples):                 
        if comp[i]['n_dec']>.5 and comp[i]['n_corr']>.5:
            correct_ID.append(i)        
    
    
    ## complete trials
    complete_ID= []
    for i in range(0,samples):                 
        if comp[i]['n_dec']>.5 :
            complete_ID.append(i) 
            
    return correct_ID,complete_ID


if __name__ == "__main__":
    
  
    samples   = 10000
    nCond     = 8
    nSubSamples = 100
    ##############################################
    ### generate dict
    ###            previous    cue     current                      
    ### cond 0 :   right (1)   YS (1)  right (2)
    ### cond 1 :   lef   (0)   YS (1)  left  (1)
    ### cond 2 :   right (1)   PS (3)  left  (1)
    ### cond 3 :   left  (0)   PS (3)  right (2)
    ### cond 4 :   right (1)   VR (0)  right (2)
    ### cond 5 :   left  (0)   VR (0)  left  (1)
    ### cond 6 :   right (1)   HR (2)  left  (1)
    ### cond 7 :   left  (0)   HR (2)  right (2)
    
    cond_dict = {}
    cond_dict[str([1,1])]=0
    cond_dict[str([0,1])]=1
    cond_dict[str([1,3])]=2
    cond_dict[str([0,3])]=3
    cond_dict[str([1,0])]=4
    cond_dict[str([0,0])]=5
    cond_dict[str([1,2])]=6
    cond_dict[str([0,2])]=7
    ################################################
     
    ### independent behavioral variables for regression
    indep_variables = np.nan*np.ones((8,3))
    indep_variables[0,:] = [-1,-1,-1]
    indep_variables[1,:] = [1,-1,-1]
    indep_variables[2,:] = [-1,1,-1]
    indep_variables[3,:] = [1,1,-1]
    indep_variables[4,:] = [-1,-1,1]
    indep_variables[5,:] = [1,-1,1]
    indep_variables[6,:] = [-1,1,1]
    indep_variables[7,:] = [1,1,1]
    #######################################################
    
    
  
    # Open the file in binary read mode and load the data into a variable
    file_path = '/behavior_RNN/comp_net1.pkl'
    with open(file_path, 'rb') as file:
        comp = pickle.load(file)
    
    
    file_path = '/behavior_RNN/trials_id_net1.pkl'
    with open(file_path, 'rb') as file:
        trials_id = pickle.load(file)
        
    file_path = '/behavior_RNN/rt_cond_net1.pkl'
    with open(file_path, 'rb') as file:
        rt_cond = pickle.load(file)
    
    corr_trials, compl_trials = get_complete_correct_trials(comp, samples)    
    weights, score_net1=get_linear_regression_rnn(nSubSamples,trials_id,rt_cond,indep_variables)                            
    
    ### normalize weights
    sum_prova = np.sum(weights[:,1:],axis=1)
    sum_prova_tile=np.transpose(np.tile(sum_prova,[6,1]))
    weights_norm_net1=weights[:,1:]/sum_prova_tile 
            
    
    ##### plot 
    _,ax=plt.subplots(2,2, figsize=(15,10))
    ax[0,1].errorbar(x=range(6), y=np.mean(weights_norm_net1,axis=0)*100,yerr=2*np.std(weights_norm_net1,axis=0)*100, capsize=8, color='k', ls='none', marker='o',markersize=10)
    ax[0,1].set_title('NET 1:Normalized Weigths', fontsize=20)
    ax[0,0].errorbar(x=[0,4,8], y=np.mean(weights_norm_net1[:,:3],axis=0)*100,yerr=2*np.std(weights_norm_net1[:,:3],axis=0)*100, capsize=8, color='k', ls='none',marker='o',markersize=10)
    ax[0,1].set_xticks( range(6),['Previous', 'Rule','Shape','PreviousANDRule','PreviousANDShape','RuleANDShape'],fontsize=15)
    ax[0,1].set_ylim([0,90])
    ax[0,0].set_ylabel('Normalized weights(%)[a.u.]',fontsize=20)
    ax[0,0].set_xticks([0,4,8],['Previous', 'Rule','Shape'],fontsize=15)
    ax[0,0].set_ylabel('Normalized weights(%) [a.u.]',fontsize=20)
    ax[0,0].set_ylim([-0.9,11])
    ax[0,0].set_title('NET 1:Normalized Weigths', fontsize=20)
    plt.suptitle('RNN: Normalized Weights of multi-linear regression', fontsize=20)
    
    
    
    ### NET 2 

    file_path = '/behavior_RNN/comp_net2.pkl'
    with open(file_path, 'rb') as file:
        comp = pickle.load(file)
    
    
    file_path = '/behavior_RNN/trials_id_net2.pkl'
    with open(file_path, 'rb') as file:
        trials_id = pickle.load(file)
        
    file_path = '/behavior_RNN/rt_cond_net2.pkl'
    with open(file_path, 'rb') as file:
        rt_cond = pickle.load(file)    
       
    corr_trials, compl_trials = get_complete_correct_trials(comp, samples)  
   
    nSubSamples = 100
    weights, score_net2 =get_linear_regression_rnn(nSubSamples,trials_id,rt_cond,indep_variables)                            
    
    ### normalize weights
    sum_prova = np.sum(weights[:,1:],axis=1)
    sum_prova_tile=np.transpose(np.tile(sum_prova,[6,1]))
    weights_norm_net2=weights[:,1:]/sum_prova_tile 
 
    ##### plot 
    #_,ax=plt.subplots(2,2, figsize=(15,10))
    ax[1,1].errorbar(x=range(6), y=np.mean(weights_norm_net2,axis=0)*100,yerr=2*np.std(weights_norm_net2,axis=0)*100, capsize=8, color='k', ls='none', marker='o',markersize=10)
    ax[1,1].set_title('NET 2:Normalized Weigths', fontsize=20)
    ax[1,0].errorbar(x=[0,4,8], y=np.mean(weights_norm_net2[:,:3],axis=0)*100,yerr=2*np.std(weights_norm_net2[:,:3],axis=0)*100, capsize=8, color='k', ls='none',marker='o',markersize=10)
    ax[1,1].set_xticks( range(6),['Previous', 'Rule','Shape','PreviousANDRule','PreviousANDShape','RuleANDShape'],fontsize=15)
    ax[1,1].set_ylim([0,90])
    ax[1,0].set_ylabel('Normalized weights(%)[a.u.]',fontsize=20)
    ax[1,0].set_xticks([0,4,8],['Previous', 'Rule','Shape'],fontsize=15)
    ax[1,0].set_ylabel('Normalized weights(%) [a.u.]',fontsize=20)
    ax[1,0].set_ylim([-0.9,11])
    ax[1,0].set_title('NET 2:Normalized Weigths', fontsize=20)
    plt.suptitle('RNN: Normalized Weights of multi-linear regression', fontsize=20)
    
    
 
