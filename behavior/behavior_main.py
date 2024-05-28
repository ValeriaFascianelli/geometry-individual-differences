# -*- coding: utf-8 -*-
"""

@author: Valeria Fascianelli
"""


import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import PolynomialFeatures
import scipy.special
import statsmodels.api as sm 


def get_multi_linear(correct_trials,rt,conditions):
    
    number_correct_trials = []
    id_correct_trials     = []
    
    for icond,jcond in enumerate(conditions):
        ## balancing right-left for correct
        tot = np.r_[[ np.all(correct_trials[itrial,0:3]==jcond) for itrial in range(np.size(correct_trials, axis=0)) ]].sum()
        number_correct_trials.append(tot)
        id_tr = np.r_[[ np.all(correct_trials[itrial,0:3]==jcond) for itrial in range(np.size(correct_trials, axis=0)) ]]
        id_correct_trials.append(np.where(id_tr)[0])
    
    tot_resamples = 100
    resample = 0
    score    = []
    weights  = []
    min_number = np.min(number_correct_trials)
    correct_trials_balanced = np.nan*np.ones(( min_number*len(conditions),np.size(correct_trials,axis=1)))
    rt_analysis   = np.nan*np.ones(( min_number*len(conditions) ))

    while resample<tot_resamples:

        for iCond, jCond in enumerate(conditions):
           
            random_sample_corr = random.sample(list(id_correct_trials[iCond]), min_number)
            
            if iCond == 0:
                correct_trials_balanced[:min_number,:] = correct_trials[random_sample_corr,:] 
                rt_analysis[:min_number] = rt[random_sample_corr]
                continue
            correct_trials_balanced[iCond*min_number:(iCond+1)*min_number,:] =  correct_trials[random_sample_corr,:] 
            rt_analysis[iCond*min_number:(iCond+1)*min_number] = rt[random_sample_corr]      
                          
        correct_sample = correct_trials_balanced[:,0:3]        
        data = correct_sample      
        rt_label = rt_analysis      
        ## Multi Linear Regression
        poly = PolynomialFeatures(interaction_only=True)
        data_int = poly.fit_transform(data)
        mod = sm.OLS(rt_label, data_int)
        res = mod.fit()
        res.summary()
        weights.append(np.abs(res.params)/np.sum(np.abs(res.params))) 
        score.append(res.rsquared)
        resample+=1    
       
    weights = np.squeeze(np.array(weights))
    
    return weights


if __name__ == '__main__':

    ### import behavioral data to reproduce Figure 6 and Suppl. Figure 2
    
    # Specify the file path of your .pkl file
    file_path_behavior = '/behavior/'
    
    ###################### MONKEY 1 (=SA)   
    file_path = file_path_behavior+'RT_cond_SA.pkl'
    
    # Open the file in binary read mode and load the data into a variable
    with open(file_path, 'rb') as file:
        RT_cond_SA = pickle.load(file)
        
    perf_cond_SA     = np.load(file_path_behavior+'perf_cond_SA.npy')
    perf_cond_err_SA = np.load(file_path_behavior+'perf_cond_err_SA.npy')
    rt_SA            = np.load(file_path_behavior+'rt_SA.npy')
     
    mean_rt_SA = []
    sem_rt_SA = []
    for iCond in range(4):
        mean_rt_SA.append(np.mean(RT_cond_SA[iCond]))
        sem_rt_SA.append(np.std(RT_cond_SA[iCond])/np.sqrt(len(RT_cond_SA[iCond])))    
    
    ## multi-linear regression
    correct_trials_SA  = np.load(file_path_behavior+'correct_trials_SA.npy')
  
    # Open the file in binary read mode and load the data into a variable
    file_path = file_path_behavior+'task_conditions_SA.pkl'
    with open(file_path, 'rb') as file:    
        task_conditions_SA = pickle.load(file)
    
    weights_SA = get_multi_linear(correct_trials_SA,  rt_SA, task_conditions_SA)
    #1: intercept, 2: previous, 3: rule, 4: shape, 5: previous&rule, 6: previious&shape, 7:rule&shape
    ### normalize weights
    sum_prova = np.sum(weights_SA[:,1:],axis=1)
    sum_prova_tile=np.transpose(np.tile(sum_prova,[6,1]))
    weights_SA_norm=weights_SA[:,1:]/sum_prova_tile
    
    
    ################## MONKEY 2 (=SH)
    file_path = file_path_behavior+'RT_cond_SH.pkl'
    
    # Open the file in binary read mode and load the data into a variable
    with open(file_path, 'rb') as file:
        RT_cond_SH = pickle.load(file)
        
    perf_cond_SH     = np.load(file_path_behavior+'perf_cond_SH.npy')
    perf_cond_err_SH = np.load(file_path_behavior+'perf_cond_err_SH.npy')
    rt_SH            = np.load(file_path_behavior+'rt_SH.npy')
     
    mean_rt_SH = []
    sem_rt_SH  = []
    for iCond in range(4):
        mean_rt_SH.append(np.mean(RT_cond_SH[iCond]))
        sem_rt_SH.append(np.std(RT_cond_SH[iCond])/np.sqrt(len(RT_cond_SH[iCond])))    
    
    ## multi-linear regression
    correct_trials_SH  = np.load(file_path_behavior+'correct_trials_SH.npy')
 
    # Open the file in binary read mode and load the data into a variable
    file_path = file_path_behavior+'task_conditions_SH.pkl'
    with open(file_path, 'rb') as file:    
        task_conditions_SH = pickle.load(file)
    
    weights_SH = get_multi_linear(correct_trials_SH, rt_SH, task_conditions_SH)
    #1: intercept, 2: previous, 3: rule, 4: shape, 5: previous&rule, 6: previious&shape, 7:rule&shape
    ### normalize weights
    sum_prova = np.sum(weights_SH[:,1:],axis=1)
    sum_prova_tile=np.transpose(np.tile(sum_prova,[6,1]))
    weights_SH_norm=weights_SH[:,1:]/sum_prova_tile   
    
    ##################################### plot section
    
    
    _,ax=plt.subplots(2,2, figsize=(15,10))
    
    ax[0,0].errorbar([0,1,2,3], perf_cond_SA, np.transpose(perf_cond_err_SA),ls='none',capsize=8,color='k') 
    ax[0,0].set_ylim([0,1])
    ax[0,0].set_xticks([0,1,2,3],['Stay', 'Shift', 'Stay', 'Shift'],fontsize=15)
    ax[0,0].set_ylabel('Behavioral Performance',fontsize=15)
    ax[0,0].set_title('Fig 5 MONKEY 1: Behavioral Performance')
    ax[0,1].plot(mean_rt_SA,'.')
    ax[0,1].errorbar(range(4), mean_rt_SA, sem_rt_SA,fmt='.', color='k', markersize=10, capsize=8)
    ax[0,1].set_ylim([310,317])##<-- monkey SA
    ax[0,1].set_ylabel('Reaction Time [ms]')
    ax[0,1].set_xticks([0,1,2,3],['Stay', 'Shift', 'Stay', 'Shift'],fontsize=15)
    ax[0,1].set_title('Fig 5 MONKEY 1: Reaction time')
    
    ax[1,0].errorbar([0,1,2,3], perf_cond_SH, np.transpose(perf_cond_err_SH),ls='none',capsize=8,color='k') 
    ax[1,0].set_ylim([0,1])
    ax[1,0].set_xticks([0,1,2,3],['Stay', 'Shift', 'Stay', 'Shift'],fontsize=15)
    ax[1,0].set_ylabel('Behavioral Performance',fontsize=15)
    ax[1,0].set_title('Fig 5 MONKEY 2: Behavioral Performance')
    ax[1,1].plot(mean_rt_SH,'.')
    ax[1,1].errorbar(range(4), mean_rt_SH, sem_rt_SH,fmt='.', color='k', markersize=10, capsize=8)
    ax[1,1].set_ylim([308,315])
    ax[1,1].set_ylabel('Reaction Time [ms]')
    ax[1,1].set_xticks([0,1,2,3],['Stay', 'Shift', 'Stay', 'Shift'],fontsize=15)
    ax[1,1].set_title('Fig 5 MONKEY 2: Reaction time')
    
    
    ## multi-linear regression analysis
    
    _,ax=plt.subplots(2,2, figsize=(15,10))
    ax[0,1].errorbar(x=range(6), y=np.mean(weights_SA_norm,axis=0)*100,yerr=2*np.std(weights_SA_norm,axis=0)*100, capsize=8, color='k', ls='none', marker='o',markersize=10)
    ax[0,1].set_title('Supp Fig 2 MONKEY 1:Normalized Weigths', fontsize=20)
    ax[0,0].errorbar(x=[0,4,8], y=np.mean(weights_SA_norm[:,:3],axis=0)*100,yerr=2*np.std(weights_SA_norm[:,:3],axis=0)*100, capsize=8, color='k', ls='none',marker='o',markersize=10)
    ax[0,1].set_xticks( range(6),['Previous', 'Rule','Shape','PreviousANDRule','PreviousANDShape','RuleANDShape'],fontsize=15)
    ax[0,1].set_ylim([0,90])
    ax[0,0].set_ylabel('Normalized weights(%)[a.u.]',fontsize=20)
    ax[0,0].set_xticks([0,4,8],['Previous', 'Rule','Shape'],fontsize=15)
    ax[0,0].set_ylabel('Normalized weights(%) [a.u.]',fontsize=20)
    ax[0,0].set_ylim([-0.9,12.5])
    ax[0,0].set_title('Fig 5 MONKEY 1:Normalized Weigths', fontsize=20)
    plt.suptitle('Fig.5F and Supp. Fig2: Normalized Weights of multi-linear regression', fontsize=20)
    
    ax[1,1].errorbar(x=range(6), y=np.mean(weights_SH_norm,axis=0)*100,yerr=2*np.std(weights_SH_norm,axis=0)*100, capsize=8, color='k', ls='none', marker='o',markersize=10)
    ax[1,1].set_title('Supp Fig 2 MONKEY 1:Normalized Weigths', fontsize=20)
    ax[1,0].errorbar(x=[0,4,8], y=np.mean(weights_SH_norm[:,:3],axis=0)*100,yerr=2*np.std(weights_SH_norm[:,:3],axis=0)*100, capsize=8, color='k', ls='none',marker='o',markersize=10)
    ax[1,1].set_xticks( range(6),['Previous', 'Rule','Shape','PreviousANDRule','PreviousANDShape','RuleANDShape'],fontsize=15)
    ax[1,1].set_ylim([0,90])
    ax[1,0].set_ylabel('Normalized weights(%)[a.u.]',fontsize=20)
    ax[1,0].set_xticks([0,4,8],['Previous', 'Rule','Shape'],fontsize=15)
    ax[1,0].set_ylabel('Normalized weights(%) [a.u.]',fontsize=20)
    ax[1,0].set_title('Fig 5 MONKEY 1:Normalized Weigths', fontsize=20)
    ax[0,0].set_ylim([-0.9,12.5])
    plt.suptitle('Fig.5F and Supp. Fig2: Normalized Weights of multi-linear regression', fontsize=20)
    

