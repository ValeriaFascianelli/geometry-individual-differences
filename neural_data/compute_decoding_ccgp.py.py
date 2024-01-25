# -*- coding: utf-8 -*-
"""
@author: Valeria Fascianelli, 2023
"""

import matplotlib.pyplot as plt
import numpy as np
import itertools
import scipy
from sklearn import svm
import random
import pickle

def get_dichotomies(n_cond, side_dim):
    
    patterns = list(range(n_cond))
    tot_dichotomies = int(scipy.special.binom(n_cond, side_dim)/2)
    control_vector = np.ones(tot_dichotomies*2)
    i  = -1
    dichotomies = []
    for j in itertools.combinations(patterns, side_dim):        
        i+=1
        ii = -1        
        if (control_vector[i]==0):
            continue
        for jj in itertools.combinations(patterns, side_dim):
            ii+=1
            if (control_vector[ii]==0) or (len(np.intersect1d(j,jj)) != 0):                
                continue
            matrix_dic = np.zeros((2,side_dim))
            matrix_dic[0,:] = j
            matrix_dic[1,:] = jj
            dichotomies.append(matrix_dic)             
            control_vector[ii] = 0            
            break
    
    dichotomies = [item.astype(int) for item in dichotomies]    
    
    return dichotomies


def get_ccgp(samples_train, samples_test, side_dim, n_cross, real_dichotomies):
    print('[###]Computing CCGP')
   
    n_cond   = np.size(samples_train, axis=0)
    n_trials = np.size(samples_train,axis=1)
    dichotomies_all = get_dichotomies(n_cond,side_dim)
    
    dichotomies = [dichotomies_all[i] for i in real_dichotomies ]#dichotomies[real_dichotomies]
    tot_dic     = len(dichotomies)
    
    ccgp_dic = np.nan*np.ones((tot_dic, side_dim*side_dim))
    for iDic in range(tot_dic):
        print('Dic ', iDic)
        ccgp_cond = []
        for iTestCondC1 in range(side_dim):        
                for iTestCondC2 in range(side_dim):
                    ccgp_cross = []
                    for iCross in range(n_cross):
                        train_condC1   = np.where(np.arange(0,side_dim)!=iTestCondC1)[0]            
                        test_sampleC1  = samples_test[dichotomies[iDic][0][iTestCondC1],:,:] 
                        train_sample_app = samples_train[dichotomies[iDic][0][train_condC1],:,:]
                        train_sampleC1   = np.reshape(train_sample_app[:,:,:],(side_dim-1*(np.size(samples_train,axis=1)), np.size(samples_train,axis=2)))
                                                
                        train_condC2  = np.where(np.arange(0,side_dim)!=iTestCondC2)[0]
                        test_sampleC2 = samples_test[dichotomies[iDic][1][iTestCondC2],:,:]
                        train_sample_app = samples_train[dichotomies[iDic][1][train_condC2],:,:]
                        train_sampleC2   = np.reshape(train_sample_app[:,:,:],(side_dim-1*(np.size(samples_train,axis=1)), np.size(samples_train,axis=2)))
                        
                        train_sample = np.concatenate((train_sampleC1, train_sampleC2))
                        test_sample  = np.concatenate((test_sampleC1, test_sampleC2))
                        
                        train_label = np.concatenate(( np.zeros((n_trials*(side_dim-1))), np.ones((n_trials*(side_dim-1))) ))
                        test_label  = np.concatenate(( np.zeros((n_trials)), np.ones((n_trials)) ))
                        
                        clf = []
                        clf = svm.SVC(kernel='linear', C=0.001) 
                        clf.fit(train_sample, train_label)
                        ccgp_cross.append(clf.score(test_sample, test_label))
                    ccgp_cond.append(np.mean(ccgp_cross))        
        ccgp_dic[iDic,:] = ccgp_cond        
    
    return ccgp_dic, dichotomies

def get_decoding_accuracy(samples_train, samples_test, side_dim, n_cross, train_perc, real_dichotomies):
    
    print('[###]Computing decoding accuracy')
   
    n_cond = np.size(samples_train, axis=0)
    n_trials = np.size(samples_train,axis=1)
    n_neurons = np.size(samples_train,axis=2)
    dichotomies_all = get_dichotomies(n_cond,side_dim)
    
    dichotomies = [dichotomies_all[i] for i in real_dichotomies ]#dichotomies[real_dichotomies]
    tot_dic     = len(dichotomies)
    
    score_dic = np.nan*np.ones((tot_dic, n_cross))
    score_train = np.nan*np.ones((tot_dic, n_cross))
    
    for iDic in range(tot_dic):
        print('Dic ', iDic)
        score_cross = []
        score_app_train = []
        for iCross in range(n_cross):  
         
            train_index = random.sample(range(0,n_trials), int(n_trials*train_perc))
            test_index = list(set(list(range(n_trials))).difference(train_index))
            
            n_train_trials = len(train_index)
            n_test_trials  = len(test_index)
            
            train_trials = samples_train[:, np.array(train_index),:] 
            test_trials  = samples_test[:, np.array(test_index), :] 
            
            train_sample_C1 = np.reshape(train_trials[dichotomies[iDic][0],:,:], (side_dim*n_train_trials, n_neurons))
            train_sample_C2 = np.reshape(train_trials[dichotomies[iDic][1],:,:], (side_dim*n_train_trials, n_neurons))
            test_sample_C1  = np.reshape(test_trials[dichotomies[iDic][0],:,:],  (side_dim*n_test_trials,  n_neurons))
            test_sample_C2  = np.reshape(test_trials[dichotomies[iDic][1],:,:],  (side_dim*n_test_trials,  n_neurons))
            
            train_sample = np.concatenate((train_sample_C1, train_sample_C2),axis=0)
            train_label  = np.concatenate((np.zeros(n_train_trials*side_dim), np.ones(n_train_trials*side_dim)),axis=0) 
            test_sample  = np.concatenate((test_sample_C1, test_sample_C2),axis=0)
            test_label   = np.concatenate((np.zeros(n_test_trials*side_dim), np.ones(n_test_trials*side_dim)),axis=0) 
                
            clf = []            
            clf = svm.SVC(kernel='linear',C=0.001)                        
            clf.fit(train_sample, train_label)           
            score_cross.append(clf.score(test_sample, test_label))     

            score_app_train.append(clf.score(train_sample, train_label))                  
            
        score_dic[iDic,:] = score_cross    
        score_train[iDic,:] = score_app_train
    return score_dic, dichotomies, score_train


if __name__ == '__main__':
    
    
    # Open the file in binary read mode and load the data into a variable
    main_path = '/neural_data/'
    monkey= 'M2' # M1=monkey 1, M2=monkey 2
    file_path = main_path+'decoder_'+monkey+'.pkl'
    with open(file_path, 'rb') as file:
        decoder = pickle.load(file)
        
    file_path = main_path+'trials_pattern_'+monkey+'.pkl'
    with open(file_path, 'rb') as file:
        trials_pattern = pickle.load(file)    
    
    file_path = main_path+'trials_pos_pattern_'+monkey+'.pkl'
    with open(file_path, 'rb') as file:
        trials_pos_pattern = pickle.load(file)   
        
    file_path = main_path+'neuron_info_'+monkey+'.pkl'
    with open(file_path, 'rb') as file:
        neuron_info = pickle.load(file)  
        
    file_path = main_path+'neuron_'+monkey+'.pkl'
    with open(file_path, 'rb') as file:
        neuron = pickle.load(file)     
        
   
    time_0      = -400 #[ms] starting time point
    delta_time  = 200  #[ms] bin width
    time_step   = 200  #[ms] bin step: if time_step=delta_time-->disjoint bins
    n_step      = 8    # total number of steps across time
    code_align  = 25 #<-- cue onset
    n_dichotomies = 4
    train_perc    = 0.8

    nNeurons_norm = len(neuron)
    dichotomies = [0,9,14,20]
    
    side_dim = 4
    n_cross_ccgp = 1 #<-- set total number of cross validation for ccgp (it might take a while-- +16 intrinsic validations across dichotomies sides)
    n_cross_dec  = 1
    tot_dic      = len(dichotomies)
    nPatterns    = 8 #<-- total number of task conditions (Figure 2D)
    
    ccgp_dichotomies = np.nan*np.ones((n_step, tot_dic, side_dim*side_dim))  
    deco_dichotomies = np.nan*np.ones((n_step, tot_dic, n_cross_dec))
    score_train = np.nan*np.ones((n_step, tot_dic, n_cross_dec))
   
       
    t=[]
    for k_time in range(n_step): 
            
        print('n_step ', k_time)        
        spike_count = []       
        time = time_0 + k_time*time_step 
        t.append(time)
        spike_count, trial_pos = decoder.get_spike_count_perpattern(neuron, trials_pattern, trials_pos_pattern, neuron_info, time, delta_time, code_align)                                                        
        trials_train_pos, trials_test_pos = decoder.get_trials_for_decoding(trial_pos)                                        
        pseudo_patterns_train, pseudo_patterns_test = decoder.generate_pseudo_recordings(spike_count, trials_train_pos, trials_test_pos)                         
        ccgp_dichotomies[k_time,:,:], _ = get_ccgp(pseudo_patterns_train, pseudo_patterns_test, side_dim, n_cross_ccgp, dichotomies)
        deco_dichotomies[k_time,:,:],_,score_train[k_time,:,:] =  get_decoding_accuracy(pseudo_patterns_train, pseudo_patterns_test, side_dim, n_cross_dec, train_perc, dichotomies) 
        
        
## important dichotomies:
## RULE:     dichotomy 0  blue
## PREVIOUS: dichotomy 1  green  
## CURRENT:  dichotomy 2  red
## SHAPE:    dichotomy 3  orange
 

ccgp_dich = np.mean(ccgp_dichotomies,axis=2)
deco_dich = np.mean(deco_dichotomies,axis=2)

### plot decoding accuracy and ccgp along time
colors=['blue', 'green','red','orange']
_,axs=plt.subplots(2,1, figsize=(10,20))
for iDich in [0,1,2,3]:
    axs[0].plot(np.asarray(t)+delta_time/2, np.mean(deco_dichotomies,axis=2)[:,iDich], color=colors[iDich],lw=5)
    axs[1].plot(np.asarray(t)+delta_time/2, np.mean(ccgp_dichotomies,axis=2)[:,iDich], color=colors[iDich],lw=5)
plt.ylim(0.2,1)
if monkey == 'M1':
    plt.suptitle('MONKEY 1')
else:
    plt.suptitle('MONKEY 2')
    

