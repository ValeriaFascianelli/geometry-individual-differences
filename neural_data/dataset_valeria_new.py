#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 16:07:04 2019

@author: valeria
"""
from scipy.io import loadmat
from sklearn import svm
import random
#from operator import itemgetter
import numpy as np
import math
import time as tm

class Dataset():
    
    def __init__(self, path_tomatdata, path_tomatrix):
        self.matlabstruct   = loadmat(path_tomatdata, squeeze_me=True, struct_as_record=False)
        self.matrixpath = path_tomatrix 
        
    def get_session_list(self):
        session_list = eval("self.matlabstruct['trials'].file")
        return session_list
    
    def get_totnumber_session(self):
        tot_session = len(eval("self.matlabstruct['trials'].file"))
        return tot_session
    
    def get_totneuron_persession(self):
        tot_neuron_persession = []
        for s in range(0,len(eval("self.matlabstruct['trials'].file"))):
            tot_neuron_persession.append(np.size(eval("self.matlabstruct['trials'].neuron[s]")))
        return tot_neuron_persession
        
    def get_tottrial(self, session_id, neuron_id):
        tot_neurons = self.get_totneuron_persession()
        if tot_neurons[session_id] == 1: # perche se ho solo una cellula nella sessione, python importa in modo sbagliato i trials
            tot_trial = np.size(eval("self.matlabstruct['trials'].number[session_id]"))
        else:
            tot_trial = np.size(eval("self.matlabstruct['trials'].number[session_id][neuron_id]"))
        return tot_trial
   
    def get_trial_id(self, session_id, neuron_id):
        tot_neurons = self.get_totneuron_persession()
        trial_id = []
        if tot_neurons[session_id] == 1:
            trial_id = eval("self.matlabstruct['trials'].number[session_id]")
        else:    
            trial_id = eval("self.matlabstruct['trials'].number[session_id][neuron_id]")
        return trial_id
       
 
    def get_spikes(self, session_id, neuron_id, code_align):
        
        listsession = self.get_session_list()
        neuron_per_session = self.get_totneuron_persession()
        eventpath = self.matrixpath+'event_'+listsession[session_id]+'.mat'
        timepath = self.matrixpath+'time_'+listsession[session_id]+'.mat'       
        event_matrix = loadmat(eventpath, squeeze_me=True, struct_as_record=False)
        time_matrix = loadmat(timepath, squeeze_me=True, struct_as_record=False)        
        event = eval("event_matrix['event']")
        time  = eval("time_matrix['time']")        
       
        if np.size(eval("self.matlabstruct['trials'].neuron[session_id]")) > 1:
           id_trials = eval("self.matlabstruct['trials'].number[session_id][neuron_id]")
        elif np.size(eval("self.matlabstruct['trials'].neuron[session_id]")) == 1:
           id_trials = eval("self.matlabstruct['trials'].number[session_id]")
          
        trial_align = []        
        id_trial_align = []
        
        if np.size(id_trials) == 1: # perche se ho solo un trial per quella cellula salto           
            #id_trials = self.matlabstruct['trials'].number[session_id] 
            return trial_align, id_trial_align
        
#        if np.size(id_trials) == 1: # perche se ho solo un trial per quella cellula salto           
#            id_trials = self.matlabstruct['trials'].number[session_id] 
#      
        ## commentato Febbraio 2020---discommentato Luglio 2020 per problema in PFo sessione 40 neuron 2
        if np.size(id_trials) == 0:
            print('Not trials for neuron {} in session {}'.format(neuron_id, session_id))
            return trial_align, id_trials
        
        id_trials = id_trials-1 # because from matlab indexing
        #print(id_trials)

        #print(neuron_id, session_id)
        index_align = np.where(event[:,id_trials]==code_align)
        
        time_align  = time[index_align[0],id_trials[index_align[1]]] 
        #p = np.r_[[s for s in eval('self.matlabstruct[\'trials\'].time[session_id][neuron_id]')]]
        p = []
      
        if neuron_per_session[session_id] == 1 or np.size(eval('self.matlabstruct[\'trials\'].time[session_id][neuron_id]'))==1:
         # stesso problema di cui sopra, se hai solo un tempo di spike in un trial python importa male tutti i valori successivi
             p = self.matlabstruct['trials'].time[session_id]
        else:
            for s in eval('self.matlabstruct[\'trials\'].time[session_id][neuron_id]'):    
                p.append(s)       
       
        
        for kk in range(len(index_align[1])):
            
            if np.size(p[index_align[1][kk]])>1:               
                trial_align.append([p[index_align[1][kk]].astype(int)-time_align[kk]])
            else:
                
                trial_align.append([p[index_align[1][kk]]-time_align[kk]])
        id_trial_align = id_trials[index_align[1]]
        
        return trial_align, id_trial_align
      
    def get_behavior(self, session_id):
        listsession = self.get_session_list()
        eventpath = self.matrixpath+'event_'+listsession[session_id]+'.mat'
        event_matrix = loadmat(eventpath, squeeze_me=True, struct_as_record=False)
        event = eval("event_matrix['event']")
        
        timepath = self.matrixpath+'time_'+listsession[session_id]+'.mat'
        time_matrix = loadmat(timepath, squeeze_me=True, struct_as_record=False)
        time = eval("time_matrix['time']")
        
        return event, time
    
    
    
    def get_neuron_data(self, session_id, neuron_id, code_align):
        check_neuron = True
        listsession = self.get_session_list()
        trial_align, id_trial_align = self.get_spikes(session_id, neuron_id, code_align)
        if len(trial_align) == 0:
            neuron = []
            check_neuron = False
            return check_neuron, neuron 
        behavior,_ = self.get_behavior(session_id)
        return check_neuron, Neuron(listsession, session_id, neuron_id, trial_align, id_trial_align, behavior)
    
    
class Neuron():
      
    def __init__(self, listsession, session_id, neuron_id, trial_align, id_trial_align, behavior):
        
        self.namesession = listsession[session_id]
        self.session_id = session_id
        self.id = neuron_id
        self.trial = trial_align
        self.id_trial = id_trial_align 
        self.behavior = behavior        
    
    def get_trials(self, trial_type):
        trials_selected = []
        pos_trials_selected = []
        for itrial, trial_real in enumerate(self.id_trial):
             if all(np.isin(trial_type, self.behavior[:,trial_real])):
                trials_selected.append(self.trial[itrial])
                pos_trials_selected.append(itrial)                
        return trials_selected, pos_trials_selected     
     
    def get_spike_counts(self, trials, time0, delta_bin):
        spike_counts = []
        #spike_counts = np.r_[[np.histogram(trials[kk], bins=1, range=[time0,time0+delta_bin]) for kk in range(len(trials))]]
        spike_counts = np.r_[[((np.asarray(trials[kk]) >= time0)*(np.asarray(trials[kk]) < time0+delta_bin)).sum() for kk in range(len(trials)) ]]
        return spike_counts    
    

class Decoder():

    def __init__(self, dataset, pattern, min_trials, n_loops):
            
        self.dataset = dataset
        self.pattern = pattern
        self.min_trials = min_trials
        self.n_loops = n_loops
 
      
    
    def get_trials_for_decoding(self, trial_pos):

        ## it gives you back the position in the trial_pos array
        trials_train_pos = []
        trials_test_pos = []
        
        for k_neuro in range(len(trial_pos)):
            #print("k_neuro {}".format(k_neuro))
            trials_train_pos_app1 = []
            trials_test_pos_app1 = []
            for k_pattern in range(len(trial_pos[k_neuro])):
                #print("pattern {}".format(k_pattern))
                n_trials = len(trial_pos[k_neuro][k_pattern])
                k_trials = int(np.floor(n_trials*0.8)) # 80% of trials used for training   
                train_trials = random.sample(range(0, n_trials), k_trials) #random.sample(trial_pos[k_neuro][k_pattern], k_trials) #random.sample(range(0, n_trials), k_trials)
                test_trials = range(0, n_trials)  #trial_pos[k_neuro][k_pattern]#range(0, n_trials)
                for i_train in range(len(train_trials)):
                    test_trials = [x for x in test_trials if x != train_trials[i_train]]    
                if len(list(set(train_trials) & set(test_trials))) > 0:
                    raise ValueError("not empty intersaction between train and test trials")
                trials_train_pos_app1.append(train_trials)
                trials_test_pos_app1.append(test_trials)
            trials_train_pos.append(trials_train_pos_app1)
            trials_test_pos.append(trials_test_pos_app1)
        return trials_train_pos, trials_test_pos    
    
        
    def split_trials_per_pattern(self, code_align):
        
    
        session_list = self.dataset.get_session_list() #<-- list of all sessions: it is an array of objects: 'sa020a11.1, sa021a11.1...'
        tot_neuron_persession = self.dataset.get_totneuron_persession() #<-- list of # of neurons per sessions
        trials_pattern = []
        neuron_info    = []
        trials_pos_pattern = []   
        not_trials = False
        neuron = []
        
        for ss in range(len(session_list)):
            print('session {}/{}:'.format(ss, len(session_list))) 
            for nn in range(tot_neuron_persession[ss]):
                trials_pattern_app1       = []
                trials_pos_pattern_app1   = []
                try:
                    check_neuron, neuron_app  = self.dataset.get_neuron_data(session_id=ss, neuron_id=nn, code_align=code_align)  
                except:
                    continue
                for pp in range(len(self.pattern)):      
                    trials_selected, pos_trials_selected = neuron_app.get_trials(trial_type=self.pattern[pp])
                    if len(trials_selected) < self.min_trials:                       
                        not_trials = True      
                        break
                    #spike_counts_app1 = neuron.get_spike_counts(trials=trials_selected, time0=time, delta_bin=delta_time)     
                    trials_pattern_app1.append(trials_selected)
                    trials_pos_pattern_app1.append(pos_trials_selected)                    
                if not_trials == True:
                    not_trials = False
                    continue
                trials_pattern.append(trials_pattern_app1)
                trials_pos_pattern.append(trials_pos_pattern_app1)
                neuron.append(neuron_app)
                neuron_info.append([ss,nn])
        return trials_pattern, trials_pos_pattern, neuron_info, neuron       
   
    def get_spike_count_perpattern(self, neuron, trials_pattern, trials_pos_pattern, neuron_info, time, delta_time, code_align):
        ## calcolo spike count
        spike_count = []   
        trial_pos = []
        for nn in range(len(trials_pattern)):
            spike_count_pattern = []
            trial_pos_app1   = []
            for pp in range(len(self.pattern)): 
                spike_counts_app1 = neuron[nn].get_spike_counts(trials=trials_pattern[nn][pp], time0=time, delta_bin=delta_time)     
                spike_count_pattern.append(spike_counts_app1)
                trial_pos_app1.append(trials_pos_pattern[nn][pp])                    
            spike_count.append(spike_count_pattern)
            trial_pos.append(trial_pos_app1)
        return spike_count, trial_pos        
    
    def generate_pseudo_recordings(self, spike_count, trials_train_pos, trials_test_pos):
        
        n_neurons = len(spike_count)
        n_patterns = len(spike_count[0])
        pseudo_patterns_train = np.zeros((n_patterns, self.n_loops, n_neurons))
        pseudo_patterns_test  = np.zeros((n_patterns, self.n_loops, n_neurons))
        
        for n_pattern in range(n_patterns):    
            #print('pattern {}:'.format(n_pattern))
            for n_neuro in range(n_neurons):                    
               # print('neuron {}:'.format(n_neuro))
                train_trials = trials_train_pos[n_neuro][n_pattern] # li scelgo solo dai train trials                   
                test_trials = trials_test_pos[n_neuro][n_pattern]   
                
                position_train = np.random.choice(train_trials, self.n_loops) #random.sample(train_trials,1)
                position_test  = np.random.choice(test_trials, self.n_loops)  #random.sample(test_trials,1)
                
                pseudo_patterns_train[n_pattern,:self.n_loops,n_neuro] = spike_count[n_neuro][n_pattern][position_train]
                pseudo_patterns_test[n_pattern,:self.n_loops,n_neuro] = spike_count[n_neuro][n_pattern][position_test]
            
        return pseudo_patterns_train, pseudo_patterns_test

   
   