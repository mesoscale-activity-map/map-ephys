# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:15:49 2020

@author: Han
"""
import numpy as np
import pandas as pd
import time

from .fitting_functions import fit_bandit, cross_validate_bandit
# from utils.plot_fitting import plot_model_comparison_predictive_choice_prob, plot_model_comparison_result
# from IPython.display import display


class BanditModelComparison:
    
    '''
    A new class that can define models, receive data, do fitting, and generate plots.
    This is the minimized module that can be plugged into Datajoint for real data.
    
    '''
    
    def __init__(self, choice_history, reward_history, model, iti=None, p_reward = None, session_num = None):
        """

        Parameters
        ----------
        choice_history, reward_history, (p_reward), (session_num)
            DESCRIPTION. p_reward is only for plotting or generative validation; session_num is for pooling across sessions
        model: only accepts one model, selected from foraging_model.Model table
            # models : list of integers or models, optional
            #     DESCRIPTION. If it's a list of integers, the models will be selected from the pre-defined models.
            #     If it's a list of models, then it will be used directly. Use the format: [forager, [para_names], [lower bounds], [higher bounds]]
            #     The default is None (using all pre-defined models).
        Returns
        -------
        None.

        """

        self.models = model
        self.fit_choice_history, self.fit_reward_history, self.p_reward, self.session_num = choice_history, reward_history, p_reward, session_num
        self.fit_iti = iti
        self.K, self.n_trials = np.shape(self.fit_reward_history)
        assert np.shape(self.fit_choice_history)[1] == self.n_trials, 'Choice length should be equal to reward length!'
        
    def fit(self, fit_method = 'DE', fit_settings = {'DE_pop_size': 16}, pool = '',
                  if_verbose = True, 
                  plot_predictive = None,  # E.g.: 0,1,2,-1: The best, 2nd, 3rd and the worst model
                  plot_generative = None):
        
        self.results_raw = []
        self.results = pd.DataFrame()
        
        if if_verbose: print('=== Model Comparison ===\nMethods = %s, %s, pool = %s' % (fit_method, fit_settings, pool!=''))
        for mm, model in enumerate(self.models):
            # == Get settings for this model ==
            forager, fit_names, fit_lb, fit_ub = model
            fit_bounds = [fit_lb, fit_ub]
            
            para_notation = None
            Km = None
                        
            # == Do fitting here ==
            #  Km = np.sum(np.diff(np.array(fit_bounds),axis=0)>0)
            
            if if_verbose: print('Model %g/%g: %15s, Km = %g ...'%(mm+1, len(self.models), forager, Km), end='')
            start = time.time()
                
            result_this = fit_bandit(forager, fit_names, fit_bounds, self.fit_choice_history, self.fit_reward_history, iti = self.fit_iti, 
                                     session_num = self.session_num, fit_method = fit_method, **fit_settings, 
                                     pool = pool, if_predictive = True) #plot_predictive is not None)
            
            if if_verbose: print(' AIC = %g, BIC = %g (done in %.3g secs)' % (result_this.AIC, result_this.BIC, time.time()-start) )
            self.results_raw.append(result_this)
            self.results = self.results.append(pd.DataFrame({'model': [forager], 'Km': Km, 'AIC': result_this.AIC, 'BIC': result_this.BIC, 
                                    'LPT_AIC': result_this.LPT_AIC, 'LPT_BIC': result_this.LPT_BIC, 'LPT': result_this.LPT, 'prediction_accuracy': result_this.prediction_accuracy,
                                    'para_names': [fit_names], 'para_bounds': [fit_bounds], 
                                    'para_notation': [para_notation], 'para_fitted': [np.round(result_this.x,3)]}, index = [mm+1]))
        
        self.trial_numbers = result_this.trial_numbers 
        
        # == Plotting == 
        if plot_predictive is not None: # Plot the predictive choice trace of the best fitting of the best model (Using AIC)
            self.plot_predictive = plot_predictive
            self.plot_predictive_choice()
        return
    
    def cross_validate(self, k_fold = 2, fit_method = 'DE', fit_settings = {'DE_pop_size': 16}, pool = '', if_verbose = True):
        
        self.prediction_accuracy_CV = pd.DataFrame()
        
        if if_verbose: print('=== Cross validation ===\nMethods = %s, %s, pool = %s' % (fit_method, fit_settings, pool!=''))
        
        for mm, model in enumerate(self.models):
            # == Get settings for this model ==
            forager, fit_names, fit_lb, fit_ub = model
            fit_bounds = [fit_lb, fit_ub]
            
            para_notation = None
            Km = None
            
            # == Do fitting here ==
            #  Km = np.sum(np.diff(np.array(fit_bounds),axis=0)>0)
            
            if if_verbose: print('Model %g/%g: %15s, Km = %g ...'%(mm+1, len(self.models), forager, Km), end = '')
            start = time.time()
                
            prediction_accuracy_test, prediction_accuracy_fit, prediction_accuracy_test_bias_only \
            = cross_validate_bandit(forager, fit_names, fit_bounds, 
                                    self.fit_choice_history, self.fit_reward_history, self.fit_iti, self.session_num, 
                                    k_fold = k_fold, **fit_settings, pool = pool, if_verbose = if_verbose) #plot_predictive is not None)
            
            if if_verbose: print('  \n%g-fold CV: Test acc.= %s, Fit acc. = %s (done in %.3g secs)' % (k_fold, prediction_accuracy_test, prediction_accuracy_fit, time.time()-start) )
            
            self.prediction_accuracy_CV = pd.concat([self.prediction_accuracy_CV, 
                                                     pd.DataFrame({'model#': mm,
                                                                   'forager': forager,
                                                                   'Km': Km,
                                                                   'para_notation': para_notation,
                                                                   'prediction_accuracy_test': [prediction_accuracy_test], 
                                                                   'prediction_accuracy_fit': [prediction_accuracy_fit],
                                                                   'prediction_accuracy_test_bias_only': [prediction_accuracy_test_bias_only]})])
            
        return

    # def plot_predictive_choice(self):
    #     plot_model_comparison_predictive_choice_prob(self)

    # def show(self):
        # pd.options.display.max_colwidth = 100
        # display(self.results_sort[['model','Km', 'AIC','log10_BF_AIC', 'model_weight_AIC', 'BIC','log10_BF_BIC', 'model_weight_BIC', 'para_notation','para_fitted']].round(2))
        
    # def plot(self):
    #     plot_model_comparison_result(self)

