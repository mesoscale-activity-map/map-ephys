"""
Created on Thu Mar 19 16:30:26 2020

@author: Han
"""
import numpy as np
import scipy.optimize as optimize
import multiprocessing as mp
# from tqdm import tqdm  # For progress bar. HH

from .bandit_model import BanditModel
global fit_history

def negLL_func(fit_value, *argss):
    '''
    Compute negative likelihood (Core func)
    '''
    # Arguments interpretation
    forager, fit_names, choice_history, reward_history, session_num, para_fixed, fit_set = argss
    
    kwargs_all = {'forager': forager, **para_fixed}  # **kargs includes all other fixed parameters
    for (nn, vv) in zip(fit_names, fit_value):
        kwargs_all = {**kwargs_all, nn:vv}

    # Put constraint hack here!!
    if 'tau2' in kwargs_all:
        if kwargs_all['tau2'] < kwargs_all['tau1']:
            return np.inf
        
    # Handle data from different sessions
    if session_num is None:
        session_num = np.zeros_like(choice_history)[0]  # Regard as one session
    
    unique_session = np.unique(session_num)
    likelihood_all_trial = []
    
    # -- For each session --
    for ss in unique_session:
        # Data in this session
        choice_this = choice_history[:, session_num == ss]
        reward_this = reward_history[:, session_num == ss]
        
        # Run **PREDICTIVE** simulation    
        bandit = BanditModel(**kwargs_all, fit_choice_history = choice_this, fit_reward_history = reward_this)  # Into the fitting mode
        bandit.simulate()
        
        # Compute negative likelihood
        predictive_choice_prob = bandit.predictive_choice_prob  # Get all predictive choice probability [K, num_trials]
        likelihood_each_trial = predictive_choice_prob [choice_this[0,:], range(len(choice_this[0]))]  # Get the actual likelihood for each trial
        
        # Deal with numerical precision
        likelihood_each_trial[(likelihood_each_trial <= 0) & (likelihood_each_trial > -1e-5)] = 1e-16  # To avoid infinity, which makes the number of zero likelihoods informative!
        likelihood_each_trial[likelihood_each_trial > 1] = 1
        
        # Cache likelihoods
        likelihood_all_trial.extend(likelihood_each_trial)
    
    likelihood_all_trial = np.array(likelihood_all_trial)
    
    if fit_set == []: # Use all trials
        negLL = - sum(np.log(likelihood_all_trial))
    else:   # Only return likelihoods in the fit_set
        negLL = - sum(np.log(likelihood_all_trial[fit_set]))
        
    # print(np.round(fit_value,4), negLL, '\n')
    # if np.any(likelihood_each_trial < 0):
    #     print(predictive_choice_prob)
    
    return negLL

def callback_history(x, **kargs):
    '''
    Store the intermediate DE results. I have to use global variable as a workaround. Any better ideas?
    '''
    global fit_history
    fit_history.append(x)
    
    return


def fit_each_init(forager, fit_names, fit_bounds, choice_history, reward_history, session_num, fit_method, callback):
    '''
    For local optimizers, fit using ONE certain initial condition    
    '''
    x0 = []
    for lb,ub in zip(fit_bounds[0], fit_bounds[1]):
        x0.append(np.random.uniform(lb,ub))
        
    # Append the initial point
    if callback != None: callback_history(x0)
        
    fitting_result = optimize.minimize(negLL_func, x0, args = (forager, fit_names, choice_history, reward_history, session_num, {}, []), method = fit_method,
                                       bounds = optimize.Bounds(fit_bounds[0], fit_bounds[1]), callback = callback, )
    return fitting_result


def fit_bandit(forager, fit_names, fit_bounds, choice_history, reward_history, session_num = None, 
               if_predictive = False, if_generative = False,  # Whether compute predictive or generative choice sequence
               if_history = False, fit_method = 'DE', DE_pop_size = 16, n_x0s = 1, pool = ''):
    '''
    Main fitting func and compute BIC etc.
    '''
    if if_history: 
        global fit_history
        fit_history = []
        fit_histories = []  # All histories for different initializations
        
    # === Fitting ===
    
    if fit_method == 'DE':
        
        # Use DE's own parallel method
        fitting_result = optimize.differential_evolution(func = negLL_func, args = (forager, fit_names, choice_history, reward_history, session_num, {}, []),
                                                         bounds = optimize.Bounds(fit_bounds[0], fit_bounds[1]), 
                                                         mutation=(0.5, 1), recombination = 0.7, popsize = DE_pop_size, strategy = 'best1bin', 
                                                         disp = False, 
                                                         workers = 1 if pool == '' else int(mp.cpu_count()),   # For DE, use pool to control if_parallel, although we don't use pool for DE
                                                         updating = 'immediate' if pool == '' else 'deferred',
                                                         callback = callback_history if if_history else None,)
        if if_history:
            fit_history.append(fitting_result.x.copy())  # Add the final result
            fit_histories = [fit_history]  # Backward compatibility
        
    elif fit_method in ['L-BFGS-B', 'SLSQP', 'TNC', 'trust-constr']:
        
        # Do parallel initialization
        fitting_parallel_results = []
        
        if pool != '':  # Go parallel
            pool_results = []
            
            # Must use two separate for loops, one for assigning and one for harvesting!
            for nn in range(n_x0s):
                # Assign jobs
                pool_results.append(pool.apply_async(fit_each_init, args = (forager, fit_names, fit_bounds, choice_history, reward_history, session_num, fit_method, 
                                                                            None)))   # We can have multiple histories only in serial mode
            for rr in pool_results:
                # Get data    
                fitting_parallel_results.append(rr.get())
        else:
            # Serial
            for nn in range(n_x0s):
                # We can have multiple histories only in serial mode
                if if_history: fit_history = []  # Clear this history
                
                result = fit_each_init(forager, fit_names, fit_bounds, choice_history, reward_history, session_num, fit_method,
                                       callback = callback_history if if_history else None)
                
                fitting_parallel_results.append(result)
                if if_history: 
                    fit_history.append(result.x.copy())  # Add the final result
                    fit_histories.append(fit_history)
            
        # Find the global optimal
        cost = np.zeros(n_x0s)
        for nn,rr in enumerate(fitting_parallel_results):
            cost[nn] = rr.fun
        
        best_ind = np.argmin(cost)
        
        fitting_result = fitting_parallel_results[best_ind]
        if if_history and fit_histories != []:
            fit_histories.insert(0,fit_histories.pop(best_ind))  # Move the best one to the first
        
    if if_history:
        fitting_result.fit_histories = fit_histories
        
    # === For Model Comparison ===
    fitting_result.k_model = np.sum(np.diff(np.array(fit_bounds),axis=0)>0)  # Get the number of fitted parameters with non-zero range of bounds
    fitting_result.n_trials = np.shape(choice_history)[1]
    fitting_result.log_likelihood = - fitting_result.fun
    
    fitting_result.AIC = -2 * fitting_result.log_likelihood + 2 * fitting_result.k_model
    fitting_result.BIC = -2 * fitting_result.log_likelihood + fitting_result.k_model * np.log(fitting_result.n_trials)
    
    # Likelihood-Per-Trial. See Wilson 2019 (but their formula was wrong...)
    fitting_result.LPT = np.exp(fitting_result.log_likelihood / fitting_result.n_trials)  # Raw LPT without penality
    fitting_result.LPT_AIC = np.exp(- fitting_result.AIC / 2 / fitting_result.n_trials)
    fitting_result.LPT_BIC = np.exp(- fitting_result.BIC / 2 / fitting_result.n_trials)
    
    # === Rerun predictive choice sequence (latent decision value; not latent value per se, but can be used to estimate Q etc.) ===
    if if_predictive:
        
        kwargs_all = {}
        for (nn, vv) in zip(fit_names, fitting_result.x):  # Use the fitted data
            kwargs_all = {**kwargs_all, nn:vv}
        
        # Handle data from different sessions
        if session_num is None:
            session_num = np.zeros_like(choice_history)[0]  # Regard as one session
        
        unique_session = np.unique(session_num)
        predictive_choice_prob = []
        fitting_result.trial_numbers = []
        
        # -- For each session --
        for ss in unique_session:
            # Data in this session
            choice_this = choice_history[:, session_num == ss]
            reward_this = reward_history[:, session_num == ss]
            fitting_result.trial_numbers.append(np.sum(session_num == ss))
            
            # Run **PREDICTIVE** simulation using the fitted data
            bandit = BanditModel(forager = forager, **kwargs_all, fit_choice_history = choice_this, fit_reward_history = reward_this)  # Into the fitting mode
            bandit.simulate()
            predictive_choice_prob.append(bandit.predictive_choice_prob)
        
        fitting_result.predictive_choice_prob = np.hstack(predictive_choice_prob)
        this_predictive_choice = np.argmax(fitting_result.predictive_choice_prob, axis = 0)
        fitting_result.prediction_accuracy = np.sum(this_predictive_choice == choice_this) / fitting_result.n_trials
        
    # === Run generative choice sequence ==  #!!!
        

    return fitting_result
            

import random

def cross_validate_bandit(forager, fit_names, fit_bounds, choice_history, reward_history, session_num = None, k_fold = 2, 
                          DE_pop_size = 16, pool = '', if_verbose = True):
    '''
    k-fold cross-validation
    '''
    
    # Split the data into k_fold parts
    n_trials = np.shape(choice_history)[1]
    trial_numbers_shuffled = np.arange(n_trials)
    random.shuffle(trial_numbers_shuffled)
    
    prediction_accuracy_test = []
    prediction_accuracy_fit = []
    prediction_accuracy_test_bias_only = []
    
    for kk in range(k_fold):
        
        test_begin = int(kk * np.floor(n_trials/k_fold))
        test_end = int((n_trials) if (kk == k_fold - 1) else (kk+1) * np.floor(n_trials/k_fold))
        test_set_this = trial_numbers_shuffled[test_begin:test_end]
        fit_set_this = np.hstack((trial_numbers_shuffled[:test_begin], trial_numbers_shuffled[test_end:]))
        
        # == Fit data using fit_set_this ==
        if if_verbose: print('%g/%g...'%(kk+1, k_fold), end = '')
        fitting_result = optimize.differential_evolution(func = negLL_func, args = (forager, fit_names, choice_history, reward_history, session_num, {}, fit_set_this),
                                                         bounds = optimize.Bounds(fit_bounds[0], fit_bounds[1]), 
                                                         mutation=(0.5, 1), recombination = 0.7, popsize = DE_pop_size, strategy = 'best1bin', 
                                                         disp = False, 
                                                         workers = 1 if pool == '' else int(mp.cpu_count()),   # For DE, use pool to control if_parallel, although we don't use pool for DE
                                                         updating = 'immediate' if pool == '' else 'deferred',
                                                         callback = None,)
            
        # == Rerun predictive choice sequence and get the prediction accuracy of the test_set_this ==
        kwargs_all = {}
        for (nn, vv) in zip(fit_names, fitting_result.x):  # Use the fitted data
            kwargs_all = {**kwargs_all, nn:vv}
        
        # Handle data from different sessions
        if session_num is None:
            session_num = np.zeros_like(choice_history)[0]  # Regard as one session
        
        unique_session = np.unique(session_num)
        predictive_choice_prob = []
        fitting_result.trial_numbers = []
        
        # -- For each session --
        for ss in unique_session:
            # Data in this session
            choice_this = choice_history[:, session_num == ss]
            reward_this = reward_history[:, session_num == ss]
            
            # Run PREDICTIVE simulation    
            bandit = BanditModel(forager = forager, **kwargs_all, fit_choice_history = choice_this, fit_reward_history = reward_this)  # Into the fitting mode
            bandit.simulate()
            predictive_choice_prob.extend(bandit.predictive_choice_prob)
            
        # Get prediction accuracy of the test_set and fitting_set
        predictive_choice_prob = np.array(predictive_choice_prob)
        predictive_choice = np.argmax(predictive_choice_prob, axis = 0)
        prediction_correct = predictive_choice == choice_history[0]
        
        # Also return cross-validated prediction_accuracy_bias (Maybe this is why Hattori's bias_only is low? -- Not exactly...)
        if 'biasL' in kwargs_all:
            bias_this = kwargs_all['biasL']
            prediction_correct_bias_only = int(bias_this <= 0) == choice_history[0] # If bias_this < 0, bias predicts all rightward choices
            prediction_accuracy_test_bias_only.append(sum(prediction_correct_bias_only[test_set_this]) / len(test_set_this))

        prediction_accuracy_test.append(sum(prediction_correct[test_set_this]) / len(test_set_this))
        prediction_accuracy_fit.append(sum(prediction_correct[fit_set_this]) / len(fit_set_this))
        

    return prediction_accuracy_test, prediction_accuracy_fit, prediction_accuracy_test_bias_only
            


