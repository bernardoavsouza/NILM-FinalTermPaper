# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 18:23:42 2021

@author: Bernardo A V de Souza
"""


from Load import Appliance as ap
import numpy as np
import pandas as pd
from tqdm import tqdm as loading_bar
from matplotlib import pyplot as plt
from pywt import wavedec as wl
from pywt import waverec as wlrec


SETTINGS = {
        'database' : 'redd',                   # Chosen dataset
        'length' : 4e4,
        'house_num' : 1,                       # Chosen house
        'win_size' : 40,
        'win_step' : 1,
        'power_thr' : 100,
        'noise_thr' : 20,
        'cluster_thr' : 100,
        'merging_thr' : 100,
        'pairing_event_thr : 230,
        'plot_num' : 0,
        }

##############################################################################
# 
#                              LOADING SETTINGS
#
##############################################################################



def main_settings():
    return  SETTINGS



##############################################################################
# 
#                              MAIN ALGORITHM
#
##############################################################################



                        ############################
                        #                          #
                        #      LOADING SIGNAL      #
                        #                          #
                        ############################



## Load the signal
def load_signal(size = 'full', SETTINGS = SETTINGS):
    signal = {}
    signal['values'] = ap(1, SETTINGS['house_num'], SETTINGS['database']).data
    signal['values'].index = range(len(signal['values']))
    
    if size != 'full':
        size = int(size)
        signal['values'] = signal['values'][:size] 
        
    signal['win'] = win(signal['values'], size=SETTINGS['win_size'], 
                        stepsize=SETTINGS['win_step'])
    signal['win var'] = signal['win'].var(1)
    signal['win mean'] = signal['win'].mean(1)
    
    for i in loading_bar(range(1), desc = "Loading signal."): pass

    return signal


## Load wavelet transform of the signal (its not working yet with a level of 
## decomposition different than 2)
def load_wl_signal(size = 'full', level = 2, win_size = SETTINGS['win_size'], 
                   kind='db1', SETTINGS = SETTINGS):
    
    signal = ap(1, SETTINGS['house_num'], SETTINGS['database']).data.values
    
    if size != 'full':
        size = int(size)
        signal = signal[:size] 
        
    if level != 2:
        raise ValueError("Function not ready for decomposition level different than 2.")
    print(win_size)
    out = np.array([])
    signal_win = win(signal, size=win_size, stepsize=win_size)
    
    for i in loading_bar(range(signal_win.shape[0]), 
                         desc="Loading wavelet signal"):
        temp = wl(signal_win.iloc[i,:], level=level, wavelet=kind)
        temp[0] = np.zeros_like(temp[0])
        temp = wlrec(temp, kind)
        out = np.concatenate((out, temp))
        
    return out
    


                        ############################
                        #                          #
                        #         DETECTORS        #
                        #                          #
                        ############################



# Main detection function
def detection(signal_variance, signal_mean, SETTINGS = SETTINGS):
    ss, ss_time = ss_detection(signal_variance, signal_mean, SETTINGS = SETTINGS)
    tup = event_detection(ss, ss_time, SETTINGS = SETTINGS)
    
    return tup


## Steady State detection
def ss_detection(signal_variance, signal_mean, SETTINGS = SETTINGS):
    ss = np.array([])
    ss_time = np.array([])
    
    for i in loading_bar(range(len(signal_variance)), 
                         desc='Detecting steady states'):
        if SETTINGS['noise_thr'] > signal_variance[i]:
            ss = np.append(ss, signal_mean[i])
            ss_time = np.append(ss_time, i)
    
    return ss, ss_time


# Detecting events
def event_detection(ss, ss_time, SETTINGS = SETTINGS):
    transient = np.array([])
    event_start = np.array([])
    event_end = np.array([])
    
    for i in loading_bar(range(1,len(ss)), desc='Detecting events'):
        if abs(ss[i] - ss[i-1]) > SETTINGS['power_thr']:
            transient = np.append(transient, ss[i]-ss[i-1])
            event_start = np.append(event_start, ss_time[i-1])     # IGNORED #
            event_end = np.append(event_end, ss_time[i])   
    
    return transient, event_start, event_end         



                        ############################
                        #                          #
                        #        CLUSTERING        #
                        #                          #
                        ############################



# Main clutering function
def clustering(transient, event_end):
    df_low = falling_events(transient, event_end)
    df_high = rising_events(transient, event_end)
    df = clustering_processing(df_low, df_high)
    means_low, clusters_low = cluster_merging_low(df)
    means_high, clusters_high = cluster_merging_high(df)
    means = {'low' : means_low,
             'high' : means_high}
    clusters = {'low' : clusters_low,
                'high' : clusters_high}
    
    if len(clusters['high'])*len(clusters['low']) == 0:
        raise ValueError("No more clusters for finite state appliances.")
    
    return means, clusters, df


# Taking falling events
def falling_events(transient, event_end):
    transient_low = transient[transient[:] < 0]
    time_low = event_end[transient[:] < 0]
    df_low = pd.DataFrame({"power": transient_low, "sample num": time_low})
    df_low = df_low.sort_values(by='power').set_index('sample num')

    return df_low


# Taking rising events
def rising_events(transient, event_end):
    transient_high = transient[transient[:] > 0]
    time_high = event_end[transient[:] > 0]
    df_high = pd.DataFrame({"power": transient_high, "sample num": time_high})
    df_high = df_high.sort_values(by='power').iloc[::-1,:].set_index('sample num')
    
    return df_high


# Processing a draft of the clusters
def clustering_processing(df_low, df_high):
    clusters = np.array([1])
    cluster_count = 1
    
    for i in range(1, df_low.shape[0]):
        if abs(df_low['power'].iloc[i-1] - df_low['power'].iloc[i]) > SETTINGS['cluster_thr']:
            cluster_count += 1
        clusters = np.append(clusters, cluster_count)
    df_low['cluster'] = clusters
    
    cluster_count += 1
    clusters = np.array([cluster_count])
    
    for i in range(1, df_high.shape[0]):
        if df_high['power'].iloc[i-1] - df_high['power'].iloc[i] > SETTINGS['cluster_thr']:
            cluster_count += 1
        clusters = np.append(clusters, cluster_count)
    df_high['cluster'] = clusters
    
    df = {'low' : df_low,
          'high' : df_high}
    
    return df


# Merging falling clusters event
def cluster_merging_low(df):
    df_low = df['low'] 
    means_low_temp = np.array([])
    
    for i in df_low['cluster'].unique():
        mean = df_low[df_low['cluster']==i]['power'].mean()
        means_low_temp = np.append(means_low_temp, mean)
        
    merging_memory = 0
    clusters_low = df_low['cluster'].unique()
    means_low = means_low_temp
    s
    for i in range(1, len(means_low_temp)):
        if abs(means_low_temp[i] - means_low_temp[i-1]) < SETTINGS['merging_thr']:
            df_low.loc[df_low['cluster'] == clusters_low[i], 'cluster'] = clusters_low[i-1] - merging_memory
            merging_memory += 1
            means_low = np.delete(means_low, i)
        else:
            merging_memory = 0
            
    return means_low, clusters_low
    

# Merging rising clusters event
def cluster_merging_high(df):
    df_high = df['high']
    means_high_temp = np.array([])
    
    for i in df_high['cluster'].unique():
        mean = df_high[df_high['cluster']==i]['power'].mean()
        means_high_temp = np.append(means_high_temp, mean)
    
    merging_memory = 0
    clusters_high = df_high['cluster'].unique()
    means_high = means_high_temp
    
    for i in range(1, len(means_high_temp)):
        if abs(means_high_temp[i] - means_high_temp[i-1]) < SETTINGS['merging_thr']:
            df_high.loc[df_high['cluster'] == clusters_high[i], 'cluster'] = clusters_high[i-1] - merging_memory
            merging_memory += 1
            means_high = np.delete(means_high, i)
        else:
            merging_memory = 0
            
    return means_high, clusters_high



                        ############################
                        #                          #
                        #     CLUSTER PAIRING      #
                        #                          #
                        ############################

 

# Main cluster pairing function    
def cluster_pairing(means, clusters):   
    on_off = generaton_cluster_pairing_df(means)
    on_off, on_off_idx = clusters_best_selection(on_off)
    means, finite_state_clusters = on_off_separation(means, clusters, on_off_idx)
    
    return means, clusters, on_off, on_off_idx
   

# Organizating it
def generaton_cluster_pairing_df(means):
    means_low = means['low']
    means_high=  means['high']
    on_off = pd.DataFrame()   
                 
    for i in range(len(means_low)):  
                                               
        for j in range(len(means_high)):
            if abs(means_low[i] + means_high[j]) < SETTINGS['cluster_thr']:
                diff = abs(means_low[i] + means_high[j])
                on_off = on_off.append(pd.DataFrame([[i, j, diff]], columns = ['low', 'high', 'diff']))

    return on_off 


# Selecting the best choices
def clusters_best_selection(on_off):
    on_off['idx'] = list(range(on_off.shape[0]))
    on_off = on_off.set_index('idx')
    on_off_idx = pd.DataFrame()
    
    while len(on_off) != 0:                                            # This while removes the repeated pairs
        idx = on_off['diff'].idxmin()
        on_off_idx = on_off_idx.append(on_off.loc[idx, ["low", 'high']])
        drops = list(on_off[on_off['high'] == on_off.loc[idx]['high']].index)
        drops = drops + list(on_off[on_off['low'] == on_off.loc[idx]['low']].index)
        
        for i in drops:
            try:
                on_off = on_off.drop(i)
            except KeyError:
                continue
    on_off_idx = on_off_idx.astype(int)
    
    return on_off, on_off_idx


# Defining on-off pair and separating the finite states clusters
def on_off_separation(means, clusters, on_off_idx):
    means_low = means['low']
    means_high=  means['high']
    clusters_low = clusters['low']
    clusters_high =  clusters['high']
    
    finite_state_clusters_low = np.delete(clusters_low, list(on_off_idx['low']))
    finite_state_clusters_high = np.delete(clusters_high, list(on_off_idx['high']))
    means_low = np.delete(means_low, list(on_off_idx['low']))
    means_high = np.delete(means_high, list(on_off_idx['high']))

    means = {'low' : means_low,
             'high' : means_high}
    finite_state_clusters = {'low' : finite_state_clusters_low,
                             'high' : finite_state_clusters_high}    
    
    return means, finite_state_clusters



                        ############################
                        #                          #
                        #       EVENT PAIRING      #
                        #                          #
                        ############################
                        
               
                        
# Main event pairing function
def event_pairing(df, clusters, on_off_idx):
    backward_pairing = {}
    forward_pairing = {}
    
    for _, i in on_off_idx.iterrows():
        times_l_h = event_pairing_start(df, clusters, i)
        forward_pairing = forward(forward_pairing, times_l_h, i)
        #backward_pairing = backward(backward_pairing, times_l_h, i)
        
    pairing = {'forward' : forward_pairing}
              #'backward' : backward_pairing}
        
    return pairing


# Sortting events to help the pairing
def sorting_events(df_low, df_high):
    df_high = df_high.sort_values(by = 'sample num')
    df_low = df_low.sort_values(by = 'sample num')
    df = {'low' : df_low,
          'high' : df_high}
    
    return df


# Preparation to Forwars and backward pairing
def event_pairing_start(df, clusters, i):
    cluster_h = clusters['high'][i['high']]
    cluster_l = clusters['low'][i['low']]
    df_low = df['low']
    df_high = df['high']
    
    time_high = list(df_high[df_high['cluster'] == cluster_h].index)
    time_low = list(df_low[df_low['cluster'] == cluster_l].index)
    
    times_l_h = {'low' : time_low,
                'high' : time_high}   
    
    return times_l_h


# Forward pairing
def forward(forward_pairing, times_l_h, i):
    time_low = times_l_h['low']
    time_high = times_l_h['high']        
    
    for j in range(len(time_high) - 1):
        forward_pairing[time_high[j]] = np.array([])
        
        for k in range(len(time_low)):
            if time_low[k] < time_high[j]:
                continue
            elif time_low[k] > time_high[j+1]:
                break
            else:
                forward_pairing[time_high[j]] = np.append(forward_pairing[time_high[j]], time_low[k])
                
        if forward_pairing[time_high[j]].size == 0:
            del forward_pairing[time_high[j]]
    
    return forward_pairing


''' >>>>>>>>>>>>>>>>>>>>>>>>
    ^ NOT WORKING PROPERLY v
    <<<<<<<<<<<<<<<<<<<<<<<<  ''' # Backward Pairing
def backward(backward_pairing, times_l_h, i):           
    time_low = times_l_h['low']
    time_high = times_l_h['high']   
    
    for j in range(len(time_low) - 1, 0, -1):
        backward_pairing[time_low[j]] = np.array([])
        
        for k in range(len(time_high)):
            if time_high[k] > time_low[j]:
                continue
            if time_high[k] < time_low[j-1]:
                break
            else:
                backward_pairing[time_low[j]] = np.append(backward_pairing[time_low[j]], time_high[k])
                
        if backward_pairing[time_low[j]].size == 0:
            del backward_pairing[time_low[j]]
    
    return backward_pairing



                        ############################
                        #                          #
''' TO DO '''           #        EVALUATION        #
                        #                          #
                        ############################
                        
               
                        
# Main evaluation function                    
def evaluation(pairing, df):
    ef_likelies = forward_evaluation(pairing, df)
    eb_likelies = backward_evaluation(pairing, df)
    likelies = combine_event_likelies(eb_likelies, ef_likelies)
    events_pairs = best_events_pairs(pairing, likelies)
    
    return events_pairs


# Ascending order evaluation
def forward_evaluation(pairing, df):
    forward_pairing = pairing['forward']  
    df_low = df['low']
    df_high = df['high']
    ef_likelies = {} 
    
    Mp = lambda x: abs(df_high.loc[i]['power'] + df_low.loc[forward_pairing[i]]['power']).iloc[x]
    Mt = lambda x: abs(forward_pairing[i] - i)[x]
    
    for i in forward_pairing.keys():
       if len(forward_pairing[i]) == 0:
           break
       else:
           mp = np.array([Mp(j) for j in range(len(forward_pairing[i]))]).mean()
           mt = np.median([Mt(j) for j in range(len(forward_pairing[i]))])
           ef_likelies[i] = np.array([])
          
           for j in forward_pairing[i]:
               omega_p = abs(df_high.loc[i]['power'] + df_low.loc[j]['power'])          # Power diff
               omega_t = abs(j - i)                                                     # Time diff
               ci = (omega_p*mp + omega_t*mt)/((omega_p**2 + omega_t**2)*(mp**2 + mt**2))**0.5
               ef_likelies[i] = np.append(ef_likelies[i], ci)
    
    return ef_likelies


# Descending order evaluation
def backward_evaluation(pairing, df):
    backward_pairing = pairing['backward']
    df_low = df['low']
    df_high = df['high']
    eb_likelies = {}
    
    Mp = lambda x: abs(df_low.loc[i]['power'] + df_high.loc[backward_pairing[i]]['power'])
    Mt = lambda x: abs(backward_pairing[i] - i)[x]
    
    for i in backward_pairing.keys():
        if len(backward_pairing[i]) == 0:
            break
        else:
            mp = np.array([Mp(j) for j in range(len(backward_pairing[i]))]).mean()
            mt = np.median([Mt(j) for j in range(len(backward_pairing[i]))])
            eb_likelies[i] = np.array([])
            
            for j in backward_pairing[i]:
                psi_p = abs(df_low.loc[i]['power'] + df_high.loc[j]['power'])           # Power diff
                psi_t = abs(j - i)                                                      # Time diff
                ci = (psi_p*mp + psi_t*mt)/((psi_p**2 + psi_t**2)*(mp**2 + mt**2))**0.5
                eb_likelies[i] = np.append(eb_likelies[i], ci)
    
    return eb_likelies

# Combine all likelies
def combine_event_likelies(eb_likelies, ef_likelies):
    likelies = {'eb' : eb_likelies,
              'ef' : ef_likelies}
    
    return likelies


''' >>>>>>>>>>>>>>>>>>>>>>>>
    ^      INCOMPLETED     v
    <<<<<<<<<<<<<<<<<<<<<<<<  ''' # Choosing the best options of pairs
def best_events_pairs(pairing, likelies):
    forward_pairing = pairing['forward'] 
    backward_pairing = pairing['backward']
    eb_likelies = likelies['eb']
    ef_likelies = likelies['ef']
    
    
    ef_pairs = {}
    
    for i in list(ef_likelies.keys()):
        ef_pairs[i] = forward_pairing[i][ef_likelies[i].argmax()]
    
    eb_pairs = {}
    
    for i in list(eb_likelies.keys()):
        eb_pairs[i] = backward_pairing[i][eb_likelies[i].argmax()]
        
    #if ef == eb: True
    #if fall_ef == fall_eb and rising_ef != rising_eb: True
    
    events_pairs = set([])
    
    return events_pairs



##############################################################################
# 
#                                  TOOLS
#
##############################################################################



## Plotting function
def plotting(signal, title, xlim = [8500, 9100], legends = ['Signal'], labels = ['Samples', 'Amplitude'], scatter = None, SETTINGS = SETTINGS):
    SETTINGS['plot_num'] += 1
    plt.figure(SETTINGS['plot_num'])
    if scatter != None:
        plt.scatter(scatter[0], scatter[1], color = 'orange')
    plt.plot(signal)
    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend([i for i in legends])
    plt.xlim(xlim)


## Windowing
def win(a, size = SETTINGS['win_size'], stepsize = 1):    
    if type(a) == pd.DataFrame or type(a) == pd.Series:
        a = a.values
    if stepsize != size:
        out = []
        max_time = a.shape[0]
        for i in range(0, max_time-size + 1, stepsize):
            temp = a[i:size+i]
            out.append(np.expand_dims(temp, 0))
        return pd.DataFrame(np.vstack(out))
    else:
        try: 
            row = int((a.shape[0] * a.shape[1])/size)
        except:
            row = int(len(a)/size)
    L = row*size
    a = a.reshape(1,-1)
    a = a[0, :L]
    a = a.reshape(row, size)
    return pd.DataFrame(a)
    
