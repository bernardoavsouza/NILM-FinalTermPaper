# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 23:37:50 2020

@author: Bernardo A V Souza
"""

from Load import Appliance as ap
from Waveletlib import InputSignal as wl
import numpy as np
from windowing import processing as win
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans as km
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix



                             # Setup
                             
database = 'redd'                   # Chosen dataset
house_num = 1                       # Chosen house
win_size = 60
win_step = 60
K = 1
level_list = [2, 4, 8]                # Decomposition level
wl_list = ['sym11', 'sym12', 'sym13', 'sym14', 'sym15',
           'sym16', 'sym17', 'sym18', 'sym19', 'sym20']


def window_define(x):    
    # Mode
    from scipy import stats
    return stats.mode(x)


appliances = {'fridge':ap('fridge', [1,2,3,5,6], database), 
              'oven':ap('electric oven', [1], database),
              'mw':ap('microwave', [1,2,3,5], database),
              "wd":ap('washer dryer', [1,2,3,4,5,6], database),
              'dw':ap('dish washer', [1,2,3,4,5,6], database),
              'light':ap('light', [1,2,3,4,5,6], database),
              'bathroom_gfi':ap('unknown', [1,3,4,5,6], database),
              'heater':ap('electric space heater', [1,5,6], database),
              'stove':ap('electric stove', [1,2,4,6], database),
              'disposal': ap('waste disposal unit', [2,3,5], database),
              'electronics': ap('CE appliance', [3,5,6], database),
              'furnace': ap('electric furnace', [3,4,5], database),
              'sa': ap('smoke alarm', [3,4], database),
              'air_cond': ap('air conditioner', [4,6], database),
              'subpanel': ap('subpanel', [5], database)}
cluster_num = {'fridge':3,'oven':2, 'mw':2, "wd":3, 'dw':2,
               'light':3,'bathroom_gfi':2, 'heater':3, 'stove':2,
               'disposal':2, 'electronics':2, 'furnace':2, 'sa':2,
               'air_cond':2, 'subpanel':2}
print("\rLoading data completed.")

# This part set the labels of the states
labels_dict = {}
counter = 0
for i in appliances.keys():
    print('\rSetting label of ' + i + '.')
    labels_dict[i] = []
    for j in range(len(appliances[i].data)):
        temp = km(n_clusters=cluster_num[i]).fit(appliances[i].data[j].values.reshape(-1,1))
        labels_dict[i].append(temp.predict(appliances[i].data[j].values.reshape(-1,1)) + counter)
    counter = counter + cluster_num[i]
print("\rAll labels have been setted.")

                            # Algorhythm

# Windowing the labels 
print('\rWindowing labels.')
labels_win_dict = {}
for i in appliances.keys():
    labels_win_dict[i] = []
    for j in range(len(appliances[i].data)):
        labels_win_dict[i].append(win(labels_dict[i][j], size=win_size, stepsize=win_step))

# Windowing the data
print("\rWindowing data.")
x_win_dict = {}
for i in appliances.keys():
    x_win_dict[i] = []
    for j in range(len(appliances[i].data)):
        x_win_dict[i].append(win(appliances[i].data[j], size=win_size, stepsize=win_step))

# Applying window_define to get the array of window's labels
labels_win = []
for i in appliances.keys():
    print('\rProcessing windows of ' + i + '.')
    for L in range(len(appliances[i].data)):
        for j in range(labels_win_dict[i][L].shape[1]):
            labels_win.append(int(window_define(labels_win_dict[i][L][j])[0]))
labels_win = np.array(labels_win)

print("\rWindowing completed.")


iteration = 1
for num in level_list:
    level = num
    for wave in wl_list:
        wl_kind = wave
        try:
            print("Processing . . . " + str(iteration) + r'/' + str(len(level_list)*len(wl_list)))
            
            # Applying Wavelet in the windows 
            wl_dict = {}
            temp = None
            temp2 = None
            for i in x_win_dict.keys():
                print('\rCalculating Wavelets in ' + i + '.')
                wl_dict[i] = []
                for L in range(len(x_win_dict[i])):
                    dt = pd.DataFrame()
                    for j in range(x_win_dict[i][L].shape[1]):
                        temp = wl(x_win_dict[i][L].iloc[:,j], level=level, kind=wl_kind)
                        temp2 = np.concatenate([k for k in temp.details])
                        temp = np.concatenate([temp2, temp.approx])
                        dt[j] = temp
                    wl_dict[i].append(dt)
                    
            # Turning all windows in one array
            print("\rPreparing to classify.")
            x_win = []
            for i in appliances.keys():
                for k in range(len(wl_dict[i])):    
                    for j in range(wl_dict[i][k].shape[1]):
                        x_win.append(list(wl_dict[i][k].iloc[:, 1]))
            x_win = np.array(x_win)
            
            # Splitting data
            x_train, x_test, y_train, y_test = train_test_split(x_win, labels_win, train_size=0.7)
            
            # Training
            print("\rTraining.")
            knn=KNeighborsClassifier(n_neighbors=K)    
            knn.fit(x_train, y_train)
            
            # Final Test
            print('\rTesting.')
            pred = knn.predict(x_test)
            result = accuracy_score(y_test,pred)
            conf = pd.DataFrame(confusion_matrix(y_test, pred))
            
            counter = 0
            temp = []
            for i in appliances.keys():
                counter = counter + cluster_num[i]
                for j in range(cluster_num[i]):
                    temp.append(i) 
            conf[str(counter)] = temp
            
            conf.to_csv(str(round(result*100,2)) + '%-' + wl_kind + '-'+ str(level) + 'level.csv')
            
        except ValueError as e:
            print(e)
        iteration +=1
            
print("\rDONE!")
  
#    -----------------------------------------------------------------------
    
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

entries = ['57.43%-rbio3.5-2level.csv', '59.43%-rbio3.1-2level.csv',
 '60.66%-sym2-2level.csv', '60.97%-rbio2.2-2level.csv',
 '62.25%-bior2.2-2level.csv', '62.38%-bior3.5-2level.csv',
 '62.9%-sym2-4level.csv', '63.03%-bior5.5-2level.csv',
 '63.6%-sym3-2level.csv', '63.79%-bior4.4-2level.csv',
 '63.97%-bior3.1-2level.csv', '64.16%-bior3.7-2level.csv',
 '64.34%-bior1.3-2level.csv', '64.67%-sym5-2level.csv',
 '64.86%-rbio1.5-2level.csv', '65.0%-rbio3.1-4level.csv',
 '65.05%-rbio3.7-2level.csv', '65.12%-rbio2.4-2level.csv',
 '65.16%-rbio1.3-2level.csv', '65.97%-bior3.1-4level.csv',
 '66.28%-rbio3.3-2level.csv', '66.29%-rbio2.6-2level.csv',
 '67.04%-bior3.3-2level.csv', '67.39%-bior2.6-2level.csv',
 '67.58%-sym8-2level.csv', '67.84%-sym7-2level.csv',
 '68.34%-rbio4.4-2level.csv', '69.39%-rbio5.5-2level.csv']

axis = np.array(['fridge', 'fridge', 'fridge', 'oven', 'oven', 'm. w.', 
                 'm. w.', 'w. d.','w. d.', 'w. d.', 'd. w.', 'd. w.', 
                 'light', 'light', 'light', 'gfi', 'gfi', 'heater',
                 'heater', 'heater', 'stove', 'stove', 'disposal',
                 'disposal', 'electronics', 'electronics', 'furnace',
                 'furnace', 'smoke a.', 'smoke a.', 'air_cond', 'air_cond',
                 'subpanel', 'subpanel'])
sns.set(font_scale=.6)
for i in entries:
    df = pd.read_csv(i)
    df = df.set_index(axis)
    df = df.iloc[:,1:-1]
    df.columns = axis
    df = df/df.max().max()
    hm = sns.heatmap(df,cmap="YlOrRd")
    fig = hm.get_figure() 
    fig.savefig(i[:-4] + '.png')
    plt.close()

#------------------------------------------------------------------------------

from Load import Appliance as ap
from Waveletlib import InputSignal as wl
import numpy as np
from windowing import processing as win
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans as km
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt

database = 'redd'                   # Chosen dataset
house_num = 1                       # Chosen house
win_size = 5
win_step = 1
epsilon = 20

signal = ap(1, 1, database).data

'''
# Windowing the data
signal_win = win(signal, size=win_size, stepsize=win_step)

# Standard Deviation
std = []
for i in range(signal_win.shape[1]):
    std.append(np.std(signal_win[i]))
'''

# Delta P
deltaP = [0]     
for k in range(1,len(signal)):
       deltaP.append(signal[k]-signal[k-1])


'''
# Defining epsilon
epsilon = {}
for i in appliances.keys():
    epsilon[i] = []
    for j in range(len(appliances[i].data)):
        epsilon[i].append(appliances[i].data[j].std())

# Getting events
time = []
transient = []
ss_time = []
for k in range(len(signal)-4):
    if std[k] >= epsilon:
        transient.append(std[k])
        time.append(k)
    elif abs(deltaP[k]) >= epsilon:
        transient.append(deltaP[k])
        time.append(k)
    else:
        ss_time.append(k)
'''

# Getting events
time = []
transient = []
ss_time = []
for k in range(len(deltaP)):
    if abs(deltaP[k]) >= epsilon:
        transient.append(deltaP[k+win_size])
        time.append(k+win_size)
    else:
        ss_time.append(k)
time = np.array(time)
transient = np.array(transient)
ss_time = np.array(ss_time)      

abs_transient = np.array([abs(i) for i in transient]).reshape(-1,1)


# Elbow method
n_clusters = 18


# Clustering
kmeans = km(n_clusters).fit(abs_transient)
prediction = kmeans.predict(np.array(abs(transient)).reshape(-1,1))
clusters_tags = np.unique(prediction)


# Functions
def grouping(transient, idx):
    ridx = []
    fidx = []
    
    temp = [idx[0]]
    for i in range(len(idx)):
        try:
            if transient[idx[i]]*transient[idx[i+1]] > 0:
                temp.append(idx[i+1])
            else:
                if transient[idx[i]] > 0:
                    ridx.append(temp)
                else:
                    fidx.append(temp)
                temp = [idx[i+1]]
        except IndexError:
            if transient[idx[i]] > 0:
                ridx.append(temp)
            else:
                fidx.append(temp)
                
    return ridx, fidx

def match(ridx, fidx):
    threshold = epsilon
    temp = []
    tempidx = []
    match_tuple = []
    if ridx[0][0] < fidx[0][0]:
        for i in range(len(ridx)):
            try:
                for j in ridx[i]:
                    for k in fidx[i]:
                        temp.append(abs(transient[j] + transient[k]))
                        tempidx.append((j,k))
                if min(temp) > threshold: break                        
                match_index = temp.index(min(temp))
                match_tuple.append(tempidx[match_index])
            except IndexError:
                pass
            temp = []   
            tempidx = []
    else:
        for i in range(len(ridx)):
            try:
                for j in ridx[i]:
                    for k in fidx[i+1]:
                        temp.append(abs(transient[j] + transient[k]))
                        tempidx.append((j,k))
                if min(temp) > threshold: break
                match_index = temp.index(min(temp))
                match_tuple.append(tempidx[match_index])
            except IndexError:
                pass
            temp = []   
            tempidx = []
    
    return match_tuple



cycles_signal = {}
for i in range(n_clusters):
    try:
        idx = np.where(prediction==i)[0]
        ridx, fidx = grouping(transient,idx)
        cycle = match(ridx, fidx)
        temp = np.zeros(len(signal))
        for t in cycle:
            temp[time[t[0]]:time[t[1]+1]] = kmeans.cluster_centers_[i]
        cycles_signal[i] = temp
        
        plt.figure()
        plt.scatter(time[idx],transient[idx],color='red')
        plt.plot(temp)
        
    except IndexError:
        pass



plt.figure()
for i in range(n_clusters):
    x = time[np.where(prediction==i)]
    y = transient[np.where(prediction==i)]
    if i <= 10:
        plt.scatter(x,y)
    else:
        plt.scatter(x,y, marker="x")

#plt.legend([i for i in range(0,18)])
    


distortions = []
K = range(1,50)
for k in K:
    kmeanModel = km(n_clusters=k)
    kmeanModel.fit(abs_transient)
    distortions.append(kmeanModel.inertia_)


plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


#-------------------------------------------------------------------------------------------------


from Load import Appliance as ap
from Waveletlib import InputSignal as wl
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans as km
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt



# FUNCTIONS
def win(a, stepsize=1, size=3, axis=0):
    out = []
    max_time = a.shape[0]
    for i in range(0, max_time-size, stepsize):
        temp = a[i:size+i]
        out.append(np.expand_dims(temp, 0))
    return pd.DataFrame(np.vstack(out))


database = 'redd'                   # Chosen dataset
house_num = 1                       # Chosen house
win_size = 3
win_step = 1
level = 1
cluster_thr = 100
merging_thr = cluster_thr
pairing_event_thr = 230
power_thr = 100
noise_thr = 20


signal = ap(1, house_num, database).data

############## trying wavelets

'''
signal = signal/max(abs(signal))
noise_thr = 350e3 #6e6  se a variancia das wavelets do sinal for tratada como o novo sinal
power_thr = 30    #20e3
power_thr = power_thr/max(abs(signal))
noise_thr = noise_thr/max(abs(signal))

for k in range(1):
    signal_wl = pd.DataFrame()
    signal_win = win(signal, size=win_size, stepsize=win_step).transpose()
    for i in range(signal_win.shape[1]):
        signal_wl[i] = np.concatenate(wl(signal_win.iloc[:,i], level=level).details)
        #approx = wl(signal_win.iloc[:,i], level=level).approx
        #signal_wl[i] = np.concatenate([details, approx])
    if k == 0:
        signal = signal_wl.var(0)
signal_variance = signal_wl.var(0)
signal_mean = signal_wl.mean(0)
signal = ap(1, house_num, database).data
'''

# EVENTS                                                                                            
signal_win = win(signal, size=win_size, stepsize=win_step)
signal_variance = signal_win.var(1)
signal_mean = signal_win.mean(1)



## Steady State detection
ss = np.array([])
ss_time = np.array([])
for i in range(len(signal_variance)):
    if noise_thr > signal_variance[i]:
        ss = np.append(ss, signal_mean[i])
        ss_time = np.append(ss_time, i)
    print("Detecting Steady States " + str(i) + " of " + str(len(signal_variance)))

## Detecting events
transient = np.array([])
event_start = np.array([])
event_end = np.array([])
for i in range(1,len(ss)):
    if abs(ss[i] - ss[i-1]) > power_thr:
        transient = np.append(transient, ss[i]-ss[i-1])
        event_start = np.append(event_start, ss_time[i-1])                      # IGNORED #
        event_end = np.append(event_end, ss_time[i])
    print("Detecting events " + str(i) + " of " + str(len(ss)-1))        

'''  PLOTTING DETECTION
plt.figure(1)
plt.plot(signal.values)
plt.scatter(ss_time, ss, color='yellow')
plt.figure(2)
plt.plot(signal.values)
plt.scatter(event_start, transient, color='orange')
plt.plot(np.zeros(200000), color='red')
'''


# CLUSTERING

## Rising and falling signals
transient_low = transient[transient[:] < 0]
time_low = event_end[transient[:] < 0]
dt_low = pd.DataFrame({"power": transient_low, "sample num": time_low})
dt_low = dt_low.sort_values(by='power').set_index('sample num')

transient_high = transient[transient[:] > 0]
time_high = event_end[transient[:] > 0]
dt_high = pd.DataFrame({"power": transient_high, "sample num": time_high})
dt_high = dt_high.sort_values(by='power').iloc[::-1,:].set_index('sample num')


## Main clustering computing
clusters = np.array([1])
cluster_count = 1
for i in range(1, dt_low.shape[0]):
    if abs(dt_low['power'].iloc[i-1] - dt_low['power'].iloc[i]) > cluster_thr:
        cluster_count += 1
    clusters = np.append(clusters, cluster_count)
dt_low['cluster'] = clusters

cluster_count += 1
clusters = np.array([cluster_count])
for i in range(1, dt_high.shape[0]):
    if dt_high['power'].iloc[i-1] - dt_high['power'].iloc[i] > cluster_thr:
        cluster_count += 1
    clusters = np.append(clusters, cluster_count)
dt_high['cluster'] = clusters


## Cluster merging
means_low = np.array([])
for i in dt_low['cluster'].unique():
    mean = dt_low[dt_low['cluster']==i]['power'].mean()
    means_low = np.append(means_low, mean)
    
merging_memory = 0
clusters_low = dt_low['cluster'].unique()
means_low_temp = means_low
for i in range(1, len(means_low)):
    if abs(means_low[i] - means_low[i-1]) < merging_thr:
        dt_low.loc[dt_low['cluster'] == clusters_low[i], 'cluster'] = clusters_low[i-1] - merging_memory
        merging_memory += 1
        means_low_temp = np.delete(means_low_temp, i)
    else:
        merging_memory = 0
     
        
means_high = np.array([])
for i in dt_high['cluster'].unique():
    mean = dt_high[dt_high['cluster']==i]['power'].mean()
    means_high = np.append(means_high, mean)

merging_memory = 0
clusters_high = dt_high['cluster'].unique()
means_high_temp = means_high
for i in range(1, len(means_high)):
    if abs(means_high[i] - means_high[i-1]) < merging_thr:
        dt_high.loc[dt_high['cluster'] == clusters_high[i], 'cluster'] = clusters_high[i-1] - merging_memory
        merging_memory += 1
        means_high_temp = np.delete(means_high_temp, i)
    else:
        merging_memory = 0


# PAIRING

## On-off event pairing
on_off_temp = pd.DataFrame()                    
for i in range(len(means_low_temp)):                                                 
    for j in range(len(means_high_temp)):
        if abs(means_low_temp[i] + means_high_temp[j]) < cluster_thr:
            diff = abs(means_low_temp[i] + means_high_temp[j])
            on_off_temp = on_off_temp.append(pd.DataFrame([[i, j, diff]], columns = ['low', 'high', 'diff']))

### Selecting the best choices
on_off_temp['idx'] = list(range(on_off_temp.shape[0]))
on_off_temp = on_off_temp.set_index('idx')

'''
on_off_temp = on_off_temp.drop([2,4])
'''  

on_off_idx = pd.DataFrame()
while len(on_off_temp) != 0:                                            # This while removes the repeated pairs
    idx = on_off_temp['diff'].idxmin()
    on_off_idx = on_off_idx.append(on_off_temp.loc[idx, ["low", 'high']])
    drops = list(on_off_temp[on_off_temp['high'] == on_off_temp.loc[idx]['high']].index)
    drops = drops + list(on_off_temp[on_off_temp['low'] == on_off_temp.loc[idx]['low']].index)
    for i in drops:
        try:
            on_off_temp = on_off_temp.drop(i)
        except KeyError:
            continue
on_off_idx = on_off_idx.astype(int)


### Defining on-off pair and separating the finite states clusters
finiteState_clusters_low = np.delete(clusters_low, list(on_off_idx['low']))
finiteState_clusters_high = np.delete(clusters_high, list(on_off_idx['high']))
means_low = np.delete(means_low, list(on_off_idx['low']))
means_high = np.delete(means_high, list(on_off_idx['high']))


 

if len(clusters_high)*len(clusters_low) == 0:
    raise ValueError("No more clusters for finite state appliances.")


### Forward and Backward Pairing
dt_high = dt_high.sort_values(by = 'sample num')
dt_low = dt_low.sort_values(by = 'sample num')

backward_pairing = {}
forward_pairing = {}
for _, i in on_off_idx.iterrows():
    cluster_h = clusters_high[i['high']]
    cluster_l = clusters_low[i['low']]
    time_high = list(dt_high[dt_high['cluster'] == cluster_h].index)
    time_low = list(dt_low[dt_low['cluster'] == cluster_l].index)
    
    #### Forward
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
    
    #### Backward (???)
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
            
#### Evaluation 
ef_likely = {} 
Mp = lambda x: abs(dt_high.loc[i]['power'] + dt_low.loc[forward_pairing[i]]['power']).iloc[x]
Mt = lambda x: abs(forward_pairing[i] - i)[x]
for i in forward_pairing.keys():
   if len(forward_pairing[i]) == 0:
       break
   else:
       mp = np.array([Mp(j) for j in range(len(forward_pairing[i]))]).mean()
       mt = np.median([Mt(j) for j in range(len(forward_pairing[i]))])
       ef_likely[i] = np.array([])
       for j in forward_pairing[i]:
           omega_p = abs(dt_high.loc[i]['power'] + dt_low.loc[j]['power'])     # Power diff
           omega_t = abs(j - i)                                                     # Time diff
           ci = (omega_p*mp + omega_t*mt)/((omega_p**2 + omega_t**2)*(mp**2 + mt**2))**0.5
           ef_likely[i] = np.append(ef_likely[i], ci)

eb_likely = {}
Mp = lambda x: abs(dt_low.loc[i]['power'] + dt_high.loc[backward_pairing[i]]['power'])
Mt = lambda x: abs(backward_pairing[i] - i)[x]
for i in backward_pairing.keys():
    if len(backward_pairing[i]) == 0:
        break
    else:
        mp = np.array([Mp(j) for j in range(len(backward_pairing[i]))]).mean()
        mt = np.median([Mt(j) for j in range(len(backward_pairing[i]))])
        eb_likely[i] = np.array([])
        for j in backward_pairing[i]:
            psi_p = abs(dt_low.loc[i]['power'] + dt_high.loc[j]['power'])
            psi_t = abs(j - i)
            ci = (psi_p*mp + psi_t*mt)/((psi_p**2 + psi_t**2)*(mp**2 + mt**2))**0.5
            eb_likely[i] = np.append(eb_likely[i], ci)

ef_pairs = {}
for i in list(ef_likely.keys()):
    ef_pairs[i] = forward_pairing[i][ef_likely[i].argmax()]

eb_pairs = {}
for i in list(eb_likely.keys()):
    eb_pairs[i] = backward_pairing[i][eb_likely[i].argmax()]

'''
if ef == eb: True

if fall_ef == fall_eb and rising_ef != rising_eb: True
'''


