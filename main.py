# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 15:41:30 2021

@author: Bernardo A V de Souza
"""

import functions as f 
from scipy.fftpack import fft
from scipy.signal.windows import hamming
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

SETTINGS = f.main_settings()

def main():
        
    
    signal = f.load_signal(SETTINGS['length'])
    transient, _, event_end = f.detection(signal['win var'], signal['win mean'])
    means, clusters, df = f.clustering(transient, event_end)
    means, clusters, on_off, on_off_idx = f.cluster_pairing(means, clusters)
    pairing = f.event_pairing(df, clusters, on_off_idx)
    #events_pairs = f.evaluation(pairing, df)
    
if __name__ == '__main__':
    main()
