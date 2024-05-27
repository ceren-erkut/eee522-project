import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import decimate,butter, lfilter
import tftb
import spkit as sp
import scipy.signal as signal

import time




def generateTFfigure(sample, fs, nfft , nperseg , alpha ):
    Xa = sp.frft(sample,alpha=alpha)
    fx,tx,Sx = signal.spectrogram(Xa,fs=fs,nperseg=nperseg,nfft=nfft,return_onesided=False)
    Sx = np.abs(Sx)

    return Sx[:int( nfft /2 + 1)], tx
