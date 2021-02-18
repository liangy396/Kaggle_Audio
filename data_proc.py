#!/usr/bin/env python
# coding: utf-8

# WARNING: Kernel fails to automatically score if more than one file is saved to disk. You can still download and manually submit prediction. To allow model/spectrograms saving, change setting below.

# In[1]:


save_to_disk = 0


# Generating Mel spectrograms for training from true positive data

# In[2]:


import csv
import librosa
import numpy as np
from skimage.transform import resize
from PIL import Image

fft = 2048
hop = 512
# Less rounding errors this way
sr = 48000
length = 10 * sr

with open('/Volumes/Software/rfcx-species-audio-detection/train_tp.csv') as f:
    reader = csv.reader(f)
    data = list(reader)

# Check minimum/maximum frequencies for bird calls
# Not neccesary, but there are usually plenty of noise in low frequencies, and removing it helps
fmin = 24000
fmax = 0

# Skip header row (recording_id,species_id,songtype_id,t_min,f_min,t_max,f_max) and start from 1 instead of 0
for i in range(1, len(data)):
    if fmin > float(data[i][4]):
        fmin = float(data[i][4])
    if fmax < float(data[i][6]):
        fmax = float(data[i][6])
# Get some safety margin
fmin = int(fmin * 0.9)
fmax = int(fmax * 1.1)
print('Minimum frequency: ' + str(fmin) + ', maximum frequency: ' + str(fmax))


print('Starting spectrogram generation')
for i in range(1, len(data)):
    # All sound files are 48000 bitrate, no need to slowly resample
    wav, sr = librosa.load('/Volumes/Software/rfcx-species-audio-detection/train/' + data[i][0] + '.flac', sr=None)  # sr: sampling rate
    
    t_min = float(data[i][3]) * sr
    t_max = float(data[i][5]) * sr
    
    # Positioning sound slice
    center = np.round((t_min + t_max) / 2)
    beginning = center - length / 2    #??
    if beginning < 0:
        beginning = 0
    
    ending = beginning + length
    if ending > len(wav):
        ending = len(wav)
        beginning = ending - length

    slice = wav[int(beginning):int(ending)]

    # Mel spectrogram generation
    # Default settings were bad, parameters are adjusted to generate somewhat reasonable quality images
    # The better your images are, the better your neural net would perform
    # You can also use librosa.stft + librosa.amplitude_to_db instead
    mel_spec = librosa.feature.melspectrogram(slice, n_fft=fft, hop_length=hop, sr=sr, fmin=fmin, fmax=fmax, power=1.5)
    mel_spec = resize(mel_spec, (224, 400))

    # Normalize to 0...1 - this is what goes into neural net
    mel_spec = mel_spec - np.min(mel_spec)
    mel_spec = mel_spec / np.max(mel_spec)

# And this 0...255 is for the saving in bmp format
    mel_spec = mel_spec * 255
    mel_spec = np.round(mel_spec)
    mel_spec = mel_spec.astype('uint8')
    mel_spec = np.asarray(mel_spec)
    
    bmp = Image.fromarray(mel_spec, 'L')
    bmp.save('/Volumes/Software/rfcx-species-audio-detection/working/' + data[i][0] + '_' + data[i][1] + '_' + str(center) + '.bmp')
    
    if i % 100 == 0:
        print('Processed ' + str(i) + ' train examples from ' + str(len(data)))
