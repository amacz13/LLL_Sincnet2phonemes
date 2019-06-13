"""
Created on Thursday, April 4 2019

@author: Life Long Learning Team
"""

import shutil
import os
import soundfile as sf
import numpy as np
import sys
import torch
# import sidekit
import readH5
import wave


def save_wav_channel(fn, wav, channel):

    '''
    Take Wave_read object as an input and save one of its
    channels into a separate .wav file.
    '''

    # Read data
    nch   = wav.getnchannels()
    depth = wav.getsampwidth()
    wav.setpos(0)
    sdata = wav.readframes(wav.getnframes())

    # Extract channel data (24-bit data not supported)
    typ = { 1: np.uint8, 2: np.uint16, 4: np.uint32 }.get(depth)
    if not typ:
        raise ValueError("sample width {} not supported".format(depth))
    if channel >= nch:
        raise ValueError("cannot extract channel {} out of {}".format(channel+1, nch))
    print ("Extracting channel {} out of {} channels, {}-bit depth".format(channel+1, nch, depth*8))
    data = np.fromstring(sdata, dtype=typ)
    ch_data = data[channel::nch]

    # Save channel to a separate file
    outwav = wave.open(fn, 'w')
    outwav.setparams(wav.getparams())
    outwav.setnchannels(1)
    outwav.writeframes(ch_data.tostring())
    outwav.close()


def ReadList(list_file):
    f=open(list_file,'r')
    lines=f.readlines()
    print("Found : ",len(lines))
    list_sig=[]
    for x in lines:
        print(x)
        list_sig.append(x.rstrip())
        f.close()
    return list_sig


def copy_folder(in_folder, out_folder):
    if not(os.path.isdir(out_folder)):
        shutil.copytree(in_folder, out_folder, ignore=ig_f)


def ig_f(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]


# Define paths and list file location
in_folder = os.path.join(os.path.dirname(__file__), '../LLL_DB/wavs')
out_folder = os.path.join(os.path.dirname(__file__), '../WAVSOUT')
h5path = os.path.join(os.path.dirname(__file__), '../LLL_DB/h5')
configFile = open("../config.scp".upper(), 'w')

# Replicate input folder structure to output folder
copy_folder(in_folder.upper(),out_folder.upper())

# Separating the 2 channels of the wav files in 2 different files & creating config file

list_sig=[]
for root, dirs, files in os.walk("../LLL_DB/wavs"):
    for filename in files:
        print("Using file : "+filename)
        configFile.write(filename.split(".")[0] + '_a.wav\n')
        configFile.write(filename.split(".")[0] + '_b.wav\n')
        wav = wave.open("../LLL_DB/wavs/"+filename)
        save_wav_channel("../WAVSOUT/"+(filename.split(".")[0] + '_a.wav').upper(), wav, 0)
        save_wav_channel("../WAVSOUT/"+(filename.split(".")[0] + '_b.wav').upper(), wav, 1)
        list_sig.append((filename.split(".")[0] + '_a.wav').upper())
        list_sig.append((filename.split(".")[0] + '_b.wav').upper())

configFile.close()


configFile = os.path.join(os.path.dirname(__file__),"../config.scp".upper())

# Read List file
list_sig = ReadList(configFile)


# Speech Data Reverberation Loop
for i in range(len(list_sig)):
    print("INLOOP")
    # Open the wav file
    wav_file=out_folder+'/'+list_sig[i]
    print(wav_file)
    [signal, fs] = sf.read(wav_file.upper())
    signal=signal.astype(np.float64)

    # Signal normalization
    signal=signal/np.abs(np.max(signal))

    fileNameWOExt = list_sig[i].split('.')[0]
    print(h5path+"/"+fileNameWOExt.lower()+".h5")
    vad = readH5.read_vad(h5path+"/"+fileNameWOExt.lower()+".h5")

    print(len(signal), " / ", len(vad))

    new_signal = np.empty((len(signal)))
    print(signal.size, " / ", new_signal.size)

    j = 0
    for l in range(0, len(vad)):
        if vad[l] == 1:
            '''
            while j % 80 != 0:
                new_signal[j] = signal[l]
                j += 1
            '''
            m = l*80
            print("Copying Signal, j=", j, " m=", m)
            new_signal[j:j+80] = signal[m:m+80]
            print("Signal copied, j=", j)
            j += 80

    sf.write("NEW/"+list_sig[i],new_signal,fs)

    '''
    # Read wrd file
    wrd_file=wav_file.replace(".wav",".wrd")
    wrd_sig=ReadList(wrd_file.upper())
    beg_sig=int(wrd_sig[0].split(' ')[0])
    end_sig=int(wrd_sig[-1].split(' ')[1])
    
    # Remove silences
    signal=signal[beg_sig:end_sig]
    

    # Save normalized speech
    file_out=out_folder+'/'+list_sig[i]

    sf.write(file_out.upper(), signal, fs)
 
    print("Done %s" % file_out)
    '''
    
# Create h5 file
'''
options=read_conf()
batch_size=int(options.batch_size)
data_folder=options.data_folder+'/'
wav_lst_tr=ReadList(list_file)
print(wav_lst_tr)
snt_tr=len(wav_lst_tr)
wlen=int(fs*float(options.cw_len)/1000.00)
cw_len=int(options.cw_len)
lab_dict=np.load(options.lab_dict).item()
class_dict_file=options.lab_dict


create_batches_rnd(batch_size,data_folder,wav_lst_tr,snt_tr,wlen,lab_dict,0.2)
'''
