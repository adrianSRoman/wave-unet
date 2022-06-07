import sys
sys.path.append('micarraylib')

from micarraylib.datasets import eigenscape_loader
import os
from tqdm import tqdm
import pickle
from random import randrange
import gc
from glob import glob
import h5py
import numpy as np
import librosa

def create_full_hdf_data(type='train'):
    FS = 44100
    A_LEN = 53929
    B_LEN = 44377
    STEREO_CH = [6, 10] # raw data ch 7 and ch 11
    
    eigen_raw_path = "/Users/adrianromanguzman/Documents/dl4genaudio/Binaural-Unet/data/eigen_raw_data/"
    eigen_binaural_path = "/Users/adrianromanguzman/Documents/dl4genaudio/Binaural-Unet/data/eigen_binaural_data/"
    save_path = '/Users/adrianromanguzman/Documents/dl4genaudio/Binaural-Unet/data/temp/'
    split_file = '/Users/adrianromanguzman/Documents/dl4genaudio/Binaural-Unet/data/processed/train_val_test.pkl'

    # data = pickle.load(open(split_file, 'rb'))
    eigen_raw = eigenscape_loader.eigenscape_raw(download=False, data_home=eigen_raw_path)

    # eigen_raw_clips = ['BusyStreet-05-Raw']
    if type == 'train':
        print("Loading training data: ")
        # Manual definition of training tracks
        eigen_raw_clips = ['Park-04']
        # eigen_raw_clips = [ 'Beach-01', 'Beach-02', 'Beach-03', 'Beach-04', 'Beach-05', 'Beach-06',
        #                     'QuietStreet-01', 'QuietStreet-02', 'QuietStreet-03', 'QuietStreet-04', 'QuietStreet-05', 'QuietStreet-06',
        #                     'PedestrianZone-01', 'PedestrianZone-02', 'PedestrianZone-03', 'PedestrianZone-04', 'PedestrianZone-05', 'PedestrianZone-06', 
        #                     'BusyStreet-01', 'BusyStreet-02', 'BusyStreet-03', 'BusyStreet-04', 'BusyStreet-05', 'BusyStreet-06',
        #                     'Park-01', 'Park-02', 'Park-03', 'Park-04', 'Park-05', 'Park-06' ]
    elif type == 'val':
        print("Loading validation data: ")
        # Manual definition of validation tracks
        eigen_raw_clips = ['Park-04']#[ 'Beach-07', 'QuietStreet-07', 'PedestrianZone-07', 'BusyStreet-07', 'Park-07' ]
    # eigen = eigenscape_loader.eigenscape(download=False, data_home=eigen_path)

    with h5py.File(f'{save_path}{type}.hdf',"w") as f:
        # for clip in tqdm(data[type]):
        for clip in eigen_raw_clips:
            print(f'Loading {clip}...')
            grp = f.create_group(clip+"-Raw")
            b_np,_ = librosa.load(eigen_binaural_path+clip+"-Binaural-30-sec.wav", sr=FS, mono=False)
            print("shape audio:", b_np.shape)
            # b_np = eigen.get_audio_numpy(clip['B'],fmt="B",fs=FS)
            a_np = eigen_raw.get_audio_numpy(clip+"-Raw",fmt="A",fs=FS)
            print("shape audio:", a_np.shape)
            a_np = a_np[STEREO_CH,0:30*FS]

            grp.create_dataset("A", shape=a_np.shape, dtype=a_np.dtype, data=a_np)
            grp.create_dataset("B", shape=b_np.shape, dtype=b_np.dtype, data=b_np)
            
            len_a = a_np.shape[1]
            
            count = 0
            
            a_list = []
            b_list = []
            
            for i in range(1024):
                start = randrange(len_a - A_LEN)
                obj = {}
                pad = (A_LEN - B_LEN)//2
                if ( start+A_LEN >= len_a) or (start + pad + B_LEN >= len_a) :
                    continue

                a_list.append([start, start+A_LEN])
                b_list.append([start+pad, start+pad+B_LEN])

                count += 1
            
            grp.attrs["length"] = count
            grp.attrs["sr"] = FS
            a_arr = np.array(a_list)
            b_arr = np.array(b_list)
            
            grp.create_dataset("a_list", shape=a_arr.shape, dtype=a_arr.dtype, data=a_arr)
            grp.create_dataset("b_list", shape=b_arr.shape, dtype=b_arr.dtype, data=b_arr)
            
            del a_np, b_np
            gc.collect()            

create_full_hdf_data(type='train')
create_full_hdf_data(type='val')
