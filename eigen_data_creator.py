from micarraylib.datasets import eigenscape_loader
import os
from tqdm import tqdm
import pickle
from random import randrange
import gc
from glob import glob
import h5py
import numpy as np

def create_full_hdf_data(type='train'):
    FS = 44100
    A_LEN = 53929
    B_LEN = 44377
    
    eigen_raw_path = "/scratch/sk8974/experiments/data/eigen_raw_data/"
    eigen_path = "/scratch/sk8974/experiments/data/eigen_data/"
    save_path = '/scratch/sk8974/experiments/data/temp/'
    split_file = '/scratch/sk8974/experiments/data/processed/train_val_test.pkl'

    data = pickle.load(open(split_file, 'rb'))
    eigen_raw = eigenscape_loader.eigenscape_raw(download=False, data_home=eigen_raw_path)
    eigen = eigenscape_loader.eigenscape(download=False, data_home=eigen_path)

    with h5py.File(f'{save_path}{type}.hdf',"w") as f:
        for clip in tqdm(data[type]):
            grp = f.create_group(clip['B'])
            b_np = eigen.get_audio_numpy(clip['B'],fmt="B",fs=FS)
            a_np = eigen_raw.get_audio_numpy(clip['A'],fmt="A",fs=FS)
    
            grp.create_dataset("A", shape=a_np.shape, dtype=a_np.dtype, data=a_np)
            grp.create_dataset("B", shape=b_np.shape, dtype=b_np.dtype, data=b_np)
            
            len_a = a_np.shape[1]
            
            assert(len_a == b_np.shape[1])
            if(a_np.shape[0]!=32):
                continue
            
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
