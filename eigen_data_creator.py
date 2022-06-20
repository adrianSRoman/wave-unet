import sys
# sys.path.append('micarraylib')

# from micarraylib.datasets import eigenscape_loader
import os
from tqdm import tqdm
import pickle
from random import randrange
import gc
from glob import glob
import h5py
import numpy as np
import librosa

import soundfile as sf

def get_filenames(path):
    return [f for f in listdir(path) if isfile(join(path, f))]

def dir_files(path):
    d = {'filename': os.path.basename(path)}
    if os.path.isdir(path):
        d['type'] = "directory"
        d['children'] = [dir_files(os.path.join(path,x)) for x in os.listdir(path)]
    else:
        d['type'] = "file"
    return d

def load_hdf():
    hdf_data = h5py.File("/Volumes/T7/Binaural_HDF_data/train.hdf", "r")
    for scene in hdf_data.keys():
        audio_stereo = hdf_data[scene]['A']
        sf.write(f"/Volumes/T7/test_stereo_{scene}.wav", np.transpose(audio_stereo), 44100, "PCM_24")
        audio_binaural = hdf_data[scene]['B']
        sf.write(f"/Volumes/T7/test_binaural_{scene}.wav", np.transpose(audio_binaural), 44100, "PCM_24")

def create_full_hdf_data(set_type='train'):
    FS = 44100
    A_LEN = 53929
    B_LEN = 44377
    STEREO_CH = [6, 10] # raw data ch 7 and ch 11
    DATA_DIR = "/Volumes/T7"
    # Eigenscape data paths
    eigen_raw_path = f"{DATA_DIR}/eigen_raw/stereo/"
    eigen_binaural_path = f"{DATA_DIR}/eigen_binaural/"
    # LOCATA training data paths
    LOCATA_tr_stereo_path = f"{DATA_DIR}/LOCATA_eval_stereo/stereo_dicit/"
    LOCATA_tr_binaural_path = f"{DATA_DIR}/LOCATA_eval_binaural/"
    # LOCATA validation data paths
    # Note: we are using "dev" data as validation set given the amount of data available
    LOCATA_vl_stereo_path = f"{DATA_DIR}/LOCATA_dev_stereo/stereo_dicit/"
    LOCATA_vl_binaural_path = f"{DATA_DIR}/LOCATA_dev_binaural/"
    # MARCo data paths
    MARCo_stereo_path = f"{DATA_DIR}/MARCo_stereo/"
    MARCo_binaural_path = f"{DATA_DIR}/MARCo_binaural/"

    save_path = f'{DATA_DIR}/Binaural_HDF_data/'
    split_file = f'{DATA_DIR}/Binaural_HDF_data/train_val_test.pkl'
 
    # We only use one to ensure matching files for data vs. label 
    eigen_files     = dir_files(eigen_raw_path) 
    LOCATA_tr_files = dir_files(LOCATA_tr_stereo_path)
    LOCATA_vl_files = dir_files(LOCATA_vl_stereo_path)
    MARCo_files     = dir_files(MARCo_stereo_path)
    # path_to_features, path_to_labels, path_to_files
    dir_set = [(eigen_raw_path,eigen_binaural_path,eigen_files),
               (LOCATA_tr_stereo_path,LOCATA_tr_binaural_path,LOCATA_tr_files),
               (LOCATA_vl_stereo_path,LOCATA_vl_binaural_path,LOCATA_vl_files),
               (MARCo_stereo_path,MARCo_binaural_path,MARCo_files)]

    print("######## Generating {} dataset ########".format(set_type))

    with h5py.File(f'{save_path}{set_type}.hdf',"w") as f:
        for dir_info in dir_set:
            feats_path, labels_path, d = dir_info
            if "LOCATA_dev" in feats_path and set_type == "train":
                continue
            elif "LOCATA_eval" in feats_path and set_type == "val":
                continue
            print(">>>>>>> Recording data from {}".format(feats_path))
            for file in d["children"]:
                clip_a, ftype = file["filename"], file["type"]
                if ".wav" not in clip_a or "._" in clip_a or ".DS_Store" in clip_a:
                    continue
                if "eigen" in feats_path:
                    if set_type == "train" and ("07" in clip_a or "08" in clip_a):
                        continue
                    elif set_type == "val" and ("07" not in clip_a):
                        continue
                print("Loading {}...".format(clip_a))
                clip_b = clip_a.replace("Stereo", "Binaural")
                grp = f.create_group(clip_b)
                a_np,sra = librosa.load(feats_path+clip_a, sr=FS, mono=False)
                b_np,srb = librosa.load(labels_path+clip_b, sr=FS, mono=False)
                b_np = b_np[:,:a_np.shape[1]]
                if "MARCo" in feats_path:
                    if set_type == "train":
                        tr_idx = (a_np.shape[1]//4)*3
                        a_np = a_np[:, :tr_idx]
                        b_np = b_np[:, :tr_idx]
                    elif set_type == "val":
                        vl_idx = (a_np.shape[1]//4)*3
                        a_np = a_np[:, vl_idx:]
                        b_np = b_np[:, vl_idx:]
                clipname = clip_b = clip_a.replace(".wav", "")
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
                # break # To use only one track per directory


create_full_hdf_data(set_type='train')
create_full_hdf_data(set_type='val')
# load_hdf()
