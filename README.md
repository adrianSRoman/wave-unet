# Binaural Wave-Unet
This repository contains work in progress to implement a neural network
capable of generating binaural audio from stereo input.

Source code and model inpired from: [Wave-U-Net](https://github.com/f90/Wave-U-Net-Pytorch)

## Organization

## Setup and requirements

## Execution

### Generating predictions from a steoreo input file

Using stereo <.hdf/.wav> data as input:
```
python predict.py --load_model checkpoints_orig/checkpoint_445440 --cuda --batch_size 4 --output_size 1 --sr 44100 --input /path/to/<stereo_input><.hdf/.wav> --output /scratch/binaural_data/test/
```

### Generating predictions from a steoreo input file + get model performance metrics (SDR, MAE)

Using stereo <.hdf/.wav> data as input:
```
python predict.py --load_model checkpoints_orig/checkpoint_445440 --cuda --batch_size 4 --output_size 1 --sr 44100 --input /path/to/<stereo_input><.hdf/.wav> --target /path/to/<binaural_label>.hdf --output /scratch/binaural_data/test/
```
