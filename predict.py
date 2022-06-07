import argparse
import os

import data.utils
import utils.model_helper as model_utils

from test import predict_song
from utils.waveunet import Waveunet

import torch.optim as optim
import librosa
import torch
import numpy as np
import soundfile as sf

def main(args):
    # MODEL
    num_features = [args.features*i for i in range(1, args.levels+1)] if args.feature_growth == "add" else \
                   [args.features*2**i for i in range(0, args.levels)]
    target_outputs = int(args.output_size * args.sr)
    model = Waveunet(args.num_in_chan, num_features, args.num_out_chan, kernel_size=args.kernel_size,
                     target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                     conv_type=args.conv_type, res=args.res)

    if args.cuda:
        model = model_utils.DataParallel(model)
        print("move model to gpu")
        model.cuda()

    print("Loading model from checkpoint " + str(args.load_model))
    optimizer = optim.Adam(model.parameters(), args.lr)
    state = model_utils.load_model(model, optimizer, args.load_model, args.cuda)
    print('Step', state['step'])

    preds = predict_song(args, args.input, model)
    output_folder = os.path.dirname(args.input) if args.output is None else args.output
    data.utils.write_wav(os.path.join(output_folder, os.path.basename(args.input) + "_" + "out_norm" + ".wav"), preds, args.sr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dry_run', action='store_true',
                        help='dry_run will run for one batch (default: False)')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Reload a previously trained model (whole task model)')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loader worker threads (default: 8)')
    parser.add_argument('--log_dir', type=str, default='logs/',
                        help='Folder to write logs into')
    parser.add_argument('--data_dir', type=str, default="data/",
                        help='Dataset path')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/',
                        help='Folder to write checkpoints into')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate (default: 3e-4)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size")
    parser.add_argument('--levels', type=int, default=6,
                        help="Number of DS/US blocks")
    parser.add_argument('--depth', type=int, default=1,
                        help="Number of convs per block")
    parser.add_argument('--sr', type=int, default=44100,
                        help="Sampling rate")
    parser.add_argument('--num_in_chan', type=int, default=2,
                        help="Number of input audio channels")
    parser.add_argument('--num_out_chan', type=int, default=2,
                        help="Number of output audio channels")
    parser.add_argument('--kernel_size', type=int, default=5,
                        help="Filter width of kernels. Has to be an odd number")
    parser.add_argument('--output_size', type=float, default=1.0,
                        help="Output duration (in sec)")
    parser.add_argument('--strides', type=int, default=4,
                        help="Strides in Waveunet")
    parser.add_argument('--patience', type=int, default=20,
                        help="Patience for early stopping on validation set")
    parser.add_argument('--example_freq', type=int, default=200,
                        help="Write an audio summary into Tensorboard logs every X training iterations")
    parser.add_argument('--loss', type=str, default="L2",
                        help="L1 or L2")
    parser.add_argument('--conv_type', type=str, default="gn",
                        help="Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn")
    parser.add_argument('--features', type=int, default=32,
                        help='Number of feature channels per layer')
    parser.add_argument('--res', type=str, default="fixed",
                        help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned")
    parser.add_argument('--feature_growth', type=str, default="double",
                        help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)")
    parser.add_argument('--input', type=str, default=os.path.join("audio_examples", "Cristina Vane - So Easy", "mix.mp3"),
                        help="Path to input mixture to be separated")
    parser.add_argument('--output', type=str, default=None, help="Output path (same folder as input path if not set)")
    parser.add_argument('--target', type=str, default=None, help="Target path (labels to be predicted used for anlyzing model performance)")

    args = parser.parse_args()

    main(args)
    # ## TRAIN PARAMETERS
    # parser = argparse.ArgumentParser()

    # parser.add_argument('--dry_run', action='store_true',
    #                     help='dry_run will run for one batch (default: False)')
    # parser.add_argument('--load_model', type=str, default=None,
    #                     help='Reload a previously trained model (whole task model)')
    # parser.add_argument('--cuda', action='store_true',
    #                     help='Use CUDA (default: False)')
    # parser.add_argument('--num_workers', type=int, default=8,
    #                     help='Number of data loader worker threads (default: 8)')
    # parser.add_argument('--log_dir', type=str, default='logs/',
    #                     help='Folder to write logs into')
    # parser.add_argument('--data_dir', type=str, default="data/",
    #                     help='Dataset path')
    # parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/',
    #                     help='Folder to write checkpoints into')
    # parser.add_argument('--lr', type=float, default=3e-4,
    #                     help='learning rate (default: 3e-4)')
    # parser.add_argument('--batch_size', type=int, default=32,
    #                     help="Batch size")
    # parser.add_argument('--levels', type=int, default=6,
    #                     help="Number of DS/US blocks")
    # parser.add_argument('--depth', type=int, default=1,
    #                     help="Number of convs per block")
    # parser.add_argument('--sr', type=int, default=44100,
    #                     help="Sampling rate")
    # parser.add_argument('--num_in_chan', type=int, default=32,
    #                     help="Number of input audio channels")
    # parser.add_argument('--num_out_chan', type=int, default=4,
    #                     help="Number of output audio channels")
    # parser.add_argument('--kernel_size', type=int, default=5,
    #                     help="Filter width of kernels. Has to be an odd number")
    # parser.add_argument('--output_size', type=float, default=1.0,
    #                     help="Output duration (in sec)")
    # parser.add_argument('--strides', type=int, default=4,
    #                     help="Strides in Waveunet")
    # parser.add_argument('--patience', type=int, default=20,
    #                     help="Patience for early stopping on validation set")
    # parser.add_argument('--example_freq', type=int, default=200,
    #                     help="Write an audio summary into Tensorboard logs every X training iterations")
    # parser.add_argument('--loss', type=str, default="L2",
    #                     help="L1 or L2")
    # parser.add_argument('--conv_type', type=str, default="gn",
    #                     help="Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn")
    # parser.add_argument('--features', type=int, default=32,
    #                     help='Number of feature channels per layer')
    # parser.add_argument('--res', type=str, default="fixed",
    #                     help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned")
    # parser.add_argument('--feature_growth', type=str, default="double",
    #                     help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)")

    # args = parser.parse_args()

    # main(args)
