# imports
import argparse
import pickle
import h5py
from tqdm import tqdm
import numpy as np
from decimal import Decimal
import museval

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

from utils.waveunet import Waveunet
from utils.data_helper import Ambisonic
import utils.model_helper as model_utils

np.seterr(divide='ignore', invalid='ignore')


# helper functions
def get_val_loss(val_loader, model, chan, criterion):
    valid_loss = 0.0
    for data, labels in tqdm(val_loader):
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()

        target = model(data)
        if chan is not None:
            loss = criterion(target[:, chan, :], labels[:, chan, :])
        else:
            loss = criterion(target, labels)
        valid_loss += loss.item()
    return valid_loss/len(val_loader)

def get_SDR(val_loader, model):

    total_SDR = None
    len_SDR = 0

    for data, labels in tqdm(val_loader):
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()

        outputs = model(data)

        for i in range(outputs.shape[0]):
            pred_b = outputs[i,:,:].cpu().detach().numpy() #shape (64,4,44300)
            real_b = labels[i,:,:].cpu().detach().numpy()

            target_sources = np.reshape(real_b, (-1, real_b.shape[1], real_b.shape[0]))
            pred_sources = np.reshape(pred_b, (-1, pred_b.shape[1], pred_b.shape[0]))
            SDR, ISR, SIR, SAR, _ = museval.metrics.bss_eval(target_sources, pred_sources)

            if total_SDR is None:
                total_SDR = SDR
            else:
                total_SDR = total_SDR + SDR

            len_SDR += 1

    return total_SDR / len_SDR

def get_SDR_numbers(val_dir,trans,model,args):
    results_dict = {}
    val_scene_names = h5py.File(val_dir, "r")

    for scene in val_scene_names.keys():
        valset = Ambisonic(path=val_dir, transform=trans, args=args, scene=scene)
        val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                pin_memory=True)

        total_err = get_SDR(val_loader, model)
        results_dict[scene] = total_err
        print(f'scene:{scene}, SDR:{total_err}')

    with open(args.log_dir + 'val_SDR_results.pkl', 'wb') as fp:
        pickle.dump(results_dict, fp)


def get_loss_numbers(val_loader, model, val_dir, trans):
    if args.loss == "L1":
        criterion = nn.L1Loss()
    elif args.loss == "L2":
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError("Couldn't find this loss!")

    results_dict = {}
    # Error on entire val data

    results_dict['val_all'] = {}

    total_val_err = get_val_loss(val_loader, model, None, criterion)
    print(f'Total_val_err : {Decimal(total_val_err):.2E}')
    results_dict['val_all']['chan_all'] = total_val_err

    for chan in range(4):
        chan_val_err = get_val_loss(val_loader, model, chan, criterion)
        print(f'chan_val_err {chan} : {Decimal(chan_val_err):.2E}')
        results_dict['val_all'][f'chan_{chan}'] = chan_val_err

    #Scene-wise data
    val_scene_names = h5py.File(val_dir, "r")

    for scene in val_scene_names.keys():
        valset = Ambisonic(path=val_dir, transform=trans, args=args, scene=scene)
        val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                pin_memory=True)

        results_dict[scene] = {}

        total_err = get_val_loss(val_loader, model, None, criterion)
        print(f'Total_{scene}_err : {Decimal(total_err):.2E}')
        results_dict[scene]['chan_all'] = total_err

        for chan in range(4):
            chan_err = get_val_loss(val_loader, model, chan, criterion)
            print(f'chan_{scene}_err {chan} : {Decimal(chan_err):.2E}')
            results_dict[scene][f'chan_{chan}'] = chan_err

    with open(args.log_dir + 'val_random_results.pkl', 'wb') as fp:
        pickle.dump(results_dict, fp)

def main(args):
    #initialize
    num_channels = [args.features * i for i in range(1, args.levels + 1)] if args.feature_growth == "add" else \
        [args.features * 2 ** i for i in range(0, args.levels)]

    target_outputs = int(args.output_size * args.sr)

    model = Waveunet(args.num_in_chan, num_channels, args.num_out_chan, kernel_size=args.kernel_size,
                     target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                     conv_type=args.conv_type, res=args.res)

    if args.cuda:
        model = model_utils.DataParallel(model)
        print("move model to gpu")
        model.cuda()

    print('model: ', model)
    print('parameter count: ', str(sum(p.numel() for p in model.parameters())))

    trans = transforms.Compose([transforms.ToTensor()])

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    val_dir = args.data_dir + 'val.hdf'

    valset = Ambisonic(path=val_dir, transform=trans, args=args)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            pin_memory=True)
    model.eval()

    if args.load_model is not None:
        print("Continuing training full model from checkpoint " + str(args.load_model))
        state = model_utils.load_model(model, optimizer, args.load_model, args.cuda)

    # run experiments
    get_SDR_numbers(val_dir,trans,model,args)

if __name__ == '__main__':
    ## TRAIN PARAMETERS
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
    parser.add_argument('--num_in_chan', type=int, default=32,
                        help="Number of input audio channels")
    parser.add_argument('--num_out_chan', type=int, default=4,
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

    args = parser.parse_args()

    main(args)