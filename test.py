import museval
from tqdm import tqdm

import numpy as np
import torch

import data.utils
import utils.model_helper as model_utils
import utils

# Added to test SDR analysis
import h5py
import librosa

def compute_model_output(model, inputs):
    '''
    Computes outputs of model with given inputs. Does NOT allow propagating gradients! See compute_loss for training.
    Procedure depends on whether we have one model for each source or not
    :param model: Model to train with
    :param compute_grad: Whether to compute gradients
    :return: Model outputs, Average loss over batch
    '''
    return model(inputs)

def predict(audio, model):
    '''
    Predict sources for a given audio input signal, with a given model. Audio is split into chunks to make predictions on each chunk before they are concatenated.
    :param audio: Audio input tensor, either Pytorch tensor or numpy array
    :param model: Pytorch model
    :return: Source predictions
    '''
    if isinstance(audio, torch.Tensor):
        is_cuda = audio.is_cuda()
        audio = audio.detach().cpu().numpy()
        return_mode = "pytorch"
    else:
        return_mode = "numpy"

    output = None
    expected_outputs = audio.shape[1]
    # Pad input if it is not divisible in length by the frame shift number
    output_shift = model.shapes["output_frames"]
    pad_back = audio.shape[1] % output_shift
    pad_back = 0 if pad_back == 0 else output_shift - pad_back
    if pad_back > 0:
        audio = np.pad(audio, [(0,0), (0, pad_back)], mode="constant", constant_values=0.0)
    target_outputs = audio.shape[1]
    outputs = np.zeros(audio.shape, np.float32)

    # Pad mixture across time at beginning and end so that neural network can make prediction at the beginning and end of signal
    pad_front_context = model.shapes["output_start_frame"]
    pad_back_context  = model.shapes["input_frames"] - model.shapes["output_end_frame"]
    framed_audio_a = librosa.util.frame(audio, frame_length=model.shapes["output_frames"], hop_length=model.shapes["output_frames"], axis=1)#53929, 53929, axis=1)
    framed_audio_a = np.transpose(framed_audio_a, (1, 0, 2))
    
    for frame_a in framed_audio_a:
        frame_a = np.pad(frame_a, [(0,0), (pad_front_context, pad_back_context)], mode="constant", constant_values=0.0)
        frame_a = frame_a[np.newaxis, :]
        frame_a = frame_a / (np.max(np.abs(frame_a)) + 1e-5) # TODO: delete once training data is unormalized
        outputs = model(torch.FloatTensor(frame_a))
        x = outputs.detach().cpu().numpy()
        x = np.squeeze(x)
        if output is None:
            output = x
        else:
            output = np.append(output, x, axis=1)
    return output #, total_SDR / len_SDR

def predict_song(args, audio_path, model):
    '''
    Predicts sources for an audio file for which the file path is given, using a given model.
    Takes care of resampling the input audio to the models sampling rate and resampling predictions back to input sampling rate.
    :param args: Options dictionary
    :param audio_path: Path to mixture audio file
    :param model: Pytorch model
    :return: Source estimates given as dictionary with keys as source names
    '''
    model.eval()
    print(f"Generating inference from file: {audio_path}")
    # Load mixture in original sampling rate
    if audio_path.endswith('.wav'):
        mix_audio,_ = data.utils.load(audio_path, sr=args.sr, mono=False)
    elif audio_path.endswith('.hdf'):
        hdf_data = h5py.File(audio_path, "r")
        scene = list(hdf_data.keys())[0]
        mix_audio = hdf_data[scene]['A']
    else:
        print("Error: file needs to be .wav or .hdf format")

    mix_channels = mix_audio.shape[0]
    mix_len = mix_audio.shape[1]
    # Adapt mixture channels to required input channels
    assert(mix_channels == args.num_in_chan)
    mix_len = mix_audio.shape[1]

    sources = predict(mix_audio, model)
    if args.target is not None: # Under development
        target_data = h5py.File(args.target, "r") #"/scratch/binaural_data/train30sec.hdf"
        scene = list(target_data.keys())[0]
        target_scene = target_data[scene]['B']
        total_err, min_error, max_error, mean_abs, min_abs, max_abs = get_performance_metrics(target_scene, sources, model)
        print(f'scene:{scene}, SDR:{total_err}, min_SDR:{min_error}, max_SDR:{max_error}, mean_abs:{mean_abs}, min_abs:{min_abs}, max_abs:{max_abs}')
    return sources


def get_performance_metrics(target_scene, pred_scene, model):
    total_SDR, min_SDR, max_SDR = None, None, None
    total_abs, min_abs, max_abs = None, None, None
    len_SDR = 0
    pad_front_context = model.shapes["output_start_frame"]
    pad_back_context = model.shapes["input_frames"] - model.shapes["output_end_frame"]
    target_framed = librosa.util.frame(target_scene, frame_length=model.shapes["output_frames"], hop_length=model.shapes["output_frames"], axis=1)
    target_framed = np.transpose(target_framed, (1, 0, 2))
    pred_framed   = librosa.util.frame(pred_scene, frame_length=model.shapes["output_frames"], hop_length=model.shapes["output_frames"], axis=1)
    pred_framed   = np.transpose(pred_framed, (1, 0, 2))
    
    for pred_frame, target_frame in zip(pred_framed, target_framed):
        target_frame = target_frame / (np.max(np.abs(target_frame)) + 1e-5) # TODO: delete once training data is unormalized
        target_frame = np.transpose(target_frame[np.newaxis, :, :], (0, 2, 1))
        pred_frame = np.transpose(pred_frame[np.newaxis, :, :], (0, 2, 1))
        SDR, ISR, SIR, SAR, _ = museval.metrics.bss_eval(target_frame, pred_frame)
        abs_diff_ch1 = np.mean(np.abs(target_frame[:,:,0] - pred_frame[:,:,0]), axis=1)[0]
        abs_diff_ch2 = np.mean(np.abs(target_frame[:,:,1] - pred_frame[:,:,1]), axis=1)[0]
        if min_SDR is None or max_SDR is None or total_SDR is None:
            min_SDR, max_SDR, total_SDR = SDR, SDR, SDR
        else:
            min_SDR = SDR if SDR < min_SDR else min_SDR
            max_SDR = SDR if SDR > max_SDR else max_SDR
            total_SDR = total_SDR + SDR
        if min_abs is None:
            min_abs = [abs_diff_ch1, abs_diff_ch2]
            max_abs = [abs_diff_ch1, abs_diff_ch2]
            total_abs = [abs_diff_ch1, abs_diff_ch2]
        else:
            min_abs = [abs_diff_ch1, abs_diff_ch2] if abs_diff_ch1 < min_abs[0] or abs_diff_ch2 < min_abs[1] else min_abs
            max_abs = [abs_diff_ch1, abs_diff_ch2] if abs_diff_ch1 > max_abs[0] or abs_diff_ch2 > max_abs[1] else max_abs
            total_abs = list( np.array(total_abs) + np.array([abs_diff_ch1, abs_diff_ch2]))
        len_SDR += 1

    return total_SDR / len_SDR, min_SDR, max_SDR, list(np.array(total_abs) / len_SDR), min_abs, max_abs

def evaluate(args, dataset, model, instruments):
    '''
    Evaluates a given model on a given dataset
    :param args: Options dict
    :param dataset: Dataset object
    :param model: Pytorch model
    :param instruments: List of source names
    :return: Performance metric dictionary, list with each element describing one dataset sample's results
    '''
    perfs = list()
    model.eval()
    with torch.no_grad():
        for example in dataset:
            print("Evaluating " + example["mix"])

            # Load source references in their original sr and channel number
            target_sources = np.stack([data.utils.load(example[instrument], sr=None, mono=False)[0].T for instrument in instruments])

            # Predict using mixture
            pred_sources = predict_song(args, example["mix"], model)
            pred_sources = np.stack([pred_sources[key].T for key in instruments])

            # Evaluate
            SDR, ISR, SIR, SAR, _ = museval.metrics.bss_eval(target_sources, pred_sources)
            song = {}
            for idx, name in enumerate(instruments):
                song[name] = {"SDR" : SDR[idx], "ISR" : ISR[idx], "SIR" : SIR[idx], "SAR" : SAR[idx]}
            perfs.append(song)

    return perfs


def validate(args, model, criterion, test_data):
    '''
    Iterate with a given model over a given test dataset and compute the desired loss
    :param args: Options dictionary
    :param model: Pytorch model
    :param criterion: Loss function to use (similar to Pytorch criterions)
    :param test_data: Test dataset (Pytorch dataset)
    :return:
    '''
    # PREPARE DATA
    dataloader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers)

    # VALIDATE
    model.eval()
    total_loss = 0.
    with tqdm(total=len(test_data) // args.batch_size) as pbar, torch.no_grad():
        for example_num, (x, targets) in enumerate(dataloader):
            if args.cuda:
                x = x.cuda()
                for k in list(targets.keys()):
                    targets[k] = targets[k].cuda()

            _, avg_loss = model_utils.compute_loss(model, x, targets, criterion)

            total_loss += (1. / float(example_num + 1)) * (avg_loss - total_loss)

            pbar.set_description("Current loss: {:.4f}".format(total_loss))
            pbar.update(1)

    return total_loss
