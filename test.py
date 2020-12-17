import os
import mir_eval
# import pretty_midi as pm
from utils import logger
from btc_model import *
from utils.mir_eval_modules import audio_file_to_features, idx2chord, idx2voca_chord, get_audio_paths
import argparse
import warnings
import json
import numpy as np
def get_all_json_files():
    res_list = []
    CE_path = '../dataset/CE200'
    for song_inedx in range(200):
        json_path = os.path.join(CE_path, str(song_inedx+1), "feature.json")
        # chord_path = os.path.join(CE_path, str(song_inedx+1), "shorthand_gt.txt")
        # res_list.append([json_path, chord_path])
        res_list.append(json_path)
    
    return res_list
os.chdir('/media/lab812/53D8AD2D1917B29C/CE/Chord-Recognition')

warnings.filterwarnings('ignore')
logger.logging_verbosity(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--voca', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--audio_dir', type=str, default='./test')
parser.add_argument('--save_dir', type=str, default='./test')
parser.add_argument('--from_json', default=True, type=lambda x: (str(x).lower() == 'true'))
args = parser.parse_args()

config = HParams.load("run_config.yaml")

if args.voca is True:
    config.feature['large_voca'] = True
    config.model['num_chords'] = 170
    # model_file = './test/btc_model_large_voca.pt'
    model_file = './test/{model_path}'.format(model_path='20-12-17_13:14_026.pth.tar')
    idx_to_chord = idx2voca_chord()
    logger.info("label type: large voca")
else:
    model_file = './test/btc_model.pt'
    idx_to_chord = idx2chord
    logger.info("label type: Major and minor")

model = BTC_model(config=config.model).to(device)
kfold_used = 4
# Load model
if os.path.isfile(model_file):
    mp3_config = config.mp3
    feature_config = config.feature
    mp3_string = "%d_%.1f_%.1f" % (mp3_config['song_hz'], mp3_config['inst_len'], mp3_config['skip_interval'])
    feature_string = "_%s_%d_%d_%d_" % ('cqt', feature_config['n_bins'], feature_config['bins_per_octave'], feature_config['hop_length'])
    z_path = os.path.join(config.path['root_path'], 'result', mp3_string + feature_string + 'mix_kfold_'+ str(kfold_used) +'_normalization.pt')
    normalization = torch.load(z_path)
    mean = normalization['mean']
    std = normalization['std']
    # mean = checkpoint['mean']
    # std = checkpoint['std']
    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint['model'])
    logger.info("restore model")

# Audio files with format of wav and mp3
audio_paths = get_audio_paths(args.audio_dir)
json_paths = get_all_json_files()

if args.from_json is True:
        # Chord recognition and save lab file
    for i, path in enumerate(json_paths):
    # for i, audio_path in enumerate(audio_paths):
        logger.info("======== %d of %d in progress ========" % (i + 1, len(json_paths)))
        # Load mp3

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        save_path = os.path.join(args.save_dir, '{:0>3d}.lab'.format(int(path.split('/')[-2])))

        # if already processed
        if os.path.exists(save_path):
            logger.info("label file %s already saved !" % save_path)
            continue
        else:
            feature_per_second = config.mp3['inst_len'] / config.model['timestep']
            with open(path, 'r') as json_file:
                features = json.load(json_file)

            chroma_stft = np.array(features['chroma_stft'])
            chroma_cqt = np.array(features['chroma_cqt'])
            chroma_cens = np.array(features['chroma_cens'])
            rms = np.array(features['rms'])
            spectral_centroid = np.array(features['spectral_centroid'])
            spectral_bandwidth = np.array(features['spectral_bandwidth'])
            spectral_contrast = np.array(features['spectral_contrast'])
            spectral_flatness = np.array(features['spectral_flatness'])
            spectral_rolloff = np.array(features['spectral_rolloff'])
            poly_features = np.array(features['poly_features'])
            tonnetz = np.array(features['tonnetz'])
            zero_crossing_rate = np.array(features['zero_crossing_rate'])

            feature = np.concatenate((chroma_stft, chroma_cqt, chroma_cens,
                            rms, spectral_centroid, spectral_bandwidth,
                            spectral_contrast, spectral_flatness,
                            spectral_rolloff, poly_features, tonnetz, 
                            zero_crossing_rate)).reshape(-1, rms.shape[1])

            song_length_second = feature.shape[1]*config.feature['hop_length']/config.mp3['song_hz']

            # feature, feature_per_second, song_length_second = audio_file_to_features(path, config)
            logger.info("audio file loaded and feature computation success : %s" % path)

            # Majmin type chord recognition
            feature = feature.T
            feature = (feature - mean) / std
            time_unit = feature_per_second
            n_timestep = config.model['timestep']

            num_pad = n_timestep - (feature.shape[0] % n_timestep)
            feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
            num_instance = feature.shape[0] // n_timestep

            start_time = 0.0
            lines = []
            with torch.no_grad():
                model.eval()
                feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
                for t in range(num_instance):
                    self_attn_output, _ = model.self_attn_layers(feature[:, n_timestep * t:n_timestep * (t + 1), :])
                    prediction, _ = model.output_layer(self_attn_output)
                    prediction = prediction.squeeze()
                    for i in range(n_timestep):
                        if t == 0 and i == 0:
                            prev_chord = prediction[i].item()
                            continue
                        if prediction[i].item() != prev_chord:
                            lines.append(
                                '%.3f %.3f %s\n' % (start_time, time_unit * (n_timestep * t + i), idx_to_chord[prev_chord]))
                            start_time = time_unit * (n_timestep * t + i)
                            prev_chord = prediction[i].item()
                        if t == num_instance - 1 and i + num_pad == n_timestep:
                            if start_time != time_unit * (n_timestep * t + i):
                                lines.append('%.3f %.3f %s\n' % (start_time, time_unit * (n_timestep * t + i), idx_to_chord[prev_chord]))
                            break

            # lab file write
            
            with open(save_path, 'w') as f:
                for line in lines:
                    f.write(line)

            logger.info("label file saved : %s" % save_path)
        
            # lab file to midi file
            

            # starts, ends, pitchs = list(), list(), list()

            # intervals, chords = mir_eval.io.load_labeled_intervals(save_path)
            # for p in range(12):
            #     for i, (interval, chord) in enumerate(zip(intervals, chords)):
            #         root_num, relative_bitmap, _ = mir_eval.chord.encode(chord)
            #         tmp_label = mir_eval.chord.rotate_bitmap_to_root(relative_bitmap, root_num)[p]
            #         if i == 0:
            #             start_time = interval[0]
            #             label = tmp_label
            #             continue
            #         if tmp_label != label:
            #             if label == 1.0:
            #                 starts.append(start_time), ends.append(interval[0]), pitchs.append(p + 48)
            #             start_time = interval[0]
            #             label = tmp_label
            #         if i == (len(intervals) - 1): 
            #             if label == 1.0:
            #                 starts.append(start_time), ends.append(interval[1]), pitchs.append(p + 48)

            # midi = pm.PrettyMIDI()
            # instrument = pm.Instrument(program=0)

            # for start, end, pitch in zip(starts, ends, pitchs):
            #     pm_note = pm.Note(velocity=120, pitch=pitch, start=start, end=end)
            #     instrument.notes.append(pm_note)

            # midi.instruments.append(instrument)
            # midi.write(save_path.replace('.lab', '.midi'))    


else:
    # Chord recognition and save lab file
    for i, audio_path in enumerate(audio_paths):
        logger.info("======== %d of %d in progress ========" % (i + 1, len(audio_paths)))
        # Load mp3

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        save_path = os.path.join(args.save_dir, os.path.split(audio_path)[-1].replace('.mp3', '').replace('.wav', '') + '.lab')

        # if already processed
        if os.path.exists(save_path):
            logger.info("label file %s already saved !" % save_path)
            continue
        else:

            feature, feature_per_second, song_length_second = audio_file_to_features(audio_path, config)
            logger.info("audio file loaded and feature computation success : %s" % audio_path)

            # Majmin type chord recognition
            feature = feature.T
            feature = (feature - mean) / std
            time_unit = feature_per_second
            n_timestep = config.model['timestep']

            num_pad = n_timestep - (feature.shape[0] % n_timestep)
            feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
            num_instance = feature.shape[0] // n_timestep

            start_time = 0.0
            lines = []
            with torch.no_grad():
                model.eval()
                feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
                for t in range(num_instance):
                    self_attn_output, _ = model.self_attn_layers(feature[:, n_timestep * t:n_timestep * (t + 1), :])
                    prediction, _ = model.output_layer(self_attn_output)
                    prediction = prediction.squeeze()
                    for i in range(n_timestep):
                        if t == 0 and i == 0:
                            prev_chord = prediction[i].item()
                            continue
                        if prediction[i].item() != prev_chord:
                            lines.append(
                                '%.3f %.3f %s\n' % (start_time, time_unit * (n_timestep * t + i), idx_to_chord[prev_chord]))
                            start_time = time_unit * (n_timestep * t + i)
                            prev_chord = prediction[i].item()
                        if t == num_instance - 1 and i + num_pad == n_timestep:
                            if start_time != time_unit * (n_timestep * t + i):
                                lines.append('%.3f %.3f %s\n' % (start_time, time_unit * (n_timestep * t + i), idx_to_chord[prev_chord]))
                            break

            # lab file write
            
            with open(save_path, 'w') as f:
                for line in lines:
                    f.write(line)

            logger.info("label file saved : %s" % save_path)
        
            # lab file to midi file
            

            # starts, ends, pitchs = list(), list(), list()

            # intervals, chords = mir_eval.io.load_labeled_intervals(save_path)
            # for p in range(12):
            #     for i, (interval, chord) in enumerate(zip(intervals, chords)):
            #         root_num, relative_bitmap, _ = mir_eval.chord.encode(chord)
            #         tmp_label = mir_eval.chord.rotate_bitmap_to_root(relative_bitmap, root_num)[p]
            #         if i == 0:
            #             start_time = interval[0]
            #             label = tmp_label
            #             continue
            #         if tmp_label != label:
            #             if label == 1.0:
            #                 starts.append(start_time), ends.append(interval[0]), pitchs.append(p + 48)
            #             start_time = interval[0]
            #             label = tmp_label
            #         if i == (len(intervals) - 1): 
            #             if label == 1.0:
            #                 starts.append(start_time), ends.append(interval[1]), pitchs.append(p + 48)

            # midi = pm.PrettyMIDI()
            # instrument = pm.Instrument(program=0)

            # for start, end, pitch in zip(starts, ends, pitchs):
            #     pm_note = pm.Note(velocity=120, pitch=pitch, start=start, end=end)
            #     instrument.notes.append(pm_note)

            # midi.instruments.append(instrument)
            # midi.write(save_path.replace('.lab', '.midi'))    

