import os
from torch import optim
from utils import logger
from audio_dataset import AudioDataset, AudioDataLoader
from btc_model import *
from baseline_models import CNN, CRNN, Bi_LSTM
from utils.hparams import HParams
import argparse
from utils.pytorch_utils import adjusting_learning_rate
from utils.mir_eval_modules import root_majmin_score_calculation, large_voca_score_calculation_json_features
import warnings
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logger.logging_verbosity(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int, help='Experiment Number', default='1')
parser.add_argument('--kfold', type=int, help='5 fold (0,1,2,3,4)',default='4')
parser.add_argument('--voca', type=bool, help='large voca is True', default=True)
parser.add_argument('--model', type=str, help='btc, cnn, crnn, bi_lstm', default='btc')
parser.add_argument('--dataset1', type=str, help='Dataset', default='CE200')
parser.add_argument('--dataset2', type=str, help='Dataset', default='uspop')
parser.add_argument('--dataset3', type=str, help='Dataset', default='robbiewilliams')
parser.add_argument('--restore_epoch', type=int, default=1000)
parser.add_argument('--early_stop', type=bool, help='no improvement during 10 epoch -> stop', default=True)
parser.add_argument('--from_json', type=bool, help='if trained from json data', default=True)
args = parser.parse_args()
os.chdir('/media/lab812/53D8AD2D1917B29C/CE/Chord-Recognition')

config = HParams.load("run_config.yaml")
if args.voca == True:
    config.feature['large_voca'] = True
    config.model['num_chords'] = 170
if args.from_json == False:
    config.model['feature_size'] = 144
# Result save path
asset_path = config.path['asset_path']
ckpt_path = config.path['ckpt_path']
result_path = config.path['result_path']
restore_epoch = args.restore_epoch
experiment_num = str(args.index)
now = time.time()
now_tuple = time.localtime(now)
date_info = '%02d-%02d-%02d_%02d:%02d'%(now_tuple[0]%100,now_tuple[1],now_tuple[2],now_tuple[3],now_tuple[4])
ckpt_file_name = date_info +'_%03d.pth.tar'
#tf_logger = TF_Logger(os.path.join(asset_path, 'tensorboard', 'idx_'+experiment_num))
subdir = 'from_json' if args.from_json else 'from_wav'
writer = SummaryWriter(log_dir=(os.path.join(asset_path, 'tensorboard', subdir, date_info)))
logger.info("==== Experiment Number : %d " % args.index)

if args.model == 'cnn':
    config.experiment['batch_size'] = 10

# Data loader
train_dataset1 = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=(args.dataset1,), num_workers=20, from_json=args.from_json, preprocessing=False, train=True, kfold=args.kfold)
# train_dataset2 = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=(args.dataset2,), num_workers=20, preprocessing=False, train=True, kfold=args.kfold)
# train_dataset3 = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=(args.dataset3,), num_workers=20, preprocessing=False, train=True, kfold=args.kfold)
# train_dataset = train_dataset1.__add__(train_dataset2).__add__(train_dataset3)
valid_dataset1 = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=(args.dataset1,), from_json=args.from_json, preprocessing=False, train=False, kfold=args.kfold)
# valid_dataset2 = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=(args.dataset2,), preprocessing=False, train=False, kfold=args.kfold)
# valid_dataset3 = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=(args.dataset3,), preprocessing=False, train=False, kfold=args.kfold)
# valid_dataset = valid_dataset1.__add__(valid_dataset2).__add__(valid_dataset3)
train_dataloader = AudioDataLoader(dataset=train_dataset1, batch_size=config.experiment['batch_size'], drop_last=False, shuffle=True)
valid_dataloader = AudioDataLoader(dataset=valid_dataset1, batch_size=config.experiment['batch_size'], drop_last=False)

# Model and Optimizer
if args.model == 'cnn':
    model = CNN(config=config.model).to(device)
elif args.model == 'crnn':
    model = CRNN(config=config.model).to(device)
elif args.model == 'bi_lstm':
    model = Bi_LSTM(config=config.model).to(device)
elif args.model == 'btc':
    model = BTC_model(config=config.model).to(device)
else: raise NotImplementedError
optimizer = optim.Adam(model.parameters(), lr=config.experiment['learning_rate'], weight_decay=config.experiment['weight_decay'], betas=(0.9, 0.98), eps=1e-9)

# Make asset directory
if not os.path.exists(os.path.join(asset_path, ckpt_path)):
    os.makedirs(os.path.join(asset_path, ckpt_path))
    os.makedirs(os.path.join(asset_path, result_path))

# Load model
if os.path.isfile(os.path.join(asset_path, ckpt_path, ckpt_file_name % restore_epoch)):
    checkpoint = torch.load(os.path.join(asset_path, ckpt_path, ckpt_file_name % restore_epoch))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    logger.info("restore model with %d epochs" % restore_epoch)
else:
    logger.info("no checkpoint with %d epochs" % restore_epoch)
    restore_epoch = 0

# Global mean and variance calculate
mp3_config = config.mp3
feature_config = config.feature
mp3_string = "%d_%.1f_%.1f" % (mp3_config['song_hz'], mp3_config['inst_len'], mp3_config['skip_interval'])
feature_string = "_%s_%d_%d_%d_" % ('cqt', feature_config['n_bins'], feature_config['bins_per_octave'], feature_config['hop_length'])
z_path = os.path.join(config.path['root_path'], 'result', mp3_string + feature_string + 'mix_kfold_'+ str(args.kfold) +'_normalization.pt')
if os.path.exists(z_path):
    normalization = torch.load(z_path)
    mean = normalization['mean']
    std = normalization['std']
    logger.info("Global mean and std (k fold index %d) load complete" % args.kfold)
else:
    mean = 0
    square_mean = 0
    k = 0
    for data in tqdm(train_dataloader):
        # one iteration is one step
        features, input_percentages, chords, collapsed_chords, chord_lens, boundaries = data
        features = features.to(device)
        mean += torch.mean(features).item()
        square_mean += torch.mean(features.pow(2)).item()
        k += 1
    square_mean = square_mean / k
    mean = mean / k
    std = np.sqrt(square_mean - mean * mean)
    normalization = dict()
    normalization['mean'] = mean
    normalization['std'] = std
    torch.save(normalization, z_path)
    logger.info("Global mean and std (training set, k fold index %d) calculation complete" % args.kfold)

current_step = 0
best_acc = 0
before_acc = 0
early_stop_idx = 0
for epoch in range(restore_epoch, config.experiment['max_epoch']):
    # Training
    model.train()
    train_loss_list = []
    total = 0.
    correct = 0.
    second_correct = 0.
    for data in tqdm(train_dataloader):
        features, input_percentages, chords, collapsed_chords, chord_lens, boundaries = data
        features, chords = features.to(device), chords.to(device)

        features.requires_grad = True
        features = (features - mean) / std

        # forward
        features = features.squeeze(1).permute(0,2,1)
        optimizer.zero_grad()
        prediction, total_loss, weights, second = model(features, chords)

        # save accuracy and loss
        total += chords.size(0)
        correct += (prediction == chords).type_as(chords).sum()
        second_correct += (second == chords).type_as(chords).sum()
        train_loss_list.append(total_loss.item())

        # writer.add_scalar(total_loss.item())

        # optimize step
        total_loss.backward()
        optimizer.step()

        current_step += 1

    # logging loss and accuracy using tensorboard
    result = {'loss/tr': np.mean(train_loss_list), 'acc/tr': correct.item() / total, 'top2/tr': (correct.item()+second_correct.item()) / total}
    for tag, value in result.items(): writer.add_scalar(tag, value, epoch+1)
    logger.info("training loss for %d epoch: %.4f" % (epoch + 1, np.mean(train_loss_list)))
    logger.info("training accuracy for %d epoch: %.4f" % (epoch + 1, (correct.item() / total)))
    logger.info("training top2 accuracy for %d epoch: %.4f" % (epoch + 1, ((correct.item() + second_correct.item()) / total)))

    # Validation
    with torch.no_grad():
        model.eval()
        val_total = 0.
        val_correct = 0.
        val_second_correct = 0.
        validation_loss = 0
        n = 0
        for data in tqdm(valid_dataloader):
            val_features, val_input_percentages, val_chords, val_collapsed_chords, val_chord_lens, val_boundaries = data
            val_features, val_chords = val_features.to(device), val_chords.to(device)

            val_features = (val_features - mean) / std

            val_features = val_features.squeeze(1).permute(0, 2, 1)
            val_prediction, val_loss, weights, val_second = model(val_features, val_chords)

            val_total += val_chords.size(0)
            val_correct += (val_prediction == val_chords).type_as(val_chords).sum()
            val_second_correct += (val_second == val_chords).type_as(val_chords).sum()
            validation_loss += val_loss.item()

            n += 1
        # logging loss and accuracy using tensorboard
        validation_loss /= n
        result = {'loss/val': validation_loss, 'acc/val': val_correct.item() / val_total, 'top2/val': (val_correct.item()+val_second_correct.item()) / val_total}
        for tag, value in result.items(): writer.add_scalar(tag, value, epoch + 1)
        logger.info("validation loss(%d): %.4f" % (epoch + 1, validation_loss))
        logger.info("validation accuracy(%d): %.4f" % (epoch + 1, (val_correct.item() / val_total)))
        logger.info("validation top2 accuracy(%d): %.4f" % (epoch + 1, ((val_correct.item() + val_second_correct.item()) / val_total)))

        current_acc = val_correct.item() / val_total

        if best_acc < val_correct.item() / val_total:
            early_stop_idx = 0
            best_acc = val_correct.item() / val_total
            logger.info('==== best accuracy is %.4f and epoch is %d' % (best_acc, epoch + 1))
            logger.info('saving model, Epoch %d, step %d' % (epoch + 1, current_step + 1))
            model_save_folder = os.path.join(asset_path, 'model', subdir)
            if not os.path.exists(model_save_folder):
                os.makedirs(model_save_folder)
            model_save_path = os.path.join(model_save_folder, ckpt_file_name % (epoch + 1))
            state_dict = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}
            torch.save(state_dict, model_save_path)
            last_best_epoch = epoch + 1

        # save model
        elif (epoch + 1) % config.experiment['save_step'] == 0:
            logger.info('saving model, Epoch %d, step %d' % (epoch + 1, current_step + 1))
            model_save_path = os.path.join(model_save_folder, ckpt_file_name % (epoch + 1))
            state_dict = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}
            torch.save(state_dict, model_save_path)
            early_stop_idx += 1
        else:
            early_stop_idx += 1

    if (args.early_stop == True) and (early_stop_idx > 9):
        logger.info('==== early stopped and epoch is %d' % (epoch + 1))
        break
    # learning rate decay
    if before_acc > current_acc:
        adjusting_learning_rate(optimizer=optimizer, factor=0.95, min_lr=5e-6)
    before_acc = current_acc

# Load model
if os.path.isfile(os.path.join(asset_path, ckpt_path, subdir, ckpt_file_name % last_best_epoch)):
    checkpoint = torch.load(os.path.join(asset_path, ckpt_path, subdir, ckpt_file_name % last_best_epoch))
    model.load_state_dict(checkpoint['model'])
    logger.info("restore model with %d epochs" % last_best_epoch)
else:
    raise NotImplementedError

# score Validation
if args.voca == True:
    score_metrics = ['root', 'thirds', 'triads', 'sevenths', 'tetrads', 'majmin', 'mirex']
    if args.from_json:
        score_list_dict1, song_length_list1, average_score_dict1 = large_voca_score_calculation_json_features(valid_dataset=valid_dataset1, config=config, model=model, model_type=args.model, mean=mean, std=std, device=device)
    else:
        score_list_dict1, song_length_list1, average_score_dict1 = large_voca_score_calculation(valid_dataset=valid_dataset1, config=config, model=model, model_type=args.model, mean=mean, std=std, device=device)
    # score_list_dict2, song_length_list2, average_score_dict2 = large_voca_score_calculation(valid_dataset=valid_dataset2, config=config, model=model, model_type=args.model, mean=mean, std=std, device=device)
    # score_list_dict3, song_length_list3, average_score_dict3 = large_voca_score_calculation(valid_dataset=valid_dataset3, config=config, model=model, model_type=args.model, mean=mean, std=std, device=device)
    for m in score_metrics:
        # average_score = (np.sum(song_length_list1) * average_score_dict1[m] + np.sum(song_length_list2) *average_score_dict2[m] + np.sum(song_length_list3) * average_score_dict3[m]) / (np.sum(song_length_list1) + np.sum(song_length_list2) + np.sum(song_length_list3))
        logger.info('==== %s score 1 is %.4f' % (m, average_score_dict1[m]))
        # logger.info('==== %s score 2 is %.4f' % (m, average_score_dict2[m]))
        # logger.info('==== %s score 3 is %.4f' % (m, average_score_dict3[m]))
        # logger.info('==== %s mix average score is %.4f' % (m, average_score))
else:
    score_metrics = ['root', 'majmin']
    score_list_dict1, song_length_list1, average_score_dict1 = root_majmin_score_calculation(valid_dataset=valid_dataset1, config=config, model=model, model_type=args.model, mean=mean, std=std, device=device)
    # score_list_dict2, song_length_list2, average_score_dict2 = root_majmin_score_calculation(valid_dataset=valid_dataset2, config=config, model=model, model_type=args.model, mean=mean, std=std, device=device)
    # score_list_dict3, song_length_list3, average_score_dict3 = root_majmin_score_calculation(valid_dataset=valid_dataset3, config=config, model=model, model_type=args.model, mean=mean, std=std, device=device)
    for m in score_metrics:
        average_score = (np.sum(song_length_list1) * average_score_dict1[m] + np.sum(song_length_list2) *average_score_dict2[m] + np.sum(song_length_list3) * average_score_dict3[m]) / (np.sum(song_length_list1) + np.sum(song_length_list2) + np.sum(song_length_list3))
        logger.info('==== %s score 1 is %.4f' % (m, average_score_dict1[m]))
        # logger.info('==== %s score 2 is %.4f' % (m, average_score_dict2[m]))
        # logger.info('==== %s score 3 is %.4f' % (m, average_score_dict3[m]))
        # logger.info('==== %s mix average score is %.4f' % (m, average_score))
