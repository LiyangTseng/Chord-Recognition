import os
import librosa
import soundfile as sf
'''
sperate source files to exclude vocal and precussion
'''

# * first separate
audio_dir = 'audios'
save_dir = 'audios/5stems' #save the separated tracks
# for file in os.listdir(os.path.join(audio_dir, 'CE200')):
#     print('separating ',file)
    # os.system('spleeter separate -i "{File}" -o {save_dir} -p spleeter:5stems'.format(File=os.path.join(audio_dir, 'CE200', file), save_dir=save_dir))   

merged_dir = 'audios/accompaniment'
if not os.path.exists(merged_dir):
    os.makedirs(merged_dir)
# * then merge
for subdirs in os.listdir('audios/5stems'):
    # print(subdirs)

    audio1_path = os.path.join(save_dir, subdirs, 'bass.wav')
    audio2_path = os.path.join(save_dir, subdirs, 'piano.wav')
    audio3_path = os.path.join(save_dir, subdirs, 'other.wav')

    y1, sample_rate1 = librosa.load(audio1_path)
    y2, sample_rate2 = librosa.load(audio2_path)
    y3, sample_rate3 = librosa.load(audio3_path)

    print('merging ', subdirs)

    sf.write(os.path.join(audio_dir, 'accompaniment', '{title}_separated.wav'.format(title=subdirs)), (y1+y2+y3)/3, int((sample_rate1+sample_rate2+sample_rate3)/3))