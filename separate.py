import os
import librosa
import soundfile as sf
'''
sperate source files to exclude vocal and precussion
'''

# * first separate
audio_dir = 'audios'
model = '4stems'
save_dir = audio_dir+'/'+model #save the separated tracks

for file in os.listdir(os.path.join(audio_dir, 'CE200')):
    # print(os.path.exists('audios/4stems'))
    
    if not os.path.exists(os.path.join(save_dir, file[:-4])):
        print('separating ',file)
        os.system('spleeter separate -i "{File}" -o {save_dir} -p spleeter:{model}'.format(File=os.path.join(audio_dir, 'CE200', file)
                                                                                            , save_dir=save_dir, model=model))   

# merged_dir = 'audios/accompaniment'
merged_dir = audio_dir+'/harmony'
if not os.path.exists(merged_dir):
    os.makedirs(merged_dir)

# * then merge
for subdirs in os.listdir(save_dir):
    if not os.path.exists(os.path.join(merged_dir, subdirs + '_separated.wav')):
        print('working on ',subdirs)
        audio1_path = os.path.join(save_dir, subdirs, 'bass.wav')
        audio2_path = os.path.join(save_dir, subdirs, 'other.wav')
        audio3_path = os.path.join(save_dir, subdirs, 'vocals.wav')
        # audio2_path = os.path.join(save_dir, subdirs, 'piano.wav')
        

        y1, sample_rate1 = librosa.load(audio1_path)
        y2, sample_rate2 = librosa.load(audio2_path)
        y3, sample_rate3 = librosa.load(audio3_path)
        # y4, sample_rate4 = librosa.load(audio4_path)

        print('merging ', subdirs)

        sf.write(os.path.join(merged_dir, '{title}_separated.wav'.format(title=subdirs)), (y1+y2+y3)/3, int((sample_rate1+sample_rate2+sample_rate3)/3))
        