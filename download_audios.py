""" 
download wav files from youtube
"""
import os 

dataset_dir = '../CE200'
data_num = 0

for _, dirnames, _ in os.walk(dataset_dir):
    data_num += len(dirnames)

for i in range(1,data_num+1):
    
    link_file_path = os.path.join(dataset_dir, str(i), 'yt_link.txt')
    
    with open(link_file_path, 'r') as link_file:
        url = link_file.readline()

        #download using youtube-dl
        download_command = 'youtube-dl  -x --audio-format wav -o "audios/CE200/{:0>3d}_%(title)s.%(ext)s" {}'.format(i, url)
        os.system(download_command)       
