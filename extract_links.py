""" 
download wav files from youtube
"""
import os 

dataset_dir = '../CE200'
links_list = []
for i in range(1,201):
    link_file_path = os.path.join(dataset_dir, str(i), 'yt_link.txt')
    # print(link_file)
    with open(link_file_path, 'r') as link_file:
        url = link_file.readline()
        links_list.append(url)

with open('../links.txt','w') as f:
    for link in links_list:
        f.write("%s" % link)

