import os
import json
"""output the submit format for AI Cup 2020
"""
output_dict = {}

os.chdir('predictions/CE200')
for file in os.listdir('.'):
    with open(file, 'r') as predict_file:
        predictions = [elem.split() for elem in predict_file.read().splitlines()]
    output_dict[file[:3]] = predictions
# print(output_dict)

with open('../../result.json', 'w') as output_file:
    json.dump(output_dict, output_file)
