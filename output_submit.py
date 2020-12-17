import os
import json
"""output the submit format for AI Cup 2020
"""
output_dict = {}

predictions_dir = 'CE200_from_audios'
os.chdir(os.path.join('predictions', predictions_dir))
for file in os.listdir('.'):
    with open(file, 'r') as predict_file:
        predictions = [elem.split() for elem in predict_file.read().splitlines()]
    for prediction in predictions:
        prediction[0] = float(prediction[0])
        prediction[1] = float(prediction[1])
    # print(predictions)
    output_dict[int(file[:3])] = predictions

with open('../../result.json', 'w') as output_file:
    json.dump(output_dict, output_file)
