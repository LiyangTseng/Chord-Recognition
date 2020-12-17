import os
import csv
import mir_eval
def get_score(gt_path, est_path):
    '''
        evalutate the results comparing to ground truth
    '''
    (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(gt_path)
    (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(est_path)
    est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(),
                                                                ref_intervals.max(), mir_eval.chord.NO_CHORD,
                                                                mir_eval.chord.NO_CHORD)
    
    (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels,
                                                                                est_intervals, est_labels)
    durations = mir_eval.util.intervals_to_durations(intervals)
    comparisons = mir_eval.chord.sevenths(ref_labels, est_labels)
    score = mir_eval.chord.weighted_accuracy(comparisons, durations)
    return score

if __name__ == '__main__':
    # * use model to estimate chord
    audios_dir = '../audio_dataset/CE200'
    if not os.path.exists(audios_dir):
        os.makedirs(audios_dir)
    save_dir = os.path.join('predictions','CE200_from_audios')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    gt_dir = '../dataset/CE200'
    
    # os.system('python test.py --voca True --audio_dir ./audios/CE200 --save_dir ./predictions/CE200')
    # os.system('python test.py --voca True --audio_dir {audios_dir} --save_dir {save_dir}'.format(audios_dir=audios_dir, save_dir=save_dir))

    # * get ground truth and prediciton 
    
    results = []
    for file in os.listdir(save_dir):
        ground_truth_path = os.path.join(gt_dir, str(int(file[0:3])), 'shorthand_gt.txt')
        prediction_path = os.path.join(save_dir, file)

                        # * id, accuracy, title
        results.append([file[0:3], get_score(ground_truth_path, prediction_path), file[4:]])

    results.sort()
    # * write results to file
    filename = 'CE200_from_audios'
    with open('score_{file}.csv'.format(file=filename), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'accuracy', 'title'])
        for result in results:        
            writer.writerow(result)        