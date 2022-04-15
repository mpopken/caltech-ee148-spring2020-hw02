import os
import json
import numpy as np
from matplotlib import pyplot as plt

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    
    intersection = max(min(box_1[2] - box_2[0], box_2[2] - box_1[0]), 0) * max(min(box_1[3] - box_2[1], box_2[3] - box_1[1]), 0)
    union = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1]) + (box_2[2] - box_2[0]) * (box_2[3] - box_2[1]) - intersection
    iou = intersection / union

    assert iou >= 0 and iou <= 1.0

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in preds.items():
        pred = list(filter(lambda p: p[4] > conf_thr, pred))
        gt = gts[pred_file]

        FP_ = [True] * len(pred)
        FN_ = [True] * len(gt)
        
        for i, g in enumerate(gt):
            for j, p in enumerate(pred):
                if compute_iou(p[:4], g) > iou_thr:
                    TP += 1
                    FP_[j] = False
                    FN_[i] = False
                    break

        FP += sum(FP_)
        FN += sum(FN_)

    '''
    END YOUR CODE
    '''

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = 'data/hw02_preds'
gts_path = 'data/hw02_annotations'

# load splits:
split_path = 'data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data. 
'''
with open(os.path.join(preds_path, 'preds_train.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 

plt.figure()
for iou_thr in [0.25, 0.5, 0.75]:
    confidence_thrs = np.arange(0.8, 1.0, 0.01)
    PR = np.array([
        compute_counts(preds_train, gts_train, iou_thr=iou_thr, conf_thr=conf_thr)
        for conf_thr in confidence_thrs
    ])

    P = np.nan_to_num(PR[:, 0] / (PR[:, 0] + PR[:, 1]), nan=1.0)
    R = PR[:, 0] / (PR[:, 0] + PR[:, 2])

    # Plot training set PR curves
    plt.plot(R, P, label=f'{iou_thr}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Train')
plt.legend()
plt.show()

if done_tweaking:
    plt.figure()
    for iou_thr in [0.25, 0.5, 0.75]:
        confidence_thrs = np.arange(0.8, 1.0, 0.01)
        PR = np.array([
            compute_counts(preds_test, gts_test, iou_thr=iou_thr, conf_thr=conf_thr)
            for conf_thr in confidence_thrs
        ])

        P = np.nan_to_num(PR[:, 0] / (PR[:, 0] + PR[:, 1]), nan=1.0)
        R = PR[:, 0] / (PR[:, 0] + PR[:, 2])

        # Plot training set PR curves
        plt.plot(R, P, label=f'{iou_thr}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Test')
    plt.legend()
    plt.show()
