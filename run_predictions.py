import os
import numpy as np
import json
import scipy as sp
from scipy import ndimage
from PIL import Image

def compute_convolution(I, T, stride=1):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''

    assert(I.dtype == np.dtype(float))
    assert(T.dtype == np.dtype(float))

    return sum([
        (I[i:I.shape[0] - T.shape[0] + i + 1:stride, j:I.shape[1] - T.shape[1] + j + 1:stride, :] * T[i, j, :]).sum(axis=2)
        for j in range(T.shape[1])
        for i in range(T.shape[0])
    ]) / np.sqrt(sum([
        (I[i:I.shape[0] - T.shape[0] + i + 1:stride, j:I.shape[1] - T.shape[1] + j + 1:stride, :] ** 2).sum(axis=2)
        for j in range(T.shape[1])
        for i in range(T.shape[0])
    ]))

def predict_boxes(heatmap, template_height, template_width, stride=1, threshold=0.95):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []
    
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            if heatmap[i, j] >= threshold and \
               heatmap[i, j] == heatmap[max(0, i-1):min(heatmap.shape[0], i+2), max(0, j-1):min(heatmap.shape[1], j+2)].max():

                output.append([i*stride, j*stride, i*stride+template_height, j*stride+template_width, heatmap[i, j]])

    return output


def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''

    # Add three examples of traffic lights of different sizes
    T = []
    T.append(np.asarray(Image.open(os.path.join(data_path, 'RL-003.jpg')), dtype=float)[199:207, 281:289, :].astype(float))
    T.append(np.asarray(Image.open(os.path.join(data_path, 'RL-001.jpg')), dtype=float)[181:193, 68:80, :].astype(float))
    T.append(np.asarray(Image.open(os.path.join(data_path, 'RL-011.jpg')), dtype=float)[72:90, 355:373, :].astype(float))
    T.append(np.asarray(Image.open(os.path.join(data_path, 'RL-006.jpg')), dtype=float)[0:20, 381:401, :].astype(float))
    T.append(np.asarray(Image.open(os.path.join(data_path, 'RL-010.jpg')), dtype=float)[17:39, 137:159, :].astype(float))

    # Run matched filtering on all templates
    output = []
    for i in range(len(T)):
        heatmap = compute_convolution(I.astype(float), T[i] / np.sqrt((T[i] ** 2).sum()))
        output.extend(predict_boxes(heatmap, T[i].shape[0], T[i].shape[1]))

    # Merge boxes
    done = False
    while not done:
        done = True
        for i in range(len(output) - 1):
            for j in range(i + 1, len(output)):
                # Check if bounding boxes overlap
                if ((output[j][0] <= output[i][0] and output[i][0] <= output[j][2]) or (output[i][0] <= output[j][0] and output[j][0] <= output[i][2])) and \
                   ((output[j][1] <= output[i][1] and output[i][1] <= output[j][3]) or (output[i][1] <= output[j][1] and output[j][1] <= output[i][3])):

                    # Keep box with higher score and start over.
                    if output[i][4] >= output[j][4]:
                        del output[j]
                    else:
                        del output[i]

                    done = False
                    break
            if not done:
                break

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0 - 1e-6) and (output[i][4] <= 1.0 + 1e-6)
        

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = 'data/RedLights2011_Medium'

# load splits: 
split_path = 'data/hw02_splits'
file_names_train = np.load(os.path.join(split_path, 'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path, 'file_names_test.npy'))

# set a path for saving predictions:
preds_path = 'data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Make predictions on the training set.
'''
preds_train = {}
for i in np.random.choice(len(file_names_train), size=5, replace=False): # range(len(file_names_train)):
    print(file_names_train[i])
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path, 'preds_train.json'), 'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
