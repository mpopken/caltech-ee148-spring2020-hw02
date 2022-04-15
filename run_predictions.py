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

    output = [
        [i*stride, j*stride, i*stride+template_height, j*stride+template_width, heatmap[i, j]]
        for j in range(heatmap.shape[1])
        for i in range(heatmap.shape[0])
        if heatmap[i, j] == heatmap[max(0, i-1):min(heatmap.shape[0], i+2), max(0, j-1):min(heatmap.shape[1], j+2)].max()
    ]

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

    # Add examples of traffic lights of different sizes
    T = []
    # T.append(np.asarray(Image.open(os.path.join(data_path, 'RL-003.jpg')))[199:207, 281:289, :])
    # T.append(np.asarray(Image.open(os.path.join(data_path, 'RL-120.jpg')))[233:245, 142:154, :])
    T.append(np.asarray(Image.open(os.path.join(data_path, 'RL-011.jpg')))[72:90, 355:373, :])
    # T.append(np.asarray(Image.open(os.path.join(data_path, 'RL-010.jpg')))[17:39, 137:159, :])

    os.makedirs('data/hw02_templates', exist_ok=True)
    for i, t in enumerate(T):
        Image.fromarray(t.astype('|u1')).save(f'data/hw02_templates/t{t.shape[0]}.jpg')
        T[i] = T[i].astype(float)

    # Run matched filtering on all templates
    output = []
    for t in T:
        heatmap = compute_convolution(I.astype(float), t / np.sqrt((t ** 2).sum()))
        Image.fromarray((heatmap * 255).astype('|u1')).save('data/heatmap.jpg')
        output.extend(predict_boxes(heatmap, t.shape[0], t.shape[1]))

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0 + 1e-6)
        

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
done_tweaking = True

'''
Make predictions on the training set.
'''
preds_train = {}    
for i in range(100): #np.random.choice(len(file_names_train), size=5, replace=False): # range(len(file_names_train)):
    print(f'{i}/100')
    # read image using PIL:
    I = Image.open(os.path.join(data_path, file_names_train[i]))

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
        print(f'{i}/{len(file_names_test)}')
        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'), 'w') as f:
        json.dump(preds_test,f)
