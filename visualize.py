import os
import numpy as np
import json
from PIL import Image, ImageDraw

data_path = 'data/RedLights2011_Medium'
preds_path = 'data/hw02_preds/preds_train.json'
viz_path = 'data/hw02_viz'
os.makedirs(viz_path, exist_ok=True)

# Load preds
with open(preds_path) as f:
    bbs = json.load(f)

threshold = 0.95

for path in bbs:
    with Image.open(os.path.join(data_path, path)) as im:
        # Draw each bounding box
        draw = ImageDraw.Draw(im)
        for box in bbs[path]:
            if box[4] >= threshold:
                draw.line([
                    (box[1], box[0]),
                    (box[3], box[0])
                ], width=1)

                draw.line([
                    (box[3], box[0]),
                    (box[3], box[2])
                ], width=1)

                draw.line([
                    (box[3], box[2]),
                    (box[1], box[2])
                ], width=1)

                draw.line([
                    (box[1], box[2]),
                    (box[1], box[0])
                ], width=1)

        # Save image in preds.
        im.save(os.path.join(viz_path, path))