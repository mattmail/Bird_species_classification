import skimage.io as io
import numpy as np
import sys
import os


ROOT_DIR = os.path.abspath("")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
from samples.coco import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)



def crop_from_mrcnn(direction):
    
    for x in os.listdir(direction):
        if (x != '.DS_Store'):
            for label in os.listdir(direction+'/'+x):
                if(label != '.DS_Store'):
                    for image_name in os.listdir(direction+'/'+x+'/'+label):
                        if(image_name != '.DS_Store'):
                            image = io.imread(direction+'/'+x+'/'+label+'/'+image_name)
                            n = image.shape[0]
                            p = image.shape[1]
                            if (len(image.shape) == 3):
                              results = model.detect([image])
                              r = results[0]
                              #birds id is 15
                              windows = r['rois'][r['class_ids'] == 15]
                              if(len(windows) > 1):
                                    selected_window = np.argmax(r['scores'][r['class_ids']==15])
                                    x0, y0, x1, y1 = windows[selected_window]
                                    # we want to have crops that are square because the input of our nn is a quare
                                    if (x1-x0 > y1-y0):
                                        new_y1 = np.min((int((y1+y0)/2) + int((x1-x0)/2), p))
                                        new_y0 = np.max((int((y1+y0)/2) - int((x1-x0)/2), 0))
                                        new_x1 = x1
                                        new_x0 = x0
                                    else:
                                        new_x1 = np.min((int((x1+x0)/2) + int((y1-y0)/2), n))
                                        new_x0 = np.max((int((x1+x0)/2) - int((y1-y0)/2), 0))
                                        new_y1 = y1
                                        new_y0 = y0


                                    image_crop = image[new_x0:new_x1,new_y0:new_y1]
                                    io.imsave(direction+'_cropped/'+x+'/'+label+'/'+image_name, image_crop)
                                    
                              elif (len(windows == 1)):
                                  x0, y0, x1, y1 = windows[0]
                                  # we want to have crops that are square because the input of our nn is a quare
                                  if (x1-x0 > y1-y0):
                                      new_y1 = np.min((int((y1+y0)/2) + int((x1-x0)/2), p))
                                      new_y0 = np.max((int((y1+y0)/2) - int((x1-x0)/2), 0))
                                      new_x1 = x1
                                      new_x0 = x0
                                  else:
                                      new_x1 = np.min((int((x1+x0)/2) + int((y1-y0)/2), n))
                                      new_x0 = np.max((int((x1+x0)/2) - int((y1-y0)/2), 0))
                                      new_y1 = y1
                                      new_y0 = y0


                                  image_crop = image[new_x0:new_x1,new_y0:new_y1]
                                  io.imsave(direction+'_cropped/'+x+'/'+label+'/'+image_name, image_crop)
                              else:
                                  io.imsave(direction+'_cropped/'+x+'/'+label+'/'+image_name, image)
                            else:
                                io.imsave(direction+'_cropped/'+x+'/'+label+'/'+image_name, image)