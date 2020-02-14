"""
Mask R-CNN
Train on the toy pantograph dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 pantograph.py train --dataset=/path/to/pantograph/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 pantograph.py train --dataset=/path/to/pantograph/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 pantograph.py train --dataset=/path/to/pantograph/dataset --weights=imagenet

    # Apply color splash to an image
    python3 pantograph.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 pantograph.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Add root to path 
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import model as modellib, utils



tools = os.path.abspath("../../")
tools = tools+'/coco-master/PythonAPI'
if tools not in sys.path:
    sys.path.append(tools)
    
# for i in sys.path:
#     print(i)

from cocotools.coco import COCO
from cocotools.cocoeval import COCOeval
from cocotools import mask as maskUtils


# # Directory to save logs and trained model
# MODEL_DIR = os.path.join(ROOT_DIR, "model")

# # Path to trained weights file
# COCO_WEIGHTS_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")

# # Directory to save logs and model checkpoints, if not provided
# # through the command line argument --logs
# DEFAULT_LOGS_DIR = os.path.join(MODEL_DIR, "logs")

############################################################
#  Configurations
############################################################


class PantographConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "pantograph"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background + pantograph

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 2

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    NUM_KEYPOINTS = 6
    MASK_SHAPE = [28, 28]
    KEYPOINT_MASK_SHAPE = [56,56]
    DETECTION_MAX_INSTANCES = 50
    TRAIN_ROIS_PER_IMAGE = 100
    MAX_GT_INSTANCES = 128
    RPN_TRAIN_ANCHORS_PER_IMAGE = 150
    USE_MINI_MASK = True
    MASK_POOL_SIZE = 14
    KEYPOINT_MASK_POOL_SIZE = 7
    LEARNING_RATE = 0.002
    WEIGHT_LOSS = True
    KEYPOINT_THRESHOLD = 0.005

    IMAGE_PADDING = True # False is not supported?
    IMAGE_RESIZE_MODE = "square" # Options = none,square,pad64,crop is not working,
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_SCALE = 0.5 # Not used yet

    RPN_ANCHOR_STRIDE = 2

    # PART_STR = ['L1','L2','L3','R3','R2','R1']
    # LIMBS = [0,1,1,2,2,3,3,4,4,5]

# Why is this here?
Person_ID = 1


############################################################
#  Dataset
############################################################

class PantographDataset(utils.Dataset):
    def __init__(self):

        self._skeleton = []
        self._keypoint_names = []
        super().__init__()
    
    def load_pantograph(self, dataset_dir, subset, class_ids=None,
                  class_map=None, return_pantograph=False):
        """Load a subset of the pantograph dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        
        # Set path to annotations and open
        ANNO_FILE = os.path.join(dataset_dir, "sample_region_data.json")
        pantograph = COCO(ANNO_FILE)

        # Set path to images
        image_dir = dataset_dir

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(pantograph.getCatIds())
            # print(class_ids)

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(pantograph.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(pantograph.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("pantograph", i, pantograph.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "pantograph", image_id=i,
                path=os.path.join(image_dir, pantograph.imgs[i]['file_name']),
                width=pantograph.imgs[i]["width"],
                height=pantograph.imgs[i]["height"],
                annotations=pantograph.loadAnns(pantograph.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))

        #the connection between 2 close keypoints
        # Change personID to catID
        self._skeleton = pantograph.loadCats(Person_ID)[0]["skeleton"]
        self._skeleton = np.array(self._skeleton,dtype=np.int32)

        self._keypoint_names = pantograph.loadCats(Person_ID)[0]["keypoints"]

        print("Skeleton:",np.shape(self._skeleton))
        print("Keypoint names:",np.shape(self._keypoint_names))

        if return_pantograph:
            return pantograph


    @property
    def skeleton(self):
        return self._skeleton
    @property
    def keypoint_names(self):
        return self._keypoint_names
        

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        # # If not a pantograph dataset image, delegate to parent class.
        # image_info = self.image_info[image_id]
        # if image_info["source"] != "pantograph":
        #     return super(self.__class__, self).load_mask(image_id)

        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "pantograph":
            return super(CocoDataset, self).load_mask(image_id)


        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "pantograph.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)


    def load_keypoints(self, image_id):
        """Load person keypoints for the given image.

        Returns:
        key_points: num_keypoints coordinates and visibility (x,y,v)  [num_person,num_keypoints,3] of num_person
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks, here is always equal to [num_person, 1]
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "pantograph":
            print("Error in loading keypoints")
            return super(CocoDataset, self).load_keypoints(image_id)

        keypoints = []
        class_ids = []
        instance_masks = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "pantograph.{}".format(annotation['category_id']))
            assert class_id in [1,2,3,4]
            if class_id:

                #load masks
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                #load keypoints
                keypoint = annotation["keypoints"]
                keypoint = np.reshape(keypoint,(-1,3))

                keypoints.append(keypoint)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            keypoints = np.array(keypoints,dtype=np.int32)
            class_ids = np.array(class_ids, dtype=np.int32)
            masks = np.stack(instance_masks, axis=2)
            return keypoints, masks, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_keypoints(image_id)


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "pantograph":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m




############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"], r["masks"])
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = PantographDataset()
    dataset_train.load_pantograph(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = PantographDataset()
    dataset_val.load_pantograph(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

# if __name__ == '__main__':
#     import argparse

#     # Parse command line arguments
#     parser = argparse.ArgumentParser(
#         description='Train Mask R-CNN to detect pantographs.')
#     parser.add_argument("command",
#                         metavar="<command>",
#                         help="'train' or 'splash'")
#     parser.add_argument('--dataset', required=False,
#                         metavar="/path/to/pantograph/dataset/",
#                         help='Directory of the pantograph dataset')
#     parser.add_argument('--weights', required=True,
#                         metavar="/path/to/weights.h5",
#                         help="Path to weights .h5 file or 'coco'")
#     parser.add_argument('--logs', required=False,
#                         default=DEFAULT_LOGS_DIR,
#                         metavar="/path/to/logs/",
#                         help='Logs and checkpoints directory (default=logs/)')
#     parser.add_argument('--image', required=False,
#                         metavar="path or URL to image",
#                         help='Image to apply the color splash effect on')
#     parser.add_argument('--video', required=False,
#                         metavar="path or URL to video",
#                         help='Video to apply the color splash effect on')
#     args = parser.parse_args()

#     # Validate arguments
#     if args.command == "train":
#         assert args.dataset, "Argument --dataset is required for training"
#     elif args.command == "splash":
#         assert args.image or args.video,\
#                "Provide --image or --video to apply color splash"

#     print("Weights: ", args.weights)
#     print("Dataset: ", args.dataset)
#     print("Logs: ", args.logs)

#     # Configurations
#     if args.command == "train":
#         config = PantographConfig()
#     else:
#         class InferenceConfig(PantographConfig):
#             # Set batch size to 1 since we'll be running inference on
#             # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
#             GPU_COUNT = 1
#             IMAGES_PER_GPU = 1
#         config = InferenceConfig()
#     config.display()

#     # Create model
#     if args.command == "train":
#         model = modellib.MaskRCNN(mode="training", config=config,
#                                   model_dir=args.logs)
#     else:
#         model = modellib.MaskRCNN(mode="inference", config=config,
#                                   model_dir=args.logs)

#     # Select weights file to load
#     if args.weights.lower() == "coco":
#         weights_path = COCO_WEIGHTS_PATH
#         # Download weights file
#         if not os.path.exists(weights_path):
#             utils.download_trained_weights(weights_path)
#     elif args.weights.lower() == "last":
#         # Find last trained weights
#         weights_path = model.find_last()
#     elif args.weights.lower() == "imagenet":
#         # Start from ImageNet trained weights
#         weights_path = model.get_imagenet_weights()
#     else:
#         weights_path = args.weights

#     # Load weights
#     print("Loading weights ", weights_path)
#     if args.weights.lower() == "coco":
#         # Exclude the last layers because they require a matching
#         # number of classes
#         model.load_weights(weights_path, by_name=True, exclude=[
#             "mrcnn_class_logits", "mrcnn_bbox_fc",
#             "mrcnn_bbox", "mrcnn_mask"])
#     else:
#         model.load_weights(weights_path, by_name=True)

#     # Train or evaluate
#     if args.command == "train":
#         train(model)
#     elif args.command == "splash":
#         detect_and_color_splash(model, image_path=args.image,
#                                 video_path=args.video)
#     else:
#         print("'{}' is not recognized. "
#               "Use 'train' or 'splash'".format(args.command))
