"""
Mask R-CNN
Train on the ICDAR15 dataset and implement text detection.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Qiao zhi
"""
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import csv
import cv2

ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn.config import Config
from mrcnn import model as modellib, utils

class ICDARConfig(Config):
	NAME = "ICDAR"
	IMAGES_PER_GPU = 1
	# Number of classes (including background)
	NUM_CLASSES = 1 + 1  # Background + text
	# Number of training steps per epoch
	# STEPS_PER_EPOCH = 100
	# Skip detections with < 90% confidence
	# DETECTION_MIN_CONFIDENCE = 0.9

	# Length of square anchor side in pixels
	RPN_ANCHOR_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)

class ICDARDataset(utils.Dataset):
	def load_annoataion(self, p):
		text_polys = []
		text_tags = []
		if not os.path.exists(p):
			print("can't find file")
			return np.array(text_polys, dtype=np.float32)
		with open(p, 'r') as f:
			reader = csv.reader(f)
			for line in reader:
				label = line[-1]
				# strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
				line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
				x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
				text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
				# Why just * or ### is True Maybe: ignore in training
				if label == '*' or label == '###':
					text_tags.append(True)
				else:
					text_tags.append(False)
		return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)

	def load_ICDAR(self, dataset_dir, gt_dir, subset):
		self.add_class("ICDAR", 1, "text")
		assert subset in ["train", "val"]
		dataset_dir = os.path.join(dataset_dir, subset)
		# dataset_gt_dir = os.path.join(gt_dir, subset)
		
		for img_path in os.listdir(dataset_dir):
			child_name = os.path.basename(img_path).split('.')[0]
			txt_fn = os.path.join(gt_dir, "gt_"+child_name+".txt")
			text_polys, text_tags = self.load_annoataion(txt_fn)
			image_path = os.path.join(dataset_dir, img_path)
			image = skimage.io.imread(image_path)
			height, width = image.shape[:2]

			self.add_image(
                "ICDAR",
                image_id=img_path,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                text_polys=text_polys,
                text_tags=text_tags)

	def load_mask(self, image_id):
		"""
		Return: mask & class_ids 
		since there only one class (text)
		"""
		image_info = self.image_info[image_id]
		if image_info["source"] != "ICDAR":
			return super(self.__class__, self).load_mask(image_id)

		info = self.image_info[image_id]
		mask = np.zeros([info["height"], info["width"], info["text_polys"].shape[0]], dtype=np.uint8)
		for i, p in enumerate(info["text_polys"]):
			if info["text_tags"][i] == True:
				continue # ignore the ### text instance
			X = p[:,0]
			Y = p[:,1]

			rr, cc = skimage.draw.polygon(Y, X)

			mask[rr, cc, i] = 1
		return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
		# return 255 * mask, np.ones([mask.shape[-1]], dtype=np.int32)
		

	def image_reference(self, image_id):
		"""Return the path of the image."""
		info = self.image_info[image_id]
		if info["source"] == "ICDAR":
			return info["path"]
		else:
			super(self.__class__, self).image_reference(image_id)

def detect(model, image_path):
	print("Running on {}".format(image_path))
	
	image = skimage.io.imread(image_path)
	r = model.detect([image], verbose=1)[0]
	bboxes = r["rois"]
	for box in bboxes:
		# cv2.polylines(image[:, :, :], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
		cv2.rectangle(image[:, :, :], (box[1], box[0]), (box[3], box[2]), color=(255, 255, 0), thickness=1)
	file_name = "res_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
	skimage.io.imsave(file_name, image)
	print("Saved to ", file_name)

def train(model):
	dataset_train = ICDARDataset()
	# dataset_train.load_ICDAR("../../training_samples", "../../training_samples", "train")
	dataset_train.load_ICDAR(args.dataset, args.gt, "train")
	dataset_train.prepare()

	dataset_val = ICDARDataset()
	dataset_val.load_ICDAR(args.dataset, args.gt, "val")
	dataset_val.prepare()

	print("Training network all")
	model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=30, layers='all')

"""
def check_data_loader():
	dl = ICDARDataset()
	dl.load_ICDAR("../../training_samples", "../../training_samples", "train")
	dl.prepare()

	mask, class_ids = dl.load_mask(0)

	print "mask: ", mask.shape
	for i in range(mask.shape[-1]):
		cv2.imwrite("mask_"+str(i)+".jpg", mask[:,:,i])
	print "class_ids: ", class_ids.shape
"""
if __name__ == '__main__':
	# check_data_loader()
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"
	import argparse

	# Parse command line arguments
	parser = argparse.ArgumentParser(
	    description='Train Mask R-CNN to detect balloons.')
	parser.add_argument("command",
	                    metavar="<command>",
	                    help="'train' or 'test'")
	parser.add_argument('--dataset', required=False,
						default="../../training_samples",
	                    metavar="/path/to/icdar/dataset/",
	                    help='Directory of the ICDAR dataset')
	parser.add_argument('--gt', required=False,
		                default="../../training_samples",
		                metavar="path/to/gt/file",
		                help="Directory of the icdar gt text files")
	parser.add_argument('--weights', required=True,
	                    metavar="/path/to/weights.h5",
	                    help="Path to weights .h5 file or 'coco'")
	parser.add_argument('--logs', required=False,
	                    default="checkpoints/",
	                    metavar="/path/to/logs/",
	                    help='Logs and checkpoints directory (default=logs/)')
	parser.add_argument('--image', required=False,
	                    metavar="path or URL to image",
	                    help='Image to apply the color splash effect on')
	args = parser.parse_args()

	if args.command == "train":
		assert args.dataset, "Argument --dataset is required for training"
	elif args.command == "splash":
		assert args.image or args.video, "Provide --image or --video to apply color splash"

	print("Dataset: ", args.dataset)
	print("Logs: ", args.logs)

	if args.command == "train":
		config = ICDARConfig()
	else:
		class InferenceConfig(ICDARConfig):
			# Set batch size to 1 since we'll be running inference on
			# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
			GPU_COUNT = 1
			IMAGES_PER_GPU = 1
		config = InferenceConfig()
	config.display()

	if args.command == "train":
		model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
	else:
		model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    # Select weights file to load
	if args.weights.lower() == "coco":
		weights_path = COCO_WEIGHTS_PATH
		# Download weights file
		if not os.path.exists(weights_path):
			utils.download_trained_weights(weights_path)
	elif args.weights.lower() == "last":
		# Find last trained weights
		weights_path = model.find_last()
	elif args.weights.lower() == "imagenet":
		# Start from ImageNet trained weights
		weights_path = model.get_imagenet_weights()
	else:
		weights_path = args.weights

	# Load weights
	print("Loading weights ", weights_path)
	if args.weights.lower() == "coco":
		# Exclude the last layers because they require a matching
		# number of classes
		model.load_weights(weights_path, by_name=True, exclude=[
	        "mrcnn_class_logits", "mrcnn_bbox_fc",
	        "mrcnn_bbox", "mrcnn_mask"])
	else:
		model.load_weights(weights_path, by_name=True)

	if args.command == "train":
		train(model)
	elif args.command == "test":
		detect(model, image_path=args.image)
	else:
		print("'{}' is not recognized. "
			"Use 'train' or 'splash'".format(args.command))