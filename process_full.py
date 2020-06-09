
# standard library imports
import argparse
import json
import os

from datetime import date
from configparser import ConfigParser

# local library imports
from code.orchestrator import process_all_images


# set up command line argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save", required=False, action="store_true",
    help="A boolean for saving the detected facial landmarks")
ap.add_argument("-d", "--draw", required=False, action="store_true",
    help="A boolean for drawing detected faces on images")
ap.add_argument("-c", "--crop", required=False, action="store_true",
    help="A boolean for croping images")
ap.add_argument("-n", "--name", required=False, 
    default="video_{}".format(date.today()),
    help="A string for the name of the video")
ap.add_argument("-r", "--rate", required=False, default=10,
    help="An integer detailing the number of frames in a video")
ap.add_argument("-o", "--output", required=False, default="./videos",
    help="A string for the file directory to output video")
args = vars(ap.parse_args())

# set up configuration parser and read config file
config = ConfigParser()
config.read("config.ini")

# initialize command line arguments and configuration variables
SAVE = args['save']
DRAW = args['draw']
CROP = args['crop']
VIDEO_PATH = os.path.join(args['output'], str(args['rate']), args['name'] + '.mp4')
FRAME_RATE = args['rate']

ORIGINAL_DIR = config['Paths']['original_dir']
MANIPULATED_DIR = config['Paths']['manipulated_dir']
PREDICTOR_PATH = config['Paths']['HOG_predictor_path']

# process all images in the input directory
face_detections = process_all_images(ORIGINAL_DIR, MANIPULATED_DIR, PREDICTOR_PATH, 
        VIDEO_PATH, FRAME_RATE, DRAW, CROP)

# if prompted then save facial detections for later usage
if SAVE:
    with open('face_data.json', 'w') as fp:
        json.dump(face_detections, fp)

