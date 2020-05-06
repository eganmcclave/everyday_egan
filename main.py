
# Standard library imports
import json
import os

from datetime import date
from configparser import ConfigParser

# Third party library imports
import argparse 

# Local library imports
from code.heif_interpreter import *
from code.HOG_implementation.facial_detection import *
from code.image_alignment import jpeg_crop_images, write_to_video


# set up command line argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--process", required=False,
  help="A path to a directory of .heic files",
  default=None)
ap.add_argument("-f", "--face-dict", required=False, 
  help="A path to a json object for an existing face dictionary", 
  default=None)
ap.add_argument("-s", "--save-bool", required=False,
  help="A boolean for saving the facial landmarks detected",
  action="store_true")
ap.add_argument("-d", "--draw_bool", required=False,
  help="A boolean for drawing detected faces on images",
  action="store_true")
ap.add_argument("-c", "--crop-bool", required=False, 
  help="A boolean for croping images",
  action="store_true")
ap.add_argument("-v", "--video-bool", required=False, 
  help="A boolean for updating the video",
  action="store_true")
ap.add_argument("-V", "--video-name", required=False, 
  help="A string for the name of the video",
  default='video_{}'.format(date.today()))
ap.add_argument("-F", "--framerate", required=False,
  help="An integer detailing the number of frames in a video",
  default=10)
ap.add_argument("-o", "--output-dir", required=False,
  help="A string for the file directory to output video",
  default="./output")
args = vars(ap.parse_args())

# set up configuration parser and read config file
config = ConfigParser()
config.read('config.ini')

# initialize command line arguments for usage
if args['process'] is not None and os.path.exists(args['process']):
  heif_photo_dir = args['process']

if args['face_dict'] is not None and os.path.exists(args['face_dict']):
  with open(args['face_dict']) as f:
    faces_dict = json.load(f)
else:
  faces_dict = None

save_bool = args['save_bool']
draw_bool = args['draw_bool']
crop_bool = args['crop_bool']
video_bool = args['video_bool']
framerate = args['framerate']
video_name = os.path.join(args['output_dir'], str(framerate), args['video_name'])

# initialize configuration arguments for usage
predictor_path = config['Paths']['HOG_predictor_path']
manip_photo_dir = config['Paths']['manipulated_dir']

if heif_photo_dir is not None:
  print("CONVERTING HEIC TO JPEG")
  convert_heif_to_jpeg_batch(heif_photo_dir, manip_photo_dir)  

if faces_dict is None:
  print("FACIAL DETECTION")
  # perform facial detection with HOG based filtering
  faces_dict = jpeg_facial_detection(predictor_path, manip_photo_dir, draw_bool=draw_bool, save_bool=save_bool)

if crop_bool and faces_dict is not None:
  print("IMAGE CROPPING")
  # crop jpeg images in given folder based on facial detection
  jpeg_crop_images(manip_photo_dir, faces_dict)
elif crop_bool and faces_dict is None:
  raise Warning("cropping did not occur because the `face_dict` object was not initialized")

if video_bool:
  print("VIDEO COMPILING")
  # compile images from given directory into a video
  write_to_video(manip_photo_dir, video_name=video_name, framerate=framerate)


