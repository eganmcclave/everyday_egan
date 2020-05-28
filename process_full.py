
# Standard library imports
import argparse 
import json
import os

from datetime import date
from configparser import ConfigParser

# Third party library imports

# Local library imports
from code.HOG_implementation.facial_detection import batch_facial_detection
from code.image_alignment import jpeg_crop_images, write_to_video


def is_valid_file(file_path):
  try:
    if not os.path.exists(file_path):
      msg = "{0!r} is not a valid file path".format(file_path)
      raise argparse.ArgumentTypeError(msg)
  except IOError as err:
    print(err)

  return file_path


# set up command line argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--file-path", required=True,
  help="A path to a directory of .heic files",
  type=lambda x: is_valid_file(x))
ap.add_argument("-d", "--draw-bool", required=False,
  help="A boolean for drawing detected faces on images",
  action="store_true")
ap.add_argument("-c", "--crop-bool", required=False,
  help="A boolean for cropping images based on facial detection",
  action="store_true")
ap.add_argument("-V", "--video-name", required=False,
  help="A string for the name of the video (default: video_{}.mp4)".format(date.today()),
  default="video_{}".format(date.today()))
ap.add_argument("-R", "--framerate", required=False,
  help="An integer detailing the number of frames per second in output video",
  default=10)
ap.add_argument("-O", "--output-dir", required=False,
  help="A string for the file directory to output the video to",
  default="./output")
args = vars(ap.parse_args())

# initialize command line arguments for usage
photo_file_dir = args['file_path']
draw_bool = args['draw_bool']
crop_bool = args['crop_bool']
framerate = args['framerate']
video_name = os.path.join(args['output_dir'], str(framerate), args['video_name'])

# set up configuration parser and read existing config file
config = ConfigParser()
config.read('config.ini')

# initialize configuration arguments for usage
predictor_path = config['Paths']['HOG_predictor_path']
manip_photo_dir = config['Paths']['manipulated_dir']

# perform facial detection with HOG based filtering
faces_dict = batch_facial_detection(predictor_path, manip_photo_dir, draw_bool, save_bool)


