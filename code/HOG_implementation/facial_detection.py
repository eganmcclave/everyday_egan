
# Standard library imports
import glob
import json
import os

# Third party library imports
import numpy as np
import dlib
import PIL

from PIL import ImageDraw
from tqdm import tqdm

# Local library imports
from ..heif_interpreter import convert_heif_to_bytes


def batch_facial_detection(predictor_path, faces_path, draw_bool=False, 
    save_bool=False, width=5, radius=3):
  """ Coordinates the facial detection for images in a directory

  :param predictor_path: A string which contains a path to the predictor object
  required for the dlib library
  :param faces_path: A string which contains the directory of images to be 
  processed
  :param draw_bool: A boolean which indicates if the facial landmarks should 
  be drawn on the image
  :param save_bool: A boolean which indicates if the facial landmark should 
  be saved as an external file
  :param width: An integer depicting the width of the lines draw for the 
  facial detection bounding box
  :param radius: An intger depicting the radius of the facial landmark points
  """

  # wrap dlib objects to catch errors
  try:
    # load in predictor and detector
    predictor = dlib.shape_predictor(predictor_path)
    detector = dlib.get_frontal_face_detector()
  except IOError as err:
    print(err)
  faces_dict = {}

  # find either .jpeg or .heic files in the given directory path
  jpeg_list = glob.glob(os.path.join(faces_path, '*.jpeg'))
  heic_list = glob.glob(os.path.join(faces_path, '*.heic'))
  pbar = tqdm(jpeg_list + heic_list)

  # iterate through the given file paths
  for file_path in pbar:
    # wrap to catch file errors
    try:
      # writing custom progress bar description for visualization purposes
      file_name = os.path.split(file_path)[1]
      pbar.set_description('Detecting faces in {}'.format(file_name))

      # utilize helper function to read in file regardless of type
      PIL_img, img_rgb = image_handler(file_path)

      # apply facial detection algorithm to the image and aggregate landmarks
      face_dict = single_facial_detection(img_rgb, file_path, detector, predictor)
      faces_dict[file_name] = face_dict

      # if indicated then draw facial landmarks
      if draw_bool:
        PIL_img = draw_face_detection(PIL_img, face_dict, width, radius)
        PIL_img.save(file_path)
    except IOError as err:
      print("{!r} had an error:".format(file_name))
      print(err)

  # if indicated then save facial landmarks
  if save_bool:
    with open('face_details.json', 'w') as f:
      json.dump(faces_dict, f)

  return faces_dict


def image_handler(file_path):
  """ Handles the different possible image types and loads them correctly

  :param file_path: A string containing a valid file path to an image
  :return: A PIL object and a numpy array containing RGB values from the 
  input image
  """

  # wrap to catch file errors
  try:
    if os.path.splitext(file_path)[1] == '.heic':
      PIL_img = convert_heif_to_bytes(file_path)
    else:
      PIL_img = PIL.Image.open(file_path)
    img_rgb = np.array(PIL_img)

    return PIL_img, img_rgb
  except IOError as err:
    print(err)


def single_facial_detection(img_rgb, file_path, detector, predictor):
  """ Applies facial detection to a numpy array and returns the coordinates

  :param img_rgb: A numpy array object containing RGB values of an image
  :param file_path: A file path to the image being detected
  :param detector: A dlib object used for detecting facial bounding box
  :param predictor: A dlib object used for predicting facial landmarks
  """

  # calculates the faces from the input image
  face_dict = {}
  dets = detector(img_rgb, 1)

  if len(dets) != 1:
    print("WARNING - {} is detected to have more than 1 face: {} faces!".format(file_path, len(dets)))

  # Iterates through the detected faces
  for i, d in enumerate(dets):
    # If the bounding box is too small then it is unlikely to be a real face
    if d.right() - d.left() > 200:
      # calculates the 68 facial landmarks and saves relevant info
      shape = predictor(img_rgb, d)
      face_dict[i] = {
        'facial_coords': [d.left(), d.top(), d.right(), d.bottom()],
        # https://bit.ly/2Stmj31 for reference on facial landmark numbers
        'facial_points': [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]
      }
  if len(dets) != 1:
    print('The passable detections are actually just {}'.format(len(face_dict.keys())))

  return face_dict


def draw_face_detection(img, face_dicts, width, radius):
  """ Orchestrates all the indivdual drawing functions

  :param img: A PIL object of the input image
  :param face_dicts: A dictionary of detected faces in input image
  :param width: An integer depicting the width of line drawn
  :param radius: An integer depicting the radius of circles drawn
  """

  # iterate through the detected faces
  for face_dict in face_dicts.values():
    img = draw_facial_coords(img, face_dict['facial_coords'], width)
    img = draw_facial_points(img, face_dict['facial_points'], width, radius)
    img = draw_facial_coords_2(img, face_dict['facial_points'], width)

  return img 


def draw_facial_coords(img, rect_coords, width):
  """ Draws the dlib detected bounding box

  :param img: A PIL object of the input image
  :param rect_coords: A list of coordinates making a bounding box
  :param width: An integer depicting the width of line drawn
  """

  draw_obj = ImageDraw.Draw(img)
  draw_obj.rectangle(rect_coords, outline='#00FF00', width=width)
  return img


def draw_facial_points(img, point_coords, width, radius):
  """ Draws the dlib detected 68 facial landmarks

  :param img: A PIL object of the input image
  :param point_coords: A list of (x,y) coordinates for points on the image
  :param width: An integer depicting the width of line drawn
  :param radius: An integer depicting the radius of circles drawn
  """
  
  # iterate through all 68 tuples and draw them on the image
  draw_obj = ImageDraw.Draw(img)
  for x, y in point_coords:
    draw_obj.ellipse(
        xy=[x-radius, y-radius, x+radius, y+radius], 
        fill='#00FF00', width=width
    )
  return img


def draw_facial_coords_2(img, point_coords, width, box_size=2000):
  """ Draws the custom bounding box based on facial landmarks

  :param img: A PIL object of the input image
  :param point_coords: A list of (x,y) coordinates for points on the image
  :param width: An integer depicting the width of line drawn
  :param box_size: An integer depicting the size of the bounding box around
  the facial landmarks.
  """

  # calculate a bounding box based on the facial landmark detected on the top
  # of the bridge of the nose
  point = point_coords[27]
  rect_coords = [
    point[0] - 0.50 * box_size, point[1] - 0.35 * box_size,
    point[0] + 0.50 * box_size, point[1] + 0.65 * box_size
  ]

  draw_obj = ImageDraw.Draw(img)
  draw_obj.rectangle(rect_coords, outline='#0000FF', width=width)
  return img


if __name__ == '__main__':

  # Standard library imports
  from configparser import ConfigParser

  # set up configuration and initialize variables
  config = ConfigParser()
  config.read('../../config.ini')

  predictor_path = config['Paths']['HOG_predictor_path']
  manip_photo_dir = config['Paths']['manipulated_dir']

  faces_dict = jpeg_facial_detection(predictor_path, manip_photo_dir, draw_bool=False)

