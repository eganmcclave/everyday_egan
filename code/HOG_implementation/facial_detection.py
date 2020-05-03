
# Standard library imports
import glob
import os

# Third party library imports
import numpy as np
import dlib
import PIL

from PIL import ImageDraw
from tqdm import tqdm

# Local library imports


def jpeg_facial_detection(predictor_path, jpeg_faces_path, draw_bool=False, width=5, radius=3):
  predictor = dlib.shape_predictor(predictor_path)
  detector = dlib.get_frontal_face_detector()

  pbar = tqdm(glob.glob(os.path.join(jpeg_faces_path, '*.jpeg')), desc="Process Images")
  for file_path in pbar:
    pbar.set_description('Processing {}'.format(os.path.split(file_path)[1]))

    PIL_img = PIL.Image.open(file_path)
    img_rgb = np.array(PIL_img)
    face_dict = single_facial_detection(img_rgb, file_path, detector, predictor)

    if draw_bool:
      PIL_img = draw_face_detection(PIL_img, face_dict, width, radius)
      PIL_img.show()
      input("Press Enter to continue...")

  return face_dict

def single_facial_detection(img_rgb, file_path, detector, predictor):

  face_dict = {}
  dets = detector(img_rgb, 1)

  if len(dets) != 1:
    raise ValueError("{file_path!r} detects more than 1 face".format(img_path))

  for d in dets:
    shape = predictor(img_rgb, d)
    face_dict = {
      'facial_coords': [d.left(), d.top(), d.right(), d.bottom()],
      # https://bit.ly/2Stmj31 for reference on facial landmark numbers
      'facial_points': [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]
    }

  return face_dict


def draw_face_detection(PIL_img, face_dict, width, radius):
    PIL_img = draw_facial_coords(PIL_img, face_dict['facial_coords'], width)
    PIL_img = draw_facial_points(PIL_img, face_dict['facial_points'], width, radius)

    return PIL_img 


def draw_facial_coords(img, rect_coords, width):
  draw_obj = ImageDraw.Draw(img)
  draw_obj.rectangle(rect_coords, outline='#00FF00', width=width)
  return img


def draw_facial_points(img, point_coords, width, radius):
  draw_obj = ImageDraw.Draw(img)
  for x, y in point_coords:
    draw_obj.ellipse(
        xy=[x-radius, y-radius, x+radius, y+radius], 
        fill='#00FF00', width=width
    )
  return img


if __name__ == '__main__':

  # Standard library imports
  from configparser import ConfigParser

  # set up configuration and initialize variables
  config = ConfigParser()
  config.read('../../config.ini')

  predictor_path = config['Paths']['HOG_predictor_path']
  manip_photo_dir = config['Paths']['manipulated_dir']

  jpeg_facial_detection(predictor_path, manip_photo_dir, draw_bool=True)

