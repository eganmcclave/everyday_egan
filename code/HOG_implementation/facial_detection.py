
# Standard library imports
import pickle 
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


def jpeg_facial_detection(predictor_path, jpeg_faces_path, draw_bool=False, save_bool=False, 
    width=5, radius=3):

  predictor = dlib.shape_predictor(predictor_path)
  detector = dlib.get_frontal_face_detector()
  faces_dict = {}

  pbar = tqdm(glob.glob(os.path.join(jpeg_faces_path, '*.jpeg')))
  for file_path in pbar:
    file_name = os.path.split(file_path)[1]
    pbar.set_description('Detecting faces in {}'.format(file_name))

    PIL_img = PIL.Image.open(file_path)
    img_rgb = np.array(PIL_img)
    face_dict = single_facial_detection(img_rgb, file_path, detector, predictor)
    faces_dict[file_name] = face_dict

    if draw_bool:
      PIL_img = draw_face_detection(PIL_img, face_dict, width, radius)
      PIL_img.save(file_path)

  if save_bool:
    with open('face_details.json', 'w') as f:
      json.dump(faces_dict, f)

  return faces_dict

def single_facial_detection(img_rgb, file_path, detector, predictor):

  face_dict = {}
  dets = detector(img_rgb, 1)

  if len(dets) != 1:
    print("WARNING - {} is detected to have more than 1 face: {} faces!".format(file_path, len(dets)))

  for i, d in enumerate(dets):
    if d.right() - d.left() > 200:
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
    for face_dict in face_dicts.values():
      img = draw_facial_coords(img, face_dict['facial_coords'], width)
      img = draw_facial_points(img, face_dict['facial_points'], width, radius)
      img = draw_facial_coords_2(img, face_dict['facial_points'], width)

    return img 


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


def draw_facial_coords_2(img, point_coords, width, box_size=2000):
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

