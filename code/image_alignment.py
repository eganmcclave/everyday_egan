
# Standard library imports
import glob
import os

# Third party library imports
import ffmpeg
import numpy
import glob
import PIL

from tqdm import tqdm

# Local library imports


def write_to_video(jpeg_photo_dir, video_name='video', framerate=5):
  (
    ffmpeg
    .input(os.path.join(jpeg_photo_dir, '*.jpeg'), pattern_type='glob', framerate=framerate)
    .output('{}.mp4'.format(video_name))
    .run()
  )


def jpeg_crop_images(jpeg_faces_path, faces_dict):
  pbar = tqdm(glob.glob(os.path.join(jpeg_faces_path, '*.jpeg')))
  for file_path in pbar:
    file_name = os.path.split(file_path)[1]
    pbar.set_description('Cropping {}'.format(file_name))

    img = PIL.Image.open(file_path)
    face_dict = faces_dict[file_name] 

    img = crop_image(img, face_dict)
    img.save(file_path)


def crop_image(image, face_dict, box_size=2000):
  point = face_dict[0]['facial_points'][27]
  coords = (
    point[0] - 0.50 * box_size, point[1] - 0.35 * box_size,
    point[0] + 0.50 * box_size, point[1] + 0.65 * box_size
  )

  image = image.crop(coords)
  return image


if __name__ == '__main__':
  
  # Standard library imports
  from configparser import ConfigParser

  # Third pary library imports
  from HOG_implementation.facial_detection import *

  # set up configuration and initialize variables
  config = ConfigParser()
  config.read('../config.ini')

  predictor_path = config['Paths']['HOG_predictor_path']
  manip_photo_dir = config['Paths']['manipulated_dir']
  
  faces_dict = batch_facial_detection(predictor_path, manip_photo_dir, draw_bool=False, save_bool=False)
  jpeg_crop_images(manip_photo_dir, faces_dict)
  write_to_video(manip_photo_dir)


