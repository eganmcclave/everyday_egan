
# Standard library imports
import glob
import os

# Third party library imports
import ffmpeg
import numpy
import glob
import PIL

from tqdm import tqdm


def write_to_video(jpeg_photo_dir, video_name='video', framerate=5):
  """ Compiles videos from a directory of .jpeg files

  :param jpeg_photo_dir: A string referencing a valid directory containing 
  .jpeg images.
  :param video_name: A string that'll become the name of the .mp4 file.
  :param framerate: An integer to depict the number of image frames per second.
  :return: None
  """

  # wrap to catch file errors
  try:
    video_path = '{}.mp4'.format(video_name)

    # grab all existing .jpeg files in provided directory and compile to video
    (
      ffmpeg
      .input(os.path.join(jpeg_photo_dir, '*.jpeg'), pattern_type='glob', framerate=framerate)
      .output(video_path)
      .run()
    )
    return video_path 
  except IOError as err:
    print(err)


def jpeg_crop_images(jpeg_faces_path, faces_dict):
  """ Coordinates the cropping of images in a directory

  :param jpeg_faces_path: A valid path to a directory of .jpeg images.
  :param faces_dict: A dictionary of image name & facial landmarks.
  :return: None
  """

  # grabs all .jpeg files from the provided directory and wraps with tqdm
  pbar = tqdm(glob.glob(os.path.join(jpeg_faces_path, '*.jpeg')))
  for file_path in pbar:
    # writing custom progress bar description for visualization purposes
    file_name = os.path.split(file_path)[1]
    pbar.set_description('Cropping {}'.format(file_name))

    # wrap to catch file errors
    try:
      # read in .jpeg file and its corresponding facial landmarks
      img = PIL.Image.open(file_path)
      face_dict = faces_dict[file_name] 

      # utilize helper function to crop the given .jpeg image
      img = crop_image(img, face_dict)
      img.save(file_path)
    except IOError as err:
      print(err)


def crop_image(image, face_dict, box_size=2000):
  """ Crops a PIL object based on facial landmarks detected from the image

  :param image: A PIL object of an image.
  :param face_dict: A dictionary of facial landmarks detected from an image.
  :param box_size: An integer depicting the size of the bounding box around
  the facial landmarks.
  """

  # calculate a bounding box based on the facial landmark detected on the top
  # of the bridge of the nose
  point = face_dict[0]['facial_points'][27]
  coords = (
    point[0] - 0.50 * box_size, point[1] - 0.35 * box_size,
    point[0] + 0.50 * box_size, point[1] + 0.65 * box_size
  )

  # with the given coordinates crop the image
  image = image.crop(coords)
  return image


def crop_image_from_file(file_path, face_dict, box_size=2000):
    """ Crops an image from the given file path based on facial landmarks. This
    function also saves the cropped image to the input file path.

    :param file_path: A string to a .jpeg file
    :param face_dict: A dict containing the detected facial coordinates
    :box_size: An integer depicting the size of the cropped image

    :return: A PIL object of the input file
    """

    if '.jpeg' not in file_path:
        raise ValueError("{file_path!r} is not a valid image".format())

    # wrap to catch file errors
    try:
        file_name = os.path.split(file_path)[1]
        img = PIL.Image.open(file_path)

        x_coord, y_coord = face_dict[0]['facial_points'][27][0:1]
        coords = (
            x_coord - 0.50 * box_size, y_coord - 0.35 * box_size,
            x_coord + 0.50 * box_size, y_coord + 0.35 * box_size
        )

        img = img.crop(coords)
        img.save(file_path)

        return img
    except IOError as err:
        print(err)


def crop_image_from_PIL(PIL_img, face_dict, box_size=2000):
    # wrap to catch file errors
    try:
        x_coord, y_coord = face_dict[0]['facial_points'][27][0:1]
        coords = (
            x_coord - 0.50 * box_size, y_coord - 0.35 * box_size,
            x_coord + 0.50 * box_size, y_coord + 0.35 * box_size
        )
        PIL_img = PIL_img.crop(coords)

        return PIL_img
    except IOError as err:
        print(err)


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


