
# Standard library imports
import glob
import os

# Third party library imports
import pyheif

from PIL import Image
from tqdm import tqdm


def convert_heif_to_PIL(heif_file_path):
  """ Converts a .heif/.heic image file into a PIL object

  :param heif_file_path: A string which leads to a valid .heif/.heic file.
  :return: A PIL object read in from the given image path.
  """
  
  # wrap to catch file errors
  try:
    # process .heif file if it exists
    heif_file = pyheif.read_heif(heif_file_path)
    PIL_image = Image.frombytes(mode=heif_file.mode, size=heif_file.size, data=heif_file.data)

    return PIL_image
  except IOError as err:
    print(err)


def convert_heif_to_jpeg_batch(heif_dir, manipulated_photos_dir):
  """ Converts a directory of .heif/.heic images into .jpeg files 

  :param heif_dir: A string of a valid directory with .heif/.heic images
  :param manipulated_photos_dir: A string of a valid directory to save
  the .jpeg files into.
  :return: None
  """

  # grabs all .heic files from the provided directory and wraps with tqdm
  pbar = tqdm(glob.glob(os.path.join(heif_dir, '*.heic')))
  for file_path in pbar:
    # writing custom progress bar description for visulization purposes
    file_name = os.path.split(file_path)[1]
    pbar.set_description('Converting {}'.format(file_name))

    # utilizes helper function to parse a given image
    convert_heif_to_jpeg(file_path, manipulated_photos_dir)


def convert_heif_to_jpeg(heif_file_path, manipulated_photos_dir):
  """ Converts a .heif/.heic image file into a PIL object and saves a .jpeg file

  :param heif_file_path: A string which leads to a valid .heif/.heic file.
  :param manipulated_photos_dir: A string of a valid directory to save
  the .jpeg files into.
  :return: A PIL object read in from the given image path and the file path to
  the saved .jpeg file.
  """

  # wrap to catch file errors
  try:
    # calculates appropriate file name and extension
    file_name = os.path.splitext(os.path.split(heif_file_path)[1])[0]
    jpeg_file_path = os.path.join(manipulated_photos_dir, file_name + '.jpeg')
    
    # process .heif file if it exists then save a .jpeg copy
    heif_file = pyheif.read_heif(heif_file_path)
    PIL_image = Image.frombytes(mode=heif_file.mode, size=heif_file.size, data=heif_file.data)
    PIL_image.save(jpeg_file_path, "JPEG")

    return PIL_image, jpeg_file_path
  except IOError as err:
    print(err)


def get_exif(jpeg_file_path):
  """ Extracts meta-data from a .jpeg image

  :param jpeg_file_path: A string which leads to a valid .jpeg file.
  :return: A dictionary containing the .jpeg image metadata.
  """

  # wrap to catch file errors
  try:
    # extract data from .jpeg file if it exists
    image = Image.open(jpeg_file_path)
    image.verify()

    return image.getexif()
  except IOError as err:
    print(err)

 
if __name__ == '__main__':

  # Standard library imports
  from configparser import ConfigParser

  # set up configration and initialize variables
  config = ConfigParser()
  config.read('../config.ini')

  heif_photo_dir = config['Paths']['original_dir']
  manip_photo_dir = config['Paths']['manipulated_dir']
  # res_1 = convert_heif_to_bytes(heif_file)
  # res_2 = convert_heif_to_jpeg(heif_file, manip_photo_dir)

  convert_heif_to_jpeg_batch(heif_photo_dir, manip_photo_dir)


