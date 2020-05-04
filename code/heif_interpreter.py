
# Standard library imports
import glob
import os

# Third party library imports
import pyheif

from PIL import Image
from tqdm import tqdm

# Local library imports


def convert_heif_to_bytes(heif_file_path):

  heif_file = pyheif.read_heif(heif_file_path)
  PIL_image = Image.frombytes(mode=heif_file.mode, size=heif_file.size, data=heif_file.data)

  return PIL_image


def convert_heif_to_jpeg_batch(heif_dir, manipulated_photos_dir):
  pbar = tqdm(glob.glob(os.path.join(heif_dir, '*.heic')))
  for file_path in pbar:
    file_name = os.path.split(file_path)[1]
    pbar.set_description('Converting {}'.format(file_name))

    convert_heif_to_jpeg(file_path, manipulated_photos_dir)


def convert_heif_to_jpeg(heif_file_path, manipulated_photos_dir):

  file_name = os.path.splitext(os.path.split(heif_file_path)[1])[0]
  jpeg_file_path = os.path.join(manipulated_photos_dir, file_name + '.jpeg')
  
  heif_file = pyheif.read_heif(heif_file_path)
  PIL_image = Image.frombytes(mode=heif_file.mode, size=heif_file.size, data=heif_file.data)
  PIL_image.save(jpeg_file_path, "JPEG")

  return PIL_image, jpeg_file_path


def get_exif(filename):
    image = Image.open(filename)
    image.verify()
    return image.getexif()

 
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


