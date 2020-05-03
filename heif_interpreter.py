
# Standard library imports
from os import path

# Third party library imports
from PIL import Image
import pyheif

# Local library imports


def convert_heif_to_bytes(heif_file_path):

  heif_file = pyheif.read_heif(heif_file_path)
  PIL_image = Image.frombytes(mode=heif_file.mode, size=heif_file.size, data=heif_file.data)

  return PIL_image


def convert_heif_to_jpeg(heif_file_path, manipulated_photos_path):

  file_name = path.splitext(path.split(heif_file_path)[1])[0]
  jpeg_file_path = path.join(manipulated_photos_path, file_name + '.jpeg')
  
  heif_file = pyheif.read_heif(heif_file_path)
  PIL_image = Image.frombytes(mode=heif_file.mode, size=heif_file.size, data=heif_file.data)
  PIL_image.save(jpeg_file_path, "JPEG")

  return PIL_image, jpeg_file_path


def get_exif(filename):
    image = Image.open(filename)
    image.verify()
    return image.getexif()

 
if __name__ == '__main__':

  # initialize variables
  manip_photo_dir = '/Users/egan/Desktop/coding/everyday_egan/manipulated_photos'
  heif_file = '/Users/egan/Desktop/coding/everyday_egan/original_photos/IMG_7807.heic'

  res = convert_heif_to_bytes(heif_file)
  # convert_heif_to_bytes(heif_file, manip_photo_dir)

  breakpoint()

