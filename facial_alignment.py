
# Standard library imports
import os

# Third party library imports
import face_alignment

# Local library imports


# TO BE CONTINUED (need system that can use CUDA gpu)
def FAN_implementation(byte_input):
  # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', verbose=True)
  fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', verbose=True)
  preds = fa.get_landmark_from_image(byte_input)
  print(preds)
  return preds


if __name__ == '__main__':
  # Local library imports
  from heif_interpreter import *

  heif_file_path = '/Users/egan/Desktop/coding/everyday_egan/original_photos/IMG_7807.heic'
  byte_input = convert_heif_to_bytes(heif_file_path)

