
# Standard library imports
import os

# Third party library imports
from tqdm import tqdm

# Local library imports
from heif_interpreter import convert_heif_to_bytes
from HOG_implementation.facial_detection import single_facial_detection
from image_alignment import crop_image_from_file, write_to_video


def process_all_images(orig_dir, manip_dir, predictor_path, detector_path):

    for orig_file_path in os.listdir(orig_dir):
        process_single_image(orig_file_path)

    video_path = write_to_video(manip_dir, video_name, framerate)


def process_recent_images():
    pass


def process_single_image(file_path):
    pass

