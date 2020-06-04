
# Standard library imports
import os

# Third party library imports
from tqdm import tqdm

# Local library imports
from heif_interpreter import convert_heif_to_PIL
from HOG_implementation.facial_detection import set_up, facial_detection_PIL
from image_alignment import crop_image_from_PIL, write_to_video


def process_all_images(orig_dir, manip_dir, predictor_path, detector_path):

    predictor, detector = set_up(predictor_path, detector_path)

    pbar = tqdm(os.listdir(orig_dir))
    for orig_fp in pbar:
        file_name = os.splitext(os.path.split(orig_fp)[1])[0]
        pbar.set_description('Processing {!r}'.format(file_name))
        manip_fp = os.path.join(manip_dir, file_name + '.jpeg')
        process_single_image_PIL(orig_fp, manip_fp, predictor, detector)

    video_path = write_to_video(manip_dir, video_name, framerate)


def process_recent_images():
    pass


def process_single_image_PIL(origin_fp, manip_fp, predictor, detector):
    PIL_img = convert_heif_to_PIL(origin_fp)
    face_dict = facial_detection_PIL(PIL_img, detector, predictor)

    res_img = crop_image_from_PIL(PIL_img, face_dict)
    res_img.save(file_path) 

