
# Standard library imports
import os

from glob import glob

# Third party library imports
from tqdm import tqdm

# Local library imports
from .input_interpreter import convert_to_PIL
from .HOG_implementation.facial_detection import set_up, facial_detection_PIL
from .image_alignment import crop_image_from_PIL, write_to_video


def process_all_images(orig_dir, manip_dir, predictor_path, video_path, frame_rate):

    face_detections = {}
    predictor, detector = set_up(predictor_path)

    pbar = tqdm(sorted(glob(os.path.join(orig_dir, "*.heic")), key=os.path.getmtime))
    for orig_fp in pbar:
        file_name = os.path.splitext(os.path.split(orig_fp)[1])[0]
        pbar.set_description("Processing {!r}".format(file_name))
        manip_fp = os.path.join(manip_dir, file_name + ".jpeg")

        face_detect = process_image_PIL(orig_fp, manip_fp, predictor, detector)
        face_detections[file_name] = face_detect

    _ = write_to_video(manip_dir, video_path, frame_rate)

    return face_detections


def process_recent_images():
    pass


def process_image_PIL(origin_fp, manip_fp, predictor, detector):
    PIL_img = convert_to_PIL(origin_fp)

    try:
        face_dict = facial_detection_PIL(PIL_img, predictor, detector)
        file_name = os.path.split(origin_fp)
    except ValueError as e:
        raise ValueError("{!r} has detected more than 1 face".format(file_name))
    except IOError as e:
        raise ValueError("{!r} has detected no faces".format(file_name))

    res_img = crop_image_from_PIL(PIL_img, face_dict, box_size=2500)
    res_img.save(manip_fp) 

