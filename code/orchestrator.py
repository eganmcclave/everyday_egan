
# Standard library imports
import os

from glob import iglob

# Third party library imports
from tqdm import tqdm

# Local library imports
from .input_interpreter import convert_to_PIL
from .HOG_implementation.facial_detection import set_up, facial_detection_PIL, draw_facial_points
from .image_alignment import crop_image_from_PIL, write_jpegs_to_video, write_numpy_to_video
from .data_storage import store_single_hdf5


def process_all_images(orig_dir, manip_dir, predictor_path, video_path, frame_rate, 
        draw, crop, box_size=2500, prev_images=None):
    """ A orchestrator function which coordinates the processing of all images in a 
    given directory and compiles them into a single video.

    :orig_dir: A string for valid directory path to input images.
    :manip_dir: A string for a valid directory path to saved processed images.
    :predictor_path: A string for a valid path to dlib predictor object.
    :video_path: A valid path for the output video to be saved to.
    :frame_rate: An integer for the framerate of the video.
    :draw: A boolean to indicate if the images should have the detected landmarks drawn.
    :crop: A boolean to indicate if the images should be cropped.
    :box_size: An integer with default 2500 to indicate the size of the cropping box.
    :return: A dictionary of dictionary of facial detections 
    """

    # initialize the facial detection dictionary and the predictor/detector
    face_detections = {}
    predictor, detector = set_up(predictor_path)

    # set a progress bar to iterate through of the input images
    valid_images = iglob(os.path.join(orig_dir, "*.*"))
    images_pbar = tqdm(sorted(valid_images, key=os.path.getmtime))
    for orig_fp in images_pbar:
        # parse the file name for custom progress bar description
        file_name = os.path.splitext(os.path.split(orig_fp)[1])[0]
        pbar.set_description("Processing {!r}".format(file_name))
        manip_fp = os.path.join(manip_dir, file_name + ".jpeg")

        # if current input image has already been processed then skip to next iter
        if prev_images is not None and file_name in prev_images:
            continue

        # process a single image using helper function and save detected faces
        face_detect = process_image_PIL(orig_fp, manip_fp, predictor, detector, draw, crop, box_size)
        face_detections[file_name] = face_detect

    # save resulting images to video
    _ = write_to_video(manip_dir, video_path, frame_rate)

    # return facial detections for later usage
    return face_detections


def process_recent_images():
    pass


def process_image_PIL(origin_fp, manip_fp, predictor, detector, draw, crop, box_size=2500):
    """ A helper function to facilitate the processing of one image

    :origin_fp: A string to a valid input image path to be processed.
    :manip_fp: A string to saved the resulting image to.
    :predictor: A dlib predictor object to assist with facial detection.
    :detector: A dlib detector object to assist with facial detection.
    :draw: A boolean to indicate if the image should have the detected landmarks drawn.
    :crop: A boolean to indicate if the image should be cropped.
    :box_size: An integer with default 2500 to indicate the size of the cropping box.
    :return: A dictionary of facial detections.
    """

    # load the image to PIL object
    PIL_img = convert_to_PIL(origin_fp)

    try:
        # detect the facial landmarks on the given image
        face_dict = facial_detection_PIL(PIL_img, predictor, detector)
        file_name = os.path.split(origin_fp)
    except ValueError as e:
        raise ValueError("{!r} has detected more than 1 face".format(file_name))
    except IOError as e:
        raise ValueError("{!r} has detected no faces".format(file_name))

    # draw the detected landmarks
    if draw:
        draw_facial_points(PIL_img, face_dict[0]['facial_points'], width=3, radius=5)

    # crop the image based on landmarks
    if crop:
        PIL_img = crop_image_from_PIL(PIL_img, face_dict, box_size=box_size)

    # save the output image
    PIL_img.save(manip_fp) 

    return face_dict

