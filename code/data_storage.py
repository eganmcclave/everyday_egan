
# Standard library imports
import os

# Third party library imports 
import h5py

# Local library imports


def store_single_hdf5(image, image_id, label, hdf5_dir="./"):
    # Interact with HDF5 file with context manager
    with h5py.File(os.path.join(hdf5_dir, f"{image_id}.h5"), 'a') as f:
        # Create a dataset in the file
        dataset = f.create_dataset(
            "image", np.shape(image), h5py.h5t.STD_U8BE, data=image
        )
        meta_set = f.create_dataset(
            "meta", np.shape(label), h5py.h5t.STD_U8BE, data=label
        )


if __name__ == "__main__":
    # Standard library imports
    from configparser import ConfigParser

    # Third party library imports
    import PIL
    import numpy as np

    # Local library imports
    from input_interpreter import convert_to_PIL


    config = ConfigParser()
    config.read('../config.ini')

    ORIGINAL_DIR = config['Paths']['original_dir']
    original_fp = os.path.join(ORIGINAL_DIR, os.listdir(ORIGINAL_DIR)[0])
    print(original_fp)
    PIL_img = convert_to_PIL(original_fp)

    breakpoint()

