
# Standard library imports
import os

# Third party library imports 
import h5py

# Local library imports


def store_single_hdf5(dataset_path, np_img, attr_dict=None):
    """ """

    # Interact with HDF5 file with context manager
    with h5py.File("face_data.h5", 'a') as f:

        # Create a dataset in the file if it does not exist in the file
        if dataset_path not in f:
            dset = f.create_dataset(
                dataset_path, np.shape(np_img), h5py.h5t.STD_U8BE, data=np_img
            )

        # Iterate through attribute dictionary and assign any new info
        for key, value in attr_dict.items():
            if key not in dset.attrs:
                dset.attrs[key] = value


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

