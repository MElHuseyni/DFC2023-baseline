from PIL import Image
import os
import numpy as np

# Set the path of the directory containing the TIFF files
tiff_dir = '/home/mahmud-elhussieni/Building-Detection-and-Estimation/BuildingSegmentationDataset/raw_data/GRSS_DFC2023/train/rgb'
# Set the path of the directory to save the PNG files
jpeg_dir = '/home/mahmud-elhussieni/Building-Detection-and-Estimation/BuildingSegmentationDataset/raw_data/GRSS_DFC2023/train/rgb_png/'

# Loop through all the files in the TIFF directory
for filename in os.listdir(tiff_dir):
    if filename.endswith('.tif'):
        # Open the multi-TIFF file
        im = Image.open(os.path.join(tiff_dir, filename))

        # Save each frame of the multi-TIFF as a separate PNG file
        for i in range(im.n_frames):
            im.seek(i)

            # Convert the image to a NumPy array for normalization
            im_array = np.array(im, dtype=np.float32)

            # Normalize the array to range between 0 and 255
            im_min, im_max = im_array.min(), im_array.max()
            im_norm = (im_array - im_min) / (im_max - im_min) * 255
            im_norm = im_norm.astype(np.uint8)  # Convert to uint8 type

            # Convert back to a PIL image
            im_converted = Image.fromarray(im_norm)

            # Set the filename of the PNG file
            jpeg_filename = os.path.splitext(filename)[0] + '.png'

            print('-------------------------------------')
            print(f"Saving {jpeg_filename}")

            # Save the current frame as a PNG file
            im_converted.save(os.path.join(jpeg_dir, jpeg_filename))
