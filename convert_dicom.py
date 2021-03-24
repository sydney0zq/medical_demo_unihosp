
import os
import numpy as np

from medical_lib import load_16bit_dicom_images, visualize_dicom_as_video
from medical_lib import norm_16bit_to_8bit, resample_slice_spacing

HU_keptwin = np.array([-1250, 250])
train_spacing = 5

# demo 1: load dicom data and visualize the images
if False:
    visual_dir = "visual_videos"
    os.makedirs(visual_dir, exist_ok=True)
    visualize_dicom_as_video("dicom_data/covid-1", output_fn=os.path.join(visual_dir, "covid-1.avi"))
    visualize_dicom_as_video("dicom_data/covid-2", output_fn=os.path.join(visual_dir, "covid-2.avi"))
    visualize_dicom_as_video("dicom_data/normal-1", output_fn=os.path.join(visual_dir, "normal-1.avi"))
    visualize_dicom_as_video("dicom_data/normal-2", output_fn=os.path.join(visual_dir, "normal-2.avi"))

# demo 2: normalize the images，from 16bit to 8bit
# take the sample normal-1 as an example
if True:
    dicom_images, dicom_spacing = load_16bit_dicom_images("dicom_data/covid-1")
    print("dicom images.shape/.dtype {}/{}, dicom_spacing {}".format(dicom_images.shape, dicom_images.dtype, dicom_spacing))

    # first we convert the 16 bit to 8 bit
    dicom_images = norm_16bit_to_8bit(dicom_images, HU_keptwin=HU_keptwin)
    print("after converting dtype, dicom images.shape/.dtype {}/{}, dicom_spacing {}".format(dicom_images.shape, dicom_images.dtype, dicom_spacing))
    
    # second we normalize the slice thickness
    dicom_images = resample_slice_spacing(dicom_images, src_spacing=dicom_spacing[0], des_spacing=train_spacing)
    print("after resample the slice thickness, dicom images.shape/.dtype {}/{}, dicom_spacing {}".format(dicom_images.shape, dicom_images.dtype, dicom_spacing))

    # third visualize the preprocessed data
    visualize_dicom_as_video(dicom_images=dicom_images, output_fn="demo2_processed.avi")





# numpyImage, numpySpacing = load_16bit_dicom_images("dicom_data/P3242223/")

# import pdb; pdb.set_trace()



























