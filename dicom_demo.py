
import os
import numpy as np
import cv2

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
    print("after resample the slice thickness, dicom images.shape/.dtype {}/{}".format(dicom_images.shape, dicom_images.dtype))

    # third visualize the preprocessed data
    # visualize_dicom_as_video(dicom_images=dicom_images, output_fn="demo2_processed.avi")

    # do cropping, here we manually set a 3D box
    z_range = [int(len(dicom_images)*0.3), int(len(dicom_images)*0.9)]
    x_range = [int(512*0.1), int(512*0.9)]
    y_range = [int(512*0.2), int(512*0.8)]
    print("cropping range: z {} x {} y {}".format(z_range, x_range, y_range))
    cropped_images = dicom_images[z_range[0]:z_range[1], y_range[0]:y_range[1], x_range[0]:x_range[1]]
    visualize_dicom_as_video(dicom_images=cropped_images, output_fn="dicom_cropped.avi")
    print("cropped dicom images shape: {}".format(cropped_images.shape))

    # do resizing to be network processed, (307, 409) -> ()
    train_size = (416, 320)
    resized_images = []
    for img in cropped_images:
        _resized_img = cv2.resize(img, train_size, interpolation=cv2.INTER_LINEAR)
        resized_images.append(_resized_img)
    resized_images = np.stack(resized_images)
    visualize_dicom_as_video(dicom_images=resized_images, output_fn="dicom_resized.avi")
    print("resized dicom images shape: {}".format(resized_images.shape))
