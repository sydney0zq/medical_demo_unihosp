import pydicom
import os
import numpy as np
import cv2
from scipy import ndimage

def imgs2vid(imgs, output_fn="test.avi", fps=5, w_index=True):
    imgs = np.asarray(imgs)
    if imgs.ndim == 3:
        _, height, width = imgs.shape
    elif imgs.ndim == 4:
        _, height, width, channels = imgs.shape
        assert channels == 3, "The number of channel (dim==3) must be 3..."
    else:
        assert False, "Invalid ndarray with the shape of {}...".format(imgs.shape)

    if imgs.ndim == 3:
        video_handler = cv2.VideoWriter(output_fn, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height), isColor=False)
    else:
        video_handler = cv2.VideoWriter(output_fn, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height), isColor=True)

    for i, img in enumerate(imgs):
        img = np.uint8(np.asarray(img))
        if w_index:
            try:
                img = cv2.putText(img, "{:03d}".format(i), (20, height-20), 
                            cv2.FONT_HERSHEY_PLAIN, 2, 255, thickness=2)
            except:
                img = cv2.putText(img.copy(), "{:03d}".format(i), (20, height-20), 
                            cv2.FONT_HERSHEY_PLAIN, 2, 255, thickness=2)
        video_handler.write(img)
    cv2.destroyAllWindows()
    video_handler.release()

def find_the_deepest_dir(root):
    listdironly = lambda d: [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]
    while True:
        ds = listdironly(root)
        if len(ds) == 0:
            ret_d = root
            break
        elif len(ds) == 1:
            root = ds[0]
        else:
            raise ValueError("The dicom diretory {} should not have more than "
                             "one directory...".format(root))
    return ret_d


def load_16bit_dicom_images(path, verbose=True):
    # 递归找到dicom真实文件最深的文件夹
    path = find_the_deepest_dir(path)
    # 去除一些无关文件，比如.DS_Store, VERSION文件等等
    dicom_file_list = sorted([s for s in os.listdir(path) if s.startswith('I')])
    print("loading `{}`, {} to {}...".format(path, dicom_file_list[0], dicom_file_list[-1]))

    slices = [pydicom.read_file(os.path.join(path, s), force=True) for s in dicom_file_list]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    #import pdb
    #pdb.set_trace()
    total_cnt = len(slices)
    slice_thickness = np.abs(slices[0].ImagePositionPatient[2]-
                             slices[1].ImagePositionPatient[2])
    # 因为导出的有时候会有重复的CT图片，因此进行去除
    ignore_cnt = 0
    if slice_thickness == 0:
        unique_slices = [slices[0]]
        start_pos = slices[0].ImagePositionPatient[2]

        for s in slices[1:]:
            if s.ImagePositionPatient[2] != start_pos:
                unique_slices.append(s)
                start_pos = s.ImagePositionPatient[2]
            else:
                ignore_cnt += 1
        slices = unique_slices
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2]-
                                 slices[1].ImagePositionPatient[2])

    if verbose:
        print ("total/ignore/reserved {}/{}/{} CTAs, slice_thickess is {}...\
               ".format(total_cnt, ignore_cnt, len(slices), slice_thickness))

    for s in slices:
        s.SliceThickness = slice_thickness

    # 将数据转换为numpy数据结构，并且划定HU窗口
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    # 设置边界外的元素为0
    image[image == -2000] = 0
    print("the origin dicom min/max is {}/{}".format(np.min(image), np.max(image)))
    # image[image <= HU_keptwin[0]] = HU_keptwin[0]
    # image[image >= HU_keptwin[1]] = HU_keptwin[1]
    # print("the filtered dicom min/max is {}/{}".format(np.min(image), np.max(image)))

    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept   # -1024
        slope = slices[slice_number].RescaleSlope           # 1
        if slice_number == 0:
            print("intercept/slope is {}/{}".format(intercept, slope))

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)

    numpySpacing = [float(slice_thickness)] + [float(x) for x in slices[0].PixelSpacing]

    # 注意这里数据处理的结果是16bit的
    numpyImage = np.array(image, dtype=np.int16)            # SxHxW
    numpySpacing = np.array(numpySpacing, dtype=np.float)
    print("the output dicom min/max is {}/{}".format(np.min(numpyImage), np.max(numpyImage)))

    return numpyImage, numpySpacing


def norm_16bit_to_8bit(dicom_16bit_imgs, HU_keptwin=None, bound_values=[0, 1]):
    # Outlier
    #mid = (0-HU_window[0])/(HU_window[1] - HU_window[0])
    #cta_image[cta_image == 0] = HU_window[0]

    dicom_16bit_imgs = (dicom_16bit_imgs-HU_keptwin[0])/(HU_keptwin[1] - HU_keptwin[0])
    dicom_16bit_imgs[dicom_16bit_imgs < 0] = bound_values[0]
    dicom_16bit_imgs[dicom_16bit_imgs >= 1] = bound_values[1]
    dicom_8bit_imgs = (dicom_16bit_imgs*255).astype('uint8')
    return dicom_8bit_imgs

def visualize_dicom_as_video(path="", output_fn="", dicom_images=None, fps=5, w_index=True):
    assert output_fn.endswith('avi'), "The output filename must be .avi suffix..."
    if dicom_images is None:
        dicom_images, spacing = load_16bit_dicom_images(path)
        min_value, max_value = np.min(dicom_images), np.max(dicom_images)
        norm_images = (dicom_images - min_value) / (max_value - min_value)
        norm_images = np.uint8(norm_images * 255)
    else:
        if dicom_images.max() > 1:
            norm_images = dicom_images
        else:
            norm_images = np.uint8(dicom_images * 255)

    if output_fn is not None:
        print ("Writing to file {}...".format(output_fn))
        imgs2vid(norm_images, output_fn)

def resample_slice_spacing(dicom_imgs, src_spacing, des_spacing):
    dicom_imgs = ndimage.interpolation.zoom(dicom_imgs, 
                                            (src_spacing/des_spacing, 1, 1),
                                            mode="nearest")
    return dicom_imgs










