
import numpy as np
import os
import SimpleITK as itk
import skimage.transform as st
import pandas as pd
import os
from skimage.measure import *
import argparse
from datetime import datetime

def normalize(img):

    ind = np.where(img != 0)
    mask = np.zeros(img.shape, dtype=np.int)
    mask[ind] = 1
    minmax = [np.percentile(img[ind], 0.5), np.percentile(img[ind], 99.5)]
    img[ind] = np.clip(img[ind], minmax[0], minmax[1])
    mean = img[ind].mean()
    std = img[ind].std()
    denominator = np.reciprocal(std)
    img = (img - mean) * denominator
    return mask * img

def normalize_cut(img):

    ind = np.where(img != 0)
    stride = [ ind[2].min() , ind[2].max() + 1]
    img = img[:, :, ind[2].min() :  ind[2].max()+1]

    ind = np.where(img != 0)
    mask = np.zeros(img.shape, dtype=np.int)
    mask[ind] = 1
    minmax = [np.percentile(img[ind], 0.5), np.percentile(img[ind], 99.5)]
    img[ind] = np.clip(img[ind], minmax[0], minmax[1])
    mean = img[ind].mean()
    std = img[ind].std()
    denominator = np.reciprocal(std)
    img = (img - mean) * denominator
    return mask * img, stride

def change_class(mask):
    for i in range(3, 11, 1):
        # print(i)
        ind = np.where(mask == i)
        if len(ind[0])!= 0:
            if i %2 == 0:
                mask[ind] =2
            else:
                mask[ind] = 3
    for i in range(11, 20, 1):
        ind = np.where(mask == i)
        if len(ind[0]) != 0:
            mask[ind] = i - 7
    return mask


def back2class20(mask):
    for i in range(12, 3, -1):
        # print(i)
        mask = np.where(mask == i, i + 7, mask)
    connect_regions = label(mask,  connectivity=None, background=0)
    props1 = regionprops(connect_regions)
    probs_class = []
    for i in range(len(props1)):
        t = props1[i]['coords'].T
        loc1 = (t[0], t[1], t[2])
        probs_class.append(np.mean(mask[loc1]))
    probs_class = np.asarray(probs_class)
    for i in range(11, 20, 1):
        ind1 = np.where(probs_class == i)[0]
        ind2 = np.where(probs_class == i+1)[0]
        if len(ind2) == 0 and len(ind1) != 0:
            for j in range(len(props1)):
                if props1[j]['centroid'][1] > props1[ind1[0]]['centroid'][1]:
                        t = props1[j]['coords'].T
                        loc1 = (t[0], t[1], t[2])
                        mask[loc1] = i - 9
        elif len(ind2) == 0 and len(ind1) == 0:
            continue

        else:
            for j in range(len(props1)):
                if props1[j]['centroid'][1] > props1[ind1[0]]['centroid'][1] and props1[j]['centroid'][1] < props1[ind2[0]]['centroid'][1] and probs_class[j] != 1:
                    t = props1[j]['coords'].T
                    loc1 = (t[0], t[1], t[2])
                    mask[loc1] = i - 9

    return mask







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str,default="./train/MR", help='The path of train set image.')
    parser.add_argument('--maskpath', type=str, default="./train/Mask", help='The path of train set mask.')
    parser.add_argument('--savepath', type=str, default="./train/process3Ddata", help='The path of train set process folder to save.')
    parser.add_argument('--process2D', type=bool, default=False, help='if process 2D data, True or False')
    parser.add_argument('--withlabel', type=bool, default=False, help='if process data with label, True or False')
    parser.add_argument('--infomation', type=str, default='info.csv', help='the file name to save the imformation of data and preprocessing')
    args = parser.parse_args()
    
    def resolve_mask_path(mask_dir, image_filename):
        """Resolve mask path by trying several naming patterns.
        Order:
          1) same name: {image_filename}
          2) mask_ + same name: mask_{image_filename}
          3) mask_ + Case→case: mask_{image_filename.replace('Case','case')}
          4) Case→case (no prefix): {image_filename.replace('Case','case')}
        Returns absolute path if found, else None.
        """
        candidates = [
            os.path.join(mask_dir, image_filename),
            os.path.join(mask_dir, f"mask_{image_filename}"),
            os.path.join(mask_dir, f"mask_{image_filename.replace('Case','case')}"),
            os.path.join(mask_dir, image_filename.replace('Case','case')),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None
    maindir = os.listdir(args.filepath)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Start preprocessing. process2D={args.process2D}, withlabel={args.withlabel}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Input image dir: {args.filepath}")
    if args.withlabel:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Input mask  dir: {args.maskpath}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Save dir        : {args.savepath}")
    infor = []
    os.makedirs(os.path.join(args.savepath, 'image'), exist_ok=True)
    if args.withlabel:
        os.makedirs(os.path.join(args.savepath, 'mask'),exist_ok=True
                                              )
    allsize = []
    allstride = []
    allorishape = []
    allresolution_size = []
    for i, name1 in enumerate(maindir):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing ({i+1}/{len(maindir)}): {name1}")
        img = itk.ReadImage(os.path.join(args.filepath, name1))
        imgdata = itk.GetArrayFromImage(img)
        space = img.GetSpacing()
        direction = img.GetDirection()
        origin = img.GetOrigin()
        orishape = imgdata.shape

        if args.withlabel:
            mask_path = resolve_mask_path(args.maskpath, name1)
            if mask_path is None:
                raise FileNotFoundError(f"Mask not found for image '{name1}'. Tried: same name, 'mask_' prefix, and Case→case in directory: {args.maskpath}")
            mask = itk.ReadImage(mask_path)
            maskdata = itk.GetArrayFromImage(mask)
            maskspace = mask.GetSpacing()
            maskdirection = mask.GetDirection()
            maskorigin = mask.GetOrigin()

        ## keep same resolution
        if args.process2D:
            newresolutionxy = 0.34482759 * 2
            newresolutionz = 4.4000001
        else:
            newresolutionxy = 0.34482759
            newresolutionz = 4.4000001
        rsize = [round(imgdata.shape[0] * space[2] / newresolutionz),
                 round(imgdata.shape[1] * space[1] / newresolutionxy),
                 round(imgdata.shape[2] * space[0] / newresolutionxy)]
        allresolution_size.append(rsize)
        space = (newresolutionxy, newresolutionxy, newresolutionz)


        ## interplot 3d
        imgdata = st.resize(imgdata, output_shape=rsize, order=1, mode='constant' ,clip=False, preserve_range=True)
        if args.withlabel:
            maskdata = st.resize(maskdata, output_shape=rsize, order=0, clip=False, mode='constant', preserve_range=True,anti_aliasing=False)

        imgdata, stride = normalize_cut(imgdata)

        if args.withlabel:
            maskdata = maskdata[:,:, stride[0]: stride[1]]

        file = itk.GetImageFromArray(imgdata.astype(np.float32))
        file.SetSpacing(space)
        file.SetOrigin(origin)
        file.SetDirection(direction)
        out_img_path = os.path.join(os.path.join(args.savepath,'image'), name1)
        itk.WriteImage(file, out_img_path)
        if args.withlabel:
            mfile = itk.GetImageFromArray(maskdata.astype(np.uint8))
            mfile.SetSpacing(maskspace)
            mfile.SetOrigin(maskorigin)
            mfile.SetDirection(maskdirection)
            out_msk_path = os.path.join(os.path.join(args.savepath, 'mask'),
                                              name1)
            itk.WriteImage(mfile, out_msk_path)
        allsize.append(np.asarray(imgdata.shape))
        allstride.append(np.asarray(stride))
        allorishape.append(np.asarray(orishape))
    allsize = np.asarray(allsize)
    allstride = np.asarray(allstride)
    allorishape = np.asarray(allorishape)
    allresolution_size = np.asarray(allresolution_size)
    pd.DataFrame(data={'name': maindir, 'shape0': allsize[:, 0], 'shape1': allsize[:, 1], 'shape2': allsize[:, 2], 'stridemin':allstride[:,0],
                       'stridemax':allstride[:,1], 'orishape0': allorishape[:, 0], 'orishape1': allorishape[:, 1], 'orishape2': allorishape[:, 2],
                       'allresolution_size0': allresolution_size[:, 0], 'allresolution_size1': allresolution_size[:, 1], 'allresolution_size2': allresolution_size[:, 2],  }).to_csv(os.path.join(args.savepath, args.infomation), index=False)
