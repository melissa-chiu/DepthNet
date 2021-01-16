# face detector: sfd
# landmark predictor: FAN
# alignment: using 5 landmarks (same as doris work)
import numpy as np
import cv2
import glob
import os
import tqdm
import argparse

from imutils import face_utils
import imutils
import dlib
import collections
import face_alignment

def face_remap(shape):
    remapped_image = cv2.convexHull(shape)
    return remapped_image

# average landmarks
mean_face_lm5p = np.array([
    [-0.17607, -0.172844],  # left eye pupil
    [0.1736, -0.17356],  # right eye pupil
    [-0.00182, 0.0357164],  # nose tip
    [-0.14617, 0.20185],  # left mouth corner
    [0.14496, 0.19943],  # right mouth corner
])

def _get_align_5p_mat23_size_256(lm):
    # legacy code
    width = 256
    mf = mean_face_lm5p.copy()

    # Assumptions:
    # 1. The output image size is 256x256 pixels
    # 2. The distance between two eye pupils is 70 pixels
    ratio = 70.0 / (
       256.0 * 0.34967
    )  # magic number 0.34967 to compensate scaling from average landmarks

    left_eye_pupil_y = mf[0][1]
    # In an aligned face image, the ratio between the vertical distances from eye to the top and bottom is 1:1.42
    ratioy = (left_eye_pupil_y * ratio + 0.5) * (1 + 1.42)
    mf[:, 0] = (mf[:, 0] * ratio + 0.5) * width
    mf[:, 1] = (mf[:, 1] * ratio + 0.5) * width / ratioy
    mx = mf[:, 0].mean()
    my = mf[:, 1].mean()
    dmx = lm[:, 0].mean()
    dmy = lm[:, 1].mean()
    mat = np.zeros((3, 3), dtype=float)
    ux = mf[:, 0] - mx
    uy = mf[:, 1] - my
    dux = lm[:, 0] - dmx
    duy = lm[:, 1] - dmy
    c1 = (ux * dux + uy * duy).sum()
    c2 = (ux * duy - uy * dux).sum()
    c3 = (dux**2 + duy**2).sum()
    a = c1 / c3
    b = c2 / c3

    kx = 1
    ky = 1

    s = c3 / (c1**2 + c2**2)
    ka = c1 * s
    kb = c2 * s

    transform = np.zeros((2, 3))
    transform[0][0] = kx * a
    transform[0][1] = kx * b
    transform[0][2] = mx - kx * a * dmx - kx * b * dmy
    transform[1][0] = -ky * b
    transform[1][1] = ky * a
    transform[1][2] = my - ky * a * dmy + ky * b * dmx
    return transform

def get_align_5p_mat23(lm5p, size):
    """Align a face given 5 facial landmarks of
    left_eye_pupil, right_eye_pupil, nose_tip, left_mouth_corner, right_mouth_corner

    :param lm5p: nparray of (5, 2), 5 facial landmarks,

    :param size: an integer, the output image size. The face is aligned to the mean face

    :return: a affine transformation matrix of shape (2, 3)
    """
    mat23 = _get_align_5p_mat23_size_256(lm5p.copy())
    mat23 *= size / 256
    return mat23


def align_given_lm5p(img, lm5p, size):
    mat23 = get_align_5p_mat23(lm5p, size)
    return cv2.warpAffine(img, mat23, (size, size)) #, cv2.warpAffine(img_d, mat23, (size, size))


def align_face_5p(img, landmarks):
    aligned_img = align_given_lm5p(img, np.array(landmarks).reshape((5, 2)), 256)
    return aligned_img

def align_one(image):
    # image: opencv image
    fa_fan = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, face_detector='sfd')
    shapes = fa_fan.get_landmarks(image)
    try:
        shape = shapes[-1] # nx68x2
    except:
        raise('Error occurs :(')

    landmarks_5 = np.array([(int(round(np.mean(shape[36:42,0]))),int(round(np.mean(shape[36:42,1])))),(int(round(np.mean(shape[42:48,0]))),int(round(np.mean(shape[42:48,1])))),(shape[33,0],shape[33,1]),(shape[48,0],shape[48,1]),(shape[54,0],shape[54,1])])
    aligned = align_face_5p(image, landmarks_5)

    return aligned

if __name__=='__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=None, required=True, help="path of testing images")
    parser.add_argument("--des", type=str, default=None, required=True, help="saved aligned images")
    opt = parser.parse_args()

    fa_fan = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, face_detector='sfd')
    fa_fan_dlib = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, face_detector='dlib')

    path = opt.src+'*/*'
    files = sorted(glob.glob(path))
    orin_num = len(files)
    print('Number of images: ', len(files))

    for f in tqdm.tqdm(files):
        image = cv2.imread(f)
        shapes = fa_fan.get_landmarks(image)
        try:
            shape = shapes[-1] # nx68x2
        except:
            print(f, ' not detected.')
            with open('not_detect.txt', 'a') as txfile:
                txfile.write(f + '\n')
            continue

        if np.asarray(shapes).shape[0] > 1:
            shapes = fa_fan_dlib.get_landmarks(image)
            try:
                shape = shapes[-1] # nx68x2
            except:
                print(f, ' dlib not detected.')
                with open('dlib_not_detect.txt', 'a') as txfile:
                    txfile.write(f + '\n')
                continue
        
        landmarks_5 = np.array([(int(round(np.mean(shape[36:42,0]))),int(round(np.mean(shape[36:42,1])))),(int(round(np.mean(shape[42:48,0]))),int(round(np.mean(shape[42:48,1])))),(shape[33,0],shape[33,1]),(shape[48,0],shape[48,1]),(shape[54,0],shape[54,1])])

        aligned = align_face_5p(image, landmarks_5)
        if aligned is None:
            print(f, ' is None after alignment.')
            with open('none_align.txt', 'a') as txfile:
                txfile.write(f + '\n')
            continue

        fold = f.replace(opt.src, opt.des)
        fold = fold.replace(fold.split('/')[-1], '')
        if not os.path.exists(fold):
            os.makedirs(fold)
        cv2.imwrite(f.replace(opt.src, opt.des), aligned)

    print('Number of images done: ', len(glob.glob(path)), ' deleted: ', orin_num-len(glob.glob(path)))