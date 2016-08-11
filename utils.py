import cv2 as cv
import numpy as np
import glob
import os.path
import warnings


def trn_to_numpy(filename):
    filename_img = filename.replace('.trn', '')
    img = cv.imread(filename_img)
    with open(filename, 'r') as f:
        file = f.read().rstrip()
        items = map(int, file.split(' '))
        numpy_array = np.asarray(items, np.uint8) * 255

        return numpy_array.reshape(img.shape[0], img.shape[1])


def xml_to_numpy(filename):
    return np.asarray(cv.cv.Load(filename)) * 255


def mask_to_proba(mask, classes=10, type='sum'):
    h, w = mask.shape
    w_per_class = w // classes
    counts = []
    for c in range(classes):
        selection = mask[0:h, c*w_per_class:(c+1)*w_per_class]
        counts.append(np.count_nonzero(selection))

    counts = np.asarray(counts)
    sum = float(np.sum(counts))
    if type == 'max':
        sum = float(np.max(counts))

    return counts/sum if sum != 0 else counts


def visualize_proba(proba, shape, color=(0, 0, 255)):
    classes = len(proba)
    h, w, ch = shape
    img = np.zeros((h, w, ch), np.uint8)
    w_per_class = w // classes
    for c, p in enumerate(proba):
        top = int(h - h*p)
        cv.rectangle(img, (c*w_per_class, top), ((c+1)*w_per_class, h),
                     color, -1)
    return img


def load_image_for_dataset(filename):
    trn_filename = '{}.trn'.format(filename)
    img = cv.imread(filename)
    mask = trn_to_numpy(trn_filename)
    proba = mask_to_proba(mask)
    cls = np.argmax(proba) if np.max(proba) != 0 else -1
    return {
        "img": img,
        "mask": mask,
        "proba": proba,
        "cls": cls
    }


def dataset_from_folder(folder, img_ext='.png', mask_ext='.trn'):
    glob_selector = '{}/*{}'.format(folder, img_ext)
    for file in glob.glob(glob_selector):
        mask_filename = '{}{}'.format(file, mask_ext)
        if not os.path.exists(mask_filename):
            msg = 'Mask file {} for image {} does not seem to exist'.format(
                mask_filename,
                file
            )
            warnings.warn(msg)
            continue
        yield load_image_for_dataset(file)

if __name__ == '__main__':
    for datum in dataset_from_folder('./plzen'):
        img = datum['img']
        max_proba = mask_to_proba(datum['mask'], type='max')
        visual = visualize_proba(max_proba, datum['img'].shape)
        weighted = cv.addWeighted(img, 0.7, visual, 0.3, 0.0)

        cv.imshow('img', weighted)
        cv.imshow('mask', datum['mask'])
        print datum['cls'], datum['proba']
        cv.waitKey(0)
