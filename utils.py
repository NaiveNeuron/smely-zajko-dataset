import glob
import os.path
import warnings
import cv2 as cv
import numpy as np


def trn_to_numpy(filename):
    filename_img = filename.replace('.trn', '')
    img = cv.imread(filename_img)
    with open(filename, 'r') as f:
        file = f.read().rstrip()
        items = map(int, file.split(' '))
        numpy_array = np.asarray(items, np.uint8) * 255

        return numpy_array.reshape(img.shape[0], img.shape[1])


def numpy_to_trn(filename, mask):
    filename_trn = "{}.trn".format(filename)
    _, mask = cv.threshold(mask, 1, 1, cv.THRESH_BINARY)
    f = open(filename_trn, 'w+')
    f.write(' '.join(str(e) for e in mask.flatten().tolist()))
    f.close


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


def visualize_mask(image, mask):
    return cv.bitwise_and(image, image, mask=mask)


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


def load_dataset(folder, img_ext='.png'):
    data = []
    data_mask = []
    for datum in dataset_from_folder(folder, img_ext):
        data.append(datum['img'])
        data_mask.append(datum['mask'])
    return np.asarray(data), np.asarray(data_mask)


def calc_PCA(data):
    # normalize data before calculating PCA
    if (data > 1).any():
        norm_data = data.astype('float32') / 255.0
    else:
        norm_data = data
    n, h, w, c = norm_data.shape
    reshaped_data = np.reshape(norm_data, (n * h * w, 3))
    conv = reshaped_data.T.dot(reshaped_data) / reshaped_data.shape[0]
    u, s, v = np.linalg.svd(conv)
    eigenvalues = np.sqrt(s)
    return eigenvalues, u


def blur_batch(batch):
    blured_batch = []
    for image in batch:
        gray_im = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blur_amount = cv.Laplacian(gray_im, cv.CV_64F).var()/1000.
        blured_im = cv.GaussianBlur(image, (3, 3), blur_amount, blur_amount)
        blured_batch.append(blured_im)
    return np.asarray(blured_batch)


def flip_batch(batch, batch_mask):
    batch_fliped = batch[:, :, ::-1, :]
    batch_mask_fliped = batch_mask[:, :, ::-1]
    return batch_fliped, batch_mask_fliped


def add_color_noise(batch, eigenvalues, eigenvectors):
    if (batch > 1).any():
        norm_data = batch.astype('float32') / 255.0
    else:
        norm_data = batch
    alpha = np.random.randn(norm_data.shape[0], 3) * 0.1
    noise = eigenvectors.dot((eigenvalues * alpha).T)
    norm_data += noise[:, np.newaxis, np.newaxis, :].T
    return norm_data


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
    data, data_mask = load_dataset('./plzen')
    eigenvalues, pc = calc_PCA(data)
    jitter_data = add_color_noise(data, eigenvalues, pc)
    pca_data = np.clip(jitter_data * 255, 0, 255).astype('uint8')

    flipped_data, flipped_mask = flip_batch(data, data_mask)

    blured_data = blur_batch(data)

    for i, datum in enumerate(dataset_from_folder('./plzen')):
        img = datum['img']
        max_proba = mask_to_proba(datum['mask'], type='max')
        visual = visualize_proba(max_proba, datum['img'].shape)
        weighted = cv.addWeighted(img, 0.7, visual, 0.3, 0.0)
        visual_mask = visualize_mask(img, datum['mask'])

        visual_flipped_mask = visualize_mask(flipped_data[i], flipped_mask[i])
        max_fliped_proba = mask_to_proba(flipped_mask[i], type='max')
        flipped_visual = visualize_proba(max_proba, flipped_data[i].shape)
        flipped_weighted = cv.addWeighted(flipped_data[i], 0.7,
                                          flipped_visual, 0.3, 0.0)


        cv.imshow('img', weighted)
        cv.imshow('mask', datum['mask'])
        cv.imshow('visualize mask', visual_mask)

        cv.imshow('fliped image', flipped_weighted)
        cv.imshow('fliped mask', flipped_mask[i])
        cv.imshow('visualize fliped mask', visual_flipped_mask)

        cv.imshow('pca image', pca_data[i])

        cv.imshow('blured image', blured_data[i])

        print datum['cls'], datum['proba']
        cv.waitKey(0)
