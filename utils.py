import glob
import os.path
import warnings
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from keras.utils.np_utils import to_categorical


def trn_to_numpy(filename):
    filename_img = filename.replace('.trn', '')
    img = cv.imread(filename_img)
    with open(filename, 'r') as f:
        file = f.read().rstrip()
        items = map(int, file.split(' '))
        numpy_array = np.asarray(items, np.uint8) * 255

        return numpy_array.reshape(img.shape[0], img.shape[1])


def numpy_to_trn(mask):
    _, mask = cv.threshold(mask, 1, 1, cv.THRESH_BINARY)
    return ' '.join(str(e) for e in mask.flatten().tolist())


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


def visualize_labels(img, mask):
    max_proba = mask_to_proba(mask, type='max')
    visual = visualize_proba(max_proba, img.shape)
    weighted = cv.addWeighted(img, 0.7, visual, 0.3, 0.0)
    return weighted


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


def load_dataset_as_numpy(folder, img_ext='.png'):
    data = []
    data_mask = []
    for datum in dataset_from_folder(folder, img_ext):
        data.append(datum['img'])
        data_mask.append(datum['mask'])
    return np.asarray(data), np.asarray(data_mask)


def calc_PCA(data):
    # normalize data before calculating PCA
    if (data > 1).any():
        norm_data = data.astype('float16') / 255.0
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
        blur_amount = cv.Laplacian(gray_im, cv.CV_64F).var() / 1000.0
        blured_im = cv.GaussianBlur(image, (3, 3), blur_amount, blur_amount)
        blured_batch.append(blured_im)
    return np.asarray(blured_batch)


def flip_batch(batch, batch_mask):
    batch_fliped = batch[:, :, ::-1, :]
    batch_mask_fliped = batch_mask[:, :, ::-1]
    return batch_fliped, batch_mask_fliped


def add_color_noise(batch, eigenvalues, eigenvectors):
    if (batch > 1).any():
        norm_data = batch.astype('float16') / 255.0
    else:
        norm_data = batch
    alpha = np.random.randn(3) * 0.1
    noise = eigenvectors.dot((eigenvalues * alpha).T)
    norm_data += noise
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


def load_augmented_dataset(folder, eigenval, eigenvectors, img_ext='.png',
                           mask_ext='.trn', resize=(240, 240)):
    X = []
    y = []
    glob_selector = '{}/*{}'.format(folder, img_ext)
    for file in glob.glob(glob_selector):
        mask_filename = '{}{}'.format(file, mask_ext)
        if not os.path.exists(mask_filename):
            continue
        arr = load_image_for_dataset(file)
        arr['img'] = cv.resize(arr['img'], resize)
        gray_im = cv.cvtColor(arr['img'], cv.COLOR_BGR2GRAY)
        blur_amount = cv.Laplacian(gray_im, cv.CV_64F).var() / 1000.0

        blured_im = cv.GaussianBlur(arr['img'], (3, 3),
                                    blur_amount, blur_amount)
        pca_im = add_color_noise(arr['img'], eigenval, eigenvectors)
        X.append((arr['img']/255.0).T)
        # sligtly blured imaged
        X.append((blured_im/255.0).T)
        # imaged with changed color values based on PCA of dataset
        X.append((pca_im).T)
        # horizontaly fliped imaged
        X.append((cv.flip(arr['img'], 1)/255.0).T)
        c = np.zeros(11)
        if arr['cls'] == -1:
            c[10] = -1
        else:
            c[arr['cls']] = 1
        for i in range(4):
            y.append(c)
    return np.array(X), np.array(y)


def blur_image(image):
    gray_im = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur_amount = cv.Laplacian(gray_im, cv.CV_64F).var() / 1000.0
    return cv.GaussianBlur(image, (3, 3), blur_amount, blur_amount)


def flip_data(data):
    flip_im = cv.flip(data['img'], 1)
    flip_mask = cv.flip(data['mask'], 1)
    flip_proba = mask_to_proba(flip_mask)
    flip_cls = np.argmax(flip_proba) if np.max(flip_proba) != 0 else -1
    return {
        "img": flip_im,
        "mask": flip_mask,
        "proba": flip_proba,
        "cls": flip_cls
    }


def format_dict_to_dataset(img, mask, proba, cls):
    return {
        "img": img,
        "mask": mask,
        "proba": proba,
        "cls": cls
    }


def augmented_dataset_from_folder(folder, eigenval, eigenvectors,
                                  img_ext='.png', mask_ext='.trn',
                                  resize=(240, 240)):
    glob_selector = '{}/*{}'.format(folder, img_ext)
    for file in glob.glob(glob_selector):
        x = []
        mask_filename = '{}{}'.format(file, mask_ext)
        if not os.path.exists(mask_filename):
            continue

        arr = load_image_for_dataset(file)
        if resize is not None:
            arr['img'] = cv.resize(arr['img'], resize)
        # arr['img'] = cv.cvtColor(arr['img'], cv.COLOR_BGR2LAB)

        blured_im = (blur_image(arr['img']))
        pca_im = add_color_noise(arr['img'], eigenval, eigenvectors)

        # original image dict
        x.append(arr)

        # sligtly blured imaged
        dict_blured = format_dict_to_dataset(blured_im, arr['mask'],
                                             arr['proba'], arr['cls'])
        x.append(dict_blured)

        # imaged with changed color values based on PCA of dataset
        pca_dict = format_dict_to_dataset(pca_im, arr['mask'],
                                          arr['proba'], arr['cls'])
        x.append(pca_dict)

        # horizontaly fliped imaged dict
        flip_arr = flip_data(arr)
        x.append(flip_arr)

        for dict_x in x:
            yield dict_x


def imshow_noax(img, normalize=True):
    """ Tiny helper to show images as uint8 and remove axis labels """
    if normalize:
        img_max, img_min = np.max(img), np.min(img)
        img = 255.0 * (img - img_min) / (img_max - img_min)
    plt.imshow(img.T.astype('uint8'))
    plt.gca().axis('off')


def show_dataset_samples(X, y, nb_samples=5):
    imgs = X[(np.random.rand(nb_samples*nb_samples) * 100).astype('uint8')]
    for i in range(nb_samples * nb_samples):
        plt.subplot(nb_samples, nb_samples, i+1)
        imshow_noax(imgs[i])
    plt.show()


def bit_to_two_cls(x):
    return to_categorical(x, nb_classes=2)
