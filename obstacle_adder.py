import glob
import cv2 as cv
import numpy as np
import utils
from random import getrandbits


img = None
new_img = None
mask = None
new_mask = None
obs = None
obs_mask = None



def obstacles_from_folder(folder, img_ext='.png'):
    glob_selector = '{}/*{}'.format(folder, img_ext)
    for file in glob.glob(glob_selector):
        yield cv.imread(file)


def load_obstacles(folder, img_ext='png'):
    obstacles = []
    obstacles_mask = []
    for obs in obstacles_from_folder(folder):
        mask = cv.cvtColor(obs, cv.COLOR_BGR2GRAY)
        _, mask = cv.threshold(mask, 10, 255, cv.THRESH_BINARY)
        obstacles.append(obs)
        obstacles_mask.append(mask)
    return np.asarray(obstacles), np.asarray(obstacles_mask)


def __insert_image(image, mask, x, y, h, w):
    global obs_mask, obs
    image = cv.copyMakeBorder(image, h, h, w, w,
                              cv.BORDER_CONSTANT,
                              value=[0, 0, 0])
    mask = cv.copyMakeBorder(mask, h, h, w, w,
                             cv.BORDER_CONSTANT,
                             value=[0, 0, 0])
    roi = image[y:y+h, x:x+w]
    mask_roi = mask[y:y+h, x:x+w]
    obs_mask_inv = cv.bitwise_not(obs_mask)
    roi_bg = cv.bitwise_and(roi, roi, mask=obs_mask_inv)
    roi_fg = cv.bitwise_and(obs, obs, mask=obs_mask)
    mask_bg = cv.bitwise_and(mask_roi, mask_roi, mask=obs_mask_inv)
    mask_fg = cv.bitwise_and(obs_mask, obs_mask, mask=obs_mask_inv)
    roi = cv.add(roi_bg, roi_fg)
    obs_mask_inv = cv.add(mask_bg, mask_fg)
    image[y:y+h, x:x+w] = roi
    mask[y:y+h, x:x+w] = obs_mask_inv
    return image[h:-h, w:-w], mask[h:-h, w:-w]


def visualize_labels(img, mask):
    max_proba = utils.mask_to_proba(mask, type='max')
    visual = utils.visualize_proba(max_proba, img.shape)
    weighted = cv.addWeighted(img, 0.7, visual, 0.3, 0.0)
    cv.imshow('labels', weighted)


# mouse callback function
def place(event, x, y, flags, param):
    global img, mask, obs, obs_mask, new_img, new_mask
    if event == cv.EVENT_LBUTTONDOWN:
        new_img = img.copy()
        new_mask = mask.copy()
        h, w = obs.shape[:2]
        new_img, new_mask = __insert_image(new_img, new_mask,
                                           x+w/2, y+h/2, h, w)

if __name__ == '__main__':
    obstacles, obstacles_mask = load_obstacles('./obstacles_cropped')
    global img, mask, obs, obs_mask, new_img, new_mask
    for i, datum in enumerate(utils.dataset_from_folder('./plzen')):
        cv.namedWindow('image')
        cv.namedWindow('mask')
        cv.setMouseCallback('image', place)
        img = datum['img']
        mask = datum['mask']
        new_img = img.copy()
        new_mask = mask.copy()
        for i, obstacle in enumerate(obstacles):
            obs = obstacle
            obs_mask = obstacles_mask[i]
            while(1):
                cv.imshow("obstacle", obs)
                cv.imshow("image", new_img)
                cv.imshow("mask", new_mask)
                visualize_labels(new_img, new_mask)
                k = cv.waitKey(1) & 0xFF
                if k == ord('+'):
                    obs = cv.resize(obs, (0, 0), fx=1.1, fy=1.1)
                    obs_mask = cv.resize(obs_mask, (0, 0), fx=1.1, fy=1.1,
                                         interpolation=cv.INTER_CUBIC)
                    print(obs.shape, obs_mask.shape)
                if k == ord('-'):
                    obs = cv.resize(obs, (0, 0), fx=0.9, fy=0.9)
                    obs_mask = cv.resize(obs_mask, (0, 0), fx=0.9, fy=0.9,
                                         interpolation=cv.INTER_AREA)
                    print(obs.shape)
                if k == ord('f'):
                    obs = cv.flip(obs, 1)
                    obs_mask = cv.flip(obs_mask, 1)
                if k == ord('s'):
                    filename = './with_obs/{}_obs.png'.format(getrandbits(32))
                    cv.imwrite(filename, new_img)
                    utils.numpy_to_trn(filename, new_mask)
                    break
                # 27 - escape
                if k == 27:
                    break
                if k == ord('q'):
                    cv.destroyAllWindows()
                    exit(0)

        cv.destroyAllWindows()
