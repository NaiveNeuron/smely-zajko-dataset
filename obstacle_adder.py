import glob
from random import getrandbits
import cv2 as cv
import numpy as np
import utils


class ObstacleAdder:

    def __init__(self):
        self.obs_mask = None
        self.obs = None
        self.img = None
        self.new_img = None
        self.mask = None
        self.new_mask = None

    def obstacles_from_folder(self, folder, img_ext='.png'):
        glob_selector = '{}/*{}'.format(folder, img_ext)
        for file in glob.glob(glob_selector):
            yield cv.imread(file)

    def load_obstacles(self, folder, img_ext='png'):
        obstacles = []
        obstacles_mask = []
        for obs in self.obstacles_from_folder(folder, img_ext):
            mask = cv.cvtColor(obs, cv.COLOR_BGR2GRAY)
            _, mask = cv.threshold(mask, 10, 255, cv.THRESH_BINARY)
            obstacles.append(obs)
            obstacles_mask.append(mask)
        return np.asarray(obstacles), np.asarray(obstacles_mask)

    def insert_image(self, image, mask, x, y, h, w):
        image = cv.copyMakeBorder(image, h, h, w, w,
                                  cv.BORDER_CONSTANT,
                                  value=[0, 0, 0])
        mask = cv.copyMakeBorder(mask, h, h, w, w,
                                 cv.BORDER_CONSTANT,
                                 value=[0, 0, 0])
        roi = image[y:y+h, x:x+w]
        mask_roi = mask[y:y+h, x:x+w]
        obs_mask_inv = cv.bitwise_not(self.obs_mask)
        roi_bg = cv.bitwise_and(roi, roi, mask=obs_mask_inv)
        roi_fg = cv.bitwise_and(self.obs, self.obs, mask=self.obs_mask)
        mask_bg = cv.bitwise_and(mask_roi, mask_roi, mask=obs_mask_inv)
        mask_fg = cv.bitwise_and(self.obs_mask,
                                 self.obs_mask,
                                 mask=obs_mask_inv)
        roi = cv.add(roi_bg, roi_fg)
        obs_mask_inv = cv.add(mask_bg, mask_fg)
        image[y:y+h, x:x+w] = roi
        mask[y:y+h, x:x+w] = obs_mask_inv
        return image[h:-h, w:-w], mask[h:-h, w:-w]

    # mouse callback function
    def place(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.new_img = self.img.copy()
            self.new_mask = self.mask.copy()
            h, w = self.obs.shape[:2]
            self.new_img, self.new_mask = self.insert_image(self.new_img,
                                                            self.new_mask,
                                                            x+w/2, y+h/2, h, w)

if __name__ == '__main__':
    obs_adder = ObstacleAdder()
    obstacles, obstacles_mask = obs_adder.load_obstacles('./obstacles_cropped')
    global img, mask, obs, obs_mask, new_img, new_mask
    for i, datum in enumerate(utils.dataset_from_folder('./plzen')):
        cv.namedWindow('image')
        cv.namedWindow('mask')
        cv.setMouseCallback('image', obs_adder.place)
        obs_adder.img = datum['img']
        obs_adder.mask = datum['mask']
        obs_adder.new_img = obs_adder.img.copy()
        obs_adder.new_mask = obs_adder.mask.copy()
        for i, obstacle in enumerate(obstacles):
            obs_adder.obs = obstacle
            obs_adder.obs_mask = obstacles_mask[i]
            while True:
                cv.imshow("obstacle", obs_adder.obs)
                cv.imshow("image", obs_adder.new_img)
                cv.imshow("mask", obs_adder.new_mask)
                label_img = utils.visualize_labels(obs_adder.new_img,
                                                   obs_adder.new_mask)
                cv.imshow('labels', label_img)
                k = cv.waitKey(1) & 0xFF
                if k == ord('+'):
                    obs_adder.obs = cv.resize(obs_adder.obs, (0, 0),
                                              fx=1.1, fy=1.1)
                    o_mask = obs_adder.obs_mask
                    o_mask = cv.resize(obs_adder.obs_mask, (0, 0),
                                       fx=1.1, fy=1.1,
                                       interpolation=cv.INTER_CUBIC)
                    print(obs_adder.obs.shape, obs_adder.obs_mask.shape)
                if k == ord('-'):
                    obs_adder.obs = cv.resize(obs_adder.obs, (0, 0),
                                              fx=0.9, fy=0.9)
                    obs_adder.obs_mask = cv.resize(obs_adder.obs_mask, (0, 0),
                                                   fx=0.9, fy=0.9,
                                                   interpolation=cv.INTER_AREA)
                    print(obs_adder.obs.shape)
                if k == ord('f'):
                    obs_adder.obs = cv.flip(obs_adder.obs, 1)
                    obs_adder.obs_mask = cv.flip(obs_adder.obs_mask, 1)
                if k == ord('s'):
                    filename = './with_obs/{}_obs.png'.format(getrandbits(32))
                    cv.imwrite(filename, obs_adder.new_img)
                    trn_content = utils.numpy_to_trn(obs_adder.new_mask)
                    with open("{}.trn".format(filename), 'w+') as f:
                        f.write(trn_content)
                    break
                # 27 - escape
                if k == 27:
                    break
                if k == ord('q'):
                    cv.destroyAllWindows()
                    exit(0)

        cv.destroyAllWindows()
