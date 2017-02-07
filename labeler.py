#!/usr/bin/env python
import argparse
import os.path as os
import cv2 as cv
import numpy as np


horz = False
mask = None
color = None
flood = False
rec_size = 5
line_points = np.array([[-1, -1], [-1, -1]])


def load_xml(filename):
    return np.asarray(cv.cv.Load(filename)) * 255


def load_trn(filename):
    filename_img = filename.replace('.trn', '')
    img = cv.imread(filename_img)
    with open(filename, 'r') as f:
        file = f.read().rstrip()
        items = map(int, file.split(' '))
        numpy_array = np.asarray(items, np.uint8) * 255

        return numpy_array.reshape(img.shape[0], img.shape[1])


def floodify(img, color, x, y):
    flags = 4
    flags | 255 << 8
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # get only path part of mask (white parts)
    tmp_img = (img == [255, 255, 255]).astype('uint8') * 255
    cv.floodFill(tmp_img, mask, (x, y), color, 0, 255, flags)
    return tmp_img


# mouse callback function
def draw(event, x, y, flags, param):
    global mask, horz, color, flood, rec_size, line_points
    if event == cv.EVENT_LBUTTONDOWN:
        color = (255, 255, 255)
        if horz is True:
            color = (255, 0, 0)
        elif flood is True:
            mask[np.nonzero(floodify(mask, color, x, y))] = 255

    if event == cv.EVENT_RBUTTONDOWN:
        color = (0, 0, 0)
        if flood is True:
            floodify(mask, color, x, y)

    if event == cv.EVENT_MOUSEMOVE and color is not None:
        cv.rectangle(mask, (x - rec_size, y - rec_size),
                           (x + rec_size, y + rec_size), color, -1)

    if event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:
        flood = False
        horz = False
        color = None


def main_event_loop(image):
    global flood, rec_size, horz, mask
    while(1):
        img = cv.addWeighted(image, 0.7, mask.astype('uint8'), 0.3, 0)
        cv.imshow("labeling board", img)
        cv.imshow("mask", mask)
        k = cv.waitKey(1) & 0xFF
        # floodfill keybind
        if k == ord('f'):
            flood = True
        # add new point to horizont keybind
        if k == ord('h'):
            horz = True
        # enlarge the brush
        if k == ord('+'):
            rec_size += 1
            print(rec_size)
        # reduce the brush
        if k == ord('-'):
            rec_size -= 1
            print(rec_size)
        # end & save keybinds
        if k == 27 or k == ord('q'):
            break


def load_mask(mask_name):
    if os.splitext(mask_name)[1] == '.trn':
        mask = load_trn(mask_name)
        mask = np.repeat(mask, 3).reshape((mask.shape[0],
                                           mask.shape[1], 3))
    elif os.splitext(mask_name)[1] == '.xml':
        mask = load_xml(mask_name)
        mask = np.repeat(mask, 3).reshape((mask.shape[0],
                                           mask.shape[1], 3))
    else:
        mask = cv.imread(mask_name, cv.CV_LOAD_IMAGE_COLOR)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    return mask


def main(image, prep_mask=None):
    global mask
    im = cv.imread(image, cv.CV_LOAD_IMAGE_COLOR)
    if prep_mask is not None:
        mask = load_mask(prep_mask)
    else:
        mask = np.zeros(im.shape)
    # create labeling board
    cv.namedWindow('labeling board')
    cv.setMouseCallback('labeling board', draw)
    main_event_loop(im)
    cv.destroyAllWindows()
    # save the created mask
    cv.imwrite("{}_mask.png".format(os.splitext(image)[0]), mask)


if __name__ == '__main__':
    desc = 'Smely-Zajko labeling tool'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('filename', metavar='filename',
                        help='Image to label.')
    parser.add_argument('-m', '--mask', nargs='?', metavar='filename',
                        help='Predifined mask.', dest='mask')
    args = parser.parse_args()
    main(args.filename, args.mask)
