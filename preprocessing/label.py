import os
import re
import sys
import time

sys.path.insert(0, os.path.abspath('.'))

import matplotlib.pyplot as plt
import numpy as np

import common

type_label = None
direction_label = ''

plt_text = None

type_dictionary = {'1': 'hollow', '2': 'full', '3': 'thin'}


def main():
    print("         Q = ignore image")
    print("         1 = label as hollow")
    print("         2 = label as full")
    print("         3 = label as thin")
    print("ARROW KEYS = label directions\n")

    global type_label
    global direction_label
    global plt_text

    unlabeled_imgs = common.get_files(common.SCREENSHOTS_DIR)

    num_labeled = 0
    for path, filename in unlabeled_imgs:
        print("Processing {}...".format(filename))

        img = plt.imread(path)

        ax = plt.gca()
        fig = plt.gcf()
        plot = ax.imshow(img)

        plt.axis('off')
        plt.tight_layout()
        plt_text = plt.text(0, 0, "")

        fig.canvas.mpl_connect('key_press_event', on_press)

        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')

        plt.show()

        if type_label and direction_label:
            dst_filename = "{}_{}_{}.png".format(
                type_dictionary[type_label], direction_label, time.strftime("%Y%m%d-%H%M%S"))

            os.rename(path, common.LABELED_DIR + dst_filename)

            direction_label = ''
            type_label = None

            num_labeled += 1

    if len(unlabeled_imgs) > 0:
        print("\nLabeled {} out of {} images ({}%).".format(
            num_labeled, len(unlabeled_imgs), 100 * num_labeled // len(unlabeled_imgs)))
        print("Finished!")
    else:
        print("\nThere are no images to label.")


def on_press(event):
    global type_label
    global direction_label

    if event.key in ['1', '2', '3']:
        type_label = event.key
        if len(direction_label) == 4:
            plt.close()
            return

    elif event.key in ['left', 'right', 'up', 'down']:
        if len(direction_label) < 4:
            direction_label += event.key[0]
        if len(direction_label) >= 4 and type_label:
            plt.close()
            return

    elif event.key == 'z':
        type_label = None
        direction_label = ''
    
    if event.key != 'q':
        if not type_label:
            t = '-'
        else:
            t = type_dictionary[type_label]

        plt_text.set_text(make_text(t, direction_label))
        plt.draw()


def make_text(type_label, direction_label):
    directions = []

    for d in direction_label:
        if d == 'd':
            directions.append('down')
        elif d == 'l':
            directions.append('left')
        elif d == 'r':
            directions.append('right')
        elif d == 'u':
            directions.append('up')

    for x in range(len(direction_label), 4):
        directions.append('-')

    return "%s: { %s, %s, %s, %s }" % (type_label, directions[0], directions[1], directions[2], directions[3])


if __name__ == "__main__":
    main()
