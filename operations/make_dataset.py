import argparse
import os
import random
import re
import sys

sys.path.insert(0, os.path.abspath('.'))

import colorful as cf
import numpy as np
import pandas as pd

import common

TRAINING_SET_RATIO = 0.9
VALIDATION_SET_RATIO = 0.5


def main(training_set_ratio):
    arrows = pd.DataFrame(np.zeros((3, 4), dtype=np.int32), index=('round', 'wide', 'narrow'),
                          columns=('down', 'left', 'right', 'up'))

    images = [(p, f) for p, f in common.get_files(common.SAMPLES_DIR) if f[-5] != 'F']

    if images:
        for _, filename in images:
            arrow_direction, arrow_type = common.arrow_labels(filename)

            arrows[arrow_direction][arrow_type] += 1

        num_samples = int(arrows.min().min() * training_set_ratio)

        print("Samples per type: {}".format(num_samples * 4))

        for t, _ in arrows.iterrows():
            print("\nProcessing {} arrows...".format(t))

            for d in arrows:
                candidates = [(p, f) for p, f in images if common.arrow_labels(f) == (d, t)]

                print("{}: {}".format(d, len(candidates)))

                training = random.sample(candidates, num_samples)
                for path, filename in training:
                    dst_dir = common.TRAINING_DIR + d + "/"
                    if not os.path.exists(dst_dir):
                        os.makedirs(dst_dir)

                    os.rename(path, dst_dir + filename)
                    os.rename(flipped(path), dst_dir + flipped(filename))

                candidates = [c for c in candidates if c not in training]

                validation = random.sample(candidates, int(len(candidates) * VALIDATION_SET_RATIO))
                for path, filename in validation:
                    dst_dir = common.VALIDATION_DIR + d + "/"
                    if not os.path.exists(dst_dir):
                        os.makedirs(dst_dir)
                        
                    os.rename(path, dst_dir + filename)
                    os.rename(flipped(path), dst_dir + flipped(filename))

                testing = [c for c in candidates if c not in validation]
                for path, filename in testing:
                    dst_dir = common.TESTING_DIR + d + "/"
                    if not os.path.exists(dst_dir):
                        os.makedirs(dst_dir)
                        
                    os.rename(path, dst_dir + filename)
                    os.rename(flipped(path), dst_dir + flipped(filename))

    show_summary()

    print("\nFinished!")


def flipped(s):
    return s[:-4] + 'F' + s[-4:]


def show_summary():
    print("\n" + cf.skyBlue("Training set"))
    print(get_summary_matrix(common.TRAINING_DIR))

    print("\n" + cf.salmon("Validation set"))
    print(get_summary_matrix(common.VALIDATION_DIR))

    print("\n" + cf.lightGreen("Testing set"))
    print(get_summary_matrix(common.TESTING_DIR))


def get_summary_matrix(directory):
    matrix = pd.DataFrame(np.zeros((4, 5), dtype=np.int32), index=(
        'round', 'wide', 'narrow', 'total'), columns=('down', 'left', 'right', 'up', 'total'))

    images = common.get_files(directory)

    for path, filename in images:
        arrow_direction, arrow_type = common.arrow_labels(filename)

        matrix[arrow_direction][arrow_type] += 1

        matrix['total'][arrow_type] += 1
        matrix[arrow_direction]['total'] += 1
        matrix['total']['total'] += 1

    return matrix


if __name__ == "__main__":
    os.system('color')

    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--ratio', type=float, default=TRAINING_SET_RATIO,
                        help="Specifies the training/validation set proportion")

    args = parser.parse_args()

    main(args.ratio)
