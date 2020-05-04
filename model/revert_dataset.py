import os
import sys

sys.path.insert(0, os.path.abspath('.'))

import common


def main():
    print("Reverting images from the training directory...")
    revert_files(common.TRAINING_DIR)

    print("Reverting images from the validation directory...")
    revert_files(common.VALIDATION_DIR)

    print("Reverting images from the testing directory...")
    revert_files(common.TESTING_DIR)

    print("Finished!")


def revert_files(src_dir):
    images = common.get_files(src_dir)

    for path, filename in images:
        os.rename(path, common.SAMPLES_DIR + filename)

    print("Reverted {} images.\n".format(len(images)))


if __name__ == "__main__":
    main()
