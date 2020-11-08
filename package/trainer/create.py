import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import pickle
import tarfile
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
from multiprocessing.pool import Pool


PADDING_X = 0
PADDING_Y = 40
PENGUIN_DIM = 175
IMAGE_DIM = 128

progress = tqdm(total=16)


def update_progress(*a):
    progress.update()


def read_file(json_filename, job_dir):
    penguins = []

    part_name_json = os.path.basename(json_filename)
    part_name = os.path.splitext(part_name_json)[0]

    if tf.io.gfile.exists(job_dir + '/' + part_name + '.pickle'):
        return

    with tf.io.gfile.GFile(json_filename, mode='r') as json_file:
        json_data = json.load(json_file)

        for i in range(len(json_data['dots'])):
            image_data = json_data['dots'][i]

            if image_data['xy'] is not None and type(image_data['xy']) is list and len(image_data['xy']) > 0 and type(image_data['xy'][0]) is list:
                with tf.io.gfile.GFile(job_dir + '/' + part_name + '/' + image_data['imName'] + '.JPG', mode='rb') as image_file:
                    raw_image = Image.open(image_file)
                    width, height = raw_image.size

                    for j in range(0, len(image_data['xy'][0]), 2):
                        xy = image_data['xy'][0][j]
                        if type(xy) is list and len(xy) == 2:
                            x, y = xy
                            x = min(
                                (max((PADDING_X + PENGUIN_DIM / 2, x)), width - PADDING_X - PENGUIN_DIM / 2))
                            y = min((max((PADDING_Y + PENGUIN_DIM / 2, y)),
                                    height - PADDING_Y - PENGUIN_DIM / 2))

                            left = x - PENGUIN_DIM / 2
                            top = y - PENGUIN_DIM / 2
                            right = x + PENGUIN_DIM / 2
                            bottom = y + PENGUIN_DIM / 2

                            penguin = raw_image.crop((left, top, right, bottom)).resize(
                                (IMAGE_DIM, IMAGE_DIM), Image.LANCZOS)
                            # penguin.show()
                            penguins.append(
                                np.array(penguin, dtype=np.uint8))

    X = np.array(penguins)
    X = X.astype('float32')
    X = (X - 127.5) / 127.5

    with tf.io.gfile.GFile(job_dir + '/' + part_name + '.pickle', mode='wb') as pickle_out:
        pickle.dump(X, pickle_out, protocol=4)


def main(job_dir, **args):
    json_files = tf.io.gfile.glob(job_dir + '/CompleteAnnotations_2016-07-11/*.json')
    json_files.reverse()

    json_files = [
        '..\\tmp\\dataset\\CompleteAnnotations_2016-07-11\\MAIVb.json'
    ]

    pool = Pool(processes=1)

    for json_file in json_files:
        read_file(json_file, job_dir)
        #pool.apply_async(read_file, args=(
        #    json_file, job_dir), callback=update_progress)

    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        help='Location to write dataset',
        required=True
    )

    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)
