import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tarfile
import argparse
from tqdm import tqdm
import urllib.request
from urllib.parse import urlparse
from multiprocessing.pool import Pool
from tensorflow.python.lib.io import file_io

src_urls = [
    'https://www.robots.ox.ac.uk/~vgg/data/penguins/CompleteAnnotations_2016-07-11.tgz',
    'https://www.robots.ox.ac.uk/~vgg/data/penguins/BAILa.tgz',
    'https://www.robots.ox.ac.uk/~vgg/data/penguins/DAMOa.tgz',
    'https://www.robots.ox.ac.uk/~vgg/data/penguins/GEORa.tgz',
    'https://www.robots.ox.ac.uk/~vgg/data/penguins/HALFb.tgz',
    'https://www.robots.ox.ac.uk/~vgg/data/penguins/HALFc.tgz',
    'https://www.robots.ox.ac.uk/~vgg/data/penguins/LOCKb.tgz',
    'https://www.robots.ox.ac.uk/~vgg/data/penguins/MAIVb.tgz',
    'https://www.robots.ox.ac.uk/~vgg/data/penguins/MAIVc.tgz',
    'https://www.robots.ox.ac.uk/~vgg/data/penguins/NEKOa.tgz',
    'https://www.robots.ox.ac.uk/~vgg/data/penguins/NEKOb.tgz',
    'https://www.robots.ox.ac.uk/~vgg/data/penguins/NEKOc.tgz',
    'https://www.robots.ox.ac.uk/~vgg/data/penguins/PETEc.tgz',
    'https://www.robots.ox.ac.uk/~vgg/data/penguins/PETEd.tgz',
    'https://www.robots.ox.ac.uk/~vgg/data/penguins/PETEe.tgz',
    'https://www.robots.ox.ac.uk/~vgg/data/penguins/PETEf.tgz',
    'https://www.robots.ox.ac.uk/~vgg/data/penguins/SPIGa.tgz'
]

progress = tqdm(total=len(src_urls))


def update_progress(*a):
    progress.update()


def download_assets(job_dir):
    pool = Pool(processes=os.cpu_count())

    for url in src_urls:
        pool.apply_async(download_url, args=(
            url, job_dir), callback=update_progress)

    pool.close()
    pool.join()


def download_url(url, job_dir):
    parsed_url = urlparse(url)
    tar_name = os.path.basename(parsed_url.path)
    folder_name = os.path.splitext(tar_name)[0]

    urllib.request.urlretrieve(url, tar_name)

    file_io.create_dir(job_dir + '/' + folder_name)

    with tarfile.open(tar_name, 'r:gz') as f:
        for filename in f.getnames():
            with f.extractfile(filename) as input_f:
                with file_io.FileIO(job_dir + '/' + filename, mode='wb+') as output_f:
                    output_f.write(input_f.read())

    os.remove(tar_name)


def main(job_dir, **args):
    download_assets(job_dir)


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
