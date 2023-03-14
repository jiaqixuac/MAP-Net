"""
The code to preprocess the Cityscapes leftImg8bit_sequence_trainvaltest downloaded from the official website
and to generate the hazy version for HazeWorld.
"""
import argparse
import os
import os.path as osp
from tqdm import tqdm

import cv2
import numpy as np

IMAGE_EXTS = ('.jpg', '.png')


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess the dataset')
    parser.add_argument('-i', '--input-dir', help='the dir that contains the original images',
                        default='./data/Cityscapes')
    parser.add_argument('-o', '--work-dir', help='the dir to save preprocessed images',
                        default='./data/HazeWorld')
    parser.add_argument('--meta-file', help='the mapping file for data preparation',
                        default=None)
    return parser.parse_args()


def resize(img: np.ndarray, min_size: int = 720):
    h, w = img.shape[:2]
    if min(h, w) <= min_size:
        return img
    if h < w:
        dim = (int(w / h * min_size), min_size)
    else:
        dim = (min_size, int(w / h * min_size))
    img = cv2.resize(img, dim)
    return img


def process_cityscapes_raw(split, input_dir, work_dir, clip_per_city=20):
    if not osp.exists(input_dir):
        print(f'Please check if the data folder exists: Cityscapes-{split}, "{input_dir}"')
        return False

    os.makedirs(work_dir, exist_ok=True)
    cities = os.listdir(input_dir)
    cities = [folder for folder in cities if osp.isdir(osp.join(input_dir, folder))]
    cities.sort()
    bar = tqdm(cities)
    bar.set_description(f'[Cityscapes/{split}]...')

    lines = []
    for city in bar:
        src_folder = osp.join(input_dir, city)
        files = os.listdir(src_folder)
        files = [file for file in files if file.endswith(IMAGE_EXTS)]
        files.sort()
        num = len(files)
        if num == 0:
            print(f" *** Warning! No image files in: {src_folder}")

        assert num % 30 == 0, "each clip in Cityscapes should have 30 frames"

        interval = max(int(num / 30 / clip_per_city), 1)
        for idx in range(clip_per_city):
            _idx = idx * interval
            dst_folder = f'{work_dir}/{city}_{_idx:04d}'
            os.makedirs(dst_folder, exist_ok=True)
            bar.set_description(f'[Cityscapes/{split}] {city}: '
                                f'{idx + 1}/{clip_per_city}')
            for i in range(30):
                img = cv2.imread(osp.join(src_folder, files[_idx * 30 + i]))
                img = resize(img)
                # img_name = osp.splitext(files[i])[0]  # bug
                img_name = osp.splitext(files[_idx * 30 + i])[0]
                # we choose to store jpg files
                dst_path = osp.join(dst_folder, img_name + '.jpg')
                cv2.imwrite(dst_path, img.astype(np.uint8))

                line = f"{split}/{city}/{files[_idx * 30 + i]} " \
                       f"{split}/{city}_{_idx:04d}/{img_name}.jpg"
                # print(line)
                lines.append(line + '\n')
    return lines


def process_cityscapes_line(line):
    src_path, dst_path, light = line.split(' ')
    split, folder, file = dst_path.split('/')
    # process the GT file
    os.makedirs(osp.join(args.work_dir, 'gt/Cityscapes', split, folder), exist_ok=True)
    src_path = osp.join(args.input_dir, 'leftImg8bit_sequence_trainvaltest/leftImg8bit_sequence', src_path)
    assert osp.exists(src_path)
    dst_path = osp.join(args.work_dir, 'gt/Cityscapes', dst_path)
    img_gt = cv2.imread(src_path)
    img_gt = resize(img_gt)
    cv2.imwrite(dst_path, img_gt.astype(np.uint8))
    # fog synthesis
    A = float(light) / 255
    img_name = osp.splitext(file)[0]
    for beta in (0.005, 0.01, 0.02, 0.03):
        # transmission
        trans_folder = osp.join(args.work_dir, 'transmission/Cityscapes', split, f"{folder}_{beta}")
        trans_file = osp.join(trans_folder, img_name + '.png')
        assert osp.exists(trans_file), f"Should download the transmission files for Cityscapes: {trans_file}"
        img_t = cv2.imread(trans_file, cv2.IMREAD_GRAYSCALE)
        img_t = img_t[:, :, None].astype(np.float32) / 255
        # haze model
        img_lq = img_gt.astype(np.float32) / 255 * img_t + A * (1 - img_t)
        img_lq = (img_lq * 255).clip(0, 255)
        # save hazy
        hazy_folder = osp.join(args.work_dir, 'hazy/Cityscapes', split, f"{folder}_{light}_{beta}")
        os.makedirs(hazy_folder, exist_ok=True)
        dst_path = osp.join(hazy_folder, img_name + '.jpg')
        cv2.imwrite(dst_path, img_lq.astype(np.uint8))
    # print(src_path, dst_path)


if __name__ == "__main__":
    args = parse_args()
    assert osp.isdir(args.input_dir), "Please make sure there is a path to the datasets."
    assert osp.isdir(osp.join(args.input_dir, 'leftImg8bit_sequence_trainvaltest/leftImg8bit_sequence')), \
        "Please make sure leftImg8bit_sequence has been downloaded and extracted."

    print(f'\nProcessing Cityscapes...\n')

    if args.meta_file:
        assert osp.exists(args.meta_file), f"No mapping file at {args.meta_file}"
        with open(args.meta_file, 'r') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            process_cityscapes_line(line.strip())
    else:
        splits = ('train', 'val', 'test')
        lines = []
        for split in splits:
            print(f"Need the mapping file to proceed since the light information is stored in the mapping file.")
            raise NotImplementedError
            input_dir = osp.join(args.input_dir, 'leftImg8bit_sequence_trainvaltest/leftImg8bit_sequence', split)
            work_dir = osp.join(args.work_dir, 'gt/Cityscapes', split)
            _lines = process_cityscapes_raw(split, input_dir, work_dir)
            lines.extend(_lines)
        with open(osp.join(args.work_dir, 'mapping_hazeworld_cityscapes.txt'), 'w') as f:
            f.writelines(lines)
