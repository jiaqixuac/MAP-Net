"""
The code to create symlinks for train and test.
Also create the meta info.
"""
import argparse
import os
import os.path as osp
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Generate symlinks for train and test')
    parser.add_argument('-d', '--dataset', help='which dataset to process',
                        default='all')
    parser.add_argument('--root-dir', help='the dir to store train and test symlinks',
                        default='./data/HazeWorld')
    parser.add_argument('--gt-dir', help='the dir that contains the ground truth images',
                        default='./data/HazeWorld/gt')
    parser.add_argument('--hazy-dir', help='the dir that contains synthetic hazy images',
                        default='./data/HazeWorld/hazy')
    parser.add_argument('--trans-dir', help='the dir that contains transmission maps',
                        default='./data/HazeWorld/transmission')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    root = osp.realpath(args.root_dir)
    gt_dir = osp.realpath(args.gt_dir)
    hazy_dir = osp.realpath(args.hazy_dir)
    trans_dir = osp.realpath(args.trans_dir)

    print(f"\nHazeWorld root:\t\t{root}")
    print(f"GT dir:\t\t\t{gt_dir}")
    print(f"Hazy dir:\t\t{hazy_dir}")
    print(f"Transmission dir:\t{trans_dir}\n")

    if args.dataset == 'all':
        datasets = os.listdir(gt_dir)
        datasets.sort()
        datasets = [x for x in datasets if x in ('Cityscapes', 'DDAD', 'UA-DETRAC', 'VisDrone', 'DAVIS', 'REDS')]
    else:
        datasets = [args.dataset]

    for folder in ('gt', 'hazy', 'transmission'):
        os.makedirs(osp.join(root, f'train/{folder}'), exist_ok=True)
        os.makedirs(osp.join(root, f'test/{folder}'), exist_ok=True)

    meta_info_train = []
    meta_info_test = []

    for dataset in datasets:
        print(f'Processing dataset...: {dataset}')
        with open(osp.join(gt_dir, dataset, 'mapping_info_GT_train.txt'), 'r') as f:
            train_lines = f.readlines()
            train_lines = [x.strip() for x in train_lines]
        with open(osp.join(gt_dir, dataset, 'mapping_info_GT_test.txt'), 'r') as f:
            test_lines = f.readlines()
            test_lines = [x.strip() for x in test_lines]

        for folder in ('gt', 'hazy', 'transmission'):
            os.makedirs(osp.join(root, f'train/{folder}', dataset), exist_ok=True)
            os.makedirs(osp.join(root, f'test/{folder}', dataset), exist_ok=True)

        split_mapping = {}
        split_folders = {}  # only for checking

        for lines in (train_lines, test_lines):
            for line in lines:
                src_split, dst_split, folder, cnt = line.split()
                if src_split not in split_mapping:
                    split_mapping[src_split] = dst_split
                    split_folders[dst_split] = []
                else:
                    assert split_mapping[src_split] == dst_split, \
                        f"{dataset}-{src_split}: {split_mapping[src_split]} vs {dst_split}"
                    assert folder not in split_folders[dst_split]
                split_folders[dst_split].append(folder)

        splits = os.listdir(osp.join(hazy_dir, dataset))
        splits = [x for x in splits if osp.isdir(osp.join(hazy_dir, dataset, x))]
        for split in splits:
            dst_split = split_mapping[split]
            folders = os.listdir(osp.join(hazy_dir, dataset, split))
            folders.sort()
            for folder in folders:
                beta = folder.split('_')[-1]

                num_files = len(os.listdir(osp.join(hazy_dir, dataset, split, folder)))  # num files in hazy

                # link gt folder
                prefix = '_'.join(folder.split('_')[:-2])
                src = osp.join(gt_dir, dataset, split, prefix)
                dst = osp.join(root, dst_split, 'gt', dataset, folder)
                assert osp.isdir(src), f"No gt dir: {src}"
                assert len(os.listdir(src)) == num_files, f'{src}: {len(os.listdir(src))}, {num_files}'
                if not osp.isdir(dst):
                    os.symlink(src, dst)

                # link hazy folder
                src = osp.join(hazy_dir, dataset, split, folder)
                dst = osp.join(root, dst_split, 'hazy', dataset, folder)
                assert osp.isdir(src), f"No hazy dir: {src}"
                assert len(os.listdir(src)) == num_files, f'{src}: {len(os.listdir(src))}, {num_files}'
                if not osp.isdir(dst):
                    os.symlink(src, dst)

                # link transmission folder
                prefix = '_'.join(folder.split('_')[:-2]) + f'_{beta}'
                src = osp.join(trans_dir, dataset, split, prefix)
                dst = osp.join(root, dst_split, 'transmission', dataset, folder)
                assert osp.isdir(src), f"No transmission dir: {src}"
                assert len(os.listdir(src)) == num_files, f'{src}: {len(os.listdir(src))}, {num_files}'
                if not osp.isdir(dst):
                    os.symlink(src, dst)

        train_folders = os.listdir(osp.join(root, 'train/gt', dataset))
        train_folders.sort()
        test_folders = os.listdir(osp.join(root, 'test/gt', dataset))
        test_folders.sort()
        print(f"[{dataset:10s} (ori)]\ttotal: {(len(train_folders) + len(test_folders)) / 4},"
              f"\ttrain: {len(train_folders) / 4},\ttest: {len(test_folders) / 4}")

        train_meta = []
        test_meta = []
        for folder in train_folders:
            files = os.listdir(osp.join(root, 'train/gt', dataset, folder))
            line = f"{dataset}/{folder} {len(files)}\n"
            train_meta.append(line)
        for folder in test_folders:
            files = os.listdir(osp.join(root, 'test/gt', dataset, folder))
            line = f"{dataset}/{folder} {len(files)}\n"
            test_meta.append(line)
        meta_info_train.extend(train_meta)
        meta_info_test.extend(test_meta)

        with open(osp.join(root, 'train', f'meta_info_GT_{dataset}.txt'), 'w') as f:
            f.writelines(train_meta)
        with open(osp.join(root, 'test', f'meta_info_GT_{dataset}.txt'), 'w') as f:
            f.writelines(test_meta)

    print(f"\n[HazeWorld Train]\t{len(meta_info_train)}")
    print(f"[HazeWorld Test ]\t{len(meta_info_test)}")
    with open(osp.join(root, 'train', f'meta_info_GT_train.txt'), 'w') as f:
        f.writelines(meta_info_train)
    with open(osp.join(root, 'test', f'meta_info_GT_test.txt'), 'w') as f:
        f.writelines(meta_info_test)
