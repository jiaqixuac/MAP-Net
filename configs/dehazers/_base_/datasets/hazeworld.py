# https://github.com/open-mmlab/mmsegmentation/blob/master/configs/_base_/datasets/ade20k.py
# https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/basicvsr/basicvsr_reds4.py

train_dataset_type = 'HWFolderMultipleGTDataset'
test_dataset_type = 'HWFolderMultipleGTDataset'  # dataset & video-level evaluation

img_norm_cfg_lq = dict(
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True,
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True,
)
img_norm_cfg_gt = dict(
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True,
    mean=[0., 0., 0.], std=[1., 1., 1.], to_rgb=True,
)
crop_size = 256
num_input_frames = 5

io_backend = 'disk'
load_kwargs = dict()

train_pipeline = [
    dict(type='GenerateFileIndices',
         interval_list=[1],
         annotation_tree_json='data/HazeWorld/train/meta_info_tree_GT_train.json'),
    dict(type='LoadImageFromFileList',
         io_backend=io_backend,
         key='lq',
         flag='unchanged',
         # channel_order='rgb',
         **load_kwargs),
    dict(type='LoadImageFromFileList',
         io_backend=io_backend,
         key='gt',
         flag='unchanged',
         **load_kwargs),
    # dict(type='LoadImageFromFileList',
    #      io_backend=io_backend,
    #      key='trans',
    #      flag='unchanged',
    #      **load_kwargs),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    # dict(type='RescaleToZeroOne', keys=['lq', 'gt', 'trans']),
    dict(type='Normalize',
         keys=['lq'],
         **img_norm_cfg_lq),
    dict(type='Normalize',
         keys=['gt'],
         **img_norm_cfg_gt),
    dict(type='PairedRandomCrop', gt_patch_size=crop_size),
    # dict(type='PairedRandomCropWithTransmission', gt_patch_size=crop_size),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
         direction='horizontal'),
    # dict(type='Flip', keys=['lq', 'gt', 'trans'], flip_ratio=0.5,
    #      direction='horizontal'),
    # dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    # dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    # dict(type='FramesToTensor', keys=['lq', 'gt', 'trans']),
    # dict(type='ToTensor', keys=['haze_light'])
    dict(type='Collect',
         keys=['lq', 'gt'],
         # keys=['lq', 'gt', 'trans', 'haze_light'],
         meta_keys=['lq_path', 'gt_path', 'dataset', 'folder', 'haze_beta', 'haze_light']),
]
test_pipeline = [
    dict(type='GenerateFileIndices',
         interval_list=[1],
         annotation_tree_json='data/HazeWorld/test/meta_info_tree_GT_test.json'),
    dict(type='LoadImageFromFileList',
         io_backend=io_backend,
         key='lq',
         flag='unchanged',
         # channel_order='rgb'
         **load_kwargs),
    dict(type='LoadImageFromFileList',
         io_backend=io_backend,
         key='gt',
         flag='unchanged',
         **load_kwargs),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize',
         keys=['lq'],
         **img_norm_cfg_lq),
    dict(type='Normalize',
         keys=['gt'],
         **img_norm_cfg_gt),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect',
         keys=['lq', 'gt'],
         meta_keys=['lq_path', 'gt_path', 'dataset', 'folder', 'haze_beta', 'haze_light']),
]

data = dict(
    workers_per_gpu=6,
    train_dataloader=dict(samples_per_gpu=4, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2),

    train=dict(
        type='RepeatDataset',
        times=10000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='data/HazeWorld/train/hazy',
            gt_folder='data/HazeWorld/train/gt',
            # trans_folder='data/HazeWorld/train/transmission',
            ann_file='data/HazeWorld/train/meta_info_GT_train.txt',
            num_input_frames=num_input_frames,
            pipeline=train_pipeline,
            test_mode=False)),
    val=dict(
        type=test_dataset_type,
        lq_folder='data/HazeWorld/test/hazy',
        gt_folder='data/HazeWorld/test/gt',
        ann_file='data/HazeWorld/test/meta_info_GT_test.txt',
        pipeline=test_pipeline,
        test_mode=False),
    test=dict(
        type=test_dataset_type,
        lq_folder='data/HazeWorld/test/hazy',
        gt_folder='data/HazeWorld/test/gt',
        ann_file='data/HazeWorld/test/meta_info_GT_test.txt',
        pipeline=test_pipeline,
        test_mode=True)
)
