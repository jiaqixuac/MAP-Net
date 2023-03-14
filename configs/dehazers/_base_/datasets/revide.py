# from https://github.com/open-mmlab/mmsegmentation/blob/master/configs/_base_/datasets/ade20k.py
# https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/basicvsr/basicvsr_reds4.py

train_dataset_type = 'SRFolderMultipleGTDataset'
test_dataset_type = 'SRFolderMultipleGTDataset'

img_norm_cfg_lq = dict(
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True,
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True,
)
img_norm_cfg_gt = dict(
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True,
    mean=[0., 0., 0.], std=[1., 1., 1.], to_rgb=True,
)
crop_size = 384
num_input_frames = 3

io_backend = 'disk'
load_kwargs = dict()

train_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1], filename_tmpl='{:05d}.JPG'),
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
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='ResizeVideo', keys=['lq', 'gt'], scales=[0.25, 0.375, 0.5, 0.625, 0.75], sample=False),
    dict(type='Normalize',
         keys=['lq'],
         **img_norm_cfg_lq),
    dict(type='Normalize',
         keys=['gt'],
         **img_norm_cfg_gt),
    dict(type='PairedRandomCrop', gt_patch_size=crop_size),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
         direction='horizontal'),
    # by jqxu: do not rotation
    # dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    # dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path', 'key']),
]
test_pipeline = [
    # folder-based
    dict(type='GenerateSegmentIndices', interval_list=[1], filename_tmpl='{:05d}.JPG'),
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
    dict(type='ResizeVideo', keys=['lq'], scales=0.5),
    dict(type='Normalize',
         keys=['lq'],
         **img_norm_cfg_lq),
    dict(type='Normalize',
         keys=['gt'],
         **img_norm_cfg_gt),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'lq_path', 'key']),
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
            lq_folder='data/REVIDE/REVIDE_indoor/Train/hazy',
            gt_folder='data/REVIDE/REVIDE_indoor/Train/gt',
            ann_file='data/REVIDE/REVIDE_indoor/Train/meta_info_GT.txt',
            num_input_frames=num_input_frames,
            pipeline=train_pipeline,
            img_extension='.JPG',
            scale=1,
            test_mode=False)),
    val=dict(
        type=test_dataset_type,
        lq_folder='data/REVIDE/REVIDE_indoor/Test/hazy',
        gt_folder='data/REVIDE/REVIDE_indoor/Test/gt',
        ann_file='data/REVIDE/REVIDE_indoor/Test/meta_info_GT.txt',
        pipeline=test_pipeline,
        img_extension='.JPG',
        scale=1,
        test_mode=True),
    test=dict(
        type=test_dataset_type,
        lq_folder='data/REVIDE/REVIDE_indoor/Test/hazy',
        gt_folder='data/REVIDE/REVIDE_indoor/Test/gt',
        ann_file='data/REVIDE/REVIDE_indoor/Test/meta_info_GT.txt',
        pipeline=test_pipeline,
        img_extension='.JPG',
        scale=1,
        test_mode=True)
)
