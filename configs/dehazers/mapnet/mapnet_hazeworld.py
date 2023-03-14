_base_ = [
    '../_base_/datasets/hazeworld.py',
    '../_base_/default_runtime.py', './mapnet_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

exp_name = 'mapnet_hazeworld_40k'

checkpoint = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'  # noqa

# model settings
model = dict(
    type='MAP',
    generator=dict(
        type='MAPNet',
        backbone=dict(
            type='ConvNeXt',
            arch='tiny',
            out_indices=[0, 1, 2, 3],
            drop_path_rate=0.0,
            layer_scale_init_value=1.0,
            gap_before_final_norm=False,
            init_cfg=dict(type='Pretrained', checkpoint=checkpoint, prefix='backbone.'),
        ),
        neck=dict(
            type='ProjectionHead',
            in_channels=[96, 192, 384, 768],
            out_channels=64,
            num_outs=4
        ),
        upsampler=dict(
            type='MAPUpsampler',
            embed_dim=32,
            num_feat=32,
        ),
        channels=32,
        num_trans_bins=32,
        align_depths=(1, 1, 1, 1),
        num_kv_frames=[1, 2, 3],
    ),

    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
)

data = dict(
    train_dataloader=dict(samples_per_gpu=2, drop_last=True),
)

# runtime settings
work_dir = f'./work_dirs/{exp_name}'
