# optimizer
optimizers = dict(
    generator=dict(
        type='AdamW',
        lr=0.0002,
        betas=(0.9, 0.999),
    )
)

# learning policy
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=1e-7,
    by_epoch=False)

# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['L1', 'PSNR', 'SSIM'], crop_border=0)

visual_config = None
