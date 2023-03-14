# https://github.com/open-mmlab/mmsegmentation/blob/master/configs/_base_/schedules/schedule_80k.py
# https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/basicvsr/basicvsr_reds4.py
# # optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# optimizer_config = dict()
# # learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
total_iters = 80000
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=8000, save_image=False, gpu_collect=True)
