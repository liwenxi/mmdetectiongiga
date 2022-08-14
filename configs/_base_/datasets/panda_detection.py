# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(2048, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

classes = ('full body', 'vehicles')

# data = dict(
#     samples_per_gpu=1,
#     workers_per_gpu=10,
#     train=dict(
#         type=dataset_type,
#         classes=classes,
#         ann_file='/ssd1_2T/panda/patch_train.json',
#         img_prefix='/ssd1_2T/panda/patch_16',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         classes=classes,
#         ann_file='/ssd1_2T/panda/patch_val.json',
#         img_prefix='/ssd1_2T/panda/patch_16',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         classes=classes,
#         ann_file='/ssd1_2T/panda/patch_val.json',
#         img_prefix='/ssd1_2T/panda/patch_16',
#         pipeline=test_pipeline))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        # ann_file='/home/wenxi/panda/patch_train.json',
        # img_prefix='/home/wenxi/panda/patch_16',
        # ann_file='/ssd1_2T/wxli/train_lin_full_v.json',
        # img_prefix='/ssd1_2T/wxli/patch_lin',
        # ann_file='/ssd1_2T/wxli/mmdetection/train_4x4.json',
        # img_prefix='/ssd1_2T/wxli/patch_4x4',
        # ann_file='/home/wenxi/panda/train_mix.json',
        # img_prefix='/home/wenxi/panda/patch_mix',
        ann_file='/ssd1_2T/wxli/tracking_train.json',
        img_prefix='/ssd1_2T/wxli/track_patch',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        # ann_file='/home/wenxi/panda/patch_val.json',
        # img_prefix='/home/wenxi/panda/patch_val_16',
        # ann_file='/ssd1_2T/wxli/mmdetection/val_4x4.json',
        # img_prefix='/ssd1_2T/wxli/patch_4x4',
        # ann_file='/home/wenxi/panda/val_mix.json',
        # img_prefix='/home/wenxi/panda/patch_mix',
        # ann_file='/ssd1_2T/wxli/val_lin_full_v.json',
        # img_prefix='/ssd1_2T/wxli/patch_lin',
        ann_file='/ssd1_2T/wxli/tracking_val.json',
        img_prefix='/ssd1_2T/wxli/track_patch',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/home/wenxi/panda/val_16.json',
        img_prefix='/home/wenxi/panda/patch_val_16',
        pipeline=test_pipeline),
    test_giga=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/home/wenxi/panda/val.json',
        img_prefix='/home/wenxi/panda/images',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')
