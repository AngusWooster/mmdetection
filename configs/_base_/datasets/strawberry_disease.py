# dataset settings
dataset_type = 'StrawberryDiseaseDataset'
data_root = 'data/strawberry_disease/2023_v1/'
data_root_training = data_root + 'train'
data_root_validation = data_root + 'valid'
data_root_test = data_root + 'test'

backend_args = None

img_scale = (640, 640)  # VGA resolution

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape','scale_factor'))
]

train_dataloader = dict(batch_size=2,
                        num_workers=2,
                        persistent_workers=True,
                        drop_last=False,
                        sampler=dict(type='DefaultSampler', shuffle=True),
                        batch_sampler=dict(type='AspectRatioBatchSampler'),
                        dataset=dict(type=dataset_type,
                                     data_root=data_root_training,
                                     ann_file = 'annotation.txt',
                                     pipeline=train_pipeline))

val_dataloader = dict(batch_size=2,
                      num_workers=2,
                      persistent_workers=True,
                      drop_last=False,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=dict(type=dataset_type,
                                   data_root=data_root_validation,
                                   ann_file='annotation.txt',
                                   test_mode=True,
                                   pipeline=test_pipeline))

test_dataloader = dict(batch_size=2,
                      num_workers=2,
                      persistent_workers=True,
                      drop_last=False,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=dict(type=dataset_type,
                                   data_root=data_root_test,
                                   ann_file='annotation.txt',
                                   test_mode=True,
                                   pipeline=test_pipeline))

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator
