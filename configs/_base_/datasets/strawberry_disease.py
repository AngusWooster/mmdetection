# dataset settings
dataset_type = 'StrawberryDiseaseDataset'
data_root = 'data/strawberry_disease/2023_v1/'
TRAINING_DATA_ROOT = data_root + 'train'
VALIDATION_DATA_ROOT = data_root + 'valid'
TEST_DATA_ROOT = data_root + 'test'

BATCH_SIZE = 20
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

train_dataloader = dict(batch_size=BATCH_SIZE,
                        num_workers=2,
                        persistent_workers=True,
                        drop_last=False,
                        sampler=dict(type='DefaultSampler', shuffle=True),
                        batch_sampler=dict(type='AspectRatioBatchSampler'),
                        dataset=dict(type=dataset_type,
                                     data_root=TRAINING_DATA_ROOT,
                                     ann_file = 'annotation.txt',
                                     pipeline=train_pipeline))

val_dataloader = dict(batch_size=BATCH_SIZE,
                      num_workers=2,
                      persistent_workers=True,
                      drop_last=False,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=dict(type=dataset_type,
                                   data_root=VALIDATION_DATA_ROOT,
                                   ann_file='annotation.txt',
                                   test_mode=True,
                                   pipeline=test_pipeline))

test_dataloader = dict(batch_size=BATCH_SIZE,
                       num_workers=2,
                       persistent_workers=True,
                       drop_last=False,
                       sampler=dict(type='DefaultSampler', shuffle=False),
                       dataset=dict(type=dataset_type,
                                    data_root=TEST_DATA_ROOT,
                                    ann_file='annotation.txt',
                                    test_mode=True,
                                    pipeline=test_pipeline))

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator
