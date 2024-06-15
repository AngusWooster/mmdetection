_base_ = './sparse-rcnn_r50_cbam_fpn_DiiHeadAdapter_voc.py'

model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))