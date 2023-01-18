_base_ = ['mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py']

dataset_type = 'CocoDataset'
classes = ('T', 'E', 'L', 'C', 'F')
img_prefix = '/home/sylvex/train/JPEGImages'
ann_file = '/home/sylvex/train/JPEGImages/output.json'
num_classes=5

data = dict(
    train=dict(
        classes=classes,
        ann_file=ann_file,
        img_prefix=img_prefix),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=ann_file,
        img_prefix=img_prefix),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=ann_file,
        img_prefix=img_prefix))

model = dict(
    roi_head=dict(
        bbox_head=dict(
                type='Shared2FCBBoxHead',
                num_classes=num_classes)))