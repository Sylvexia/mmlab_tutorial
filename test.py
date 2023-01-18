from mmdet.apis import init_detector, inference_detector
from mmdet.models import BaseDetector

# config_file = 'checkpoints/yolov3_mobilenetv2_320_300e_coco.py'
# checkpoint_file = 'checkpoints/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth'

# config_file = 'checkpoints/yolox_tiny_8x8_300e_coco.py'
# checkpoint_file = 'checkpoints/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'

model = init_detector(config_file, checkpoint_file,
                      device='cuda:0')  # device='cuda:0' or 'cpu'

model : BaseDetector # get the model attributes

result = inference_detector(model, 'image/cats.jpg')

model.show_result('image/cats.jpg', result, show=True) # show result in a new window

model.show_result('image/cats.jpg', result, out_file='result.jpg') # or save the visualization results to image files