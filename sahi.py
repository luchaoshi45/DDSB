from sahi.models.deformable_detr import *
# import required functions, classes
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
import os

MODEL_PATH = 'checkpoints/Deformable-DETR-Resnet50-epoch49.pth'
MODEL_PATH = 'checkpoints/DDSA-SAHI-80+1.pth'
detection_model = Deformable_detrDetectionModel(
    model_path=MODEL_PATH,
    confidence_threshold=0.5,
    image_size = (448, 448),  # （h,w）
    device="cuda" # or 'cuda:0'
)

jpg = ['demo_data/'+ pth for pth in os.listdir('demo_data/') if '.jpg' in pth]
i = 0
for jpg in jpg:
    i = i+1
    result = get_sliced_prediction(
        jpg,
        detection_model,
        slice_height = 448,
        slice_width = 448,
        overlap_height_ratio = 0.2,
        overlap_width_ratio = 0.2,
        postprocess_type = 'GREEDYNMM',
        postprocess_match_metric='IOU',
        postprocess_match_threshold=0.1,
        postprocess_class_agnostic=True,
    )
    result.export_visuals(export_dir="demo_data/out/",text_size=1, rect_th=1,file_name=str(i))
    

