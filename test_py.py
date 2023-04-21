'''
import argparse
def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--lr_min', default=2e-7, type=float) # 控制setp()函数实现 epoch!=last_epoch
    # ***** 1 batch_size *****
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    # ***** 2 epochs *****
    parser.add_argument('--epochs', default=100, type=int)
    # ***** 3 lr_drop *****
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    
    # ***** use_wass *****
    parser.add_argument('--distance_loss', default='l1', type=str)
    parser.add_argument('--iou_loss', default='giou', type=str)

    # 优化器
    parser.add_argument('--optimizer', default='AdamW', type=str)
    parser.add_argument('--img_size', default=896, type=int)

    # mixup
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--augment', default='None', type=str)

    # ***** 1 backbone *****
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    # ***** 2 coco_path *****
    parser.add_argument('--coco_path', default='../dataset/VisDrone_coco/', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    # ***** 3 output_dir *****  默认 'runs/' + args.backbone
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    # ***** 4 resume *****
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # ***** 5 start_epoch ***** 会被 checkpoint 覆盖
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # **************************************不常修改**********************************************
    

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--ciou_loss_coef', default=2, type=float) # 这里可以尝试
    parser.add_argument('--wass_loss_coef', default=6, type=float) # 这里可以尝试
    parser.add_argument('--kld_loss_coef', default=6, type=float) # 这里可以尝试
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # **************************************不常修改**********************************************
    return parser




from sahi.slicing import *
from datasets import build_dataset, get_coco_api_from_dataset
from torch.utils.data import DataLoader
import torch
import util.misc as utils
from datasets.data_prefetcher import data_prefetcher
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    dataset_val= build_dataset(image_set='val', args=args)

    sampler_val= torch.utils.data.RandomSampler(dataset_val)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_val, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_val, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    print_freq = 100
    prefetcher = data_prefetcher(data_loader_train, device, prefetch=True)
    #samples, targets = prefetcher.next()






    # image_height, image_width = 1080, 1920
    # image_height, image_width = 480, 360
    # image_height, image_width = 540, 960
    samples, targets = prefetcher.next()
    img = samples.tensors[0]
    mask = samples.mask[0]
    tar =  targets[0]['boxes']
    #targets[0].

    image_height, image_width = img.shape[1], img.shape[2]
    slice_height, slice_width = 448, 448
    overlap_height_ratio, overlap_width_ratio = 0.2, 0.2
    bk = get_slice_bboxes(image_height, image_width,
                          slice_height, slice_width,
                          overlap_height_ratio, overlap_width_ratio)

    #dataset_train
    #coco 格式 xywh
    kk = torch.tensor([image_width, image_height, image_width, image_height], device=device) #x1y1x2y2
    tar_unscale = kk*tar

    re = annotation_inside_slice({'bbox':tar_unscale[0]}, bk[0])
    a = 1
'''
'''
from sahi.models.deformable_detr import *

# import required functions, classes
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict


detection_model = Deformable_detrDetectionModel(
    model_path='checkpoints/Deformable-DETR-Resnet50-epoch80.pth',
    confidence_threshold=0.5,
    image_size = (896, 896),  # （h,w）
    device="cuda" # or 'cuda:0'
)

result = get_sliced_prediction(
    "demo_data/0000360_00001_d_0000713.jpg",
    detection_model,
    slice_height = 896,
    slice_width = 896,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)

result.export_visuals(export_dir="demo_data/")

# Image("demo_data/prediction_visual.png")

object_prediction_list = result.object_prediction_list
print(object_prediction_list[0])
'''
'''
import cv2 
img = cv2.imread('runs/analyze/DDSA-res_query-25_loss.png')

cv2.imwrite('DDSA-res_query-25_loss.png', img)
'''

import numpy as np
import matplotlib.pyplot as plt
#创建数据
x = np.linspace(0, 1, 100)

x1 = x
x2 = (1 - x)
y1 = np.log(x1/x2)

plt.plot(x, y1)

#显示出所有设置
plt.savefig("inverse_sigmoid.png")