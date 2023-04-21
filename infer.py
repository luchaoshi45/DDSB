import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch
from models import build_model
from PIL import Image
import os
import torchvision
from datasets.coco import get_dataset_classes
from util.draw import *

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)

    # 检测的图像路径
    source_dir = 'demo/images'
    source_dir = '../dataset/VisDrone2019-DET/VisDrone2019-DET-val/images/'
    parser.add_argument('--source_dir', default=source_dir,
                        help='path where to save, empty for no saving')
    # 检测结果保存路径
    parser.add_argument('--output_dir', default='demo/outputs',
                        help='path where to save, empty for no saving')

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
    parser.add_argument('--sahi', action='store_true')
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--augment', default='None', type=str)
    # 这里 0 index 是不希望被使用的类别
    parser.add_argument('--num_classes', default=11, type=int)

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
    #parser.add_argument('--output_dir', default='',
    #                    help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    # ***** 4 resume *****
    parser.add_argument('--resume', default='checkpoints/Deformable-DETR-Resnet50-epoch80.pth', help='resume from checkpoint')
    # ***** 5 start_epoch ***** 会被 checkpoint 覆盖'
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default="True")
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


def main(args):
    print(args)
    device = torch.device(args.device)
    model, criterion, postprocessors = build_model(args)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'],False)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("parameters:", n_parameters)

    image_Totensor = torchvision.transforms.ToTensor()
    image_file_path = os.listdir(args.source_dir)

    for image_item in image_file_path:
        print("inference_image:", image_item)
        image_path = os.path.join(args.source_dir, image_item)
        image = Image.open(image_path)
        #WIDTH = 800
        #KKKK = image.size[1] / image.size[0]
        #image = image.resize((WIDTH, int(KKKK*WIDTH)), Image.LANCZOS) #resize image with high-quality
        FIX = 896
        image = image.resize((FIX, FIX), Image.LANCZOS) #resize image with high-quality
        image_tensor = image_Totensor(image)
        image_tensor = torch.reshape(image_tensor,
                                    [-1, image_tensor.shape[0], 
                                    image_tensor.shape[1], 
                                    image_tensor.shape[2]])
        image_tensor = image_tensor.to(args.device)
        time1 = time.time()
        inference_result = model(image_tensor)
        time2 = time.time()
        print("inference_time:", time2 - time1)
        show_save_one_img(image_item, image, image_tensor, inference_result, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)