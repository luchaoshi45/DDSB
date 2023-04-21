# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from datasets.coco import mixup_data
from sahi.predict import get_sliced_prediction
from util import box_ops, draw
from sahi.models.deformable_detr import Deformable_detrDetectionModel


def train_one_epoch(args, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    if args.mixup:
        samples, targets_a, targets_b, lam = mixup_data(samples, targets, alpha=1.0)

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        weight_dict = criterion.weight_dict
        #loss_dict = 0
        #losses = 0
        if args.mixup:
            loss_dict_a = criterion(outputs, targets_a)
            loss_dict_b = criterion(outputs, targets_b)
            losses_a = sum(loss_dict_a[k] * weight_dict[k] for k in loss_dict_a.keys() if k in weight_dict)
            losses_b = sum(loss_dict_b[k] * weight_dict[k] for k in loss_dict_b.keys() if k in weight_dict)
            loss_dict = {**loss_dict_a, **loss_dict_b}
            losses = lam * losses_a + (1 - lam) * losses_b
        else:
            loss_dict = criterion(outputs, targets)
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, args):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    coco_evaluator.coco_eval[iou_types[0]].params.maxDets=[1, 100, 500]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(args.output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # results = postprocessors['bbox'](outputs, orig_target_sizes)
        results = coco_postprocessors(outputs, orig_target_sizes, args=args)

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator


@torch.no_grad()
def evaluate_sahi(model, criterion, postprocessors, data_loader, base_ds, device, args):
    model.eval()
    criterion.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(['bbox'])
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    coco_evaluator.coco_eval[iou_types[0]].params.maxDets=[1, 100, 500]
    
    '''#################  sahi  ################# '''
    detection_model = Deformable_detrDetectionModel(
        model_path=args.resume,
        confidence_threshold=0.3,
        image_size = (args.img_size, args.img_size),  # （h,w）
        device=args.device, # or 'cuda:0'
        args=args)
    
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        '''outputs00 = model(samples)
        # 这里batch = 1 直接取
        # 从dataset_val 取出 图像 str 路径
        #################  sahi  ################# '''
        # 一定要每一个片都进行nms
        outputs = get_sliced_prediction(
            str(data_loader.dataset.root) + '/' + data_loader.dataset.coco.imgs[int(targets[0]['image_id'])]['file_name'],
            detection_model,
            slice_height = args.img_size,
            slice_width = args.img_size,
            overlap_height_ratio = 0.2,
            overlap_width_ratio = 0.2,
            perform_standard_pred = True,
            postprocess_type = 'NMS',
            postprocess_match_metric='IOS',
            postprocess_match_threshold=0.7,
            postprocess_class_agnostic=False,
            verbose=0,
            merge_buffer_length=1, # 每个 siled img（有检查目标存在） 都执行nms
        )
        # outputs.export_visuals(text_size=1,rect_th=1,export_dir="./",file_name='src_outputs')
        '''#################### 转化结果 ####################'''
        outputs = sahi_2_detr(outputs, targets, args)
        '''#################### 绘图 ####################'''
        # draw.show_save(outputs, data_loader.dataset, targets, outdir='runs/out/', confidence=0.5)
        draw.analyze_vis(outputs, data_loader.dataset, targets, outdir='runs/out/', confidence=0.3)
        '''#################### 计算损失 ####################'''
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        
        '''#################### mAP ####################'''
        # results = postprocessors['bbox'](outputs, orig_target_sizes)
        results = coco_postprocessors(outputs, orig_target_sizes, args=args)
        # results = coco_postprocessors_single(outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    return stats, coco_evaluator

@torch.no_grad()
def sahi_2_detr(outputs, targets, args):
    '''x1 y1 x2 y2 -> cx cy w h
    转化为标准输出
    尺度缩放到 0 - 1
    '''
    all_bbox = []
    all_logits = []
    for obj in outputs.object_prediction_list:
        all_bbox.append([(obj.bbox.minx+obj.bbox.maxx)/2., (obj.bbox.miny+obj.bbox.maxy)/2., obj.bbox.maxx-obj.bbox.minx, obj.bbox.maxy-obj.bbox.miny])
        all_logits.append(obj.out_logits.unsqueeze_(0)) 

    all_bbox = torch.tensor(all_bbox, dtype=torch.float32, device=args.device)
    # 尺度缩放到 0 - 1
    image_h_w = targets[0]['orig_size']
    image_w_h = torch.tensor([image_h_w[1], image_h_w[0]], dtype=image_h_w.dtype, device=image_h_w.device) 
        
    '''没有预测物体时处理'''
    _OBJECT_NUM = len(all_bbox)
    if _OBJECT_NUM != 0:
        all_bbox = torch.div(all_bbox, image_w_h.repeat(1, 2))
        all_logits = torch.cat(all_logits, dim=0) 
        outputs = {'pred_logits': all_logits[None,:,:], 'pred_boxes': all_bbox[None,:,:]}
    else:
        temp1 = torch.tensor([-100], dtype=torch.float32, device=args.device)
        temp2 = torch.tensor([0], dtype=torch.float32, device=args.device)
        outputs = {'pred_logits': temp1.repeat(1, 1, args.num_classes), 'pred_boxes': temp2.repeat(1, 1, 4)}
    return outputs


@torch.no_grad()
def coco_postprocessors(outputs, target_sizes, args):
    """ Perform the computation
    Parameters:
        outputs: raw outputs of the model
        target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                        For evaluation, this must be the original image size (before any data augmentation)
                        For visualization, this should be the image size after data augment, but before padding
    """

    # out_logits 1 300 11     out_bbox 1 300 4 
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
    prob = out_logits.sigmoid()
    # 选择 100 个目标
    # maxdet = out_logits.shape[1]
    MAXDET = 500

    if out_logits.shape[1]*args.num_classes < MAXDET:
        MAXDET = out_logits.shape[1]*args.num_classes 

    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), MAXDET, dim=1)
    # 得到 score,labels,boxes
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
 
    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]
    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    
    return results

@torch.no_grad()
def coco_postprocessors_single(outputs, target_sizes):
    '''专门针对单标签优化'''
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
    prob = out_logits.sigmoid()
    MAXDET = 500

    if out_logits.shape[1] < MAXDET:
        MAXDET = out_logits.shape[1] 
    # 对于只有一个标签的情况
    prob, labels = prob.max(2)
    scores, topk_indexes = torch.topk(prob, MAXDET, dim=1)
    labels = torch.gather(labels, 1, topk_indexes)
    boxes = torch.gather(out_bbox, 1, topk_indexes.unsqueeze(-1).repeat(1,1,4))
    boxes = box_ops.box_cxcywh_to_xyxy(boxes)
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]
    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    
    return results