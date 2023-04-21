import cv2,os
import numpy as np
import random
import torch
from torchvision.ops.boxes import batched_nms
from PIL import Image
import time
from datasets.coco import get_dataset_classes

class BbxScoLab(object):
    def __init__(self, boxes=None, scores=None, labels=None) -> None:
        self.boxes = boxes if boxes is not None else []
        self.scores = scores if scores is not None else []
        self.labels = labels if labels is not None else []

    def append(self, boxe, score, label):
        self.boxes.append(boxe)
        self.scores.append(score)
        self.labels.append(label)

    def pop(self,index):
        return self.boxes.pop(index), self.scores.pop(index), self.labels.pop(index)
    
    def get_val(self, index):
        return self.boxes[index], self.scores[index], self.labels[index]
    
    def get_attr(self): 
        return self.boxes, self.scores, self.labels
    
    def len(self):
        return len(self.boxes)   

def out2post(outputs, targets, to_numpy=True):
    '''输出转化为后处理格式'''
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
    # sigmoid 可以多分类的
    prob = out_logits.sigmoid()

    MAXDET = 500
    num_classes = len(get_dataset_classes())+1
    if out_logits.shape[1]*num_classes < MAXDET:
        MAXDET = out_logits.shape[1]*num_classes 
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), MAXDET, dim=1)
    # 得到 score,labels,boxes
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = box_cxcywh_to_xyxy(out_bbox,-1)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
    img_h, img_w = targets[0]['orig_size'].unbind(-1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
    boxes = boxes * scale_fct[None, None, :]

    if to_numpy:
        boxes = boxes[0].cpu().numpy()
        scores = scores[0].cpu().numpy()
        labels = labels[0].cpu().numpy()
    else:
        boxes = boxes[0]
        scores = scores[0]
        labels = labels[0]

    return boxes, scores , labels

def score_filter(boxes, scores, labels, confidence=0.3):
    bboxes = []
    bscores = []
    blabels = []
    for boxes,scores,labels in zip(boxes, scores, labels):
        if scores >= confidence:
            bboxes.append(boxes)
            bscores.append(scores)
            blabels.append(labels)
    # return np.array(bboxes), np.array(bscores), np.array(blabels) 
    return bboxes, bscores, blabels 

def iou(box0: np.ndarray, box1: np.ndarray):
    # x y x y 
    xy_max = np.minimum(box0[2:], box1[2:])
    xy_min = np.maximum(box0[:2], box1[:2])

    # 计算交集
    inter = np.clip(xy_max-xy_min, a_min=0, a_max=np.inf)
    inter = inter[0]*inter[1]

    # 计算并集
    area_0 = (box0[2]-box0[0])*(box0[3]-box0[1])
    area_1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    union = area_0 + area_1- inter
    return inter/union

def iou_match(pre_bsl, tar_bsl, iou_th=0.5):
    '''iou 阈值匹配 gt pre'''
    tp_bsl = BbxScoLab()
    fp_bsl = BbxScoLab()
    for i in range(pre_bsl.len()):
        boxe, score, label = pre_bsl.get_val(i)
        tp_bsl_len = tp_bsl.len()
        for j in range(tar_bsl.len()):
            tar_boxe, tar_score, tar_label = tar_bsl.get_val(j)
            if iou(boxe, tar_boxe) >= iou_th and (label == tar_label):
                tp_bsl.append(boxe, score, label) # tp_bsl 保存tp
                tar_bsl.pop(j) # tar_bsl 去除gt 已经匹配了不可以匹配低置信度 只会执行一次
                break # 跳出循环
        if tp_bsl.len() - tp_bsl_len == 0: # 没有匹配到
            fp_bsl.append(boxe, score, label)
    return tp_bsl, fp_bsl ,tar_bsl

@torch.no_grad()
def plot_one_box(x, img, color=None, label=None, line_thickness=1):
    '''绘制单个框'''
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color if color is not None else [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # color = [0, 255, 0]
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, color=[255,255,255], thickness=tf, lineType=cv2.LINE_AA)

def plot_one_img(image, boxes, scores, labels, color=None, putText=True):
    image = np.array(image)
    if len(labels) != 0:
        for box, score, label in zip(boxes,scores,labels):
            if putText: 
                label = get_dataset_classes()[label-1]
                text = f"{label} {score:.3f}"
            else:
                text = None
            plot_one_box(box, image, color=color, label=text)
    return image

def save_specified_imgs(img_path, tp_bsl, fp_bsl, tar_bsl, outdir='runs/out/', specified=None):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    image = Image.open(img_path)
    image_name = img_path.split('/')[-1]

    MD_DIR = ['TP', 'FP', 'FN', 'TP+FP', 'TP+FN', 'FP+FN', 'TP+FP+FN'] if specified is None else specified
    DIRS = [outdir+midd+'/' for midd in MD_DIR]
    for dir in DIRS:
        if not os.path.exists(dir):
            os.mkdir(dir)

    image_TP = plot_one_img(image, tp_bsl.boxes, tp_bsl.scores, tp_bsl.labels, color=(0,255,0))
    image_FP = plot_one_img(image, fp_bsl.boxes, fp_bsl.scores, fp_bsl.labels, color=(255,0,0))
    image_FN = plot_one_img(image, tar_bsl.boxes, tar_bsl.scores, tar_bsl.labels, color=(0,0,255), putText=False)

    image_TP_FP = plot_one_img(image_TP, fp_bsl.boxes, fp_bsl.scores, fp_bsl.labels, color=(255,0,0))
    image_TP_FN = plot_one_img(image_TP, tar_bsl.boxes, tar_bsl.scores, tar_bsl.labels, color=(0,0,255), putText=False)
    image_FP_FN = plot_one_img(image_FP, tar_bsl.boxes, tar_bsl.scores, tar_bsl.labels, color=(0,0,255), putText=False)
    image_TP_FP_FN = plot_one_img(image_TP_FP, tar_bsl.boxes, tar_bsl.scores, tar_bsl.labels, color=(0,0,255), putText=False)

    Image.fromarray(image_TP).save(DIRS[0] + image_name)
    Image.fromarray(image_FP).save(DIRS[1] + image_name)
    Image.fromarray(image_FN).save(DIRS[2] + image_name)
    Image.fromarray(image_TP_FP).save(DIRS[3] + image_name)
    Image.fromarray(image_TP_FN).save(DIRS[4] + image_name)
    Image.fromarray(image_FP_FN).save(DIRS[5] + image_name)
    Image.fromarray(image_TP_FP_FN).save(DIRS[6] + image_name)

@torch.no_grad()
def analyze_vis(outputs, dataset_val, targets, outdir='runs/out/', confidence=0.3, iou_th=0.5):
    img_path = str(dataset_val.root) + '/' + dataset_val.coco.imgs[int(targets[0]['image_id'])]['file_name']

    # pre 格式转化 缩放
    boxes, scores, labels = out2post(outputs, targets)
    # 滤除置信度小于 confidence 的框
    boxes, scores, labels = score_filter(boxes, scores, labels, confidence)
    pre_bsl = BbxScoLab(boxes, scores, labels)

    # tar 格式转化 缩放
    img_h, img_w = targets[0]['orig_size'].unbind(-1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
    tar_bsl = BbxScoLab(list((box_cxcywh_to_xyxy(targets[0]['boxes'] ,-1) * scale_fct[None,:]).cpu().numpy()),
                        list(np.ones(len(targets[0]['labels']),dtype=np.float32)),
                        list(targets[0]['labels'].cpu().numpy()))
    
    # iou 阈值匹配 gt pre
    tp_bsl, fp_bsl ,tar_bsl = iou_match(pre_bsl, tar_bsl, iou_th=iou_th)

    specified = ['TP', 'FP', 'FN', 'TP+FP', 'TP+FN', 'FP+FN', 'TP+FP+FN']
    save_specified_imgs(img_path, tp_bsl, fp_bsl, tar_bsl, outdir=outdir, specified=specified)



@torch.no_grad()
def show_save(outputs, dataset_val, targets, outdir='runs/out/', confidence=0.3, apply_nms=False, iou=0.5):
    # 只是支持 batch = 1
    # dataset_val + targets 确定图像路径
    '''绘图 _OBJECT_NUM != '''
    ims_path = str(dataset_val.root) + '/' + dataset_val.coco.imgs[int(targets[0]['image_id'])]['file_name']
    image = Image.open(ims_path)
    image_h, image_w = targets[0]['orig_size'].unbind(-1)

    if (outputs['pred_logits']).shape[2] != 0:
        probas = outputs['pred_logits'].softmax(-1)[0, :, :].cpu()
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0,].cpu(),
                                            (image_w, image_h))
        scores, boxes = filter_boxes(probas, bboxes_scaled, confidence=confidence, apply_nms=apply_nms, iou=iou)
        scores = scores.data.numpy()
        boxes = boxes.data.numpy()
            
        for i in range(boxes.shape[0]):
            class_id = scores[i].argmax()
            label = get_dataset_classes()[class_id-1]
            confidence = scores[i].max()
            text = f"{label} {confidence:.3f}"
            #print(text)
            image = np.array(image)
            plot_one_box(boxes[i], image, label=text)
        if boxes.shape[0] != 0:
            image = Image.fromarray(image)
            
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    image.save(outdir + dataset_val.coco.imgs[int(targets[0]['image_id'])]['file_name'])


@torch.no_grad()
def show_save_one_img(image_item, image, image_tensor, inference_result, args):
    '''显示单张图像'''
    # 是用 0-1-10 11个  1 - 10 
    probas = inference_result['pred_logits'].softmax(-1)[0, :, :].cpu()
    bboxes_scaled = rescale_bboxes(inference_result['pred_boxes'][0,].cpu(),
                                        (image_tensor.shape[3], image_tensor.shape[2]))
    scores, boxes = filter_boxes(probas, bboxes_scaled, confidence=0.5)
    scores = scores.data.numpy()
    boxes = boxes.data.numpy()
    for i in range(boxes.shape[0]):
        class_id = scores[i].argmax()
        label = get_dataset_classes()[class_id-1]
        confidence = scores[i].max()
        text = f"{label} {confidence:.3f}"
        print(text)
        image = np.array(image)
        plot_one_box(boxes[i], image, label=text)
                
    #cv2.imshow("images", cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    #cv2.waitKey()
    if boxes.shape[0] != 0:
        image = Image.fromarray(image)
    image.save(os.path.join(args.output_dir, image_item))


@torch.no_grad()
def box_cxcywh_to_xyxy(x,dim=1):
    x_c, y_c, w, h = x.unbind(dim)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=dim)

@torch.no_grad()
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

@torch.no_grad()
def filter_boxes(scores, boxes, confidence=0.7, apply_nms=True, iou=0.5):
    '''使用 nms confidence 过滤框'''
    keep = scores.max(-1).values > confidence
    scores, boxes = scores[keep], boxes[keep]

    if apply_nms:
        top_scores, labels = scores.max(-1)
        keep = batched_nms(boxes, top_scores, labels, iou)
        scores, boxes = scores[keep], boxes[keep]
    return scores, boxes


@torch.no_grad()
def get_dataset_color():
    '''获得数据集类别的颜色'''
    calss = get_dataset_classes()
    color_list = []
    for i in (range(len(calss))):
        add = 255//len(calss) * i
        start = [(random.randint(0, 255)+add) % 255 for _ in range(3)]
        start = [(50+add) % 255 , (100+add) % 255 , (20+add) % 255 ]
        color_list.append(start)
    return color_list

@torch.no_grad()
def vis_one_img(img: np.array, bbox_list: list, label_list: list=None, putText: bool=True):
    '''绘制单张图像'''
    # 获得颜色
    if label_list is None:
        color = [random.randint(0, 255) for _ in range(3)]
    else:
        color_list = get_dataset_color()
    # 绘图
    for i in range(len(bbox_list)):
        bbox = bbox_list[i]
        bbox = [bbox[i] if i<2 else bbox[i]+bbox[i-2] for i in range(4)]
        if label_list is not None:
            color = color_list[label_list[i]-1]
            label = get_dataset_classes()[label_list[i]-1] if putText else None
        plot_one_box(bbox, img, color, label)
    return img