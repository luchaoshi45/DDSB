import cv2, os
import argparse
from main import get_args_parser
from datasets import build_dataset
from util.draw import vis_one_img


def id2ann(img_id: int):
    '''根据 imgid 找到 图像 和 标签'''
    img_name = dataset_val.coco.loadImgs(img_id)[0]['file_name']
    img_path = str(dataset_val.root) + '/' + img_name
    img = cv2.imread(img_path)
    annIds = dataset_val.coco.getAnnIds(imgIds=img_id,iscrowd=None)
    anns = dataset_val.coco.loadAnns(annIds)
    bbox_list = [ann['bbox'] for ann in anns] # len(bbox) 4 list
    label_list = [ann['category_id'] for ann in anns]  #len(label)  list
    return img, bbox_list, label_list

def vis_dataset(dataset, out_pth: str='vis_coco'):
    '''绘制数据集'''
    for img_id in dataset.ids:
        # 绘制单张图像
        img, bbox_list, label_list = id2ann(img_id)
        img = vis_one_img(img, bbox_list, label_list, putText=True)
        # 保存图像
        if not os.path.exists(out_pth):
            os.mkdir(out_pth)
        cv2.imwrite(out_pth + '/' + dataset_val.coco.loadImgs(img_id)[0]['file_name'], img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR', parents=[get_args_parser()])
    args = parser.parse_args()
    dataset_val = build_dataset(image_set='val', args=args)
    vis_dataset(dataset_val,out_pth='runs/vis_coco')
    '''
    img_id = 1
    out_pth = 'vis_coco'
    img, bbox_list, label_list = id2ann(img_id)
    img = vis_one_img(img, bbox_list, label_list)

    if not os.path.exists(out_pth):
            os.mkdir(out_pth)
    cv2.imwrite( out_pth + '/' + dataset_val.coco.loadImgs(img_id)[0]['file_name'], img)
    '''
 