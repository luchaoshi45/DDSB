import numpy as np
import random
from PIL import Image
import torch
# from datasets.coco import CLASSES

class No_Augmentation(object):
    def __init__(self):
        pass
    def __call__(self, img, target):

        # show(img, target['boxes'], target['labels'], 0)
        return img, target
    
class SmallObjectAugmentation(object):
    def __init__(self, thresh=64*64, q_thresh = 16*16,prob=0.5, copy_times=3, epochs=30, all_objects=False, one_object=False):
        """
        sample = {'img':img, 'annot':annots}
        img = [height, width, 3]
        annot = [xmin, ymin, xmax, ymax, label]
        thresh：the detection threshold of the small object. If annot_h * annot_w < thresh, the object is small
        prob: the prob to do small object augmentation
        epochs: the epochs to do
        """
        self.thresh = thresh
        self.q_thresh = q_thresh
        self.prob = prob
        self.copy_times = copy_times
        self.epochs = epochs
        self.all_objects = all_objects
        self.one_object = one_object
        # if self.all_objects or self.one_object:
            # self.copy_times = 1

    def issmallobject(self, h, w):
        if h * w <= self.thresh:
            return True
        else:
            return False
        
    def isqualityobject(self, h, w):
        if h * w >= self.q_thresh:
            return True
        else:
            return False
    def issmall_rara(self, h, w, label):
        INDEX = [3, 7, 8] # bicycle tricycle awning-tricycle 
        return self.issmallobject(h, w) and label in INDEX and self.isqualityobject(h, w)


    def compute_overlap(self, annot_a, annot_b):
        if annot_a is None: return False
        left_max = max(annot_a[0], annot_b[0])
        top_max = max(annot_a[1], annot_b[1])
        right_min = min(annot_a[2], annot_b[2])
        bottom_min = min(annot_a[3], annot_b[3])
        inter = max(0, (right_min-left_max)) * max(0, (bottom_min-top_max))
        if inter != 0:
            return True
        else:
            return False

    def donot_overlap(self, new_annot, annots):
        for annot in annots:
            if self.compute_overlap(new_annot, annot): return False
        return True

    def create_copy_annot(self, h, w, annot, bboxs): 
        '''
        annot 一维列表 float64 copy_annot
        bboxs 二维列表 float64 all_annot
        '''
        annot = list(map(int, annot))
        annot_h, annot_w = annot[3] - annot[1], annot[2] - annot[0]
        for epoch in range(self.epochs):
            # 随机中心的坐标
            random_x, random_y = np.random.randint(int(annot_w / 2), int(w - annot_w / 2)), \
                                 np.random.randint(int(annot_h / 2), int(h - annot_h / 2))
            # 目标的边界的(x1=xmin, y1=ymin, x2=xmax, y2=ymax)
            xmin, ymin = random_x - annot_w / 2, random_y - annot_h / 2
            xmax, ymax = xmin + annot_w, ymin + annot_h
            if xmin < 0 or xmax > w or ymin < 0 or ymax > h:
                continue
            # new_annot = np.array([xmin, ymin, xmax, ymax].astype(np.int64)
            new_annot = list(map(int, [xmin, ymin, xmax, ymax]))

            if self.donot_overlap(new_annot, bboxs) is False:
                continue

            return new_annot
        return None

    def add_patch_in_img(self, annot, copy_annot, image):
        copy_annot = list(map(int, copy_annot))
        image = np.array(image)
        image[annot[1]:annot[3], annot[0]:annot[2], :] = image[copy_annot[1]:copy_annot[3], copy_annot[0]:copy_annot[2], :]
        # Image.fromarray(image).save('new.jpg')
        return Image.fromarray(image)

    def __call__(self, img, annots):
        if self.all_objects and self.one_object: return img, annots
        if np.random.rand() > self.prob: return img, annots

        # img, annots = sample['img'], sample['annot']
        h, w= img.size[1], img.size[0]  #  注意   这里是PIL （W,H）  np和coco 标注 是（H，W）

        small_object_list = list()
        for idx in range(annots['boxes'].shape[0]):
            annot = annots['boxes'][idx]
            annot_h, annot_w = annot[3] - annot[1], annot[2] - annot[0]
            if self.issmall_rara(annot_h, annot_w, annots['labels'][idx]):
                small_object_list.append(idx)

        l = len(small_object_list)
        # No Small Object
        if l == 0: return img, annots

        # Refine the copy_object by the given policy
        # Policy 2:
        copy_object_num = np.random.randint(0, l)
        # Policy 3:
        if self.all_objects:
            copy_object_num = l
        # Policy 1:
        if self.one_object:
            copy_object_num = 1

        random_list = random.sample(range(l), copy_object_num)
        annot_idx_of_small_object = [small_object_list[idx] for idx in random_list]
        # select_annots = annots[annot_idx_of_small_object, :]  # 这里需要修改
        select_annots = annots['boxes'][annot_idx_of_small_object, :]

        annots['boxes'] = annots['boxes'].tolist()  # boxes labels area iscrowd
        annots['labels'] = annots['labels'].tolist()  # boxes labels area iscrowd 
        annots['area'] = annots['area'].tolist()  # boxes labels area iscrowd
        annots['iscrowd'] = annots['iscrowd'].tolist()  # boxes labels area iscrowd

        for idx in range(copy_object_num):
            # annot_idx_of_small_object[idx]
            annot = select_annots[idx].tolist()
            annot_h, annot_w = annot[3] - annot[1], annot[2] - annot[0]

            if self.issmallobject(annot_h, annot_w) is False: continue

            for i in range(self.copy_times):
                
                new_annot = self.create_copy_annot(h, w, annot, annots['boxes'])
                if new_annot is not None:
                    img = self.add_patch_in_img(new_annot, annot, img)
                    
                    annots['boxes'].append(new_annot)
                    annots['labels'].append(annots['labels'][annot_idx_of_small_object[idx]])
                    annots['area'].append(annots['area'][annot_idx_of_small_object[idx]])
                    annots['iscrowd'].append(annots['iscrowd'][annot_idx_of_small_object[idx]])

        # return {'img': img, 'annot': np.array(annots)}
        annots['boxes'] = torch.Tensor(annots['boxes'])
        annots['labels'] = torch.LongTensor(annots['labels'])
        annots['area'] = torch.LongTensor(annots['area'])
        annots['iscrowd'] = torch.LongTensor(annots['iscrowd'])
        

        # show(img, annots['boxes'], annots['labels'], copy_object_num*self.copy_times)
        # 对类别增强
        # 什么是小目标，coco 评估中如何定义的
        return img, annots