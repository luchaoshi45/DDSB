# OBSS SAHI Tool
# Code written by MiraBit, 2023.

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.import_utils import check_package_minimum_version, check_requirements

import argparse
import numpy as np
import torch
import datasets.samplers as samplers
from models import build_model
import torchvision.transforms as T
from util import box_ops

logger = logging.getLogger(__name__)

from datasets.coco import get_dataset_classes


class Deformable_detrDetectionModel(DetectionModel):

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[Any] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        category_mapping: Optional[Dict] = None,
        category_remapping: Optional[Dict] = None,
        load_at_init: bool = True,
        image_size: int = None,
        args = None,
    ):
        if args is not None:
            self.args = args
        else:
            pass
        super().__init__(
            model_path,
            model,
            config_path,
            device,
            mask_threshold,
            confidence_threshold,
            category_mapping,
            category_remapping,
            load_at_init,
            image_size,
        )

    def check_dependencies(self) -> None:
        check_requirements(["torch"])

    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """
        try:
           
            model, criterion, postprocessors = build_model(self.args)
            model.to(self.device)
            checkpoint = torch.load(self.model_path, map_location='cpu')
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
            unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
            if len(missing_keys) > 0:
                print('Missing Keys: {}'.format(missing_keys))
            if len(unexpected_keys) > 0:
                print('Unexpected Keys: {}'.format(unexpected_keys))
            self.set_model(model)

        except Exception as e:
            raise TypeError("model_path is not a valid Deformable_detrDetectionModel model path: ", e)

    def set_model(self, model: Any):
        """
        Sets the underlying Deformable_detrDetectionModel model.
        Args:
            model: Any
                A Deformable_detrDetectionModel model
        """

        if model.__class__.__module__ not in ["models.deformable_detr", "models.common"]:
            raise Exception(f"Not a yolov5 model: {type(model)}")

        model.conf = self.confidence_threshold
        self.model = model

        # set category_mapping
        if self.category_mapping is None:
            category_names = {str(i): get_dataset_classes()[i] for i in range(len(get_dataset_classes()))}
            self.category_mapping = category_names

    @torch.no_grad()
    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """

        ######################################################################################
        # 放前面 后面有 resize 全图 resize 了  记录的是最原始的尺度 没有经过管道
        self.IMG_ZOOM_SCALE = np.array(image.shape[0:2]) # 与 self.image 不同 因为 有 全图推理
        ######################################################################################

        # use dataset and use batch
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")
        
        if False:
            pass
        # else str 读取单张图像 分割预测   
        else:
            # 为了兼容性 不对 sahi 工程代码修改，对图像的预处理 在这里进行
            # (w, h)
            pipeline = T.Compose([
                # T.RandomResize(self.image_size, max_size=1333), 
                T.ToTensor(),
                T.Resize(self.image_size),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image = pipeline(image)
            image = image.to(self.device)
            
            
            image = torch.reshape(image, [-1, image.shape[0], image.shape[1], image.shape[2]])
            prediction_result = self.model(image)

        self._original_predictions = prediction_result
        
        

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return len(self.category_mapping)

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        return False

    @property
    def category_names(self):
        return list(self.category_mapping.values())

    @torch.no_grad()
    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions
        '''
        temp1 = torch.tensor([-10,1.,-10,-10,-10,-10,-10,-10,-10,-10,-10], dtype=torch.float32, device='cuda')
        temp2 = torch.tensor([0.2,0.2,0.1,0.1], dtype=torch.float32, device='cuda')
        original_predictions = {'pred_logits': temp1.repeat(1, 100, 1), 'pred_boxes': temp2.repeat(1, 100, 1)}
        '''
        # compatilibty for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        # handle all predictions
        ##########################################################
        ''' object_prediction_list_per_image 存放全部的运算结果 '''
        ''' 首先执行detr的两个函数 然后执行偏移 '''
        ##########################################################
        object_prediction_list_per_image = []

        '''
        # 注意我使用 softmax detr 使用 sigmoid
        all_bbox =   original_predictions['pred_boxes'] # b(1) n 4   
        all_prob, all_label =  original_predictions['pred_logits'].softmax(-1).max(-1) # n 1     n 1
        # all_bbox.shape[1]
        SLICED_MAXDET = 11
        topk_values, topk_indexes  = torch.topk(all_prob, SLICED_MAXDET, dim=1) # 对置信度排序

        # 过滤100个目标中置信度  小于阈值目标  的score和index
        values = []
        indexes = []
        for v,i in zip(topk_values[0], topk_indexes[0]):
            if v > self.confidence_threshold:
                values.append(v)
                indexes.append(i)
        topk_values, topk_indexes = torch.tensor(values)[None,:].to(self.device), torch.tensor(indexes)[None,:].to(self.device)

        topk_label = torch.gather(all_label, 1, topk_indexes)
        topk_boxes = torch.gather(all_bbox, 1, topk_indexes.unsqueeze(-1).repeat(1,1,4)) # repeat 要维度一致
        topk_boxes = box_ops.box_cxcywh_to_xyxy(topk_boxes)
        self.out_logits = torch.gather(original_predictions['pred_logits'], 1, topk_indexes.unsqueeze(-1).repeat(1,1,original_predictions['pred_logits'].shape[2]))
        
        # and from relative [0, 1] to absolute [0, height] coordinates
        target_sizes = torch.tensor(np.array([self.IMG_ZOOM_SCALE])).to(self.device)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        topk_boxes = topk_boxes * scale_fct[:, None, :]
        results = torch.cat((topk_boxes, topk_values[:,:,None], topk_label[:,:,None]), 2)
        '''

        
        # out_logits 1 300 11     out_bbox 1 300 4 
        out_logits, out_bbox = original_predictions['pred_logits'], original_predictions['pred_boxes']
        prob = out_logits.sigmoid()

        # 选择 100 个目标
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        
        # 过滤100个目标中置信度  小于阈值目标  的score和index
        values = []
        indexes = []
        for v,i in zip(topk_values[0], topk_indexes[0]):
            if v > self.confidence_threshold:
                values.append(v)
                indexes.append(i)
        topk_values, topk_indexes = torch.tensor(values)[None,:].to(self.device), torch.tensor(indexes)[None,:].to(self.device)
        
        # 得到 score,labels,boxes
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        # 保存 选择好的 out_logits b c_slect 11
        self.out_logits = torch.gather(out_logits, 1, topk_boxes.unsqueeze(-1).repeat(1,1,out_logits.shape[2]))
 
        # and from relative [0, 1] to absolute [0, height] coordinates
        target_sizes = torch.tensor(np.array([self.IMG_ZOOM_SCALE])).to(self.device)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        results = torch.cat((boxes, scores[:,:,None], labels[:,:,None]), 2)
        

        for image_ind, image_predictions_in_xyxy_format in enumerate(results):
            out_logits = self.out_logits[image_ind]

            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            object_prediction_list = []


            
            # process predictions,
            for prediction, one_out_logits in zip(image_predictions_in_xyxy_format.cpu().detach().numpy(),out_logits):
                x1 = prediction[0]
                y1 = prediction[1]
                x2 = prediction[2]
                y2 = prediction[3]
                bbox = [x1, y1, x2, y2]
                score = prediction[4]
                category_id = int(prediction[5])
                category_id = category_id - 1 ############ 舍弃 0 标号
                category_name = self.category_mapping[str(category_id)]

                # fix negative box coords
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = max(0, bbox[2])
                bbox[3] = max(0, bbox[3])

                # fix out of image box coords
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                # ignore invalid predictions
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                    continue

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    bool_mask=None,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                    out_logits = one_out_logits
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image


