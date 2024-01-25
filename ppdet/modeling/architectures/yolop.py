# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved. 
#
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

import ppdet.modeling
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
import paddle.nn.functional as F
import paddle
from paddle import nn
from ppdet.modeling.backbones.swin_transformer import SwinTransformer, MODEL_cfg
__all__ = ['YOLOP']


@register
class YOLOP(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']

    def __init__(self,
                 backbone='YOLOv8CSPDarkNet',
                 backbonec='YOLOv8CSPDarkNet',
                 neck='YOLOCSPPAN',
                 yolo_head='YOLOv8Head',
                 post_process='BBoxPostProcess',
                 for_mot=False):
        """
        YOLOv8

        Args:
            backbone (nn.Layer): backbone instance
            neck (nn.Layer): neck instance
            yolo_head (nn.Layer): anchor_head instance
            for_mot (bool): whether return other features for multi-object tracking
                models, default False in pure object detection models.
        """
        super(YOLOP, self).__init__()
        self.backbone = backbone
        # if backbonec=='swin_T_224':
        #     self.backbonec = SwinTransformer(**MODEL_cfg[backbonec])
        # else:
        #     self.backbonec = SwinTransformer(**MODEL_cfg[backbonec])
        self.neck = neck
        self.yolo_head = yolo_head
        self.post_process = post_process
        self.for_mot = for_mot
        # self.weight_layer = nn.Conv2D(256+512+768+512, 1, kernel_size=5, stride=5)
        # self.weight_layer = nn.Conv2D(896, 1, kernel_size=5, stride=5)
        # self.weight_layer = nn.Conv2D(128+256+512, 1, kernel_size=5, stride=5)
        # self.weight_layer = nn.Conv2D(96 + 192 + 384, 1, kernel_size=10, stride=10)

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])

        # fpn
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        # head
        kwargs = {'input_shape': neck.out_shape}
        yolo_head = create(cfg['yolo_head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "yolo_head": yolo_head,
        }

    def preprocess(self, slice_size=4, fuse=False):
        scale_factor = 1/slice_size
        self.inputs_down = self.inputs.copy()
        self.inputs_down['image'] = F.interpolate(self.inputs_down['image'],
                                                  scale_factor=scale_factor,
                                                  mode='bicubic')
        # new_bboxs = []
        # for gt_bbox in self.inputs_down['gt_bbox']:
        #     new_bbox = [pos*scale_factor for pos in gt_bbox]
        #     new_bboxs.append(new_bbox)
        if self.training:
            self.inputs_down['gt_bbox'] *= scale_factor
            self.inputs_down['im_shape'] *= scale_factor

        self.inputs_coord = self.get_coord_inputs(self.inputs, slice_size)
        if fuse:
            for k in list(self.inputs_down.keys()):
                if k not in ['epoch_id', 'num_gpus', 'curr_iter']:
                    if self.inputs_down[k].shape[1] == 0:
                        if len(self.inputs_down[k].shape)==2:
                            self.inputs_down[k] = paddle.empty([len(self.inputs_coord)+1, 0])
                        elif len(self.inputs_down[k].shape)==3:
                            self.inputs_down[k] = paddle.empty([len(self.inputs_coord)+1, 0, self.inputs_down[k].shape[2]])
                    else:
                        self.inputs_down[k] = paddle.concat([input_coord[k] for input_coord in self.inputs_coord]+[self.inputs_down[k]], axis=0)

    def get_coord_masks(self):
        # mask = paddle.zeros((2560, 2560))
        # ones = paddle.ones((640, 640))
        masks = []
        for i in range(4):
            for j in range(4):
                # temp_mask = mask.clone()
                # temp_mask[i * 640:(i + 1) * 640, j * 640:(j + 1) * 640] = ones
                # masks.append(temp_mask.unsqueeze(0).unsqueeze(0))
                masks.append([i * 640, j * 640, (i + 1) * 640, (j + 1) * 640])
        return masks

    def get_coord_extent(self, crop_size=640, stride= 640):
        """
        crop_size: size of the cropped image
        stride: stride of the cropping
        """
        coordinates = []
        for i in range(0, 2560, stride):
            for j in range(0, 2560, stride):
                coordinates.append([(i, j), (i + crop_size, j + crop_size)])
        return coordinates

    def get_coord_bbox(self, bboxs, gts, coord_ext):
        new_bboxes = []
        new_gts = []
        new_gts_patch = []
        for coord in coord_ext:
            (xs, ys), (xe, ye) = coord
            coord = paddle.Tensor(np.array([xs, ys, xs, ys]), place=bboxs.place)
            if bboxs.shape[1]==0:
                new_bboxes.append(bboxs)
                gt = gts
                new_gts.append(gt)
                # print('gt.shape', gt.shape)
                new_gt_patch = paddle.to_tensor([0], place=bboxs.place)
                new_gts_patch.append(new_gt_patch)
                # new_bboxes, new_gts, new_gts_patch
            else:
                bboxs_off = bboxs-coord

                mask_x_exceed = (bboxs_off[:, :, 0] < 1) & (bboxs_off[:, :, 2] < 1)
                mask_x_exceed = (1 - mask_x_exceed.astype('int32')).unsqueeze(-1)
                bboxs_off = bboxs_off*mask_x_exceed

                mask_y_exceed = (bboxs_off[:, :, 1] < 1) & (bboxs_off[:, :, 3] <1)
                mask_y_exceed = (1 - mask_y_exceed.astype('int32')).unsqueeze(-1)
                bboxs_off = bboxs_off*mask_y_exceed

                mask_x_exceed = (bboxs_off[:, :, 0] > 639) & (bboxs_off[:, :, 2] > 639)
                mask_x_exceed = (1 - mask_x_exceed.astype('int32')).unsqueeze(-1)
                bboxs_off = bboxs_off*mask_x_exceed

                mask_y_exceed = (bboxs_off[:, :, 1] > 639) & (bboxs_off[:, :, 3] > 639)
                mask_y_exceed = (1 - mask_y_exceed.astype('int32')).unsqueeze(-1)
                bboxs_off = bboxs_off*mask_y_exceed

                mask_less = bboxs_off > 639
                bboxs_off = paddle.where(mask_less, paddle.ones_like(bboxs_off)*640, bboxs_off)
                mask_less = bboxs_off < 1
                bboxs_off = paddle.where(mask_less, paddle.zeros_like(bboxs_off), bboxs_off)
                new_bboxes.append(bboxs_off)

                mask = paddle.sum(bboxs_off, axis=-1).unsqueeze(-1)
                mask[mask>0] = 1
                gt = gts*mask
                new_gts.append(gt)
                gt_patch = paddle.max(gt, axis=1)
                new_gt_patch = (gt_patch>0).astype('int32')
                new_gts_patch.append(new_gt_patch)
        return new_bboxes, new_gts, new_gts_patch

    def get_coord_imge(self, images, coord_ext):
        coord_images = []
        for mask_extent in coord_ext:
            coord_image = paddle.slice(images, axes=[2, 3], starts=mask_extent[0], ends=mask_extent[1])
            coord_images.append(coord_image)
        return coord_images

    def get_coord_inputs(self, inputs, slice_size):
        mask_extents = self.get_coord_extent()
        # B = self.inputs['image'].shape[0]
        inputs_coord = {}

        inputs_coord['image'] = self.get_coord_imge(self.inputs['image'], mask_extents)

        inputs_coord['im_id'] = [inputs['im_id']]*16
        inputs_coord['curr_iter'] = [inputs['curr_iter']] * 16
        inputs_coord['im_shape'] = [inputs['im_shape']/4] * 16
        inputs_coord['scale_factor'] = [inputs['scale_factor']] * 16

        if self.training:
            inputs_coord['gt_bbox'], inputs_coord['gt_class'], inputs_coord['gt_patch'] = self.get_coord_bbox(
                self.inputs['gt_bbox'], self.inputs['gt_class'], mask_extents)
            inputs_coord['pad_gt_mask'] = [inputs['pad_gt_mask']] * 16
            inputs_coord['epoch_id'] = [inputs['epoch_id']] * 16
            inputs_coord['num_gpus'] = [inputs['num_gpus']] * 16


        inputs_coord_final = []
        for i in range(16):
            input_coord_final = {}
            for k,v in inputs_coord.items():
                input_coord_final[k] = inputs_coord[k][i]
            inputs_coord_final.append(input_coord_final)
        return inputs_coord_final

    def features_fuse(self, features, dst_size):
        features_rs = []
        for feature in features:
            feature_rs = F.interpolate(feature, dst_size)
            features_rs.append(feature_rs)
        features_fuse = paddle.concat(features_rs, axis=1)
        return features_fuse

    def _forward(self):
        return self._forward_all()


    def _forward_all(self):
        self.preprocess(fuse=True)
        body_feats = self.backbone(self.inputs_down)
        neck_feats = self.neck(body_feats, self.for_mot)

        if self.training:
            yolo_losses = self.yolo_head(neck_feats, self.inputs_down)
            return yolo_losses
        else:
            yolo_head_outs = self.yolo_head(neck_feats)
            post_outs = self.yolo_head.post_process(yolo_head_outs,
                                                    # self.inputs_down['im_shape'],
                                                    self.inputs_down['scale_factor'])

            if not isinstance(post_outs, (tuple, list)):
                # if set exclude_post_process, concat([pred_bboxes, pred_scores]) not scaled to origin
                # export onnx as torch yolo models
                return post_outs
            else:
                # if set exclude_nms, [pred_bboxes, pred_scores] scaled to origin
                bbox, bbox_num = post_outs  # default for end-to-end eval/infer
                return {'bbox': bbox, 'bbox_num': bbox_num}


    def _forward_16(self):
        self.preprocess()
        down_feats = self.backbonec(self.inputs_down)
        down_feats = self.features_fuse(down_feats, down_feats[-1].shape[-2:])
        down_weight = self.weight_layer(down_feats)
        down_weight = down_weight.reshape((down_weight.shape[0], down_weight.shape[1], -1)).transpose((2, 0, 1))
        down_weight = F.sigmoid(down_weight)

        if self.training:
            yolo_losses = {
                'loss':0,
                'loss_cls':0,
                'loss_iou':0,
                'loss_dfl':0,
                'loss_l1':0,
                'loss_patch':0,
            }
            for idx in range(16):
                loss_patch = F.sigmoid_focal_loss(down_weight[idx],
                                                paddle.to_tensor(self.inputs_coord[idx]['gt_patch'].numpy().astype(np.float32), place=down_weight[idx].place),
                                                # self.inputs_coord[idx]['gt_patch'].to(down_weight[idx].place()),
                                                reduction='mean')
                yolo_losses['loss_patch'] += loss_patch

                body_feats = self.backbone(self.inputs_coord[idx])
                neck_feats = self.neck(body_feats, self.for_mot)
                yolo_loss = self.yolo_head(neck_feats, self.inputs)
                yolo_loss['loss'] += loss_patch
                # return yolo_loss
                for k,v in yolo_loss.items():
                    yolo_losses[k] += v
            return yolo_losses
        else:
            for idx in range(16):
                if down_weight[idx] > 0.5:
                    body_feats = self.backbone(self.inputs_coord[idx])
                    neck_feats = self.neck(body_feats, self.for_mot)

                    yolo_head_outs = self.yolo_head(neck_feats)
                    post_outs = self.yolo_head.post_process(yolo_head_outs,
                                                            self.inputs['im_shape'],
                                                            self.inputs['scale_factor'])

                    if not isinstance(post_outs, (tuple, list)):
                        # if set exclude_post_process, concat([pred_bboxes, pred_scores]) not scaled to origin
                        # export onnx as torch yolo models
                        return post_outs
                    else:
                        # if set exclude_nms, [pred_bboxes, pred_scores] scaled to origin
                        bbox, bbox_num = post_outs  # default for end-to-end eval/infer
                        return {'bbox': bbox, 'bbox_num': bbox_num}

        # for idx in range(16):
        #     loss_patch = F.sigmoid_focal_loss(down_weight[idx],
        #                                     paddle.to_tensor(self.inputs_coord[idx]['gt_patch'].numpy().astype(np.float32), place=down_weight[idx].place),
        #                                     # self.inputs_coord[idx]['gt_patch'].to(down_weight[idx].place()),
        #                                     reduction='mean')
        #     yolo_losses['loss_patch'] += loss_patch
        #     body_feats = self.backbone(self.inputs_coord[idx])
        #     neck_feats = self.neck(body_feats, self.for_mot)
        #
        #     if self.training:
        #         yolo_loss = self.yolo_head(neck_feats, self.inputs)
        #         yolo_loss['loss'] += loss_patch
        #         # return yolo_loss
        #         for k,v in yolo_loss.items():
        #             yolo_losses[k] += v
        #     else:
        #         yolo_head_outs = self.yolo_head(neck_feats)
        #         post_outs = self.yolo_head.post_process(yolo_head_outs,
        #                                                 self.inputs['im_shape'],
        #                                                 self.inputs['scale_factor'])
        #
        #         if not isinstance(post_outs, (tuple, list)):
        #             # if set exclude_post_process, concat([pred_bboxes, pred_scores]) not scaled to origin
        #             # export onnx as torch yolo models
        #             return post_outs
        #         else:
        #             # if set exclude_nms, [pred_bboxes, pred_scores] scaled to origin
        #             bbox, bbox_num = post_outs  # default for end-to-end eval/infer
        #             return {'bbox': bbox, 'bbox_num': bbox_num}
        # if self.training:
        #     return yolo_losses


    def _forward_select(self):
        self.preprocess()
        down_feats = self.backbonec(self.inputs_down)
        down_feats = self.features_fuse(down_feats, down_feats[-1].shape[-2:])
        down_weight = self.weight_layer(down_feats)
        down_weight = down_weight.reshape((down_weight.shape[0], down_weight.shape[1], -1)).transpose((2, 0, 1))
        down_weight = F.sigmoid(down_weight)

        if self.training:
            yolo_losses = {
                'loss':0,
                'loss_cls':0,
                'loss_iou':0,
                'loss_dfl':0,
                'loss_l1':0,
                'loss_patch':0,
            }
            for idx in range(16):
                loss_patch = F.sigmoid_focal_loss(down_weight[idx],
                                                paddle.to_tensor(self.inputs_coord[idx]['gt_patch'].numpy().astype(np.float32), place=down_weight[idx].place),
                                                # self.inputs_coord[idx]['gt_patch'].to(down_weight[idx].place()),
                                                reduction='mean')
                yolo_losses['loss_patch'] += loss_patch

                if down_weight[idx] > 0.5:
                    body_feats = self.backbone(self.inputs_coord[idx])
                    neck_feats = self.neck(body_feats, self.for_mot)
                    yolo_loss = self.yolo_head(neck_feats, self.inputs)
                    yolo_loss['loss'] += loss_patch
                    # return yolo_loss
                    for k,v in yolo_loss.items():
                        yolo_losses[k] += v
            return yolo_losses
        else:
            for idx in range(16):
                if down_weight[idx] > 0.5:
                    body_feats = self.backbone(self.inputs_coord[idx])
                    neck_feats = self.neck(body_feats, self.for_mot)

                    yolo_head_outs = self.yolo_head(neck_feats)
                    post_outs = self.yolo_head.post_process(yolo_head_outs,
                                                            self.inputs['im_shape'],
                                                            self.inputs['scale_factor'])

                    if not isinstance(post_outs, (tuple, list)):
                        # if set exclude_post_process, concat([pred_bboxes, pred_scores]) not scaled to origin
                        # export onnx as torch yolo models
                        return post_outs
                    else:
                        # if set exclude_nms, [pred_bboxes, pred_scores] scaled to origin
                        bbox, bbox_num = post_outs  # default for end-to-end eval/infer
                        return {'bbox': bbox, 'bbox_num': bbox_num}

        # for idx in range(16):
        #     loss_patch = F.sigmoid_focal_loss(down_weight[idx],
        #                                     paddle.to_tensor(self.inputs_coord[idx]['gt_patch'].numpy().astype(np.float32), place=down_weight[idx].place),
        #                                     # self.inputs_coord[idx]['gt_patch'].to(down_weight[idx].place()),
        #                                     reduction='mean')
        #     yolo_losses['loss_patch'] += loss_patch
        #     body_feats = self.backbone(self.inputs_coord[idx])
        #     neck_feats = self.neck(body_feats, self.for_mot)
        #
        #     if self.training:
        #         yolo_loss = self.yolo_head(neck_feats, self.inputs)
        #         yolo_loss['loss'] += loss_patch
        #         # return yolo_loss
        #         for k,v in yolo_loss.items():
        #             yolo_losses[k] += v
        #     else:
        #         yolo_head_outs = self.yolo_head(neck_feats)
        #         post_outs = self.yolo_head.post_process(yolo_head_outs,
        #                                                 self.inputs['im_shape'],
        #                                                 self.inputs['scale_factor'])
        #
        #         if not isinstance(post_outs, (tuple, list)):
        #             # if set exclude_post_process, concat([pred_bboxes, pred_scores]) not scaled to origin
        #             # export onnx as torch yolo models
        #             return post_outs
        #         else:
        #             # if set exclude_nms, [pred_bboxes, pred_scores] scaled to origin
        #             bbox, bbox_num = post_outs  # default for end-to-end eval/infer
        #             return {'bbox': bbox, 'bbox_num': bbox_num}
        # if self.training:
        #     return yolo_losses


    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
