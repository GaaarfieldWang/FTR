import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from sam2.build_sam import build_sam2_camera_predictor, build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class SAM2Model:
    def __init__(self, args, cuda_idx=0):
        self.image_crop_size = args.image_crop_size
        self.masked_region_num = args.masked_region_num
        self.max_area = args.max_area
        self.min_area = args.min_area
        self.clip_range = args.clip_range
        self.reverse_sort = args.reverse_sort
        self.color_type = args.color_type
        self.add_original_frame = args.add_original_frame

        # Use sam2 as the mask generator
        sam2 = build_sam2(args.sam2_config, args.sam2_model_path, device="cuda:{}".format(str(cuda_idx)), apply_postprocessing=False)
        sam2.eval()
        self.mask_generator = SAM2AutomaticMaskGenerator(sam2, pred_iou_thresh=args.pred_iou_thresh,
                                                        stability_score_thresh=args.stability_score_thresh,
                                                        points_per_side=args.points_per_side,
                                                        points_per_batch=args.points_per_batch)
        
        # Use sam2 as the mask predictor
        self.camera_predictor = build_sam2_camera_predictor(args.sam2_config, args.sam2_model_path, device="cuda:{}".format(str(cuda_idx)))

        self.plot = args.plot_segment
        self.timer = args.segment_timer

    def mask_filter(self, masks, image, initial_flag=False, overlap_threshold=0.95):
        if len(masks) == 0:
            print('No mask provided in automatic mask generator!')
            return []

        masks = sorted(masks, key=(
            lambda x: x['area']), reverse=self.reverse_sort)
        filtered_masks = []
        for mask in masks:
            if mask['area'] > self.max_area or mask['area'] < self.min_area:
                continue
            if np.sum(mask['segmentation'][self.clip_range[0]: self.clip_range[1],
                      self.clip_range[0]: self.clip_range[1]]) == 0:
                continue

            contain_flag = False
            for previous_mask in filtered_masks:
                if np.sum(mask['segmentation'] * previous_mask['segmentation']) > overlap_threshold * np.sum(
                        mask['segmentation']):
                    contain_flag = True
                    break
            if contain_flag:
                continue

            filtered_masks.append(mask)

        if len(filtered_masks) == 0:
            print('No mask detected in automatic mask generator! Use white mask instead.')
            white_mask = copy.deepcopy(masks[0])
            white_mask['segmentation'] = np.ones_like(
                white_mask['segmentation'])
            filtered_masks = filtered_masks + [white_mask]

        if len(filtered_masks) > self.masked_region_num:
            filtered_masks = filtered_masks[:self.masked_region_num]
        elif len(filtered_masks) < self.masked_region_num:
            black_mask = copy.deepcopy(masks[0])
            black_mask['segmentation'] = np.zeros_like(
                black_mask['segmentation'])
            filtered_masks = filtered_masks + \
                             [black_mask] * (self.masked_region_num - len(filtered_masks))

        if self.add_original_frame:
            white_mask = copy.deepcopy(masks[0])
            white_mask['segmentation'] = np.ones_like(
                white_mask['segmentation'])
            filtered_masks = filtered_masks + [white_mask]

        return filtered_masks

    def _generate_image_mask(self, image, initial_flag=False, dtype=np.uint8):
        start = time.time()
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        image = image.astype(np.uint8)
        assert image.shape[0] in [1, 3], "Image can only be gray or rgb"
        if image.shape[0] == 1:
            image = np.concatenate([image, image, image], axis=0)
        image = np.transpose(image, [1, 2, 0])
        # input image of mask_generator.generate: np.array, shape = (h, w, c)
        masks = self.mask_generator.generate(image)
        # masks[0]['segmentation']: np.array, shape = (84, 84)
        masks = self.mask_filter(masks, image, initial_flag)
        # len(masks) = self.masked_region_num + self.add_original_frame
        ret_masks = []
        for i in range(self.masked_region_num):
            ret_masks.append(masks[i]['segmentation'])

        if len(masks) == 0:
            print("No mask detected!")
            plt.imshow(image)
            plt.savefig("./figures/alert.jpg")
            if self.color_type == "rgb":
                black_image = np.zeros((3, self.image_crop_size, self.image_crop_size))
                total_images = [image.transpose((2, 0, 1))] + [black_image] * (self.masked_region_num - 1)
                if self.add_original_frame:
                    total_images = total_images + [image.transpose((2, 0, 1))]
                total_images = np.concatenate(total_images, axis=0).astype(dtype)
                return total_images
            elif self.color_type == "gray":
                black_image = np.zeros((1, self.image_crop_size, self.image_crop_size))
                gray_image = image.mean(axis=2).astype("uint8")
                total_images = [np.expand_dims(gray_image, axis=0)] + [black_image] * (self.masked_region_num - 1)
                if self.add_original_frame:
                    total_images = total_images + [np.expand_dims(gray_image, axis=0)]
                total_images = np.concatenate(total_images).astype(dtype)
                return total_images

        if self.color_type == "rgb":
            total_masks = []
            for item in masks:
                m = np.expand_dims(item["segmentation"], axis=0)
                m = np.expand_dims(np.concatenate(
                    [m for _ in range(3)], axis=0), axis=0)
                total_masks.append(m)
            total_masks = np.concatenate(total_masks, axis=0)
            total_images = np.concatenate(
                [np.expand_dims(image.transpose((2, 0, 1)), axis=0) for _ in range(len(masks))])
            masked_images = (total_masks * total_images).reshape(len(masks) * 3, self.image_crop_size,
                                                                 self.image_crop_size)
            zero_position_sum = np.sum(
                masked_images.reshape((3, self.masked_region_num + 1 if self.add_original_frame else self.masked_region_num, self.image_crop_size, self.image_crop_size))[:, 0, :, :], axis=(0, 1, 2))
            if zero_position_sum == 0:
                print("zero position is black")
                masked_images[:3, :, :] = image.transpose((2, 0, 1))
                plt.imshow(masked_images[:3, :, :].transpose(1, 2, 0))
                plt.savefig("./figures/zero.jpg")
        elif self.color_type == "gray":
            total_masks = []
            for item in masks:
                m = np.expand_dims(item["segmentation"], axis=0)
                total_masks.append(m)
            total_masks = np.concatenate(total_masks, axis=0)
            gray_image = image.mean(axis=2).astype("uint8")
            total_images = np.concatenate(
                [np.expand_dims(gray_image, axis=0) for _ in range(len(masks))])
            masked_images = total_masks * total_images

        if self.timer:
            print(time.time() - start)
        return masked_images.astype(dtype), np.stack(ret_masks).astype(np.bool8)
    
    def segment_image(self, image):
        return self._generate_image_mask(image)
    
    # Below are modules in sam2
    def load_first_frame(self, img):
        '''
        img: np.array, shape = (h, w, c)
        '''
        return self.camera_predictor.load_first_frame(img)
    
    def add_new_masks(self, frame_idx, obj_ids, masks):
        '''
        frame_idx: int
        obj_ids: list of int
        masks: list of mask, mask: shape = (h, w)

        return: out_frame_idx, out_obj_ids, out_video_res_masks
        out_frame_idx: int
        out_obj_ids: list of int, ids of tracking objects
        out_video_res_masks: torch.Tensor, shape = (len(out_obj_ids), 1, h, w)
        '''
        assert len(obj_ids) == len(masks), "objects and masks must be the same len"
        out_frame_idx, out_obj_ids, out_video_res_masks = 0, [], []
        for obj_id, mask in zip(obj_ids, masks):
            out_frame_idx, out_obj_ids, out_video_res_masks = self.camera_predictor.add_new_mask(frame_idx, obj_id, mask)
        return out_frame_idx, out_obj_ids, out_video_res_masks
    
    def track(self, img):
        '''
        img: np.array, shape = (h, w, c)

        return: out_obj_ids, out_video_res_masks
        out_obj_ids: list of int, ids of tracking objects
        out_video_res_masks: np.array, shape = (len(out_obj_ids), h, w), dtype=np.bool8
        '''
        out_obj_ids, out_video_res_masks = self.camera_predictor.track(img)
        # out_video_res_masks: torch.Tensor, shape = (len(out_obj_ids), 1, h, w), dtype=torch.float32
        # to np.array, shape = (len(out_obj_ids), h, w), dtype=np.bool8
        out_video_res_masks = out_video_res_masks.squeeze(1).cpu().numpy()
        out_video_res_masks = np.where(out_video_res_masks < 0, 0, 1) 
        return out_obj_ids, out_video_res_masks.astype(np.bool8)