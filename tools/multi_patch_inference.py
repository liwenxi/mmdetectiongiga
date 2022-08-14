import glob
import os
import pickle
import time
import warnings
from argparse import ArgumentParser
from itertools import product
from math import ceil
from pathlib import Path

import mmcv
import numpy as np
import torch
import tqdm
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
import json

from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector


def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmcv.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    if 'pretrained' in config.model:
        config.model.pretrained = None
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None
    config.model.train_cfg = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def get_multiscale_patch(sizes, steps, ratios):
    """Get multiscale patch sizes and steps.

    Args:
        sizes (list): A list of patch sizes.
        steps (list): A list of steps to slide patches.
        ratios (list): Multiscale ratios. devidie to each size and step and
            generate patches in new scales.

    Returns:
        new_sizes (list): A list of multiscale patch sizes.
        new_steps (list): A list of steps corresponding to new_sizes.
    """
    assert len(sizes) == len(steps), 'The length of `sizes` and `steps`' \
                                     'should be the same.'
    new_sizes, new_steps = [], []
    size_steps = list(zip(sizes, steps))
    for (size, step), ratio in product(size_steps, ratios):
        new_sizes.append(int(size / ratio))
        new_steps.append(int(step / ratio))
    return new_sizes, new_steps


def slide_window(width, height, sizes, steps, img_rate_thr=0.6):
    """Slide windows in images and get window position.

    Args:
        width (int): The width of the image.
        height (int): The height of the image.
        sizes (list): List of window's sizes.
        steps (list): List of window's steps.
        img_rate_thr (float): Threshold of window area divided by image area.

    Returns:
        np.ndarray: Information of valid windows.
    """
    assert 1 >= img_rate_thr >= 0, 'The `in_rate_thr` should lie in 0~1'
    windows = []
    # Sliding windows.
    for size, step in zip(sizes, steps):
        size_w, size_h = size
        step_w, step_h = step

        x_num = 1 if width <= size_w else ceil((width - size_w) / step_w + 1)
        x_start = [step_w * i for i in range(x_num)]
        if len(x_start) > 1 and x_start[-1] + size_w > width:
            x_start[-1] = width - size_w

        y_num = 1 if height <= size_h else ceil((height - size_h) / step_h + 1)
        y_start = [step_h * i for i in range(y_num)]
        if len(y_start) > 1 and y_start[-1] + size_h > height:
            y_start[-1] = height - size_h

        start = np.array(list(product(x_start, y_start)), dtype=np.int64)
        windows.append(np.concatenate([start, start + size], axis=1))
    windows = np.concatenate(windows, axis=0)

    # Calculate the rate of image part in each window.
    img_in_wins = windows.copy()
    img_in_wins[:, 0::2] = np.clip(img_in_wins[:, 0::2], 0, width)
    img_in_wins[:, 1::2] = np.clip(img_in_wins[:, 1::2], 0, height)
    img_areas = (img_in_wins[:, 2] - img_in_wins[:, 0]) * \
                (img_in_wins[:, 3] - img_in_wins[:, 1])
    win_areas = (windows[:, 2] - windows[:, 0]) * \
                (windows[:, 3] - windows[:, 1])
    img_rates = img_areas / win_areas
    if not (img_rates >= img_rate_thr).any():
        img_rates[img_rates == img_rates.max()] = 1
    return windows[img_rates >= img_rate_thr]


def merge_results(results, offsets, iou_thr=0.1, device='cpu'):
    """Merge patch results via nms.

    Args:
        results (list[np.ndarray]): A list of patches results.
        offsets (np.ndarray): Positions of the left top points of patches.
        iou_thr (float): The IoU threshold of NMS.
        device (str): The device to call nms.

    Retunrns:
        list[np.ndarray]: Detection results after merging.
    """
    assert len(results) == offsets.shape[0], 'The `results` should has the ' \
                                             'same length with `offsets`.'
    merged_results = []
    for results_pre_cls in zip(*results):
        tran_dets = []
        for dets, offset in zip(results_pre_cls, offsets):
            dets[:, :2] += offset
            dets[:, 2:4] += offset
            tran_dets.append(dets)
        tran_dets = np.concatenate(tran_dets, axis=0)

        # ************
        merged_results.append(tran_dets)
        # ************

        # if tran_dets.size == 0:
        #     merged_results.append(tran_dets)
        # else:
        #     tran_dets = torch.from_numpy(tran_dets)
        #     tran_dets = tran_dets.to(device)
        #     nms_dets, _ = nms(tran_dets[:, :4].contiguous(), tran_dets[:, -1].contiguous(),
        #                               iou_thr)
        #     merged_results.append(nms_dets.cpu().numpy())
    return merged_results


def inference_detector_by_patches(model,
                                  img,
                                  sizes,
                                  steps,
                                  ratios,
                                  merge_iou_thr,
                                  bs=10):
    """inference patches with the detector.
    Split huge image(s) into patches and inference them with the detector.
    Finally, merge patch results on one huge image by nms.
    Args:
        model (nn.Module): The loaded detector.
        img (str | ndarray or): Either an image file or loaded image.
        sizes (list): The sizes of patches.
        steps (list): The steps between two patches.
        ratios (list): Image resizing ratios for multi-scale detecting.
        merge_iou_thr (float): IoU threshold for merging results.
        bs (int): Batch size, must greater than or equal to 1.
    Returns:
        list[np.ndarray]: Detection results.
    """

    # if isinstance(img, (list, tuple)):
    #     is_batch = True
    # else:
    #     img = [img]
    #     is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    cfg = cfg.copy()
    # set loading pipeline type
    cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    if not isinstance(img, np.ndarray):
        img = mmcv.imread(img)

    height, width = img.shape[:2]
    windows = slide_window(width, height, [(6144, 3072)], [(6144 - 1000, 3072 - 1000)])


    results = []
    start = 0

    while True:
        # prepare patch data
        patch_datas = []
        if (start + bs) > len(windows):
            end = len(windows)
        else:
            end = start + bs
        for window in windows[start:end]:
            x_start, y_start, x_stop, y_stop = window
            # patch_width = x_stop - x_start
            # patch_height = y_stop - y_start
            patch = img[y_start:y_stop, x_start:x_stop]
            # prepare data

            data = dict(img=patch)

            data = test_pipeline(data)
            patch_datas.append(data)

        data = collate(patch_datas, samples_per_gpu=len(patch_datas))
        # just get the actual data from DataContainer
        data['img_metas'] = [
            img_metas.data[0] for img_metas in data['img_metas']
        ]
        data['img'] = [img.data[0] for img in data['img']]
        if next(model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            for m in model.modules():
                assert not isinstance(
                    m, RoIPool
                ), 'CPU inference with RoIPool is not supported currently.'

        # forward the model
        with torch.no_grad():
            results.extend(model(return_loss=False, rescale=True, **data))

        if end >= len(windows):
            break
        start += bs

    results = merge_results(
        results,
        windows[:, :2],
        iou_thr=merge_iou_thr,
        device=device)
    return results


def parse_args():
    parser = ArgumentParser()
    # parser.add_argument('img_path', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--patch_sizes',
        type=int,
        nargs='+',
        default=[1024],
        help='The sizes of patches')
    parser.add_argument(
        '--patch_steps',
        type=int,
        nargs='+',
        default=[824],
        help='The steps between two patches')
    parser.add_argument(
        '--img_ratios',
        type=float,
        nargs='+',
        default=[1.0],
        help='Image resizing ratios for multi-scale detecting')
    parser.add_argument(
        '--merge_iou_thr',
        type=float,
        default=0.3,
        help='IoU threshould for merging results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):
    all_result = []
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a huge image by patches

    root = "/media/wzh/wxli/PANDA/images_test"

    paths = glob.glob(os.path.join(root, '*jpg'))
    paths.sort()
    for img in tqdm.tqdm(paths):
        result = inference_detector_by_patches(model, img, args.patch_sizes,
                                               args.patch_steps, args.img_ratios,
                                               args.merge_iou_thr)

        all_result.append(result)



    with open('./person_bbox_test.json') as f:
        gt = json.load(f)
    name_id_map = {}
    for name in gt.keys():
        short_name = os.path.basename(name)
        name_id_map[short_name] = gt[name]['image id']
    # print(name_id_map)

    coco_result = []
    root = "./images_test"
    # root = "/data3/wxli/panda/images_test"

    paths = glob.glob(os.path.join(root, '*jpg'))
    paths.sort()
    for img, boxes in zip(paths, all_result):
        name = os.path.basename(img)
        for item in boxes[0]:
            # print(item.shape)
            item = item.astype(np.float64)
            box = {}
            box['image_id'] = name_id_map[name]
            box['category_id'] = 2
            box['bbox'] = [item[0], item[1], item[2]-item[0], item[3]-item[1]]
            box['score'] = item[4]
            coco_result.append(box)
        for item in boxes[1]:
            # print(item.shape)
            item = item.astype(np.float64)
            box = {}
            box['image_id'] = name_id_map[name]
            box['category_id'] = 1
            box['bbox'] = [item[0], item[1], item[2]-item[0], item[3]-item[1]]
            box['score'] = item[4]
            coco_result.append(box)
    with open("cascade_sw_2class_blur.json", "w") as f:
        save_data = json.dumps(coco_result, indent=4)
        f.write(save_data)


if __name__ == '__main__':
    args = parse_args()
    main(args)
