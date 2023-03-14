# https://github.com/open-mmlab/mmediting/blob/master/mmedit/datasets/pipelines/augmentation.py
import os
import os.path as osp
import random

import numpy as np
import mmcv

from ..registry import PIPELINES


@PIPELINES.register_module()
class GenerateFileIndices:
    """Generate frame indices for the HazeWorld dataset. It also performs temporal
    augmentation with random interval.
    Note that GenerateFileIndices does not rename original frame names to filename_tmpl

    Required keys: lq_path, gt_path, key, num_input_frames, sequence_length
    Added or modified keys:  lq_path, gt_path, interval, reverse

    Args:
        interval_list (list[int]): Interval list for temporal augmentation.
            It will randomly pick an interval from interval_list and sample
            frame index with the interval.
        annotation_tree_json(int): Annotation tree for loading.
    """

    def __init__(self,
                 interval_list,
                 annotation_tree_json=None):
        self.interval_list = interval_list
        self.annotation_tree = mmcv.fileio.load(annotation_tree_json) \
            if annotation_tree_json is not None else None

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        # key example: 'DAVIS/aerobatics_242_0.005'
        clip_name = results['key']
        interval = np.random.choice(self.interval_list)

        self.sequence_length = results['sequence_length']
        num_input_frames = results.get('num_input_frames',
                                       self.sequence_length)

        # randomly select a frame as start
        if self.sequence_length - num_input_frames * interval < 0:
            raise ValueError('The input sequence is not long enough to '
                             'support the current choice of [interval] or '
                             '[num_input_frames].')
        start_frame_idx = np.random.randint(
            0, self.sequence_length - num_input_frames * interval + 1)
        end_frame_idx = start_frame_idx + num_input_frames * interval
        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # add the corresponding file paths
        lq_path_root = results['lq_path']
        gt_path_root = results['gt_path']
        if self.annotation_tree is not None:
            frames = self.annotation_tree[clip_name]
        else:
            frames = os.listdir(osp.join(lq_path_root, clip_name))
        frames.sort()
        lq_path = [
            osp.join(lq_path_root, clip_name, str(frames[v]))
            for v in neighbor_list
        ]
        gt_path = [
            osp.join(gt_path_root, clip_name, str(frames[v]))
            for v in neighbor_list
        ]

        results['lq_path'] = lq_path
        results['gt_path'] = gt_path
        results['interval'] = interval

        tm_path_root = results[f'trans_path']
        if tm_path_root != 'None':
            tm_path = []
            for v in neighbor_list:
                basename, ext = osp.splitext(osp.basename(frames[v]))
                tm_path.append(osp.join(tm_path_root, clip_name, str(basename) + '.png'))
            results[f'trams_path'] = tm_path

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(interval_list={self.interval_list})')
        return repr_str


@PIPELINES.register_module()
class ResizeVideo:
    """Resize the 720p image to 480p"""

    def __init__(self,
                 keys,
                 scales=None,
                 sample=False,
                 interpolation='bilinear',
                 backend=None):
        assert keys, 'Keys should not be empty.'
        self.keys = keys
        if not isinstance(scales, (list, tuple)):
            scales = [scales]
        self.scales = scales
        self.sample = sample
        if sample:
            assert scales is not None
            self.scales = [np.min(scales), np.max(scales)]
        self.interpolation = interpolation
        self.backend = backend

    def _resize(self, img, scale=None):
        h, w, _ = img.shape
        img, self.scale_factor = mmcv.imrescale(
            img,
            scale,
            return_scale=True,
            interpolation=self.interpolation,
            backend=self.backend)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        return img

    def __call__(self, results):
        if self.sample:
            scale = np.random.uniform(self.scales[0], self.scales[1])
        else:
            scale = random.choice(self.scales)
        for key in self.keys:
            if isinstance(results[key], list):
                results[key] = [
                    self._resize(v, scale) for v in results[key]
                ]
            else:
                results[key] = self._resize(results[key], scale)

        results['scale_factor'] = self.scale_factor
        results['interpolation'] = self.interpolation
        results['backend'] = self.backend

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(keys={self.keys}, size_factor={self.size_factor}, '
            f'interpolation={self.interpolation})')
        return repr_str
