# https://github.com/open-mmlab/mmediting/blob/master/mmedit/datasets/sr_folder_multiple_gt_dataset.py
import os
import os.path as osp
from collections import defaultdict

import mmcv
from mmedit.utils import get_root_logger

from .base_dh_dataset import BaseDHDataset
from .registry import DATASETS

AVAILABLE_DATASETS = ('Cityscapes', 'DDAD', 'UA-DETRAC', 'VisDrone', 'DAVIS', 'REDS')
AVAILABLE_BETAS = (0.005, 0.01, 0.02, 0.03)
AVAILABLE_METRICS = ('L1', 'PSNR', 'SSIM')  # the same as in models


@DATASETS.register_module()
class HWFolderMultipleGTDataset(BaseDHDataset):
    """General dataset for video dehazing, used for recurrent networks.

    The dataset loads several LQ (Low-Quality) frames and GT (Ground-Truth)
    frames. Then it applies specified transforms and finally returns a dict
    containing paired data and other information.

    This dataset takes an annotation file specifying the sequences used in
    training or test. If no annotation file is provided, it assumes all video
    sequences under the root directory is used for training or test.

    In the annotation file (.txt), each line contains:

        1. folder name;
        2. number of frames in this sequence (in the same folder)

    Examples:

    ::

        Cityscapes/aachen_0000_218_0.005 30
        DAVIS/bear_239_0.005 82
        REDS/000_192_0.005 100
        UA-DETRAC/MVI_20011_229_0.005 95
        DDAD/000000_236_0.005 50
        VisDrone/uav0000013_00000_v_222_0.005 90

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        pipeline (list[dict | callable]): A sequence of data transformations.
        trans_folder (str | :obj:`Path`): Path to a transmission folder.
        ann_file (str): The path to the annotation file. If None, we assume
            that all sequences in the folder is used. Default: None
        num_input_frames (None | int): The number of frames per iteration.
            If None, the whole clip is extracted. If it is a positive integer,
            a sequence of 'num_input_frames' frames is extracted from the clip.
            Note that non-positive integers are not accepted. Default: None.
        test_mode (bool): Store `True` when building test dataset.
            Default: `True`.
    """

    def __init__(self,
                 lq_folder,
                 gt_folder,
                 pipeline,
                 trans_folder=None,
                 ann_file=None,
                 num_input_frames=None,
                 test_mode=True):
        super().__init__(pipeline, test_mode)

        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.trans_folder = str(trans_folder)
        self.ann_file = ann_file

        if num_input_frames is not None and num_input_frames <= 0:
            raise ValueError('"num_input_frames" must be None or positive, '
                             f'but got {num_input_frames}.')
        self.num_input_frames = num_input_frames

        self.data_infos = self.load_annotations()

    def _load_annotations_from_file(self):
        data_infos = []

        ann_list = mmcv.list_from_file(self.ann_file)
        for ann in ann_list:
            key, sequence_length = ann.strip().split(' ')
            if self.num_input_frames is None:
                num_input_frames = sequence_length
            else:
                num_input_frames = self.num_input_frames
            dataset = key.split('/')[-2]
            folder = key.split('/')[-1]
            haze_beta = float(folder.split('_')[-1])
            haze_light = float(folder.split('_')[-2]) / 255
            data_info = dict(
                lq_path=self.lq_folder,
                gt_path=self.gt_folder,
                trans_path=self.trans_folder,
                key=key,
                num_input_frames=int(num_input_frames),
                sequence_length=int(sequence_length),
                #
                dataset=dataset,
                folder=folder,
                haze_beta=haze_beta,
                haze_light=haze_light)
            data_infos.append(data_info)

        return data_infos

    def load_annotations(self):
        """Load annoations for the dataset.

        Returns:
            dict: Returned dict for LQ and GT pairs.
        """

        if self.ann_file is not None:
            return self._load_annotations_from_file()
        else:
            # May encounter some unexpected errors especially when with ceph storage.
            raise NotImplementedError

        logger = get_root_logger()
        datasets = os.listdir(self.lq_folder)
        assert datasets == os.listdir(self.gt_folder)
        datasets.sort()
        logger.info(f'Datasets ({len(datasets)}): {datasets}')
        data_infos = []
        for dataset in datasets:
            folders = os.listdir(osp.join(osp.join(self.lq_folder, dataset)))
            assert folders == os.listdir(osp.join(osp.join(self.gt_folder, dataset)))
            folders.sort()

            for folder in folders:
                lq_folder = osp.join(self.lq_folder, dataset, folder)
                sequence_length = len(os.listdir(lq_folder))
                if self.num_input_frames is None:
                    num_input_frames = sequence_length
                else:
                    num_input_frames = self.num_input_frames
                haze_beta = float(folder.split('_')[-1])
                haze_light = float(folder.split('_')[-2]) / 255
                data_info = dict(
                    lq_path=self.lq_folder,
                    gt_path=self.gt_folder,
                    trans_path=self.trans_folder,
                    key=osp.join(dataset, folder),
                    num_input_frames=num_input_frames,
                    sequence_length=sequence_length,
                    #
                    dataset=dataset,
                    folder=folder,
                    haze_beta=haze_beta,
                    haze_light=haze_light)
                data_infos.append(data_info)

        return data_infos

    def evaluate(self, results, logger=None):
        """Evaluate with different metrics.
        For the HazeWorld dataset, we separately report the evaluation metrics for different dataset/beta.

        Args:
            results (list[tuple]): The output of forward_test() of the model.

        Return:
            dict: Evaluation results dict.
        """
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        eval_results = [res['eval_result'] for res in results]  # a list of dict
        eval_result = defaultdict(list)  # a dict of list

        metrics = [metric for metric in AVAILABLE_METRICS if metric in eval_results[0].keys()]

        for dataset in AVAILABLE_DATASETS:
            for haze_beta in AVAILABLE_BETAS:
                for metric in metrics:
                    eval_result[f'{dataset}/{haze_beta}/{metric}'] = list()

        datasets = []
        haze_betas = []
        for i in range(len(results)):
            res = eval_results[i]
            dataset = results[i]['dataset']
            haze_beta = results[i]['haze_beta']
            for metric, val in res.items():
                eval_result[metric].append(val)
                eval_result[f'{dataset}/{haze_beta}/{metric}'].append(val)
                eval_result[f'{dataset}/avg/{metric}'].append(val)
                if metric not in metrics:
                    metrics.append(metric)
            if dataset not in datasets:
                datasets.append(dataset)
            if haze_beta not in haze_betas:
                haze_betas.append(haze_beta)
        datasets = [dataset for dataset in AVAILABLE_DATASETS if dataset in datasets]
        haze_betas = [haze_beta for haze_beta in AVAILABLE_BETAS if haze_beta in haze_betas]
        # print(f'\n{metrics}\n')

        # average the results
        if not self.test_mode:
            fmt_eval_result = {}
            for metric in metrics:
                values = [sum(eval_result[f'{dataset}/avg/{metric}']) / len(eval_result[f'{dataset}/avg/{metric}'])
                          for dataset in datasets]
                fmt_eval_result[metric] = sum(values) / len(values)
            return fmt_eval_result

        fmt_eval_result = {}
        for dataset in datasets + ['Average']:
            for haze_beta in haze_betas:
                for metric in metrics:
                    assert len(eval_result[f'{dataset}/{haze_beta}/{metric}']) == len(
                        eval_result[f'{dataset}/{haze_betas[0]}/{metrics[0]}'])
            num_folders = len(eval_result[f'{dataset}/{haze_betas[0]}/{metrics[0]}'])
            assert len(eval_result[f'{dataset}/avg/{metrics[0]}']) == num_folders * len(haze_betas)
            if dataset == ['Average']:
                # average across datasets
                assert num_folders == len(datasets)
            key = f'[{dataset:10s} ({num_folders})]\t'
            info = ''
            for haze_beta in haze_betas + ['avg']:
                info += f'{haze_beta}: '
                for metric in metrics:
                    # averaged per dataset
                    val = sum(eval_result[f'{dataset}/{haze_beta}/{metric}']) / len(
                        eval_result[f'{dataset}/{haze_beta}/{metric}'])
                    # average across datasets
                    if dataset != 'Average' and haze_beta != 'avg':
                        eval_result[f'Average/{haze_beta}/{metric}'].append(val)
                        eval_result[f'Average/avg/{metric}'].append(val)
                    if metric in ('L1', 'PSNR', 'SSIM'):
                        info += f'{val:.4f}/'
                info = f"{info.rstrip('/')},\t"
            fmt_eval_result[key] = info

        return fmt_eval_result
