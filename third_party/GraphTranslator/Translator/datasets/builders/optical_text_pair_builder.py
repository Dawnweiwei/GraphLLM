"""Dataset builder for optical-network GraphTranslator data."""

import logging

import torch.distributed as dist

from common.dist_utils import is_dist_avail_and_initialized
from common.registry import registry
from datasets.builders.base_dataset_builder import BaseDatasetBuilder
from datasets.datasets.optical_text_pair_datasets import OpticalTextPairDataset


@registry.register_builder("optical_caption")
class OpticalCaptionBuilder(BaseDatasetBuilder):
    DATASET_CONFIG_DICT = {
        "stage1": "train/pretrain_optical_stage1.yaml",
        "stage1_focused": "train/pretrain_optical_stage1_focused.yaml",
        "stage1_focused_v2": "train/pretrain_optical_stage1_focused_v2.yaml",
        "stage2": "train/pretrain_optical_stage2.yaml",
        "stage2_qa_1k": "train/pretrain_optical_stage2_qa_1k.yaml",
        "stage2_qa_20k": "train/pretrain_optical_stage2_qa_20k.yaml",
        "stage2_qa_20k_focused": "train/pretrain_optical_stage2_qa_20k_focused.yaml",
        "stage2_qa_20k_focused_v2": "train/pretrain_optical_stage2_qa_20k_focused_v2.yaml",
        "generate": "train/generate_optical.yaml",
        "generate_qa_32": "train/generate_optical_qa_32.yaml",
        "generate_qa_1k": "train/generate_optical_qa_1k.yaml",
        "generate_qa_1k_focused": "train/generate_optical_qa_1k_focused.yaml",
        "generate_qa_1k_focused_v2": "train/generate_optical_qa_1k_focused_v2.yaml",
    }

    def __init__(self, dataset_config, cfg):
        self.data_type = "optical_feature"
        self.dataset_config = dataset_config
        self.runners_config = cfg.run_cfg
        self.args = cfg.args
        self.train_dataset_cls = OpticalTextPairDataset

    def build_datasets(self):
        if is_dist_avail_and_initialized():
            dist.barrier()

        logging.info("Building optical datasets...")
        return self.build()

    def build(self):
        return {
            "train": self.train_dataset_cls(
                cfg=self.dataset_config,
                mode="train",
            )
        }
