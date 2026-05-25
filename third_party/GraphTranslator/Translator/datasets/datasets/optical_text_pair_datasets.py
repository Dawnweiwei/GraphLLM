"""Optical-network text-pair datasets for GraphTranslator."""

import numpy as np
import pandas as pd

from datasets.datasets.base_dataset import BatchIterableDataset


class OpticalTextPairDataset(BatchIterableDataset):
    def __init__(self, cfg, mode):
        super(OpticalTextPairDataset, self).__init__(cfg, mode)
        self.summary_embeddings = pd.read_csv(cfg["datasets_dir"], sep="\t")
        self.row_count = self.summary_embeddings.shape[0]
        self.stage = cfg.get("stage", "stage2")
        self.max_text_chars = int(cfg.get("max_text_chars", 1000))

    def _train_data_parser(self, data):
        row = data[0]
        sample_id = row[0]
        embedding = np.array(row[1].split(","), dtype=np.float32)
        producer_text = str(row[2])
        question = str(row[3])
        answer = str(row[4])

        if self.stage == "stage1":
            text_input = producer_text[: self.max_text_chars]
            return sample_id, embedding, text_input, question

        producer_text = producer_text[: self.max_text_chars]
        return sample_id, embedding, producer_text, question, answer

    def __len__(self):
        return self.row_count
