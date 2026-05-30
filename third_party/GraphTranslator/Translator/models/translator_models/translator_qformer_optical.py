"""Q-Former translator config entry for optical-network stage-1 alignment."""

from common.registry import registry
from models.translator_models.translator_qformer_arxiv import TranslatorQformerArxiv


@registry.register_model("translator_optical")
class TranslatorQformerOptical(TranslatorQformerArxiv):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_optical": "train/pretrain_optical_stage1.yaml",
        "pretrain_optical_focused": "train/pretrain_optical_stage1_focused.yaml",
        "pretrain_optical_focused_v2": "train/pretrain_optical_stage1_focused_v2.yaml",
    }
