from src.models.dual_head_unet import DualHeadUNet
from src.models.two_decoder_unet import TwoDecoderUNet
from src.models.two_stage_refinement_unet import TwoStageRefinementUNet

def build_model(model_name, base_channels=32):
    if model_name == "dual_head_unet":
        return DualHeadUNet(base_channels=base_channels)
    if model_name == "two_decoder_unet":
        return TwoDecoderUNet(base_channels=base_channels)
    if model_name == "two_stage_refinement_unet":
        return TwoStageRefinementUNet(base_channels=base_channels)
    raise ValueError(f"Unknown model: {model_name}")
