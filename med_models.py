from pathlib import Path

import streamlit as st
import torch
from pytorch_grad_cam import GradCAM

from med_config import resolve_torch_device
from med_model_arch import ECAResNet50, load_state_dict_compatible


@st.cache_resource
def load_model_and_cam(model_path_text, requested_device=None, model_version_token=None):
    # model_version_token is part of cache key so overwritten checkpoint files are reloaded.
    del model_version_token
    model_path = Path(model_path_text)
    device, device_policy = resolve_torch_device(requested_device)
    model = ECAResNet50(num_classes=7).to(device)
    if not model_path.exists():
        st.error(f"模型权重不存在：{model_path}")
        st.stop()

    checkpoint_payload = torch.load(str(model_path), map_location=device)
    load_info = load_state_dict_compatible(model, checkpoint_payload, strict=False)
    missing_keys = load_info.get("missing_keys", [])
    unexpected_keys = load_info.get("unexpected_keys", [])
    if missing_keys or unexpected_keys:
        st.error("模型权重与当前网络结构不匹配，请更换正确的 checkpoint。")
        if missing_keys:
            st.caption(f"Missing keys ({len(missing_keys)}): {missing_keys[:10]}")
        if unexpected_keys:
            st.caption(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:10]}")
        st.stop()
    if load_info["legacy_mapped"] > 0:
        st.caption(f"已兼容加载历史权重格式（映射键数: {load_info['legacy_mapped']}）")
    model.eval()
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    return model, cam, device, device_policy
