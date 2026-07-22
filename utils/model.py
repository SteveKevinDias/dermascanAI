"""
Model architectures + inference pipeline that exactly mirror
segmentation-skin.ipynb and classification-skin.ipynb, so checkpoints
produced by those notebooks load here with no modification.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

# Expected filenames in the Hugging Face repo. Adjust here if your repo uses
# different names for the checkpoint files.
HF_FILENAMES = {
    "best_unet.pt": "segmentation",
    "best_efficientnet_b3.pt": "efficientnet_b3",
    "best_convnext_small.pt": "convnext_small",
    "best_efficientnetv2_m.pt": "efficientnetv2_m",
    "class_names.json": "class_names",
    "ensemble_config.json": "ensemble_config",
}


def fetch_from_hub(repo_id, filename, hf_token=None, cache_dir=None):
    """Downloads a single file from a Hugging Face Hub model repo and
    returns the local cached path. Returns None if the file isn't found
    (e.g. repo doesn't have that particular checkpoint)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=hf_token,
            cache_dir=cache_dir,
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None


def sync_artifacts_from_hub(repo_id, local_seg_dir, local_cls_dir, hf_token=None):
    """Downloads whichever expected files exist in the given HF repo into
    the local artifacts folders (segmentation / classification), skipping
    any that are already present locally. Returns a list of filenames that
    were successfully made available (downloaded or already local)."""
    local_seg_dir = Path(local_seg_dir)
    local_cls_dir = Path(local_cls_dir)
    local_seg_dir.mkdir(parents=True, exist_ok=True)
    local_cls_dir.mkdir(parents=True, exist_ok=True)

    available = []
    for filename in HF_FILENAMES:
        dest_dir = local_seg_dir if filename == "best_unet.pt" else local_cls_dir
        dest_path = dest_dir / filename
        if dest_path.exists():
            available.append(filename)
            continue
        cached_path = fetch_from_hub(repo_id, filename, hf_token=hf_token)
        if cached_path is not None:
            import shutil
            shutil.copy(cached_path, dest_path)
            available.append(filename)
    return available

CLASS_INFO = {
    "MEL":  {"name": "Melanoma",                       "risk": "high"},
    "BCC":  {"name": "Basal Cell Carcinoma",            "risk": "high"},
    "SCC":  {"name": "Squamous Cell Carcinoma",          "risk": "high"},
    "AK":   {"name": "Actinic Keratosis",                "risk": "medium"},
    "BKL":  {"name": "Benign Keratosis-like Lesion",     "risk": "low"},
    "DF":   {"name": "Dermatofibroma",                   "risk": "low"},
    "VASC": {"name": "Vascular Lesion",                  "risk": "low"},
    "NV":   {"name": "Melanocytic Nevus (mole)",         "risk": "low"},
}


# ----------------------------------------------------------------------------
# U-Net (identical to segmentation-skin.ipynb / classification-skin.ipynb)
# ----------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        chs = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc1 = ConvBlock(in_ch, chs[0])
        self.enc2 = ConvBlock(chs[0], chs[1])
        self.enc3 = ConvBlock(chs[1], chs[2])
        self.enc4 = ConvBlock(chs[2], chs[3])
        self.bottleneck = ConvBlock(chs[3], chs[4])
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(chs[4], chs[3], 2, stride=2)
        self.dec4 = ConvBlock(chs[4], chs[3])
        self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.dec3 = ConvBlock(chs[3], chs[2])
        self.up2 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.dec2 = ConvBlock(chs[2], chs[1])
        self.up1 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.dec1 = ConvBlock(chs[1], chs[0])

        self.head = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


def load_unet(checkpoint_path, device=DEVICE):
    model = UNet().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


UNET_INPUT_SIZE = 384


def unet_preprocess(image_rgb, size=UNET_INPUT_SIZE):
    h, w = image_rgb.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(image_rgb, (int(w * scale), int(h * scale)))
    pad_h, pad_w = size - resized.shape[0], size - resized.shape[1]
    padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    norm = (padded / 255.0 - np.array(MEAN)) / np.array(STD)
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, scale, (h, w)


@torch.no_grad()
def predict_mask(unet_model, image_rgb, device=DEVICE, size=UNET_INPUT_SIZE):
    tensor, scale, (h, w) = unet_preprocess(image_rgb, size)
    logits = unet_model(tensor.to(device))
    prob_map = torch.sigmoid(logits)[0, 0].cpu().numpy()
    resized_h, resized_w = int(h * scale), int(w * scale)
    prob_crop = prob_map[:resized_h, :resized_w]
    prob_full = cv2.resize(prob_crop, (w, h))
    mask = (prob_full > 0.5).astype(np.uint8)
    return mask, prob_full


def crop_with_margin(image_rgb, mask, margin_frac=0.15):
    ys, xs = np.where(mask > 0)
    h, w = image_rgb.shape[:2]
    if len(xs) == 0:
        return image_rgb, (0, 0, w, h)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0, y1 - y0
    x0 = max(0, int(x0 - bw * margin_frac))
    x1 = min(w, int(x1 + bw * margin_frac))
    y0 = max(0, int(y0 - bh * margin_frac))
    y1 = min(h, int(y1 + bh * margin_frac))
    return image_rgb[y0:y1, x0:x1], (x0, y0, x1, y1)


# ----------------------------------------------------------------------------
# Classification backbones (timm, matching classification-skin.ipynb)
# ----------------------------------------------------------------------------
MODEL_SPECS = [
    {"name": "efficientnet_b3", "timm_name": "efficientnet_b3", "img_size": 300},
    {"name": "convnext_small", "timm_name": "convnext_small", "img_size": 224},
    {"name": "efficientnetv2_m", "timm_name": "tf_efficientnetv2_m", "img_size": 320},
]


def build_classifier(timm_name, num_classes, dropout_p=0.3, device=DEVICE):
    import timm
    model = timm.create_model(timm_name, pretrained=False, num_classes=num_classes, drop_rate=dropout_p)
    return model.to(device)


def load_classifiers(artifacts_dir, num_classes, device=DEVICE):
    """Loads whichever of the three backbone checkpoints are present.
    Returns dict[name] -> {"model":..., "img_size":...}"""
    artifacts_dir = Path(artifacts_dir)
    loaded = {}
    for spec in MODEL_SPECS:
        ckpt = artifacts_dir / f"best_{spec['name']}.pt"
        if not ckpt.exists():
            continue
        model = build_classifier(spec["timm_name"], num_classes, device=device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        model.eval()
        loaded[spec["name"]] = {"model": model, "img_size": spec["img_size"]}
    return loaded


def build_val_transform(img_size):
    return A.Compose([
        A.Resize(int(img_size * 1.14), int(img_size * 1.14)),
        A.CenterCrop(img_size, img_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


TTA_TRANSFORMS = ["original", "hflip", "vflip", "rot90"]


def apply_tta_variant(image_np, variant):
    if variant == "original":
        return image_np
    if variant == "hflip":
        return cv2.flip(image_np, 1)
    if variant == "vflip":
        return cv2.flip(image_np, 0)
    if variant == "rot90":
        return cv2.rotate(image_np, cv2.ROTATE_90_CLOCKWISE)
    raise ValueError(variant)


@torch.no_grad()
def predict_probs_single(model, image_rgb, img_size, tta_enabled=True, device=DEVICE):
    tfm = build_val_transform(img_size)
    variants = TTA_TRANSFORMS if tta_enabled else ["original"]
    probs_sum = None
    for variant in variants:
        img = apply_tta_variant(image_rgb, variant)
        tensor = tfm(image=img)["image"].unsqueeze(0).to(device)
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).float().cpu().numpy()
        probs_sum = probs if probs_sum is None else probs_sum + probs
    return (probs_sum / len(variants))[0]


def run_ensemble(classifiers, ensemble_weights, image_rgb, tta_enabled=True, device=DEVICE):
    """classifiers: dict name-> {model, img_size}; ensemble_weights: dict name->weight"""
    per_model_probs = {}
    for name, info in classifiers.items():
        per_model_probs[name] = predict_probs_single(
            info["model"], image_rgb, info["img_size"], tta_enabled, device
        )
    names = list(classifiers.keys())
    weights = np.array([ensemble_weights.get(n, 1.0 / len(names)) for n in names])
    weights = weights / weights.sum()
    combined = sum(w * per_model_probs[n] for w, n in zip(weights, names))
    return combined, per_model_probs


# ----------------------------------------------------------------------------
# Grad-CAM (best single backbone, mirrors classification-skin.ipynb section 11)
# ----------------------------------------------------------------------------
def get_target_layer_candidates(model, name):
    """Return a list of plausible target layers to try, in priority order,
    so small version/architecture differences don't hard-fail Grad-CAM."""
    candidates = []
    if "efficientnet" in name:
        for attr in ("conv_head", "bn2", "blocks"):
            layer = getattr(model, attr, None)
            if layer is not None:
                candidates.append(layer[-1] if attr == "blocks" else layer)
    if "convnext" in name:
        stages = getattr(model, "stages", None)
        if stages is not None:
            try:
                last_block = stages[-1].blocks[-1]
                for attr in ("conv_dw", "dwconv"):
                    layer = getattr(last_block, attr, None)
                    if layer is not None:
                        candidates.append(layer)
                candidates.append(last_block)
            except Exception:
                pass
        head = getattr(model, "head", None)
        if head is not None:
            candidates.append(head)
    # last-resort: last Conv2d found anywhere in the model
    if not candidates:
        last_conv = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is not None:
            candidates.append(last_conv)
    return candidates


def compute_gradcam(model, name, image_rgb, img_size, device=DEVICE):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    tfm = build_val_transform(img_size)
    tensor = tfm(image=image_rgb)["image"].unsqueeze(0).to(device)

    candidates = get_target_layer_candidates(model, name)
    if not candidates:
        raise RuntimeError(f"Could not find any candidate target layer for model '{name}'")

    last_err = None
    for target_layer in candidates:
        try:
            cam = GradCAM(model=model, target_layers=[target_layer])
            grayscale_cam = cam(input_tensor=tensor)[0]
            img_resized = cv2.resize(image_rgb, (img_size, img_size)) / 255.0
            overlay = show_cam_on_image(img_resized.astype(np.float32), grayscale_cam, use_rgb=True)
            return overlay
        except Exception as e:  # try next candidate layer
            last_err = e
            continue
    raise RuntimeError(
        f"Grad-CAM failed for all {len(candidates)} candidate layer(s) on '{name}': {last_err}"
    ) from last_err


# ----------------------------------------------------------------------------
# Artifact loading helpers
# ----------------------------------------------------------------------------
def load_class_names(artifacts_dir):
    p = Path(artifacts_dir) / "class_names.json"
    if p.exists():
        data = json.loads(p.read_text())
        return data["class_names"], data.get("melanoma_index")
    return DEFAULT_CLASS_NAMES, DEFAULT_CLASS_NAMES.index("MEL")


def load_ensemble_config(artifacts_dir, model_names):
    p = Path(artifacts_dir) / "ensemble_config.json"
    if p.exists():
        data = json.loads(p.read_text())
        return dict(zip(data["member_names"], data["weights"])), data.get("tta_enabled", True)
    return {n: 1.0 / len(model_names) for n in model_names}, True