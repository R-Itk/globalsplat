"""Rendering losses: VGG-19 perceptual loss + the combined LossComputer.

Moved verbatim from the original ``loss.py``. The unused ``RenderingCriterion``
class was dropped (the training loop uses ``LossComputer``). The VGG ``.mat``
download is unchanged here; see README for how to cache it instead of fetching
at runtime.
"""
from pathlib import Path

import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from torchvision.models import vgg19


class PerceptualLoss(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.vgg = self._build_vgg()
        self._load_weights()
        self._setup_feature_blocks()
        # ImageNet mean in [0,255]; a constant, registered as a non-persistent
        # buffer so it lives on the right device but is NOT written to / required
        # from checkpoints (it is re-created identically at every init).
        self.register_buffer(
            "vgg_mean",
            torch.tensor([123.6800, 116.7790, 103.9390]).reshape(1, 3, 1, 1),
            persistent=False,
        )

    def _build_vgg(self):
        """Create VGG model with average pooling instead of max pooling."""
        model = vgg19()
        # Replace max pooling with average pooling
        for i, layer in enumerate(model.features):
            if isinstance(layer, nn.MaxPool2d):
                model.features[i] = nn.AvgPool2d(kernel_size=2, stride=2)

        return model.to(self.device).eval()

    def _load_weights(self):
        """Load pre-trained VGG weights (MatConvNet .mat)."""
        weight_file = Path("./metric_checkpoint/imagenet-vgg-verydeep-19.mat")
        weight_file.parent.mkdir(exist_ok=True, parents=True)

        if not weight_file.exists():
            url = "https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat"
            try:
                # Robust, resumable download with a clear failure instead of a
                # silent os.system('wget') that can leave a truncated file.
                torch.hub.download_url_to_file(url, str(weight_file), progress=True)
            except Exception as exc:  # network/offline/permission issues
                raise RuntimeError(
                    f"Failed to download VGG-19 perceptual weights from {url}. "
                    f"Place the .mat at {weight_file} manually (see README) and retry. "
                    f"Original error: {exc}"
                ) from exc


        # Load MatConvNet weights
        vgg_data = scipy.io.loadmat(weight_file)
        vgg_layers = vgg_data["layers"][0]

        # Layer indices and filter sizes
        layer_indices = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        filter_sizes = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]

        # Transfer weights to PyTorch model
        with torch.no_grad():
            for i, layer_idx in enumerate(layer_indices):
                # Set weights
                weights = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][0]).permute(3, 2, 0, 1)
                self.vgg.features[layer_idx].weight = nn.Parameter(weights, requires_grad=False)

                # Set biases
                biases = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][1]).view(filter_sizes[i])
                self.vgg.features[layer_idx].bias = nn.Parameter(biases, requires_grad=False)

    def _setup_feature_blocks(self):
        """Create feature extraction blocks at different network depths."""
        output_indices = [0, 4, 9, 14, 23, 32]
        self.blocks = nn.ModuleList()

        # Create sequential blocks
        for i in range(len(output_indices) - 1):
            block = nn.Sequential(*list(self.vgg.features[output_indices[i]:output_indices[i + 1]]))
            self.blocks.append(block.to(self.device).eval())

        # Freeze all parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

    def _extract_features(self, x):
        """Extract features from each block."""
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

    def _preprocess_images(self, images):
        """Convert images to VGG input format."""
        return images * 255.0 - self.vgg_mean

    @staticmethod
    def _compute_error(real, fake):
        return torch.mean(torch.abs(real - fake))

    def extract_target(self, target_img):
        """Preprocess a target image and extract its VGG features.

        Returns ``(target_img_p, target_features)``. When the same ground-truth
        is scored against several predictions in one step (e.g. the A/B subset
        branches share one target), compute this once and pass it back into
        ``forward`` via ``target_cache`` so the target features are not
        recomputed per branch.
        """
        target_img_p = self._preprocess_images(target_img)
        target_features = self._extract_features(target_img_p)
        return target_img_p, target_features

    def forward(self, pred_img, target_img, target_cache=None):
        """Compute perceptual loss between prediction and target.

        ``target_cache`` is an optional ``(target_img_p, target_features)`` tuple
        from :meth:`extract_target`; when given, the (identical) target features
        are reused instead of recomputed.
        """
        # Preprocess + extract features for the target (reuse if cached).
        if target_cache is None:
            target_img_p, target_features = self.extract_target(target_img)
        else:
            target_img_p, target_features = target_cache

        pred_img_p = self._preprocess_images(pred_img)
        pred_features = self._extract_features(pred_img_p)

        # Pixel-level error
        e0 = self._compute_error(target_img_p, pred_img_p)

        # Feature-level errors with scaling factors
        e1 = self._compute_error(target_features[0], pred_features[0]) / 2.6
        e2 = self._compute_error(target_features[1], pred_features[1]) / 4.8
        e3 = self._compute_error(target_features[2], pred_features[2]) / 3.7
        e4 = self._compute_error(target_features[3], pred_features[3]) / 5.6
        e5 = self._compute_error(target_features[4], pred_features[4]) * 10 / 1.5

        # Combine all errors and normalize
        total_loss = (e0 + e1 + e2 + e3 + e4 + e5) / 255.0

        return total_loss


class LossComputer(nn.Module):
    def __init__(self, lpips_w = 0.,l2_w= 1., perc_w = 0.5):
        super().__init__()
        self.lpips_w = lpips_w
        self.perc_w = perc_w
        self.l2_w = l2_w
        if self.lpips_w > 0.0:
            self.lpips_loss_module = self._init_frozen_module(lpips.LPIPS(net="vgg"))
        if self.perc_w  > 0.0:
            self.perceptual_loss_module = self._init_frozen_module(PerceptualLoss())

    def _init_frozen_module(self, module):
        """Helper method to initialize and freeze a module's parameters."""
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
        return module

    def forward(
            self,
            rendering,
            target,
            perc_target_cache=None,
            return_perc_target_cache=False,
    ):
        """
        Calculate various losses between rendering and target images.

        Args:
            rendering: [b*T, 3, h, w], value range [0, 1]
            target: [b*T, 3, h, w], value range [0, 1]
            perc_target_cache: optional ``(target_p, target_features)`` from
                ``PerceptualLoss.extract_target`` to reuse across calls that
                share the same target (e.g. the A/B subset branches).
            return_perc_target_cache: if True, also return the perceptual target
                cache (computed here if not supplied) so a sibling call can
                reuse it instead of recomputing the target's VGG features.

        Returns:
            ``res`` dict, or ``(res, perc_target_cache)`` when
            ``return_perc_target_cache`` is True.
        """
        res = {}
        # Handle alpha channel if present
        if target.size(1) == 4:
            target, _ = target.split([3, 1], dim=1)

        zero = rendering.new_zeros(())

        l2_loss = zero
        if self.l2_w > 0.0:
            l2_loss = F.mse_loss(rendering, target)
            res["l2_loss"] = l2_loss.detach()

        lpips_loss = zero
        if self.lpips_w > 0.0:
            # Scale from [0,1] to [-1,1] as required by LPIPS
            lpips_loss = self.lpips_loss_module(
                rendering * 2.0 - 1.0, target * 2.0 - 1.0
            ).mean()
            res["lpips_loss"] = lpips_loss.detach()

        perceptual_loss = zero
        if self.perc_w > 0.0:
            # Compute the target's VGG features once and reuse across calls that
            # share the same ground truth (saves a full VGG pass on the GT).
            if perc_target_cache is None:
                perc_target_cache = self.perceptual_loss_module.extract_target(target)
            perceptual_loss = self.perceptual_loss_module(
                rendering, target, target_cache=perc_target_cache
            )
            res["perceptual_loss"] = perceptual_loss.detach()

        res["loss"] = (
            self.l2_w * l2_loss
            + self.lpips_w * lpips_loss
            + self.perc_w * perceptual_loss
        )
        if return_perc_target_cache:
            return res, perc_target_cache
        return res
