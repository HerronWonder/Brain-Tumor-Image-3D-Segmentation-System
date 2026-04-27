import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss, DiceLoss


class CompoundBAEMLoss(nn.Module):
    """Dice + CE + edge Dice for BAEM training.

    Expected model output: (segmentation_logits, edge_prediction).
    """

    def __init__(
        self,
        num_classes: int,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        lambda_edge: float = 0.3,
        smooth: float = 1e-5,
        include_background: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.lambda_edge = lambda_edge
        self.smooth = smooth
        self.include_background = include_background

        self.seg_dice = DiceLoss(to_onehot_y=True, softmax=True)
        self.seg_ce = nn.CrossEntropyLoss()

        laplacian_kernel = torch.zeros((1, 1, 3, 3, 3), dtype=torch.float32)
        laplacian_kernel[0, 0, 1, 1, 1] = -6.0
        laplacian_kernel[0, 0, 0, 1, 1] = 1.0
        laplacian_kernel[0, 0, 2, 1, 1] = 1.0
        laplacian_kernel[0, 0, 1, 0, 1] = 1.0
        laplacian_kernel[0, 0, 1, 2, 1] = 1.0
        laplacian_kernel[0, 0, 1, 1, 0] = 1.0
        laplacian_kernel[0, 0, 1, 1, 2] = 1.0
        self.register_buffer("laplacian_kernel", laplacian_kernel, persistent=False)

    def _generate_edge_target(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim == 5 and labels.size(1) == 1:
            labels = labels[:, 0]

        labels = labels.long()
        one_hot = F.one_hot(labels, num_classes=self.num_classes).permute(0, 4, 1, 2, 3)

        if self.include_background:
            class_maps = one_hot
        else:
            class_maps = one_hot[:, 1:] if self.num_classes > 1 else one_hot

        class_maps = class_maps.float()
        class_count = class_maps.shape[1]

        if class_count == 0:
            return class_maps.new_zeros((labels.shape[0], 1, *labels.shape[1:]))

        kernel = self.laplacian_kernel.repeat(class_count, 1, 1, 1, 1)
        edge_response = F.conv3d(class_maps, kernel, padding=1, groups=class_count)
        edge_target = (edge_response.abs() > 0).any(dim=1, keepdim=True).float()
        return edge_target

    def _binary_dice_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = prediction.float()
        target = target.float()

        intersection = (prediction * target).sum(dim=(1, 2, 3, 4))
        denominator = prediction.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
        dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice_score.mean()

    def forward(self, prediction, labels: torch.Tensor) -> torch.Tensor:
        if isinstance(prediction, (tuple, list)):
            segmentation_logits = prediction[0]
            edge_prediction = prediction[1] if len(prediction) > 1 else None
        else:
            segmentation_logits = prediction
            edge_prediction = None

        seg_logits_fp32 = segmentation_logits.float()
        seg_dice_loss = self.seg_dice(seg_logits_fp32, labels)
        seg_ce_loss = self.seg_ce(seg_logits_fp32, labels.squeeze(1).long())

        total_loss = self.lambda_dice * seg_dice_loss + self.lambda_ce * seg_ce_loss

        if edge_prediction is not None:
            edge_target = self._generate_edge_target(labels).to(edge_prediction.device)
            edge_dice_loss = self._binary_dice_loss(edge_prediction.float(), edge_target)
            total_loss = total_loss + self.lambda_edge * edge_dice_loss

        return total_loss

def build_loss_function():
    """
    构建复合损失函数：Dice Loss 关注区域重合度，Cross Entropy 关注体素级分类精度。
    这种组合是医学多分类分割的当前最优解(SOTA)配置。
    """
    return DiceCELoss(to_onehot_y=True, softmax=True)