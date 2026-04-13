from monai.losses import DiceCELoss

def build_loss_function():
    """
    构建复合损失函数：Dice Loss 关注区域重合度，Cross Entropy 关注体素级分类精度。
    这种组合是医学多分类分割的当前最优解(SOTA)配置。
    """
    return DiceCELoss(to_onehot_y=True, softmax=True)