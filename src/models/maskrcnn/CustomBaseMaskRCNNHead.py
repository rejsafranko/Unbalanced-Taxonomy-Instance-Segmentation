class CustomBaseMaskRCNNHead(nn.Module):
    """
    Implementacija osnovne Mask R-CNN glave s prilagođenim izračunom gubitka i integracijom modula za dinamičko balansiranje težina.
    """

    @configurable
    def __init__(self, *, loss_weight: float = 1.0, vis_period: int = 0):
        """
        Args:
            loss_balancer: Instanca klase `MaskRCNNLossBalancer`.
            loss_weight (float): Množitelj za gubitak.
            vis_period (int): Period vizualizacije.
        """
        super().__init__()
        self.vis_period = vis_period
        self.loss_weight = loss_weight
        self.loss_balancer = MaskRCNNLossBalancer(
            class_frequencies={
                "Can": 194,
                "Other": 1397,
                "Bottle": 344,
                "Bottle cap": 226,
                "Cup": 150,
                "Lid": 63,
                "Plastic bag": 697,
                "Pop tab": 75,
                "Straw": 108,
                "Cigarette": 457,
            },
            total_samples=3711
        )

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "vis_period": cfg.VIS_PERIOD,
        }

    def forward(self, x, instances: List[Instances]):
        x = self.layers(x)
        if self.training:
            recalls = self.loss_balancer.recalls if hasattr(self.loss_balancer, 'recalls') else None
            return {"loss_mask": mask_rcnn_loss(x, instances, self.loss_balancer, self.vis_period) * self.loss_weight}
        else:
            mask_rcnn_inference(x, instances)
            return instances

    def layers(self, x):
        """
        Neural network layers that makes predictions from input features.
        """
        raise NotImplementedError