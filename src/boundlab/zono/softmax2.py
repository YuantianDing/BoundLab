


def softmax2_linearizer(x: torch.Tensor, ub: torch.Tensor, lb: torch.Tensor) -> ZonoBounds:
    """Linearizer for softmax2(x, y) = x / (1 + x exp(y)).
    """
    
