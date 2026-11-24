"""Hardware-independent estimates, not measured GPU instruction counts."""
import math


class FlopBudgetReached(Exception):
    """The next complete operation would exceed the training compute budget."""


def forward_flops(model, config):
    """Count convolution/linear multiply-adds on one dummy forward pass."""
    import torch
    from torch import nn

    operations = 0

    def count(layer, inputs, output):
        nonlocal operations
        if isinstance(layer, nn.Conv2d):
            products = math.prod(layer.kernel_size) * layer.in_channels // layer.groups
        else:
            products = layer.in_features
        operations += 2 * output.numel() * products

    hooks = [layer.register_forward_hook(count) for layer in model.modules()
             if isinstance(layer, (nn.Conv2d, nn.Linear))]
    was_training = model.training
    try:
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            model(torch.zeros((1, *config["input_shape"]), device=device),
                  torch.zeros((1, config["context_dim"]), device=device))
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    return operations


class FlopCounter:
    convention = ("estimated-v1: Conv2d/Linear multiply-add=2 FLOPs, backward=2x online forward, "
                  "Adam=14 FLOPs/trainable parameter/update; excludes biases, activations, pooling, "
                  "loss, clipping, scalar overhead, environment and data-generation costs")

    def __init__(self, forward, parameters, budget=None):
        self.forward = forward
        self.parameters = parameters
        self.budget = budget
        self.training = False
        self.history_complete = True
        self.costs = dict(action_inference=0, online_forward=0, target_forward=0, backward=0, adam=0)
        self.pretraining = 0
        self.evaluation = 0

    @property
    def total(self):
        return sum(self.costs.values())

    def require(self, cost):
        if self.budget is not None and self.total + cost > self.budget:
            raise FlopBudgetReached()

    def update_cost(self, batch_size):
        return 4 * batch_size * self.forward + 14 * self.parameters

    def record_update(self, batch_size, pretraining=False):
        self.costs["online_forward"] += batch_size * self.forward
        self.costs["target_forward"] += batch_size * self.forward
        self.costs["backward"] += 2 * batch_size * self.forward
        self.costs["adam"] += 14 * self.parameters
        if pretraining:
            self.pretraining += self.update_cost(batch_size)

    def summary(self):
        return {"convention": self.convention, "forward_flops_per_state": self.forward,
                "training_flops": self.total if self.history_complete else None,
                "history_complete": self.history_complete, "training_breakdown": dict(self.costs),
                "pretraining_flops": self.pretraining, "evaluation_flops": self.evaluation,
                "max_train_flops": self.budget, "estimated": True}
