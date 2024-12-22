import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """LoRA layer implementation"""

    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Return LoRA delta: (x @ A) @ B * scaling
        return (x @ self.lora_A) @ self.lora_B * self.scaling


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation"""

    def __init__(self, linear_layer, rank=4, alpha=1.0):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features, linear_layer.out_features, rank=rank, alpha=alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


class LoRATimesNet(nn.Module):
    """TimesNet adapted with LoRA for fine-tuning"""

    def __init__(self, base_model, rank=4, alpha=1.0, target_modules=None):
        super().__init__()
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha

        # Default target modules to adapt if none specified
        if target_modules is None:
            target_modules = ["projection", "predict_linear"]

        # Add LoRA to target modules
        self._add_lora_layers(target_modules)

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

    def _add_lora_layers(self, target_modules):
        """Replace target linear layers with LoRA versions"""
        for name, module in self.base_model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    # Check if there's a period in the name
                    if "." in name:
                        parent_name, child_name = name.rsplit(".", 1)
                        parent = self.base_model
                        for part in parent_name.split("."):
                            parent = getattr(parent, part)
                    else:
                        # The module is at the top level
                        parent = self.base_model
                        child_name = name
                    # Replace the module with LoRA version
                    setattr(
                        parent, child_name, LoRALinear(module, self.rank, self.alpha)
                    )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.base_model(
            x_enc=x_enc,
            x_mark_enc=x_mark_enc,
            x_dec=x_dec,
            x_mark_dec=x_mark_dec,
        )

    def get_lora_parameters(self):
        """Get only LoRA parameters for optimization"""
        params = []
        for module in self.modules():
            if isinstance(module, LoRALayer):
                params.extend([module.lora_A, module.lora_B])
        return params
