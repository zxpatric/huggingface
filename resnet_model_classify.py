import torch
from .configuration_resnet import ResnetConfig

from transformers import PreTrainedModel
from timm.models.resnet import BasicBlock, Bottleneck, ResNet

class ResnetModelForImageClassification(PreTrainedModel):
    config_class = ResnetConfig

    def __init__(self, config):
        super().__init__(config)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.model = ResNet(
            block_layer,
            config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

    def forward(self, tensor, labels=None):
        logits = self.model(tensor)
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
    
import timm

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d = ResnetModelForImageClassification(resnet50d_config)
resnet50d.model.load_state_dict(pretrained_model.state_dict())    