import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification
from labels import LABEL2ID, ID2LABEL


class WeightedTokenClassificationModel(nn.Module):
    """Wrapper for token classification with weighted loss."""
    
    def __init__(self, model_name: str, class_weights=None):
        super().__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        self.class_weights = class_weights
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # Don't use built-in loss
        )
        
        logits = outputs.logits
        
        if labels is not None:
            # Compute weighted cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
                ignore_index=-100
            )
            loss = loss_fct(logits.view(-1, len(LABEL2ID)), labels.view(-1))
            
            # Return in same format as HF model
            return type('Output', (), {'loss': loss, 'logits': logits})()
        else:
            return outputs
    
    def save_pretrained(self, *args, **kwargs):
        """Delegate to underlying model."""
        return self.model.save_pretrained(*args, **kwargs)
    
    def to(self, *args, **kwargs):
        """Override to() to move both model and weights."""
        super().to(*args, **kwargs)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(*args, **kwargs)
        return self


def create_model(model_name: str, class_weights=None):
    """Create model with optional class weights for handling imbalanced data."""
    if class_weights is not None:
        return WeightedTokenClassificationModel(model_name, class_weights)
    else:
        return AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
