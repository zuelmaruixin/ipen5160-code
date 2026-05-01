from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel

try:
    from torchcrf import CRF
except ImportError:  # pragma: no cover - handled at runtime in Colab if needed
    try:
        from TorchCRF import CRF  # type: ignore
    except ImportError:
        CRF = None


class AspectSentimentMultiTaskModel(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        num_labels: int,
        alpha: float = 0.2,
        dropout: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.sentiment_classifier = nn.Linear(hidden_size, num_labels)
        self.rating_regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        self.alpha = alpha
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        sentiment_labels: Optional[torch.Tensor] = None,
        rating_labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_repr = self.dropout(outputs.last_hidden_state[:, 0, :])
        sentiment_logits = self.sentiment_classifier(cls_repr)
        rating_pred = self.rating_regressor(cls_repr).squeeze(-1)

        result = {
            "sentiment_logits": sentiment_logits,
            "rating_pred": rating_pred,
        }
        if sentiment_labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            sentiment_loss = loss_fct(sentiment_logits, sentiment_labels)
            result["sentiment_loss"] = sentiment_loss
            total_loss = sentiment_loss
            if rating_labels is not None:
                rating_loss = nn.functional.mse_loss(rating_pred, rating_labels.float())
                result["rating_loss"] = rating_loss
                total_loss = sentiment_loss + self.alpha * rating_loss
            result["loss"] = total_loss
        return result


class BertCrfForSequenceLabeling(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        num_tags: int,
        alpha: float = 0.2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if CRF is None:
            raise ImportError(
                "torchcrf is not installed. Install it in Colab before using the CRF model."
            )
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.emission = nn.Linear(hidden_size, num_tags)
        self.crf = CRF(num_tags=num_tags, batch_first=True)
        self.rating_regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        self.alpha = alpha

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        rating_labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor | list[list[int]]]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.emission(sequence_output)
        decoded = self.crf.decode(emissions, mask=attention_mask.bool())
        cls_repr = sequence_output[:, 0, :]
        rating_pred = self.rating_regressor(cls_repr).squeeze(-1)

        result: dict[str, torch.Tensor | list[list[int]]] = {
            "emissions": emissions,
            "decoded_tags": decoded,
            "rating_pred": rating_pred,
        }
        if labels is not None:
            crf_loss = -self.crf(
                emissions,
                labels,
                mask=attention_mask.bool(),
                reduction="mean",
            )
            result["crf_loss"] = crf_loss
            total_loss = crf_loss
            if rating_labels is not None:
                rating_loss = nn.functional.mse_loss(rating_pred, rating_labels.float())
                result["rating_loss"] = rating_loss
                total_loss = crf_loss + self.alpha * rating_loss
            result["loss"] = total_loss
        return result
