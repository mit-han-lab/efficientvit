from typing import Tuple

import torch
import torch.nn as nn

from efficientvit.models.efficientvit.sam import EfficientViTSam


class EvitSAMBoxDecoder(nn.Module):
    def __init__(
        self,
        model: EfficientViTSam,
        return_single_mask: bool,
    ) -> None:
        super().__init__()
        self.mask_decoder = model.mask_decoder
        self.model = model
        self.img_size = model.image_size[0]
        self.return_single_mask = return_single_mask

    @staticmethod
    def resize_longest_image_size(input_image_size: torch.Tensor, longest_side: int) -> torch.Tensor:
        input_image_size = input_image_size.to(torch.float32)
        scale = longest_side / torch.max(input_image_size)
        transformed_size = scale * input_image_size
        transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
        return transformed_size

    def _embed_points(self, point_coords: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        point_embedding = self.model.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.model.prompt_encoder.not_a_point_embed.weight * (point_labels == -1)

        for i in range(self.model.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.model.prompt_encoder.point_embeddings[i].weight * (
                point_labels == i
            )

        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)

        # forward with coords in SAM
        coords /= self.img_size
        corner_embedding = self.model.prompt_encoder.pe_layer._pe_encoding(coords)

        corner_embedding[:, 0, :] += self.model.prompt_encoder.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.model.prompt_encoder.point_embeddings[3].weight

        return corner_embedding

    def select_masks(self, masks: torch.Tensor, iou_preds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        best_idx = torch.argmax(iou_preds, dim=1)
        masks = masks[torch.arange(masks.shape[0]), best_idx, :, :].unsqueeze(1)
        iou_preds = iou_preds[torch.arange(masks.shape[0]), best_idx].unsqueeze(1)

        return masks, iou_preds

    @torch.no_grad()
    def forward(
        self,
        image_embeddings: torch.Tensor,
        boxes: torch.Tensor,
    ):
        sparse_embedding = self._embed_boxes(boxes)
        dense_embedding = self.model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            image_embeddings.shape[0], -1, image_embeddings.shape[-2], image_embeddings.shape[-1]
        )

        masks, iou_preds = self.model.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
        )

        if self.return_single_mask:
            masks, iou_preds = self.select_masks(masks, iou_preds)

        return masks, iou_preds
