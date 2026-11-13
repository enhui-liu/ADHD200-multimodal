import torch

from adhd200_multimodal.models import MidFusionClassifier


def test_mid_fusion_classifier_returns_logits_for_each_subject() -> None:
    model = MidFusionClassifier(
        phenotype_dim=5,
        image_channels=1,
        phenotype_embedding_dim=8,
        imaging_embedding_dim=12,
        fusion_hidden_dim=10,
    )
    model.eval()

    logits = model(torch.randn(2, 5), torch.randn(2, 1, 16, 16, 16))

    assert logits.shape == (2, 2)
