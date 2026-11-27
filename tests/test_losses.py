import unittest

import torch

from sem_fracture.losses import (
    ClassBalancedCrossEntropyLoss,
    compute_effective_number_weights,
)


class ClassBalancedLossTests(unittest.TestCase):
    def test_weights_shape_and_dtype(self):
        weights = compute_effective_number_weights([589, 711, 102], beta=0.999)
        self.assertEqual(weights.shape, (3,))
        self.assertEqual(weights.dtype, torch.float32)

    def test_weights_normalized_mean_one(self):
        weights = compute_effective_number_weights([589, 711, 102], beta=0.999)
        self.assertAlmostEqual(weights.mean().item(), 1.0, places=5)

    def test_minority_class_receives_higher_weight(self):
        weights = compute_effective_number_weights([589, 711, 102], beta=0.999)
        # Mixed (102) > Brittle (589) > Ductile (711)
        self.assertGreater(weights[2].item(), weights[0].item())
        self.assertGreater(weights[0].item(), weights[1].item())

    def test_sem_dataset_expected_weights(self):
        weights = compute_effective_number_weights([589, 711, 102], beta=0.999)
        expected = [0.464, 0.406, 2.130]
        for actual, exp in zip(weights.tolist(), expected):
            self.assertAlmostEqual(actual, exp, delta=0.002)

    def test_beta_zero_produces_equal_weights(self):
        weights = compute_effective_number_weights([10, 100, 1000], beta=0.0)
        for w in weights.tolist():
            self.assertAlmostEqual(w, 1.0, places=6)

    def test_invalid_beta_raises(self):
        for bad_beta in (-0.1, 1.0, 1.5):
            with self.assertRaises(ValueError):
                compute_effective_number_weights([10, 20], beta=bad_beta)
        with self.assertRaises(TypeError):
            compute_effective_number_weights([10, 20], beta=True)

    def test_invalid_class_counts_raises(self):
        with self.assertRaises(ValueError):
            compute_effective_number_weights([], beta=0.999)
        for bad_counts in ([0, 10], [-5, 10]):
            with self.assertRaises(ValueError):
                compute_effective_number_weights(bad_counts, beta=0.999)
        for non_int_counts in ([10.5, 20], [True, 20]):
            with self.assertRaises(TypeError):
                compute_effective_number_weights(non_int_counts, beta=0.999)

    def test_arbitrary_number_of_classes(self):
        weights = compute_effective_number_weights([50, 100, 200, 400, 800], beta=0.99)
        self.assertEqual(weights.shape, (5,))
        self.assertAlmostEqual(weights.mean().item(), 1.0, places=5)

    def test_loss_forward_and_backward(self):
        criterion = ClassBalancedCrossEntropyLoss([589, 711, 102], beta=0.999)
        logits = torch.randn(4, 3, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 0])
        loss = criterion(logits, targets)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all().item())


if __name__ == "__main__":
    unittest.main()
