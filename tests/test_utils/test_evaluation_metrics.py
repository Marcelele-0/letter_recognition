import unittest
import numpy as np
import torch
from letter_recognition.utils.evaluation_metrics import calculate_class_weights
from letter_recognition.utils.evaluation_metrics import calculate_loss_and_accuracy
import torch.nn.functional as F

class TestCalculateClassWeights(unittest.TestCase):
    def test_torch_tensor_input(self):
        labels = torch.tensor([0, 1, 0, 1])
        expected_weights = torch.tensor([0.5, 0.5])  # Changed to torch tensor
        actual_weights = calculate_class_weights(labels)

        max_delta = 1e-6  # Adjust as needed

        self.assertTrue(torch.allclose(actual_weights, expected_weights, atol=max_delta))  # Changed to torch.allclose

    def test_numpy_array_input(self):
        labels = np.array([0, 1, 1, 0])
        expected_weights = torch.tensor([0.5, 0.5])  # Changed to torch tensor

        self.assertTrue(torch.allclose(calculate_class_weights(labels), expected_weights))  # Changed to torch.allclose

    def test_unsupported_data_type(self):
        labels = "invalid"

        with self.assertRaises(ValueError):
            calculate_class_weights(labels)

    def test_object_dtype_labels(self):
        labels = np.array(["A", "B", "A", "B"])
        expected_weights = torch.tensor([0.5, 0.5])
        label_map = {"A": 0, "B": 1}
        integer_labels = np.array([label_map[label] for label in labels])

        self.assertTrue(torch.allclose(calculate_class_weights(integer_labels), expected_weights)) 

    def test_empty_labels(self):
        labels = np.array([])

        with self.assertRaises(ValueError):
            calculate_class_weights(labels)

    def test_single_class_labels(self):
        labels = np.array([0, 0, 0, 0])
        expected_weights = torch.tensor([1.0])
        actual_weights = calculate_class_weights(labels)

        max_delta = 1e-6  # Adjust as needed

        self.assertTrue(torch.allclose(actual_weights, expected_weights, atol=max_delta))

    def test_multi_class_labels(self):
        labels = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        with self.assertRaises(ValueError):
            calculate_class_weights(labels)

    def test_large_class_counts(self):
        labels = np.array([0] * 1000000 + [1] * 1000000 + [2] * 1000000)  # Three classes with large counts
        expected_weights = torch.tensor([1/3, 1/3, 1/3])  # Expected weights for balanced classes
        actual_weights = calculate_class_weights(labels)

        max_delta = 1e-6

        self.assertTrue(torch.allclose(actual_weights, expected_weights, atol=max_delta)) 

class TestCalculateLossAndAccuracy(unittest.TestCase):
    def test_loss_and_accuracy(self):
        # Przygotowanie danych testowych
        outputs = torch.tensor([[0.2, 0.5, 0.3], [0.7, 0.1, 0.2]])
        labels = torch.tensor([[0, 1, 0], [1, 0, 0]])

        # Obliczanie straty i dokładności przy użyciu funkcji testowanej
        loss, accuracy = calculate_loss_and_accuracy(outputs, labels)

        # Obliczanie oczekiwanej wartości straty
        expected_loss = F.cross_entropy(outputs, torch.argmax(labels, dim=1))

        # Sprawdzenie poprawności obliczonej wartości straty
        self.assertAlmostEqual(loss.item(), expected_loss.item(), delta=1e-5)

        # Obliczanie oczekiwanej wartości dokładności
        expected_accuracy = torch.tensor(1.0)  # W przypadku danych testowych, oba przykłady są poprawnie sklasyfikowane

        # Sprawdzenie poprawności obliczonej wartości dokładności
        self.assertAlmostEqual(accuracy, expected_accuracy.item(), delta=1e-5)

if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    unittest.main()
