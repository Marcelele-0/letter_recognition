import unittest
import numpy as np
import torch
from letter_recognition.utils.evaluation_metrics import calculate_class_weights
from letter_recognition.utils.evaluation_metrics import calculate_loss_and_accuracy
import unittest
import numpy as np
import torch

import torch.nn.functional as F



class TestCalculateClassWeights(unittest.TestCase):
    def test_tensor_input(self):
        labels_onehot = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0]])
        weights = calculate_class_weights(labels_onehot)

        expected_weights = torch.tensor([0.2, 0.4, 0.4])

        self.assertTrue(torch.allclose(weights, expected_weights, atol=1e-5))

    def test_numpy_input(self):
        labels_onehot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
        weights = calculate_class_weights(labels_onehot)

        expected_weights = torch.tensor([0.4, 0.2, 0.4]) 

        self.assertTrue(torch.allclose(weights, expected_weights, atol=1e-5))

    def test_invalid_input(self):
        labels_onehot = "Invalid input"

        with self.assertRaises(ValueError):
            calculate_class_weights(labels_onehot)

    def test_zero_class(self):
        labels_onehot = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]])

        with self.assertRaises(ValueError):
            calculate_class_weights(labels_onehot)  

    def test_empty_input(self):
        labels_onehot = torch.tensor([])

        with self.assertRaises(ValueError):
            calculate_class_weights(labels_onehot)


    def test_too_deep_input(self):
        labels_onehot = torch.tensor([[[0, 1, 0], [1, 0, 0]], [[0, 0, 1], [0, 1, 0]]])

        with self.assertRaises(ValueError):
            calculate_class_weights(labels_onehot)
    
    def test_negative_input(self):
        labels_onehot = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

        with self.assertRaises(ValueError):
            calculate_class_weights(labels_onehot)

    def test_too_large_input(self):
        labels_onehot = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 2]])

        with self.assertRaises(ValueError):
            calculate_class_weights(labels_onehot)

    def test_too_shallow_input(self):
        labels_onehot = torch.tensor([0, 1, 0])

        with self.assertRaises(ValueError):
            calculate_class_weights(labels_onehot)

    def test_long_input(self):
        labels_onehot = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]]*1000)
        weights = calculate_class_weights(labels_onehot)
        
        expected_weights = torch.tensor([0.3333, 0.3333, 0.3333], dtype=torch.float32)

        self.assertTrue(torch.allclose(weights, expected_weights, atol=1e-3))



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
