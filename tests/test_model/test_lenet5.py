import unittest
import torch
import torch.optim as optim
import torch.nn.functional as F


from letter_recognition.model.lenet5 import LeNet5

class TestLeNet5(unittest.TestCase):
    def test_model_initialization(self):
        # Test inicjalizacji modelu
        model = LeNet5()
        self.assertIsInstance(model, LeNet5)

    def test_forward_pass(self):
        # Test przekazywania danych przez model
        model = LeNet5()
        input_data = torch.randn(1, 1, 28, 28)  # Losowe dane wejściowe o rozmiarze 28x28
        output = model(input_data)
        self.assertEqual(output.shape, torch.Size([1, 26]))  # Oczekiwany rozmiar wyjścia to 1x26

    def test_training_process(self):
        # Test procesu uczenia modelu
        model = LeNet5()
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        criterion = F.cross_entropy

        # Losowe dane treningowe
        input_data = torch.randn(32, 1, 28, 28)
        target = torch.randint(0, 26, (32,))

        # Przebieg jednej iteracji uczenia
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        self.assertTrue(loss.item() >= 0)  # Upewniamy się, że wartość funkcji straty jest nieujemna
        self.assertTrue(len(optimizer.param_groups) > 0)  # Upewniamy się, że istnieją parametry optymalizatora

if __name__ == '__main__':
    unittest.main()
