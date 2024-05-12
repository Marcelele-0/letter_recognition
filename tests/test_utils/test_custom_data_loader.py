import unittest
from letter_recognition.utils.custom_data_loader import CustomDataLoader
import torch

class TestCustomDataLoader(unittest.TestCase):
    def test_load_data(self):
        # Tworzymy obiekt klasy CustomDataLoader
        data_loader = CustomDataLoader()

        # Wczytujemy dane
        data_loader.load_data()

        # Sprawdzamy, czy obiekty TensorDataset dla danych treningowych i walidacyjnych zostały utworzone
        self.assertIsNotNone(data_loader.train_dataset)
        self.assertIsNotNone(data_loader.val_dataset)

        # Sprawdzamy, czy dane wczytane do obiektów TensorDataset nie są puste
        self.assertTrue(len(data_loader.train_dataset) > 0)
        self.assertTrue(len(data_loader.val_dataset) > 0)

        # Sprawdzamy, czy dane mają oczekiwany typ danych
        self.assertIsInstance(data_loader.train_dataset.tensors[0], torch.Tensor)
        self.assertIsInstance(data_loader.train_dataset.tensors[1], torch.Tensor)
        self.assertIsInstance(data_loader.val_dataset.tensors[0], torch.Tensor)
        self.assertIsInstance(data_loader.val_dataset.tensors[1], torch.Tensor)

    def test_get_data_loader(self):
        # Tworzymy obiekt klasy CustomDataLoader
        data_loader = CustomDataLoader(batch_size=64)

        # Pobieramy DataLoader dla danych treningowych i walidacyjnych
        train_loader, val_loader = data_loader.get_data_loader()

        # Sprawdzamy, czy DataLoader został utworzony
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)

        # Sprawdzamy, czy DataLoader zawiera dane
        self.assertTrue(len(train_loader.dataset) > 0)
        self.assertTrue(len(val_loader.dataset) > 0)
    

        # Sprawdzamy, czy DataLoader ma prawidłowy rozmiar batcha
        self.assertEqual(train_loader.batch_size, 64)
        self.assertEqual(val_loader.batch_size, 64)

    def test_data_shapes(self):
        # Load the data
        batch_size = 64  # Ustawiamy rozmiar batcha
        data_loader = CustomDataLoader(batch_size=batch_size)
        train_loader, val_loader = data_loader.get_data_loader()

        # Check the shape of training data
        for images, labels in train_loader:
            self.assertEqual(images.shape[1:], (1, 28, 28), "Invalid shape of training images")
            self.assertEqual(labels.shape[1], 26, "Invalid shape of training labels")
            break


        # Check the shape of validation data
        for images, labels in val_loader:
            self.assertEqual(images.shape[1:], (1, 28, 28), "Invalid shape of validation images")
            self.assertEqual(labels.shape[1], 26, "Invalid shape of validation labels")
            break  # Stop the loop after the first iteration

if __name__ == '__main__':
    unittest.main()
