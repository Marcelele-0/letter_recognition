import unittest
from letter_recognition.utils.custom_data_loader import CustomDataLoader

class TestCustomDataLoader(unittest.TestCase):
    def test_load_data(self):
        # Tworzymy obiekt klasy CustomDataLoader
        data_loader = CustomDataLoader()

        # Wczytujemy dane
        data_loader.load_data()

        # Sprawdzamy, czy obiekty TensorDataset dla danych treningowych i walidacyjnych zostały utworzone
        self.assertIsNotNone(data_loader.train_dataset)
        self.assertIsNotNone(data_loader.val_dataset)

        # Sprawdzamy, czy dane treningowe i walidacyjne mają odpowiednie rozmiary
        self.assertEqual(len(data_loader.train_dataset), 151166)
        self.assertEqual(len(data_loader.val_dataset), 37792)

    def test_get_data_loader(self):
        # Tworzymy obiekt klasy CustomDataLoader
        data_loader = CustomDataLoader()

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

if __name__ == '__main__':
    unittest.main()
