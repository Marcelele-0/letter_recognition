# Letter Recognition - Środowisko Conda

## Tworzenie środowiska

Aby utworzyć środowisko Conda z pliku `env.yaml`, wykonaj poniższe kroki:

1. Upewnij się, że masz zainstalowaną Condę.
2. Uruchom terminal i przejdź do katalogu zawierającego plik `env.yaml`.
3. Wykonaj polecenie:
   ```bash
   conda env create -f env.yaml
   ```
4. Po zakończeniu instalacji aktywuj środowisko:
   ```bash
   conda activate letter_recognition
   ```

## Usuwanie środowiska

Jeśli chcesz usunąć środowisko, użyj polecenia:
```bash
conda remove --name letter_recognition --all
```

