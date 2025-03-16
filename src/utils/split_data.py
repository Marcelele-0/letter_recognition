def calculate_split_sizes(total_samples: int, ratios: tuple) -> tuple:
    """
    Calculate the train, validation, and test split sizes.

    Args:
        total_samples (int): Total number of samples.
        ratios (tuple): A tuple containing the ratios for train, validation, and test splits.

    Returns:
        tuple: A tuple containing the sizes of train, validation, and test splits.
    """
    train_ratio, val_ratio, _ = ratios
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size
    return train_size, val_size, test_size