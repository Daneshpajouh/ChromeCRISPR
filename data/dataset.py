"""
CRISPR Dataset for ChromeCRISPR.

This module contains the main dataset class for loading and processing
CRISPR/Cas9 data with proper train/validation/test splits.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


class CRISPRDataset(Dataset):
    """
    Dataset class for CRISPR/Cas9 on-target activity prediction.

    This dataset handles:
    - Sequence encoding (one-hot encoding)
    - GC content calculation
    - Biological features
    - Train/validation/test splits
    - Data normalization
    """

    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        use_gc: bool = True,
        use_bio_features: bool = False,
        normalize: bool = True,
        transform: Optional[callable] = None,
        target_column: str = 'activity',
        sequence_column: str = 'sequence'
    ):
        """
        Initialize the CRISPR dataset.

        Args:
            data_path: Path to the data file (CSV, HDF5, or parquet)
            split: Dataset split ('train', 'val', 'test')
            use_gc: Whether to include GC content features
            use_bio_features: Whether to include biological features
            normalize: Whether to normalize features
            transform: Optional data transformation
            target_column: Name of the target column
            sequence_column: Name of the sequence column
        """
        self.data_path = Path(data_path)
        self.split = split
        self.use_gc = use_gc
        self.use_bio_features = use_bio_features
        self.normalize = normalize
        self.transform = transform
        self.target_column = target_column
        self.sequence_column = sequence_column

        # Load and process data
        self._load_data()
        self._process_data()

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _load_data(self):
        """Load data from file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # Load based on file extension
        if self.data_path.suffix == '.csv':
            self.data = pd.read_csv(self.data_path)
        elif self.data_path.suffix == '.h5' or self.data_path.suffix == '.hdf5':
            self.data = pd.read_hdf(self.data_path)
        elif self.data_path.suffix == '.parquet':
            self.data = pd.read_parquet(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

        self.logger.info(f"Loaded {len(self.data)} samples from {self.data_path}")

    def _process_data(self):
        """Process and prepare the data."""
        # Validate required columns
        required_columns = [self.sequence_column, self.target_column]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Clean data
        self._clean_data()

        # Encode sequences
        self._encode_sequences()

        # Calculate GC content if needed
        if self.use_gc:
            self._calculate_gc_content()

        # Process biological features if needed
        if self.use_bio_features:
            self._process_bio_features()

        # Prepare features
        self._prepare_features()

        # Normalize if requested
        if self.normalize:
            self._normalize_features()

        # Convert to tensors
        self._convert_to_tensors()

    def _clean_data(self):
        """Clean the dataset."""
        # Remove rows with missing values
        initial_size = len(self.data)
        self.data = self.data.dropna(subset=[self.sequence_column, self.target_column])

        # Remove invalid sequences
        self.data = self.data[self.data[self.sequence_column].str.len() == 21]

        # Remove sequences with invalid characters
        valid_chars = set('ACGT')
        self.data = self.data[
            self.data[self.sequence_column].apply(
                lambda x: all(c in valid_chars for c in x.upper())
            )
        ]

        final_size = len(self.data)
        removed = initial_size - final_size
        if removed > 0:
            self.logger.warning(f"Removed {removed} invalid samples during cleaning")

    def _encode_sequences(self):
        """Encode DNA sequences to one-hot encoding."""
        # Define nucleotide mapping
        nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        # Encode sequences
        encoded_sequences = []
        for sequence in self.data[self.sequence_column]:
            sequence = sequence.upper()
            encoded = np.zeros((len(sequence), 4))
            for i, nucleotide in enumerate(sequence):
                if nucleotide in nucleotide_map:
                    encoded[i, nucleotide_map[nucleotide]] = 1
            encoded_sequences.append(encoded)

        self.encoded_sequences = np.array(encoded_sequences)

    def _calculate_gc_content(self):
        """Calculate GC content for each sequence."""
        gc_contents = []
        for sequence in self.data[self.sequence_column]:
            sequence = sequence.upper()
            gc_count = sequence.count('G') + sequence.count('C')
            gc_content = gc_count / len(sequence)
            gc_contents.append(gc_content)

        self.gc_contents = np.array(gc_contents).reshape(-1, 1)

    def _process_bio_features(self):
        """Process additional biological features."""
        bio_features = []

        for _, row in self.data.iterrows():
            features = []

            # Position features (if available)
            if 'position' in row:
                features.append(row['position'] / 1000000)  # Normalize position
            else:
                features.append(0.5)  # Default value

            # Chromosome features (if available)
            if 'chromosome' in row:
                # Simple chromosome encoding
                chrom = str(row['chromosome']).replace('chr', '')
                try:
                    chrom_num = int(chrom) if chrom.isdigit() else 23  # X/Y as 23
                    features.append(chrom_num / 23)  # Normalize
                except:
                    features.append(0.5)
            else:
                features.append(0.5)

            # Strand features (if available)
            if 'strand' in row:
                features.append(1.0 if row['strand'] == '+' else 0.0)
            else:
                features.append(0.5)

            # Additional features can be added here
            features.extend([0.0, 0.0])  # Placeholder for future features

            bio_features.append(features)

        self.bio_features = np.array(bio_features)

    def _prepare_features(self):
        """Prepare final feature matrix."""
        features_list = [self.encoded_sequences]

        if self.use_gc:
            features_list.append(self.gc_contents)

        if self.use_bio_features:
            features_list.append(self.bio_features)

        # Concatenate all features
        self.features = np.concatenate(features_list, axis=2)

        # Prepare targets
        self.targets = self.data[self.target_column].values

    def _normalize_features(self):
        """Normalize features."""
        # Don't normalize one-hot encoded sequences
        if self.use_gc:
            # Normalize GC content
            scaler = MinMaxScaler()
            self.features[:, :, 4] = scaler.fit_transform(self.features[:, :, 4].reshape(-1, 1)).reshape(self.features.shape[0], -1)

        if self.use_bio_features:
            # Normalize biological features
            start_idx = 5 if self.use_gc else 4
            scaler = StandardScaler()
            bio_features = self.features[:, :, start_idx:].reshape(-1, self.features.shape[2] - start_idx)
            normalized_bio = scaler.fit_transform(bio_features)
            self.features[:, :, start_idx:] = normalized_bio.reshape(self.features.shape[0], -1, self.features.shape[2] - start_idx)

    def _convert_to_tensors(self):
        """Convert numpy arrays to PyTorch tensors."""
        self.features = torch.FloatTensor(self.features)
        self.targets = torch.FloatTensor(self.targets)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample by index."""
        features = self.features[idx]
        target = self.targets[idx]

        if self.transform:
            features = self.transform(features)

        return features, target

    def get_feature_dim(self) -> int:
        """Get the feature dimension."""
        return self.features.shape[2]

    def get_sequence_length(self) -> int:
        """Get the sequence length."""
        return self.features.shape[1]

    @classmethod
    def create_splits(
        cls,
        data_path: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42,
        **kwargs
    ) -> Tuple['CRISPRDataset', 'CRISPRDataset', 'CRISPRDataset']:
        """
        Create train/validation/test splits from a single dataset.

        Args:
            data_path: Path to the data file
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_state: Random seed for reproducibility
            **kwargs: Additional arguments for dataset initialization

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

        # Load full dataset
        full_dataset = cls(data_path, split='full', **kwargs)

        # Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size

        # Create splits
        train_data, val_data, test_data = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_state)
        )

        return train_data, val_data, test_data

    def get_statistics(self) -> Dict[str, float]:
        """Get dataset statistics."""
        return {
            'num_samples': len(self),
            'sequence_length': self.get_sequence_length(),
            'feature_dim': self.get_feature_dim(),
            'target_mean': float(self.targets.mean()),
            'target_std': float(self.targets.std()),
            'target_min': float(self.targets.min()),
            'target_max': float(self.targets.max()),
            'gc_content_mean': float(self.gc_contents.mean()) if self.use_gc else None
        }
