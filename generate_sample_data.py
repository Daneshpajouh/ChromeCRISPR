#!/usr/bin/env python3
"""
Generate Sample CRISPR Data

This script generates sample CRISPR/Cas9 data for demonstration and testing
purposes. The data includes realistic sequences and activity scores.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging


def generate_crispr_sequences(n_samples: int, sequence_length: int = 21) -> list:
    """
    Generate realistic CRISPR guide RNA sequences.

    Args:
        n_samples: Number of sequences to generate
        sequence_length: Length of each sequence

    Returns:
        List of DNA sequences
    """
    nucleotides = ['A', 'C', 'G', 'T']

    # Generate random sequences
    sequences = []
    for _ in range(n_samples):
        sequence = ''.join(np.random.choice(nucleotides, sequence_length))
        sequences.append(sequence)

    return sequences


def calculate_gc_content(sequence: str) -> float:
    """
    Calculate GC content of a DNA sequence.

    Args:
        sequence: DNA sequence

    Returns:
        GC content as a fraction
    """
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence)


def generate_activity_scores(sequences: list, base_correlation: float = 0.7) -> list:
    """
    Generate realistic activity scores based on sequence features.

    Args:
        sequences: List of DNA sequences
        base_correlation: Base correlation between features and activity

    Returns:
        List of activity scores
    """
    activities = []

    for sequence in sequences:
        # Calculate sequence features
        gc_content = calculate_gc_content(sequence)

        # Position-specific features (PAM site importance)
        pam_score = 0
        if sequence.endswith('GG'):  # Common PAM site
            pam_score = 0.8
        elif sequence.endswith('AG'):
            pam_score = 0.6
        elif sequence.endswith('CG'):
            pam_score = 0.4
        else:
            pam_score = 0.2

        # GC content preference (optimal around 40-60%)
        gc_penalty = 1.0 - abs(gc_content - 0.5) * 2

        # Base activity score
        base_score = 0.5

        # Combine features
        activity = base_score + (pam_score * 0.3) + (gc_penalty * 0.2)

        # Add some noise
        noise = np.random.normal(0, 0.1)
        activity += noise

        # Clamp to [0, 1]
        activity = np.clip(activity, 0.0, 1.0)

        activities.append(activity)

    return activities


def generate_bio_features(sequences: list) -> dict:
    """
    Generate biological features for sequences.

    Args:
        sequences: List of DNA sequences

    Returns:
        Dictionary of biological features
    """
    n_samples = len(sequences)

    # Generate random biological features
    positions = np.random.randint(1, 1000000, n_samples)
    chromosomes = np.random.choice([f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY'], n_samples)
    strands = np.random.choice(['+', '-'], n_samples)

    return {
        'position': positions,
        'chromosome': chromosomes,
        'strand': strands
    }


def main():
    """Main function to generate sample data."""
    parser = argparse.ArgumentParser(description='Generate sample CRISPR data')
    parser.add_argument('--output', type=str, default='data/processed/crispr_dataset.csv',
                       help='Output file path')
    parser.add_argument('--n-samples', type=int, default=10000,
                       help='Number of samples to generate')
            parser.add_argument('--sequence-length', type=int, default=21,
                       help='Length of each sequence')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set random seed
    np.random.seed(args.seed)
    logger.info(f"Set random seed to {args.seed}")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {args.n_samples} CRISPR sequences...")

    # Generate sequences
    sequences = generate_crispr_sequences(args.n_samples, args.sequence_length)

    # Generate activity scores
    activities = generate_activity_scores(sequences)

    # Generate biological features
    bio_features = generate_bio_features(sequences)

    # Calculate GC content
    gc_contents = [calculate_gc_content(seq) for seq in sequences]

    # Create DataFrame
    data = {
        'sequence': sequences,
        'activity': activities,
        'gc_content': gc_contents,
        'position': bio_features['position'],
        'chromosome': bio_features['chromosome'],
        'strand': bio_features['strand']
    }

    df = pd.DataFrame(data)

    # Save to file
    df.to_csv(output_path, index=False)

    # Print statistics
    logger.info(f"Generated {len(df)} samples")
    logger.info(f"Activity range: {df['activity'].min():.3f} - {df['activity'].max():.3f}")
    logger.info(f"Activity mean: {df['activity'].mean():.3f}")
    logger.info(f"Activity std: {df['activity'].std():.3f}")
    logger.info(f"GC content range: {df['gc_content'].min():.3f} - {df['gc_content'].max():.3f}")
    logger.info(f"GC content mean: {df['gc_content'].mean():.3f}")
    logger.info(f"Data saved to {output_path}")

    # Create train/val/test splits for demonstration
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))

    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]

    # Save splits
    train_path = output_path.parent / 'train_data.csv'
    val_path = output_path.parent / 'val_data.csv'
    test_path = output_path.parent / 'test_data.csv'

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Train set: {len(train_df)} samples")
    logger.info(f"Validation set: {len(val_df)} samples")
    logger.info(f"Test set: {len(test_df)} samples")
    logger.info(f"Split files saved to {output_path.parent}")


if __name__ == "__main__":
    main()
