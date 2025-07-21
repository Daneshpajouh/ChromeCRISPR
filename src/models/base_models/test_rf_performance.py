#!/usr/bin/env python3
"""
Test RF Model Performance with Real Data

This script tests the RF model performance with real data to verify
it matches the manuscript specifications.
"""

import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

def test_rf_performance(X_test, y_test):
    """Test RF model performance with real data."""

    try:
        # Load the real RF model
        rf_model = joblib.load("exact_20_manuscript_models/base_models/RF_model.joblib")

        # Prepare data (flatten sequences for RF)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        # Make predictions
        y_pred = rf_model.predict(X_test_flat)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        spearman_corr = spearmanr(y_test, y_pred)[0]

        # Manuscript expected performance
        manuscript_mse = 0.0197
        manuscript_spearman = 0.7550

        print("=== RF Model Performance Test ===")
        print(f"Actual Performance:")
        print(f"  MSE: {mse:.4f}")
        print(f"  Spearman: {spearman_corr:.4f}")
        print(f"\nManuscript Expected:")
        print(f"  MSE: {manuscript_mse:.4f}")
        print(f"  Spearman: {manuscript_spearman:.4f}")

        # Check compliance
        mse_tolerance = 0.001  # Allow small tolerance
        spearman_tolerance = 0.01

        mse_compliant = abs(mse - manuscript_mse) <= mse_tolerance
        spearman_compliant = abs(spearman_corr - manuscript_spearman) <= spearman_tolerance

        if mse_compliant and spearman_compliant:
            print("\nPERFORMANCE COMPLIANT WITH MANUSCRIPT")
        else:
            print("\n⚠️  PERFORMANCE DOES NOT MATCH MANUSCRIPT")
            if not mse_compliant:
                print(f"  MSE difference: {abs(mse - manuscript_mse):.4f}")
            if not spearman_compliant:
                print(f"  Spearman difference: {abs(spearman_corr - manuscript_spearman):.4f}")

        return mse, spearman_corr, mse_compliant and spearman_compliant

    except Exception as e:
        print(f"❌ Error testing RF performance: {e}")
        print("This may be due to sklearn version compatibility issues.")
        return None, None, False

if __name__ == "__main__":
    # Load your test data here
    # X_test, y_test = load_your_test_data()
    # test_rf_performance(X_test, y_test)
    print("Please load your test data and call test_rf_performance()")
