#!/usr/bin/env python3
"""
ChromeCRISPR Model Validation Suite - Complete 20 Model Testing Framework
========================================================================

This comprehensive validation script tests all 20 ChromeCRISPR models against 
exact manuscript specifications with ±0.001 tolerance for MSE and Spearman correlation.

Author: ChromeCRISPR Validation Team
Date: January 2025
Version: 1.0.0

CRITICAL REQUIREMENTS:
- NO HALLUCINATIONS: All data from actual manuscript/cluster logs
- EXACT MATCHING: ±0.001 tolerance for all metrics
- COMPLETE COVERAGE: All 20 models validated
- BMC COMPLIANT: Publication-ready documentation

MODEL CATEGORIES (20 Total):
1. Base Models (5): Random Forest, CNN, GRU, LSTM, BiLSTM
2. Base + GC (4): CNN+GC, GRU+GC, LSTM+GC, BiLSTM+GC
3. Deep Models (4): deepCNN, deepGRU, deepLSTM, deepBiLSTM
4. Deep + GC (4): deepCNN+GC, deepGRU+GC, deepLSTM+GC, deepBiLSTM+GC
5. ChromeCRISPR Hybrid (3): CNN_GRU+GC, CNN_LSTM+GC, CNN_BiLSTM+GC
"""

import os
import sys
import json
import pickle
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
