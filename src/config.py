# src/config.py

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models_v2")
MODELS_DIR_TEMPORAL = os.path.join(PROJECT_ROOT, "model_temporal") 


SEEDS = [42, 123, 777, 2023, 31415]
