# Hull Tactical - PROMETHEUS v3 Artifacts

Upload this entire folder as a Kaggle dataset named `hull-artifacts-v3`.

## Files

| File | Description | Size |
|------|-------------|------|
| `config.pkl` | Position sizing config | 138 B |
| `feature_cols.pkl` | 187 feature column names | 1.5 KB |
| `lgb_models.pkl` | 5x LightGBM models | 216 KB |
| `scaler.pkl` | RobustScaler for features | 5 KB |
| `recent_data.parquet` | Last 300 rows for warmup | 550 KB |
| `xgb_model_0.json` - `xgb_model_4.json` | 5x XGBoost models | ~1.6 MB |

## Config

```python
{
    'base_position': 1.0,
    'min_position': 0.2,
    'max_position': 1.8,
    'scale_factor': 80.0,
    'risk_aversion': 50.0
}
```

## Model Architecture

- **Ensemble**: 5x LightGBM + 5x XGBoost (10 models total)
- **Features**: 187 features including PROMETHEUS insights
- **Position Sizing**: Kelly criterion with uncertainty scaling

## PROMETHEUS v3 Features (computed at inference)

1. Variance compression detection
2. Anomalous dimension Δ (crash predictor)
3. Critical slowing down (AC1)
4. Market temperature
5. Order parameters (V/M/S coherence)
6. Cross-domain correlation surge
7. Sentiment-volatility interaction
8. Economic surprise
9. Interest rate regime
10. Dempster-Shafer belief fusion
11. Value clustering consensus
12. CIC confidence functional

## Usage

```python
ARTIFACTS_DIR = Path('/kaggle/input/hull-artifacts-v3')
```

---
*PROMETHEUS v3 × Hull Tactical - December 2024*
