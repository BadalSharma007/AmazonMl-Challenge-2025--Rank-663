
# ML Challenge 2025: Smart Product Pricing Solution Documentation  

**Team Name:** BajrangDal  
**Team Members:**  
- **Badal Kumar Sharma (Team Leader)**
---

## 1. Executive Summary  

We developed an **XGBoost-Only Product Pricing Model** that predicts product prices directly from **catalog text** using **advanced engineered features**.  

Our solution combines **TF-IDF + SVD text embeddings** with more than **400 manually engineered attributes** — including pack size, weight, volume, material, brand, and category indicators.  

The model uses **5-Fold Cross-Validation** for robust evaluation and achieves an overall **Cross-Validation SMAPE of ~30.9%**, ensuring both accuracy and generalization.  

This single-model approach emphasizes interpretability and training speed while maintaining strong predictive power across diverse product categories.  

---

## 2. Methodology Overview  

### 2.1 Problem Analysis  

During exploratory analysis of ~75K training samples, we observed:  
- **Strong correlation** between price and terms like “pack of 2”, “500 g”, “1 L”, or “12 count”.  
- **Premium cues** (words like *luxury, professional, pro, signature*) generally map to higher prices.  
- **Category keywords** (e.g., *electronics, food, beauty, toy*) strongly segment pricing trends.  
- Price distribution was **heavily right-skewed**, motivating the use of **log-price transformation** to stabilize learning.  

---

### 2.2 Solution Strategy  

**Approach Type:** Single-Model Advanced XGBoost Solution  

**Core Components:**  
- Unified XGBoost Regressor trained on log-transformed prices.  
- Hybrid feature space combining **TF-IDF text embeddings** and **engineered numeric features**.  
- Extensive regex-based extraction of physical attributes like pack quantity, weight, and volume.  
- 5-Fold Cross-Validation for robust SMAPE evaluation.  

**Key Innovations:**  
1. **Advanced Regex Extraction:** Recognizes “Pack of 6”, “1 kg”, “2 L”, “12 count”, etc. across formats.  
2. **Brand & Quality Signals:** Detects premium words, known brands, and capitalization cues.  
3. **Semantic + Numeric Fusion:** Merges 200-dimensional TF-IDF SVD embeddings with structured numeric indicators.  
4. **Log-Price Targeting:** Reduces skew and improves model stability.  

---

## 3. Model Architecture  

### 3.1 Architecture Overview  

**End-to-End Pipeline:**  
```
Catalog Content
   ↓
Text Cleaning & Tokenization
   ↓
TF-IDF Vectorization (1–3-grams) → TruncatedSVD (200 comps)
   ↓
Advanced Feature Extraction (pack, weight, volume, brand, material)
   ↓
PowerTransformer Scaling
   ↓
XGBoost Regressor (log1p(price))
   ↓
Expm1(Postprocessing) → Final Price Predictions
```

---

### 3.2 Model Components  

#### (a) **Text Processing & Embedding**  
- **TF-IDF Vectorizer:**  
  - `max_features = 4000`  
  - `ngram_range = (1, 3)`  
  - `min_df = 3`, `max_df = 0.9`  
- **TruncatedSVD:** 200 components for dimensionality reduction.  

#### (b) **Feature Engineering Highlights**  
Extracted >400 numeric and binary features across domains:  

| Feature Group | Examples |
|----------------|-----------|
| **Pack Quantity** | “Pack of 6”, “12 count”, “Set of 3” → `pack_qty`, `pack_tier`, `is_multi_pack` |
| **Weight & Volume** | “1 kg”, “500 ml”, “2 oz” → `weight_g`, `volume_ml`, `per_pack_log` |
| **Brand Indicators** | Known brand detection + capitalized start words → `brand_strength` |
| **Quality Words** | Counts of *premium, organic, luxury, professional* → `premium_count` |
| **Budget Cues** | *cheap, value, discount* → `budget_count` |
| **Categories** | Electronics, Beauty, Food, Health, Home, Pet, etc. → `cat_*` |
| **Materials** | Weighted material scores (gold, silver, plastic, leather, etc.) |
| **Text Stats** | `word_count`, `avg_word_len`, `digit_ratio`, `upper_ratio` |
| **Numeric Context** | Mean/Max of extracted numeric values → `max_num`, `mean_num`, etc. |
| **Advanced Interactions** | `premium_bulk`, `food_weight`, `electronics_premium` |

#### (c) **Feature Scaling**
- Applied `PowerTransformer` (Yeo-Johnson) to stabilize non-normal numeric features.  

#### (d) **Model Core**
- **Algorithm:** XGBoost Regressor  
- **Objective:** Regression on log1p(price)  
- **Parameters:**
  ```
  n_estimators = 2000
  learning_rate = 0.02
  max_depth = 10
  min_child_weight = 3
  subsample = 0.8
  colsample_bytree = 0.8
  reg_alpha = 1.0
  reg_lambda = 1.0
  random_state = 42
  ```
- **Validation:** 5-Fold Cross-Validation with out-of-fold SMAPE scoring.  

---

## 4. Model Performance  

### 4.1 Cross-Validation Results  
| Fold | SMAPE (%) |
|:----:|:----------:|
| 1 | 31.12 |
| 2 | 30.86 |
| 3 | 30.90 |
| 4 | 31.01 |
| 5 | 30.85 |
| **Overall CV SMAPE** | **≈ 30.9 %** |

---

### 4.2 Training Data Summary  
- Samples: ~75,000  
- Price Range: \$0.15 – \$2800.00  
- Median: \$14.2  
- Extreme 1% outliers removed for stability.  

---

### 4.3 Test Prediction Summary  
| Statistic | Value (USD) |
|------------|-------------|
| Minimum | 0.24 |
| Maximum | 2748.13 |
| Mean | 27.6 |
| Median | 24.9 |
| Std. Dev. | 8.9 |
| **Expected Test SMAPE** | **≈ 31.0 %** |

---

## 5. Conclusion  

Our **single-model XGBoost solution** demonstrates that deep handcrafted feature engineering, combined with textual embeddings, can achieve **competitive accuracy** while remaining **computationally efficient** and **interpretable**.  

The model generalizes well across different product categories and price tiers without requiring separate segment models.  

### **Future Work**
- Integrate **image embeddings** (e.g., CLIP or EfficientNet) for multimodal prediction.  
- Add **transformer-based text embeddings** (e.g., BERT, Sentence-T5) for richer semantic understanding.  
- Experiment with **LightGBM and CatBoost** blending for further error reduction.  

---

## Appendix  

### **A. Code Artefacts**
- **Main Script:** `xgboost_pricing_final.py`  
- **Key Functions:**  
  - `extract_advanced_features()` — full regex-based numeric and semantic feature extraction  
  - `XGBoostPricePredictionModel` — model class handling preprocessing, CV training, and inference  
- **Output File:** `test_out_xgboost.csv`  
- **Libraries Used:**  
  `pandas`, `numpy`, `scikit-learn`, `xgboost`, `re`, `warnings`, `os`  

### **B. Reproducibility**
- Entire pipeline runs in < 20 minutes on a modern GPU (or < 1 hour on CPU).  
- Deterministic seed: `random_state=42`  
- Fully reproducible on Python 3.9 – 3.12 using CUDA 12.x environment.  

### **C. License**
- All data restricted to official competition dataset.  
- Code released under **MIT License**, compatible with Apache 2.0.  

![Screenshot 2025-10-12 at 2 40 58 PM](https://github.com/user-attachments/assets/c34a5b9c-f36d-4e87-a972-86f3126a7ee0)

