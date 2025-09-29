# Hurricane Ian POI Visit Forecasting (D14 Task)

## 📑 Abstract
Natural disasters like hurricanes cause abrupt and severe disruptions in human mobility and business activity, posing significant challenges for short-term forecasting models that support disaster response and resource allocation. 

Traditional statistical and deep learning approaches often fail to adapt to the context-sensitive shifts in visitation patterns during such events. To address these limitations, we introduce a disaster-aware mobility prediction framework that integrates **large language models (LLMs)** with **retrieval-augmented generation (RAG)**. 

The framework uses prompting to seamlessly integrate high-resolution **point-of-interest (POI)-level mobility data** with **geospatial contextual signals**, such as evacuation orders. It also employs **parameter-efficient fine-tuning** to create LLMs tailored specifically for mobility forecasting tasks during disasters. 

Experimental results using mobility data during Hurricane Ian in four Florida cities show that the proposed LLM-based framework consistently outperforms traditional methods in most scenarios, indicating its strong ability to comprehend disrupted mobility patterns. These findings highlight the potential of integrating structured mobility data with contextual language comprehension to foster more resilient and adaptable forecasts in disaster situations. 

---

## 🗂 Data
- **Source:** SafeGraph visit patterns dataset  
- **Geographic scope:** Four Florida cities (Tampa, Miami, Orlando, Cape Coral)  
- **POI metadata:** placekey, city, location name, top category, latitude, longitude  
- **Temporal windows:** Before, during, Hurricane Ian landfall  
- **Forecasting setup:**
  - **Input sequence:**  
    $$
    v^p_{k-n+1:k-1} = [v^p_{k-13}, \dots, v^p_{k-1}]
    $$
  - **Forecast target:**  
    $$
    v^p_k \quad (n = 14)
    $$
---

## 📂 Repository Structure
```
├── arima_models/          # Scripts for ARIMA-based forecasting
├── dl/                    # Scripts for deep learning models (RNN, LSTM, GRU)
├── llms/                  # Scripts for Large Language Model baselines
├── prophet_model/         # Scripts for Prophet forecasting
├── rag/                   # Retrieval-Augmented Generation (RAG) forecasting
├── tuned_llm/             # Fine-tuned LLM implementations
├── utilities/             # Helper functions (data loaders, metrics, plotting, I/O utils)
└── README.md              # Project documentation
```

---

## 📝 Full Prompt Format for LLM Forecasting

This section describes how each training or inference instance is converted into a structured **prompt** and fed into the LLM.  

### 1. Training Prompt
Includes the **ground-truth label (`y_true_d14`)** for supervised fine-tuning.  

### 2. Inference Prompt
Model must generate the prediction. No ground truth is shown.  

---

## ⚙️ Prompt Generation Functions

### `create_rag_training_prompt(row)`
- Builds a **training prompt** with ground truth.  
- Adds evacuation context when `rag_active == True`.  
- **Mandatory** if `evacuation_severity >= 3`.  

### `create_rag_inference_prompt(row)`
- Builds an **inference prompt** without ground truth.  
- Same evacuation rules as training.  

---

## 📊 Evaluation
**Metrics used:**  
- MAE (Mean Absolute Error)  
- RMSE (Root Mean Squared Error)  
- RMSLE (Root Mean Squared Logarithmic Error)  

Outputs include per-POI error analysis, top-10 best/worst cases, and visualizations.  

---

## 🛠 Implementation Details (LLM+LoRA_RAG)

- **Quantized inference** with 4-bit NF4 via `bitsandbytes`.  
- Tokenization: model’s **SentencePiece vocabulary**, with EOS as padding.  
- **Data split:** 70/10/20 for training/validation/test.  
- **Fine-tuning:** LoRA adapters on attention & feed-forward projections.  
  - Rank = 16, α = 32, Dropout = 0.1  
- **Training:** 3 epochs, evaluation every 100 steps.  
- **Best model** selected by lowest validation loss.  

---

## 🛠 Implementation Details (Deep Learning Models)

- **Models:** RNN, LSTM, GRU with learnable **city** and **category** embeddings (dim=16 each).  
- **Data split:** 70/10/20 (train/validation/test).  
- **Normalization:** Input sequences standardized with `StandardScaler` (optional via `--normalize`).  

- **Training:**  
  - Hidden size = 64, Layers = 2, Dropout = 0.2  
  - Optimizer: Adam (`lr=1e-3`)  
  - Loss: MSE (predictions clamped ≥0, rounded to int for reporting)  
  - Batch size = 32, Epochs = 30  
  - Early stopping with patience = 5  
  - LR scheduler: ReduceLROnPlateau (patience=5, factor=0.7)  

- **Evaluation metrics:**  
  MAE, RMSE, RMSLE, SMAPE (mean & median), Exact-match accuracy.  

- **Best model:**  
  Per type saved by lowest validation loss (`best_<model>.pth`).  

- **Reports generated:**  
  - Per-city metrics (`*_city_metrics.csv`)  
  - Per-category metrics (`*_category_metrics.csv`)  
  - Overall summary JSON (`*_summary.json`)  
  - Dataset summary JSON (`*_dataset_summary.json`)  
  - Combined multi-model city metrics (`all_models_city_metrics.csv`, `all_models_city_metrics_wide.csv`)  

---

## 🛠 Implementation Details (Classical Models:ARIMA)

- **Forecast horizon:** D14 (train on days 1–13).  
- **Model search:** up to **8 ARIMA (p,d,q)** candidates, selected by **lowest AIC**.  
- **Fallback:** last observed value if ARIMA fails.  
- **Prediction:** clamped [0, 10,000], rounded to integer.  
- **Parallelization:** configurable (`--parallel`, default auto ≤8 workers).  
- **Metrics:** MAE, RMSE, RMSLE, sMAPE (mean/median), median error.  
- **Epochs/iterations:** single-step forecast per placekey (no retraining epochs).  
---

## 🛠 Implementation Details (Classical Models:Prophet)


- **Horizon:** D14 (train on first **13 days**, predict day **14**).
- **Prophet config:**
  - `yearly_seasonality=False`
  - `weekly_seasonality=False`
  - `daily_seasonality=False`
  - `changepoint_prior_scale=0.1`
  - `seasonality_prior_scale=0.1`
  - `interval_width=0.8`  *(80% CI)*
  - `uncertainty_samples=100`
- **Fallback logic:**  
  - Constant series → last value (naive)  
  - All zeros → 0 (naive)  
  - Any Prophet error → last value (naive)
- **Validation metrics:** `MAE`, `RMSE` , RMSLE, sMAPE (mean/median), median error
- **Execution controls:** `CITIES=[Tampa, Miami, Orlando, Cape Coral]`, `--sample` not used (full data).


## 🚀 How to Run
1. Prepare train/test JSONL datasets.  
2. Run classical/deep baselines:  
   ```bash
   python arima_tampa_14.py --input ts_daily_panel/tampa_daily_panel.parquet --target d14
   ```  
3. Run LLM inference (example with LLaMA-3.1-8B):  
   To run the forecasting pipeline with the **LLaMA-8B** model on Tampa test data:

```bash
python llama8b_forecast.py \
  --input prepared_data/tampa_2025_09_01/data/tampa_test.jsonl \
  --output tampa_test_final_predictions1.csv \
  --city tampa \
  --batch-size 64 \
  --generate-reports 
``` 
---

## 📈 Results: Mistral Variants Across Cities

| City       | Model                | MAE    | RMSE    | RMSLE |
|------------|----------------------|--------|---------|-------|
| **Tampa**      | Mistral              | 3.72   | 26.12   | 0.75 |
|            | Mistral + RAG        | 4.73   | 25.37   | 0.86 |
|            | **Mistral + LoRA**       | **2.76** | 18.29   | 0.79 |
|            | Mistral + LoRA + RAG | 2.80   | **14.42** | **0.79** |
| **Orlando**    | Mistral              | 5.48   | 27.95   | 0.83 |
|            | Mistral + RAG        | 6.29   | 36.72   | 0.89 |
|            | Mistral + LoRA       | 3.50   | 18.00   | **0.68** |
|            | **Mistral + LoRA + RAG** | **3.47** | **16.60** | 0.69 |
| **Miami**      | Mistral              | 4.25   | 22.53   | 0.85 |
|            | Mistral + RAG        | 4.76   | 24.74   | 0.88 |
|            | Mistral + LoRA       | 2.31   | 5.70    | 0.70 |
|            | **Mistral + LoRA + RAG** | **2.19** | **5.16**  | **0.68** |
| **Cape Coral** | Mistral              | 5.24   | 23.85   | 0.87 |
|            | Mistral + RAG        | 7.42   | 31.34   | 1.06 |
|            | Mistral + LoRA       | 2.50   | 7.02    | 0.78 |
|            | **Mistral + LoRA + RAG** | **2.16** | **5.01**  | **0.77** |

---

## 🌆 Orlando – Sample Predictions (Our Framework)

- **Best cases:** low-traffic POIs (stable, easy to predict).  
- **Worst cases:** high-volume, event-driven POIs (airports, theme parks, tourist restaurants).  

### 🔹 10 Best Predictions
| Rank | Placekey            | Location Name                                | Category                                   | True Visits | Predicted | Abs. Error |
|------|---------------------|----------------------------------------------|--------------------------------------------|-------------|-----------|------------|
| 1    | 223-222@8fy-7x8-wkz | Orlando Auto Mall                            | Automobile Dealers                          | 0           | 0         | 0          |
| 2    | 223-222@8fy-7yv-z9f | Dsign Factory                                | Printing & Related Support Activities       | 0           | 0         | 0          |
| 3    | 22y-222@8fy-7wz-m8v | Osceola Gynecology                           | Offices of Physicians                       | 0           | 0         | 0          |
| 4    | 222-223@8fy-7zy-rzf | V & N Complete Auto Repair                   | Automotive Repair and Maintenance           | 0           | 0         | 0          |
| 5    | zzy-222@8fy-8mg-cdv | Cam Miller Realty                            | Real Estate Agents and Brokers              | 0           | 0         | 0          |
| 6    | 222-222@8fy-82q-qfz | Adolescent Substance Abuse Program           | Elementary & Secondary Schools              | 0           | 0         | 0          |
| 7    | 222-222@8fy-8kf-z4v | Sofrito Latin Cafe                           | Restaurants and Other Eating Places         | 4           | 4         | 0          |
| 8    | 225-225@8fy-8n3-xt9 | AdventHealth Medical Group Surgery Lake Nona | Offices of Physicians                       | 0           | 0         | 0          |
| 9    | zzy-229@8fy-8bj-835 | Bright Light Paper                           | Florists                                    | 5           | 5         | 0          |
| 10   | 229-223@8fy-8bv-djv | Flying Window Tinting                        | Building Finishing Contractors              | 0           | 0         | 0          |

### 🔹 10 Worst Predictions
| Rank | Placekey            | Location Name                                | Category                                   | True Visits | Predicted | Abs. Error |
|------|---------------------|----------------------------------------------|--------------------------------------------|-------------|-----------|------------|
| 1    | 225-22w@8fy-7z9-2tv | Orlando International Airport                | Support Activities for Air Transportation   | 5808        | 6544      | 736        |
| 2    | zzw-22r@8fy-8jw-qcq | Hagrid's Magical Creatures Motorbike Ride    | Amusement Parks and Arcades                 | 398         | 16        | 382        |
| 3    | zzy-22c@8fy-d6k-bkz | Woody's Lunch Box                            | Amusement Parks and Arcades                 | 318         | 40        | 278        |
| 4    | zzy-22q@8fy-8m2-k9f | Big Thunder Mountain Railroad                | Amusement Parks and Arcades                 | 359         | 97        | 262        |
| 5    | zzw-223@8fy-8kn-j35 | Disney Springs                               | Lessors of Real Estate                      | 3490        | 3704      | 214        |
| 6    | 222-235@8fy-8jx-dn5 | The Hello Kitty Shop                         | Book Stores and News Dealers                | 178         | 0         | 178        |
| 7    | zzy-222@8fy-83t-fvf | Dunkin'                                      | Restaurants and Other Eating Places         | 179         | 1         | 178        |
| 8    | zzw-22j@8fy-8jw-qcq | The Amazing Adventures of Spider-Man         | Amusement Parks and Arcades                 | 181         | 10        | 171        |
| 9    | zzw-222@8fy-8k6-9xq | Raptor Encounter                             | Amusement Parks and Arcades                 | 208         | 81        | 127        |
| 10   | zzy-223@8fy-7z9-rff | Orlando Intl Airport – Airside 2             | Support Activities for Air Transportation   | 580         | 684       | 104        |

---

## 📊 Category-Wise Performance (Orlando)

| **Category** | **Count** | **Mean AE** |
|--------------|-----------|-------------|
| Amusement Parks and Arcades | 40 | 39.23 |
| Warehousing and Storage | 3 | 13.67 |
| Lessors of Real Estate | 55 | 10.78 |
| Motion Picture and Video Industries | 3 | 7.67 |
| General Merchandise Stores (incl. Warehouse Clubs & Supercenters) | 33 | 7.46 |
| Promoters of Performing Arts, Sports, and Similar Events | 26 | 6.89 |
| Office Supplies, Stationery, and Gift Stores | 26 | 6.46 |
| Colleges, Universities, and Professional Schools | 9 | 6.44 |
| Traveler Accommodation | 86 | 5.43 |
| General Medical and Surgical Hospitals | 17 | 5.12 |
| Gasoline Stations | 48 | 4.35 |
| Other Amusement and Recreation Industries | 120 | 3.23 |

---

## ✨ Contributions

This study presents the following contributions:

- We design a novel **LLM-based forecasting framework** that integrates structured mobility sequences with external disaster-specific context.  
- We evaluate our framework using mobility data in Florida during Hurricane Ian, benchmarking against statistical (ARIMA and Prophet) and deep learning (RNN, LSTM, and GRU) baselines, and demonstrate improved performance in capturing disrupted visitation patterns.  

---
