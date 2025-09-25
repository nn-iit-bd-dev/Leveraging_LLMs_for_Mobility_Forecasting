# Hurricane Ian POI Visit Forecasting (D14 Task)

## 📌 Overview
This project studies the problem of **forecasting daily visits to Points of Interest (POIs)** in Florida during **Hurricane Ian (2022)**.  
The primary task is to **predict the 14th day’s visit count** at each POI, given the previous 13 days of observed visits.  

We experiment with **classical time-series models, deep learning methods, and large language models (LLMs)**, and extend them with **fine-tuning and retrieval-augmented generation (RAG)** to incorporate hurricane-specific contextual knowledge.

---

## 🗂 Data
- **Source:** SafeGraph visit patterns dataset  
- **Geographic scope:** Four Florida cities (Tampa, Miami, Orlando, Cape Coral)  
- **POI metadata:** placekey, city, location name, top category, latitude, longitude  
- **Temporal windows:** Before, during, Hurricane Ian landfall  
- **Forecasting setup:**  
  - Input sequence:  
    \\[
    v^p_{k-n+1:k-1} = [v^p_{k-13}, \\dots, v^p_{k-1}]
    \\]  
  - Forecast target:  
    \\[
    v^p_k \\quad (n=14)
    \\]

---

## ⚙️ Methods
We compared multiple approaches:  

1. **Classical Models**  
   - ARIMA  
   - Prophet  

2. **Deep Learning Models**  
   - LSTM  
   - RNN
   - GRU

3. **Large Language Models (LLMs)**  
   - Zero/few-shot forecasting with LLaMA-3.1-8B and Mistral-7B  
   - Fine-tuning with **LoRA** adapters for D14 forecasting  
   - **RAG-enhanced prompts** using hurricane-specific evacuation and landfall context  

---

## 🧪 Key Findings
- **Classical & deep models** perform reasonably but fail to adapt well when hurricane disruptions change visit patterns.  
- **LLMs without fine-tuning** show limited accuracy.  
- **Fine-tuned LLMs (LoRA)** substantially improve D14 predictions.  
- **RAG-enhanced prompts** (e.g., “Evacuation order announced 2 days ago”) allow LLMs to incorporate real-world context and further boost accuracy.  

---


## 📂 Repository Structure
```
## Project Structure

├── arima_models/          # Scripts for ARIMA-based forecasting
├── dl/                    # Scripts for deep learning models (RNN, LSTM, GRU)
├── llms/                  # Scripts for Large Language Model baselines
├── prophet_model/         # Scripts for Prophet forecasting
├── rag/                   # Retrieval-Augmented Generation (RAG) forecasting
├── tuned_llm/             # Fine-tuned LLM implementations
├── utilities/             # Helper functions (e.g., data loaders, metrics, plotting, I/O utils)
└── README.md              # Project documentation


---
```

## Full Prompt Format for LLM Forecasting

This document describes how each training or inference instance is converted into a structured **prompt** and fed into the LLM.  

---

# Sample Data Instance

This example illustrates a single **train/test record** in the hurricane mobility forecasting dataset.

---

## Metadata
- **Placekey**: `zzw-226@8dj-jy9-x89`  
- **City**: Tampa  
- **Location Name**: Gr8Physiques Fitness Solutions  
- **Top Category**: Other Amusement and Recreation Industries  
- **Latitude**: 28.004637  
- **Longitude**: -82.59652  

---

## Temporal Information
- **Series Start Date**: 2022-09-19  
- **Landfall Date**: 2022-09-28  
- **Target Date (Day 14)**: 2022-10-02  
- **Actual Target Date**: 2022-10-02  
- **Time Periods Used**: after; before; landfall  
- **Target Days After Landfall**: 4  

---

## Input Data (13-day History)
`prev_13_values` =  
```
[11.0, 5.0, 9.0, 6.0, 5.0, 7.0, 3.0, 3.0, 4.0, 0.0, 0.0, 13.0, 6.0]
```

---

## Output Data (Ground Truth)
- **y_true_d14**: 6.0  

---

## RAG / Evacuation Context
- **FIPS County Code**: 12057  
- **County Name**: Hillsborough  
- **Evacuation Severity**: 3 (MANDATORY evacuation)  
- **Evacuation Days Since**: 5  
- **RAG Active**: true  

# Prompt Generation Functions

This document describes the functions used to generate training and inference prompts for the forecasting model.

---

## 1. `create_rag_training_prompt(row)`

This function builds a **training prompt** that includes the ground-truth label (`y_true_d14`) for supervised fine-tuning.

### Inputs
A dictionary-like row from the dataset containing:
- Metadata (`location_name`, `top_category`, `city`, `series_start_date`, `landfall_date`, `target_date_d14`)
- Visit history (`prev_13_values`)
- Evacuation info (`rag_active`, `evacuation_severity`, `evacuation_days_since`)
- Ground truth (`y_true_d14`)

### Logic
1. Construct a base instruction and metadata block.  
2. If `rag_active == True`, add an **evacuation context line**:  
   - `"MANDATORY"` if `evacuation_severity >= 3`  
   - `"Voluntary"` otherwise  
   - Days since order taken from `evacuation_days_since`  
3. Append 13 daily visit counts.  
4. Append the **prediction with ground truth** (e.g., `PREDICTION: 6.0`).  

### Example Output (with RAG)
```
You are an expert forecaster. Given 13 daily visit counts and evacuation context, predict day 14. 
Reply ONLY: 'PREDICTION: <number>'.
Location: Gr8Physiques Fitness Solutions
Category: Other Amusement and Recreation Industries
City: Tampa
SeriesStart: 2022-09-19
Landfall: 2022-09-28
TargetDate: 2022-10-02
Evacuation Context: MANDATORY evacuation active (5 days since effective)
Visits(1-13): 11.0, 5.0, 9.0, 6.0, 5.0, 7.0, 3.0, 3.0, 4.0, 0.0, 0.0, 13.0, 6.0
PREDICTION: 6.0
```

---

## 2. `create_rag_inference_prompt(row)`

This function builds an **inference prompt** where the model must generate the prediction. No ground truth is shown.

### Inputs
Same as above, except `y_true_d14` is not appended.  

### Logic
1. Construct base instruction and metadata.  
2. Conditionally add evacuation context (same rules as training).  
3. Append 13 daily visits.  
4. End with a blank `PREDICTION:` for the model to complete.  

### Example Output (with RAG)
```
You are an expert forecaster. Given 13 daily visit counts and evacuation context, predict day 14. 
Reply ONLY: 'PREDICTION: <number>'.
Location: Gr8Physiques Fitness Solutions
Category: Other Amusement and Recreation Industries
City: Tampa
SeriesStart: 2022-09-19
Landfall: 2022-09-28
TargetDate: 2022-10-02
Evacuation Context: MANDATORY evacuation active (5 days since effective)
Visits(1-13): 11.0, 5.0, 9.0, 6.0, 5.0, 7.0, 3.0, 3.0, 4.0, 0.0, 0.0, 13.0, 6.0
PREDICTION:
```

---

## Key Notes
- `rag_active == True` controls whether evacuation context is inserted.  
- **Mandatory orders** are recognized when `evacuation_severity >= 3`.  
- Training prompts include the **answer**, inference prompts do not.  
- Visit sequences are always shown as **13 comma-separated floats**.  
## 📊 Evaluation
Metrics used:  
- MAE (Mean Absolute Error)  
- RMSE (Root Mean Squared Error)
- RMSLE (Root Mean Squared Logarithmic Error)


Outputs include per-POI error analysis, top-10 best/worst cases, and visualizations.  

---
# Implementation Details

For efficiency, we used **quantized inference** with **4-bit NF4** via the `bitsandbytes` library.  
Tokenization employed the model’s **SentencePiece vocabulary**, with the **EOS token** used for padding.  

---

## Data Preparation
- Dataset split into **70/10/20** for **training / validation / testing**.  

---

## Fine-Tuning
We applied **PEFT with LoRA** adapters.  

- **Adapter locations**: attention and feed-forward projections  
  - $q_{\mathrm{proj}}, k_{\mathrm{proj}}, v_{\mathrm{proj}}, o_{\mathrm{proj}}, \mathrm{gate}_{\mathrm{proj}}, \mathrm{up}_{\mathrm{proj}}, \mathrm{down}_{\mathrm{proj}}$  
- **Hyperparameters**:  
  - Rank = 16  
  - $\alpha = 32$  
  - Dropout = 0.1  

This reduced the number of trainable parameters to a **small fraction of the base model** while preserving representational capacity.

---

## Training
- Epochs: **3**  
- Evaluation: run every **100 steps**  
- Best model: selected by **lowest validation loss** 
---

##  How to Run
1. Prepare train/test JSONL datasets (per city or multi-city).  
2. Run classical/deep baselines:  
   ```bash
   python arima_tampa_14.py --input ts_daily_panel/tampa_daily_panel.parquet --target d14
   ```  
3. Run LLM inference (example with LLaMA-3.1-8B):  
   ```bash
   python run_llama.py --input prepared_data/tampa_test.jsonl --output results/tampa_predictions.csv
   ```  

---


### 📈 Results: Mistral Variants Across Cities

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


## Orlando – Sample Predictions: Our Framework

Below are examples of the **10 best** and **10 worst** model predictions for Orlando POIs.  
- *Best cases* are low-traffic POIs (stable, easy to predict).  
- *Worst cases* are high-volume, event-driven POIs (airports, theme parks, tourist restaurants).  

### 🔹 10 Best Predictions
| Rank | Placekey              | Location Name                                | Category                                   | True Visits | Predicted | Abs. Error |
|------|-----------------------|----------------------------------------------|--------------------------------------------|-------------|-----------|------------|
| 1    | 223-222@8fy-7x8-wkz   | Orlando Auto Mall                            | Automobile Dealers                          | 0           | 0         | 0          |
| 2    | 223-222@8fy-7yv-z9f   | Dsign Factory                                | Printing & Related Support Activities       | 0           | 0         | 0          |
| 3    | 22y-222@8fy-7wz-m8v   | Osceola Gynecology                           | Offices of Physicians                       | 0           | 0         | 0          |
| 4    | 222-223@8fy-7zy-rzf   | V & N Complete Auto Repair                   | Automotive Repair and Maintenance           | 0           | 0         | 0          |
| 5    | zzy-222@8fy-8mg-cdv   | Cam Miller Realty                            | Real Estate Agents and Brokers              | 0           | 0         | 0          |
| 6    | 222-222@8fy-82q-qfz   | Adolescent Substance Abuse Program           | Elementary & Secondary Schools              | 0           | 0         | 0          |
| 7    | 222-222@8fy-8kf-z4v   | Sofrito Latin Cafe                           | Restaurants and Other Eating Places         | 4           | 4         | 0          |
| 8    | 225-225@8fy-8n3-xt9   | AdventHealth Medical Group Surgery Lake Nona | Offices of Physicians                       | 0           | 0         | 0          |
| 9    | zzy-229@8fy-8bj-835   | Bright Light Paper                           | Florists                                    | 5           | 5         | 0          |
| 10   | 229-223@8fy-8bv-djv   | Flying Window Tinting                        | Building Finishing Contractors              | 0           | 0         | 0          |

### 🔹 10 Worst Predictions
| Rank | Placekey              | Location Name                                | Category                                   | True Visits | Predicted | Abs. Error |
|------|-----------------------|----------------------------------------------|--------------------------------------------|-------------|-----------|------------|
| 1    | 225-22w@8fy-7z9-2tv   | Orlando International Airport                | Support Activities for Air Transportation   | 5808        | 6544      | 736        |
| 2    | zzw-22r@8fy-8jw-qcq   | Hagrid's Magical Creatures Motorbike Ride    | Amusement Parks and Arcades                 | 398         | 16        | 382        |
| 3    | zzy-22c@8fy-d6k-bkz   | Woody's Lunch Box                            | Amusement Parks and Arcades                 | 318         | 40        | 278        |
| 4    | zzy-22q@8fy-8m2-k9f   | Big Thunder Mountain Railroad                | Amusement Parks and Arcades                 | 359         | 97        | 262        |
| 5    | zzw-223@8fy-8kn-j35   | Disney Springs                               | Lessors of Real Estate                      | 3490        | 3704      | 214        |
| 6    | 222-235@8fy-8jx-dn5   | The Hello Kitty Shop                         | Book Stores and News Dealers                | 178         | 0         | 178        |
| 7    | zzy-222@8fy-83t-fvf   | Dunkin'                                      | Restaurants and Other Eating Places         | 179         | 1         | 178        |
| 8    | zzw-22j@8fy-8jw-qcq   | The Amazing Adventures of Spider-Man         | Amusement Parks and Arcades                 | 181         | 10        | 171        |
| 9    | zzw-222@8fy-8k6-9xq   | Raptor Encounter                             | Amusement Parks and Arcades                 | 208         | 81        | 127        |
| 10   | zzy-223@8fy-7z9-rff   | Orlando Intl Airport – Airside 2             | Support Activities for Air Transportation   | 580         | 684       | 104        |

# Category-Wise Performance

This section summarizes forecasting performance across different business categories of Orlando.  
We report the number of POIs in each category (**Count**) and the average forecasting error (**Mean Absolute Error, MAE**).  

---

## Performance Table

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

## ✨ Contributions
- Built a full **data preparation pipeline** for hurricane-aware POI visit forecasting.  
- Implemented and compared **classical, deep learning, and LLM-based approaches**.  
- Designed **generalized prompt templates** and **LoRA fine-tuning** for forecasting tasks.  
- Incorporated **geospatial RAG knowledge** (evacuation orders, landfall dates) into LLM forecasts.  

---


```
