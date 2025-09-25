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

## 📊 Evaluation
Metrics used:  
- MAE (Mean Absolute Error)  
- RMSE (Root Mean Squared Error)
- RMSLE (Root Mean Squared Logarithmic Error)


Outputs include per-POI error analysis, top-10 best/worst cases, and visualizations.  

---

## 📂 Repository Structure
```
## Project Structure

├── arima_models/          # Scripts and outputs for ARIMA-based forecasting
├── dl/                    # Deep learning models (RNN, LSTM, GRU)
├── llms/                  # Large Language Model baselines
├── prophet_model/         # Prophet forecasting scripts and results
├── rag/                   # Retrieval-Augmented Generation (RAG) forecasting
├── tuned_llm/             # Fine-tuned LLM implementations and experiments
└── README.md              # Project documentation

```

---

## 🚀 How to Run
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





## ✨ Contributions
- Built a full **data preparation pipeline** for hurricane-aware POI visit forecasting.  
- Implemented and compared **classical, deep learning, and LLM-based approaches**.  
- Designed **generalized prompt templates** and **LoRA fine-tuning** for forecasting tasks.  
- Incorporated **geospatial RAG knowledge** (evacuation orders, landfall dates) into LLM forecasts.  

---


```
