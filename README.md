# Diamond Price Prediction: Comparing 5 Regression Algorithms with 10-Fold CV

**Assignment 3 – Machine Learning Fundamentals (AIC354), COMSATS University Islamabad, Lahore Campus**

**Student:** Ebbad Ur Rehman &nbsp;|&nbsp; **Registration No.:** FA24-BSE-032 &nbsp;|&nbsp; **Instructor:** Dr. Rao Muhammad Adeel Nawab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebbad-dev/Diamond-ML-Problem/blob/main/diamond_price_kfold_10fold.ipynb)

**Live app:** https://diamond-price-predicton.streamlit.app/

A regression experiment that predicts diamond prices (US $) from the
[Kaggle Diamonds dataset](https://www.kaggle.com/datasets/shivam2503/diamonds) (53,940 diamonds, full dataset).
Five scikit-learn algorithms that were not used in the Car Price Prediction notebook are compared with
**10-fold cross validation**, using the same folds, the same pre-processing pipeline and the same metrics (RMSE, MAE, R²).
The selected model is served by a **Streamlit** web app.

| File | Purpose |
|---|---|
| `diamond_price_kfold_10fold.ipynb` | The experiment: data loading, pre-processing, 10-fold CV of 5 algorithms, timing / CPU / RAM / GPU monitoring, model selection, saving the model, application phase |
| `diamonds.csv` | The full dataset (53,940 rows) |
| `diamond_price_model.pkl` | Trained final model (whole scikit-learn pipeline) |
| `model_info.json` | Cross-validation results and feature information shown in the app |
| `app.py` | Streamlit web app |
| `requirements.txt` | Packages for the app (Streamlit Community Cloud installs these) |

## Algorithms

| # | Algorithm | Family |
|---|---|---|
| 1 | Ridge Regression | Linear (L2-regularised) |
| 2 | Extra Trees | Ensemble – bagging |
| 3 | Hist Gradient Boosting | Ensemble – boosting (histogram based) |
| 4 | AdaBoost | Ensemble – boosting |
| 5 | MLP Regressor | Neural network |

## Results (10-fold cross validation, 53,794 unique diamonds)

| Rank | Algorithm | Mean RMSE | Std RMSE | Mean MAE | R² | Time, 10 folds (4-core CPU) |
|---|---|---|---|---|---|---|
| 1 | Extra Trees | **$543** | $20 | **$266** | **0.981** | ≈ 60 s |
| 2 | Hist Gradient Boosting | $551 | $27 | $290 | 0.981 | ≈ 10 s |
| 3 | MLP Neural Network | $682 | $144 | $329 | 0.969 | ≈ 265 s |
| 4 | Ridge Regression | $1,136 | $36 | $735 | 0.919 | ≈ 2 s |
| 5 | AdaBoost | $1,403 | $26 | $1,028 | 0.876 | ≈ 45 s |

**Deployed model: Hist Gradient Boosting.** Extra Trees is the most accurate, but its trained model is 682 MB:
too large for GitHub (100 MB file limit) and for a Streamlit Community Cloud app (~1 GB RAM).
Hist Gradient Boosting's model is 0.37 MB, trains ~6× faster and its RMSE is only 1.5% higher.
The notebook makes this choice automatically (Step 10.2).

## 1. Run the experiment on Google Colab

1. Click the **Open in Colab** badge above (or https://colab.research.google.com → **File → Upload notebook**).
2. **Runtime → Change runtime type → CPU** → **Runtime → Run all**.
   Step 2.1 downloads the dataset automatically (Kaggle, then this GitHub repo); only if both fail, click **Choose Files** and upload `diamonds.csv`.
3. Read the timing / resource table in **Step 9.4**. In **Step 9.5**, set `DOWNLOAD_RESULTS = True` and re-run the cell to download `cv_results_CPU.csv`.
4. **GPU investigation:** **Runtime → Change runtime type → T4 GPU** → **Run all** again (downloads `cv_results_GPU.csv`).
   Optionally run a third time with `USE_CUML_ACCEL = True` in Step 0.2 (RAPIDS cuML runs supported estimators on the GPU).
5. In **Step 9.6**, set `COMPARE_RUNTIMES = True`, run the cell and upload the CSV(s) from the other run(s) to get the CPU vs GPU speed-up table.
   Copy the times into the *Recorded Colab Execution Times* table under Step 9.6.

## 2. Run the app locally (optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 3. Deploy on Streamlit Community Cloud

1. Push this folder to a **public GitHub repository** (it already is, if you are reading this on GitHub).
2. Go to https://share.streamlit.io and sign in with your GitHub account.
3. Click **Create app → Deploy a public app from GitHub**:
   - **Repository:** `<your-username>/<this-repo>`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** choose a name, e.g. `diamond-price-predictor`
   - **Advanced settings → Python version:** 3.12 or 3.13
4. Click **Deploy**. The first build takes 2–3 minutes.
5. Copy the app URL (e.g. `https://diamond-price-predictor.streamlit.app`) into the **first cell of the notebook** and the top of this README.

> **If you retrain the model on Colab** and replace `diamond_price_model.pkl` / `model_info.json` with your own files, change the
> `scikit-learn==...` line in `requirements.txt` to the version printed at the end of notebook Step 10.3. A pickled model must be
> loaded with the same scikit-learn version that created it.

## Submission checklist (Google Classroom)

- [ ] Live app link in the **first cell** of the notebook
- [ ] Notebook run on Colab (outputs visible), including the recorded CPU / GPU execution times
- [ ] `diamonds.csv` (dataset)
- [ ] `diamond_price_model.pkl` (trained model)
- [ ] `app.py`, `requirements.txt`, `model_info.json`
