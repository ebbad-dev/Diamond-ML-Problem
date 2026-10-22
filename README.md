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

## Results (10-fold cross validation on Google Colab, 53,794 unique diamonds)

| Rank | Algorithm | Mean RMSE | Std RMSE | Mean MAE | R² | Colab CPU time | Colab T4 GPU time |
|---|---|---|---|---|---|---|---|
| 1 | Extra Trees | **$543** | $20 | **$266** | **0.981** | 129.4 s | 169.4 s |
| 2 | Hist Gradient Boosting | $551 | $28 | $290 | 0.981 | 11.7 s | 11.9 s |
| 3 | MLP Neural Network | $682 | $144 | $329 | 0.969 | 203.3 s | 273.7 s |
| 4 | Ridge Regression | $1,136 | $36 | $735 | 0.919 | 3.2 s | 1.8 s |
| 5 | AdaBoost | $1,403 | $26 | $1,028 | 0.876 | 34.7 s | 34.9 s |

**Deployed model: Hist Gradient Boosting.** Extra Trees is the most accurate, but its trained model is 682 MB:
too large for GitHub (100 MB file limit) and for a Streamlit Community Cloud app (~1 GB RAM).
Hist Gradient Boosting's model is 0.37 MB, trains ~11× faster and its RMSE is only 1.5% higher.
The notebook makes this choice automatically (Step 10.2).

**GPU:** on the T4 runtime the GPU stayed at 0% utilisation for every algorithm. scikit-learn runs on the CPU only,
so the GPU did not speed anything up; the time differences come from the two runs using different Colab machines.

| Folder / file | Contents |
|---|---|
| `diamond_price_kfold_10fold.ipynb` | **Final notebook**, run on the Colab **T4 GPU** runtime, including the CPU vs GPU comparison |
| `colab_runs/diamond_price_kfold_10fold_CPU_run.ipynb` | The same experiment run on the Colab **CPU** runtime |
| `cv_results_CPU.csv`, `colab_runs/cv_results_*.csv` | Timing and accuracy results of each Colab run |

## 1. Run the experiment on Google Colab

1. Click the **Open in Colab** badge above, then **File → Save a copy in Drive**.
2. **Run 1 (CPU):** **Runtime → Change runtime type → CPU** → **Runtime → Run all** (~10–15 min).
   The dataset is downloaded automatically, and at the end `cv_results_CPU.csv` is downloaded to your computer.
3. **Run 2 (GPU):** **Runtime → Change runtime type → T4 GPU** → **Runtime → Run all**.
   In **Step 9.6**, upload `cv_results_CPU.csv` when asked: the notebook shows the CPU vs GPU speed-up and prints a
   ready-to-paste table for *Recorded Colab Execution Times*.
4. *(Optional)* Run 3 with `USE_CUML_ACCEL = True` in Step 0.2 on the GPU runtime.
5. **File → Download → Download .ipynb** and submit that file (it contains all outputs).

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
