# ⛏️ Mintek Flotation Plant AI Optimizer

> **Mintek SCI Grad Hackathon (2025)**  
> **Challenge:** "How can AI and machine learning improve recovery rates in flotation circuits?"  
> **Deliverable:** An algorithm prototype and predictive machine learning model to optimize chemical reagent dosing in real-time.  
> **Author:** Bonga Manzini (UNISA BSc Informatics Student)

---

## 📖 The Story Behind This Project

In industrial mining and froth flotation plants, checking the quality of iron and silica concentrate usually requires sending physical samples to a laboratory, which takes hours. During that time, the plant could be wasting expensive chemical reagents or producing ore with too many silica impurities.

For the **Mintek SCI Grad Hackathon (2025)**, I built an AI model that predicts silica impurity percentages ahead of time using continuous sensor streams (pulp pH, pulp density, air flow, and feed grade). This allows mining engineers to adjust reagent dosing in real time to maximize iron recovery while keeping silica under penalty limits (< 2.0%).

---

## 📊 Dataset & Metrics

- **Dataset:** [Quality Prediction in a Mining Process (Kaggle)](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process)
- **Observations:** 737,453 continuous sensor records sampled at 20-second intervals over 6 months of active plant operations.
- **Target Variables:** `% Silica Concentrate` (impurity) and `% Iron Concentrate` (valuable output).

---

## 🚀 Running the Model Locally

```bash
# Evaluate against real plant database
python3 flotation_plant_optimizer.py --feed-silica 16.5 --target-impurity 2.0 --eval-dataset
```

---

## 🤝 Contributing & Pull Requests

Feedback from mining engineers, data scientists, and fellow students is warmly welcomed!
1. Fork the repo.
2. Create your branch (`git checkout -b feature/improved-xgboost`).
3. Commit your changes and open a Pull Request!
