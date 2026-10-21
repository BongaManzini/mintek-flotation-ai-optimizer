# Mintek Flotation Plant AI Optimizer

An industrial mineral processing AI pipeline engineered to predict silica impurity percentages in iron ore concentrate and optimize chemical reagent dosing (Amina collector and Starch depressant flow rates) in real-time.

Developed for the **Mintek Graduate Innovation Hackathon: AI for Mineral Processing Optimization Challenge**.

## Problem Statement

In industrial froth flotation plants, assaying iron and silica concentrate quality typically requires laboratory physical sampling that takes hours. By leveraging process sensor telemetry (pulp pH, pulp density, column air flow, column level, feed grade), this AI model predicts silica concentration ahead of time, allowing engineers to dynamically adjust reagent flow to maximize iron recovery while keeping silica below contract penalty limits (< 2.0%).

## Kaggle Dataset

* **Dataset**: [Quality Prediction in a Mining Process](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process)
* **Records**: 737,453 continuous plant sensor observations sampled at 20-second intervals over 6 months.

## Usage

```bash
# Evaluate against real plant database
python flotation_plant_optimizer.py --feed-silica 16.5 --target-impurity 2.0 --eval-dataset
```
