# Multimodal Deep Learning for Modeling ATCO Command Lifecycle in Terminal Airspace

This repository contains the official implementation for the research project:

**"Multimodal Deep Learning for Modeling ATCO Command Lifecycle and Workload Prediction in Terminal Airspace"**  
by Kaizhen Tan, Tongji University.

## 🧠 Project Overview

This project aims to model the **lifecycle of ATCO commands** in terminal airspace by jointly predicting:

- ⏱ **Time Offset**: the delay between ATCO voice command and aircraft maneuver
- 🗣 **Duration**: the time length of ATCO command speech segments

The proposed model is a **CNN-Transformer hybrid architecture**, trained on real-world data from **Singapore Changi Airport’s TMA**, integrating:

- Structured features: flight parameters, command types, environment variables
- Historical trajectory sequences
- Airspace snapshots and historical maneuver images

A multimodal ensemble strategy is applied to enhance prediction robustness and interpretability.

## 🧩 Features

- ✅ Joint modeling of **Time Offset** and **Duration** as ATCO lifecycle targets
- 🧠 **CNN-Transformer hybrid model** with structured, sequential, and visual inputs
- 📈 Built-in baseline comparison using **LightGBM** and **TabPFN**
- 📊 SHAP & Grad-CAM explainability for structured and image features
- 🧪 End-to-end training pipeline with data augmentation and ensemble inference
