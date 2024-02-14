# 🚧 Under Construction! 🚧

## Introduction

This project is part of the [ReMAP](https://h2020-remap.eu/) initiative, with its primary objective focused on enhancing the Remaining Useful Life (RUL) prediction of complex aeronautical elements. Specifically, it aims to upscale the prediction capabilities for multi-stiffener composite panels (MSPs) by leveraging historical data from single-stiffener panel histories (SSPs).

## Overview

- **Run-to-Failure Data**: Strain measurements from MSPs subjected to compression-compression fatigue are used to construct damage sensitive features, i.e., health indicators (HIs).
- **Feature-Level Fusion**: Genetic Algorithms are employed to fuse HIs, enhancing monotonicity and prognosability attributes.
- **Ensemble Approach**: An ensemble approach aggregates RUL predictions by training diverse sub-models and combining predictions with a dynamic weighting strategy based on Fuzzy Similarity Analysis (FSA).
- **Regression Techniques**: Support Vector Regression (SVR) and Long Short-Term Memory Network (LSTMN) are considered as regression techniques to map input data to RUL output.
- **Comparison of ML and DL Ensembles**: Comparative analysis between machine learning (ML) and deep learning (DL) ensembles based on standard evaluation metrics such as RMSE, MAE, MAPE, CRA.
