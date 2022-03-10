# Conditional Self-supervised for histopathology images

This repository contains the code for the paper [Interpretable Prediction of Lung Squamous Cell Carcinoma Recurrence With Self-supervised Learning]().

## Introduction
In this study, we explore the morphological features of LSCC recurrence and metastasis with novel SSL method, based on conditional SSL. We propose a sampling mechanism within contrastive SSL framework for histopathology images that avoids overfitting to batch effects. 

The 2D UMAP projection of tile representations, trained by different sampling
in self-supervised learning. Tiles from 8 slides with mostly LSCC tumor content
are highlighted with different colors. Left: model trained by MoCo contrastive
learning with uniform sampling. It shows that tiles within each slide cluster
together. Right: model trained with proposed conditional contrastive learning.
The tiles from each slide are less clustered together.
![UMAP](./plots/umap.png)

The Kaplan-Meier curves shows rates of recurrence-free patients over time in
sub-cohorts of test set with different criterion. Two sub-cohorts stratified with the predicted
recurrence risk by our Cox regression. The high risk cohort includes the top half
patients of highest estimated risks; the low risk cohort includes the lower half.
![progression](./plots/progression_plot.png)


## Self-supervised learning

