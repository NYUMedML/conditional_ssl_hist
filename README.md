# Conditional Self-supervised for histopathology images

This repository contains the code for the paper [Interpretable Prediction of Lung Squamous Cell Carcinoma Recurrence With Self-supervised Learning](https://arxiv.org/pdf/2203.12204.pdf).

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

<p align="center">
<img src="./plots/progression_plot.png" width="400"/>
</p>

## Data 

### TCGA-LUSC
Download the TCGA-LUSC whole slide image from this [filter](https://portal.gdc.cancer.gov/repository?facetTab=files&filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.project.project_id%22%2C%22value%22%3A%5B%22TCGA-LUSC%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.data_format%22%2C%22value%22%3A%5B%22svs%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.experimental_strategy%22%2C%22value%22%3A%5B%22Tissue%20Slide%22%5D%7D%7D%5D%7D). 

### CPTAC-LSCC
Download the TCGA-LSCC whole slide image from [here](https://wiki.cancerimagingarchive.net/display/Public/CPTAC-LSCC).

## Preprocessing

To preprocess the WSIs, run the code in preprocessing folder.

`python process_tcga.py --followup_path {followup_table} --wsi_path {directory_of_WSIs} --refer_img {color_norm_img} --s {proportion_of_tissue}`

`python process_cptac.py --followup_path {followup_table} --wsi_path {directory_of_WSIs} --refer_img {color_norm_img} --s {proportion_of_tissue}`

## Self-supervised learning

Run the command to train the Inception V4 with conditional SSL on two-layer sampling.

`torchrun train.py --data_dir {data_dir} --split_dir {annotation_dir} --batch_slide_num {number of slides in batch} --cos --out_dir {output_dir}`

Pretrained weight can be downloaded [here](https://drive.google.com/drive/folders/1Uc7JZZRkBNxoKkDmy-fcLsy9cUz_ixcr?usp=sharing).

## Extract features

To extract features, we first extract the tile representations with SSL pretrained Inception V4.

`python extract_embeddings.py --feature_extractor_dir {checkpoint of pretrained feature extractor} --subtype_model_dir {subtype model} --root_dir {tiles directory} --split_dir {annotation files} --out_dir {output directory}`

Then we fit the clusters of the tile reprenstations in the training data, and assign the clusters to tiles in the validation and test set. After clustering each tile, we aggregate tile
probabilities with average pooling on clusters to generate the slide-level features. Run the following commends:

`python get_clusters.py --data_dir {data_dir} --cluster_type {method_of_clustering} --n_cluster {number_of_clusters} --out_dir {out_dir}`

## Survival model (Cox-PH)

We run the Cox-PH regression on the extracted slide-level features and the time and status of recurrence.

The triplet of features and
slide labels $\{(v_j , y_j , t_j)\}^N_{j=1}$ will be used, where $v_j$ is the vector of cluster features, $y_j$ is the
binary label indicating LSCC recurrence, and $t_j$ encodes the recurrence-free followup times
for the patient. i.e. If a patient was not observed to have recurrence during the followup
period, we use the length of followup time $t_j$ as the time of censoring. Each $t_j$ is computed
with a granularity of 6 months. We fit a Cox regression model with L2-norm regularization
using $\{(v_j , y_j , t_j)\}^N_{j=1}$  to compute the proportional hazard function of recurrence $\lambda (t|v)$.

`python cox.py --data_dir {data_dir} --cluster_name {cluster_model_checkpoint} --noramlize {pooling_method_over_slides}`

## Reference

<blockquote>
    <p>@misc{https://doi.org/10.48550/arxiv.2203.12204,
  doi = {10.48550/ARXIV.2203.12204},
  url = {https://arxiv.org/abs/2203.12204},
  author = {Zhu, Weicheng and Fernandez-Granda, Carlos and Razavian, Narges},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Interpretable Prediction of Lung Squamous Cell Carcinoma Recurrence With Self-supervised Learning},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
</p>
</blockquote>
