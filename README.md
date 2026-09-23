# Cross-Regional Seismic Event Classification Based on Domain Adaptation

This repository accompanies the manuscript:

**Cross-Regional Seismic Event Classification Based on Domain Adaptation**

## Overview

Deep learning has shown strong potential for automated seismic event classification. However, differences in geological structures, wave propagation conditions, station distributions, and noise environments can lead to substantial domain shifts between regions, limiting the generalization ability of models trained on one region when applied to another.

This study investigates domain adaptation for cross-regional classification of **earthquake (EQ)** and **explosion (EX)** events. Three domain adaptation approaches are evaluated under a unified classification framework:

- Domain-Adversarial Neural Network (DANN)
- Deep Adaptation Network (DAN)
- Deep Subdomain Adaptation Network (DSAN)

Conventional fine-tuning is used as the baseline.

## Dataset

Experiments are conducted using seismic waveform data from the **US_EQ_EX dataset**.

Four regions are considered:

- BASE
- MSH
- ENAM
- HLP

Cross-regional experiments are constructed by treating one region as the source domain and another as the target domain.

## Methods

The workflow includes:

1. Waveform preprocessing
2. Signal-to-noise ratio screening
3. Time-domain data augmentation
4. STFT-based time-frequency representation
5. Feature extraction using a convolutional neural network
6. Cross-regional domain adaptation using DANN, DAN, and DSAN
7. Evaluation using Accuracy and Macro-F1

## Repository Structure

The repository is organized as follows:

```text
Cross-Regional-Seismic-Event-Classification/
├── README.md
├── LICENSE
├── requirements.txt
├── FT/
├── DANN/
├── DAN/
├── DSAN/
├── target_label_ratio_analysis/
├── STFT_visualization/
├── DA_ACC_Macro_analysis/
├── region_visualization/
├── magnitude_recall_analysis/
└── negative_transfer_analysis/
```

## Code Availability

The source code used in this study is publicly available in this repository.

## Citation

Citation information will be updated after the manuscript is published.

## License

License information will be added when the source code is released.

## Contact

For questions regarding this work, please contact the authors through the corresponding author information provided in the manuscript.
