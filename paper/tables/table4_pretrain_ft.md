# Table 3: Transfer Learning Membership Leakage (Pretrain -> Fine-tune)

Public pretraining (10,000 compounds for 60 epochs) followed by private fine-tuning (10,000 compounds for 20 epochs). Evaluated on 8 leave-one-out shadow targets using `nll_sum`.

*Note: There is no from-scratch control trained at the exact same size and epochs in this repository.*

| Attack Method | ROC-AUC | TPR @ 1% FPR (%) | TPR @ 0.1% FPR (%) |
|---|---|---|---|
| Loss Thresholding (LOSS) | 0.640 ± 0.003 | 2.31 ± 0.26 | 0.34 ± 0.08 |
| Offline LiRA | 0.801 ± 0.003 | 4.36 ± 0.51 | 0.43 ± 0.26 |
| Online LiRA | 0.751 ± 0.004 | 2.00 ± 0.22 | 0.22 ± 0.05 |
