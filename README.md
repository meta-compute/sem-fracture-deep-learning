**# SEM Fracture Classification Using CNNs and Vision Transformers**



**\*\*Brittle, ductile and mixed fracture classification from scanning electron microscopy images\*\***



**## Abstract**



This study evaluates transfer learning for three-class classification of steel fracture surfaces in scanning electron microscopy (SEM) images. ResNet-18, DeiT-Tiny, TinyViT-11M and TinyViT-21M were trained using a common dataset split and preprocessing pipeline. Both frozen-backbone classification and selective fine-tuning were evaluated. The best model, fine-tuned **\*\*TinyViT-21M\*\***, achieved **\*\*99.29% validation accuracy\*\***, **\*\*0.9810 macro-F1\*\*** and **\*\*0.9500 F1 for the Mixed class\*\*** on 280 validation images. It outperformed the fine-tuned ResNet-18 baseline (97.86% accuracy; 0.9525 macro-F1) on the same split and exceeded the **\*\*99.10% accuracy\*\*** reported by the original study using a **\*\*ResNet-50 binary classifier\*\***.



**## 1. Dataset and prior work**



The images come from the **\*\*[SEM Image dataset]\(**&#x68;ttps\://doi.org/10.5281/zenodo.1551059&#x30;**)\*\*** released by Campari and colleagues. The collection covers fracture surfaces of several steels examined after mechanical testing in hydrogen and reference atmospheres. The working dataset contains **\*\*1,961 labeled images\*\***, grouped as follows:



\| Class | Train | Validation | Reserved test | Total |

\|:--|--:|--:|--:|--:|

\| Brittle | 589 | 117 | 116 | 822 |

\| Ductile | 711 | 142 | 143 | 996 |

\| Mixed | 102 | 21 | 20 | 143 |

\| **\*\*Total\*\*** | **\*\*1,402\*\*** | **\*\*280\*\*** | **\*\*279\*\*** | **\*\*1,961\*\*** |



The dataset's \`Labels.csv\` also contained a non-image \`desktop.ini\` entry, which was excluded. Only **\*\*7.3%\*\*** of valid images belong to the Mixed class.



**### Original study**



Campari et al. ([*\*Computational Materials Science\**, 2026]\(https\://doi.org/10.1016/j.commatsci.2025.114366)) investigated automated identification of **\*\*brittle and ductile\*\*** fracture morphology. They compared pretrained CNNs, including ResNet-50, EfficientNet-B4, VGG-19 and DenseNet-121. Their best reported result was **\*\*99.10% accuracy with ResNet-50\*\*** on a curated **\*\*binary\*\*** classification dataset. The work also employed Grad-CAM and sliding-window analysis to investigate fracture morphology.



The published binary dataset and evaluation protocol differ from those used here. In particular, the original binary classification excluded ambiguous transition images, whereas this study retains **\*\*Mixed\*\*** as an explicit third class. The original paper reports 2,222 curated binary images; the working three-class archive analyzed here contains 1,961 valid images.



**## 2. Hypothesis and methods**



The hypothesis is that **\*\*pretrained vision-transformer and hybrid CNN–Transformer models can provide competitive or improved SEM fracture classification relative to CNN baselines, particularly when Mixed fracture morphology is explicitly represented\*\***. TinyViT is a hybrid architecture; DeiT-Tiny represents the pure transformer baseline in these experiments.



All experiments used the same \`dataset_split.csv\` and preprocessing:



\- **\*\*Split:\*\*** 1,402 training, 280 validation and 279 reserved test images. Groups were derived from filename prefixes and kept disjoint across the three partitions. These prefixes have **\*\*not\*\*** been verified as physical specimen identifiers.

\- **\*\*Input:\*\*** grayscale SEM images converted to three channels and resized to **\*\*224 × 224\*\*** pixels with bilinear interpolation; ImageNet mean/std normalization.

\- **\*\*Augmentation:\*\*** random horizontal and vertical flips (each probability 0.5), training only.

\- **\*\*Training:\*\*** ImageNet-1K pretrained backbones, batch size **\*\*16\*\***, class-weighted cross-entropy and AdamW. Weights for Brittle, Ductile and Mixed were **\*\*0.7934\*\***, **\*\*0.6573\*\*** and **\*\*4.5817\*\***, respectively.

\- **\*\*Strategy:\*\*** train a new three-class classification head with the backbone frozen, then selectively fine-tune upper layers. Head-only learning rate: **\*\*1 × 10⁻³\*\***; fine-tuned layers: **\*\*1 × 10⁻⁵\*\***; head during fine-tuning: **\*\*1 × 10⁻⁴\*\***; weight decay: **\*\*1 × 10⁻⁴\*\***.

\- **\*\*Selection:\*\*** checkpoint with the highest **\*\*validation macro-F1\*\***, which gives equal weight to the three classes. Training ran on Apple Silicon using PyTorch MPS.



For selective fine-tuning, ResNet-18 used its last residual block plus classifier; DeiT-Tiny used its final two transformer blocks plus classifier; TinyViT-11M and TinyViT-21M used the final stage plus classifier. Frozen normalization statistics were retained. The TinyViT attention modules in the trainable final stage were kept in training mode during optimization to preserve gradient flow through relative-position biases. Gradient clipping at a maximum norm of 1.0 was used for TinyViT fine-tuning.



**## 3. Results**



**\*\*Table 1. Best validation checkpoint for each trained configuration.\*\*** No test-set metrics are reported. FT = partial fine-tuning.



\| Model | Parameters | Training | Accuracy | Macro-F1 | Mixed F1 |

\|:--|--:|:--|--:|--:|--:|

\| ResNet-18 | 11.18M | Head only | 97.14% | 0.9434 | 0.8696 |

\| ResNet-18 | 11.18M | FT | 97.86% | 0.9525 | 0.8837 |

\| DeiT-Tiny | 5.52M | Head only | 95.36% | 0.8873 | 0.7143 |

\| DeiT-Tiny | 5.52M | FT | 95.36% | 0.8954 | 0.7391 |

\| TinyViT-11M | 10.55M | Head only | 97.50% | 0.9396 | 0.8444 |

\| TinyViT-11M | 10.55M | FT | 98.93% | 0.9721 | 0.9268 |

\| TinyViT-21M | 20.62M | Head only | 98.21% | 0.9554 | 0.8837 |

\| **\*\*TinyViT-21M\*\*** | **\*\*20.62M\*\*** | **\*\*FT\*\*** | **\*\*99.29%\*\*** | **\*\*0.9810\*\*** | **\*\*0.9500\*\*** |



**### 3.1 TinyViT-21M**



The best TinyViT-21M checkpoint was obtained at **\*\*epoch 3\*\*** of a restarted fine-tuning run (seed 43), initialized from the best head-only checkpoint (epoch 6). It correctly classified **\*\*278/280\*\*** validation images. In the head-only configuration, it classified **\*\*275/280\*\*** correctly.



**\*\*Table 2. Confusion matrix of fine-tuned TinyViT-21M.\*\*** Rows are true labels; columns are predictions.



\| Actual / predicted | Brittle | Ductile | Mixed |

\|:--|--:|--:|--:|

\| Brittle | **\*\*117\*\*** | 0 | 0 |

\| Ductile | 0 | **\*\*142\*\*** | 0 |

\| Mixed | 0 | 2 | **\*\*19\*\*** |



\| Class | Precision | Recall | F1 |

\|:--|--:|--:|--:|

\| Brittle | 1.0000 | 1.0000 | 1.0000 |

\| Ductile | 0.9861 | 1.0000 | 0.9930 |

\| Mixed | **\*\*1.0000\*\*** | **\*\*0.9048\*\*** | **\*\*0.9500\*\*** |



Compared with the head-only model, fine-tuning corrected three Ductile images previously predicted as Mixed: \`316L_AR_21_19.png\`, \`316L_AR_21_20.png\` and \`X65_AR_12_25.png\`. The two remaining errors were **\*\*Mixed → Ductile\*\***: \`316L_AR_04_15.png\` and \`mX65_H2_26_29.png\`. The two corrected \`316L_AR_21\` images belong to the same filename-derived group.



![TinyViT-21M: head-only and fine-tuned confusion matrices and validation macro-F1]\(results/tinyvit_21m_finetune/plots/tinyvit_21m_finetuning_comparison.png)



*\*Figure 1. Confusion matrices and validation macro-F1 from the TinyViT-21M notebook. The curve corresponds to the restarted fine-tuning run.\**



Fine-tuning was initialized from the best head-only checkpoint. Two training runs were conducted using random seeds 42 and 43. The best model was obtained with seed 43 at epoch 3, achieving a validation macro-F1 of **\*\*0.9810\*\***.



**### 3.2 Held-out Test Results**



The fine-tuned TinyViT-21M achieved **\*\*96.06% accuracy (268/279)\*\*** and **\*\*0.8921 macro-F1\*\*** on the held-out test set, compared with 99.29% accuracy and 0.9810 macro-F1 on validation.



\| Class | Precision | Recall | F1 |

\|---|---:|---:|---:|

\| Brittle | 0.9746 | 0.9914 | 0.9829 |

\| Ductile | 0.9789 | 0.9720 | 0.9754 |

\| Mixed | 0.7368 | 0.7000 | 0.7179 |



Of the 11 misclassified images, six were Mixed fractures classified as Brittle or Ductile, while five Brittle/Ductile images were classified as Mixed. Errors were concentrated in 100Cr6 and 316L samples, particularly under AR conditions.



The results indicate strong classification performance for Brittle and Ductile fractures, while **\*\*generalization to Mixed fracture morphology remains the primary limitation\*\***.



**### 3.3 Comparison with TinyViT-11M**



Fine-tuned TinyViT-11M achieved **\*\*98.93% accuracy\*\***, **\*\*0.9721 macro-F1\*\***, and **\*\*0.9268 Mixed F1\*\***, using **\*\*10.55M parameters\*\***. Fine-tuned TinyViT-21M achieved one additional correct validation prediction while nearly doubling parameter count. The difference is therefore small relative to the dataset size.



![TinyViT-11M: head-only and fine-tuned classification comparison]\(results/tinyvit_11m_finetune/plots/tinyvit_finetuning_comparison.png)



*\*Figure 2. TinyViT-11M confusion matrices, per-class F1 and fine-tuning performance from the corresponding notebook.\**




### 3.4 Augmentation comparison: TinyViT-21M and ResNet-18

A further experiment compared augmented and non-augmented training for TinyViT-21M and ResNet-18, with class-weighted loss enabled in the non-augmented baseline. The three classes are **Brittle, Ductile and Mixed**. The reported validation set contains 63 images (21 per class), while the held-out test set contains 60 images (20 per class).

The results below are an additional experiment and should be interpreted separately from the 280-image validation and 279-image test evaluation reported in Sections 3.1–3.2. In particular, the test-set sizes differ, so these results should not be merged with the earlier full-split results.

**Table 3. Held-out test performance with and without augmentation.**

| Model | Training setup | Accuracy | Macro-F1 |
|:--|:--|--:|--:|
| TinyViT-21M | No augmentation; class weights | 86.67% | 0.8638 |
| TinyViT-21M | Augmented | 81.67% | 0.8126 |
| ResNet-18 | No augmentation; class weights | 86.67% | 0.8666 |
| ResNet-18 | Augmented | 88.33% | 0.8840 |

**Purpose of the balanced evaluation.** The original dataset is strongly skewed: Brittle accounts for 822/1,961 images (41.9%), Ductile for 996/1,961 (50.8%), and Mixed for only 143/1,961 (7.3%). Mixed images were augmented during training to increase their representation, while Brittle and Ductile retained substantially more distinct original images. The additional evaluation was therefore deliberately constructed with equal numbers of images per class (21 per class in validation and 20 per class in the held-out test subset). This creates a balanced evaluation distribution in which Mixed is much more prevalent, proportionally, than in the original dataset, while Brittle and Ductile are less prevalent. The purpose is to examine whether the trained models can recognise all three fracture classes when the evaluation class distribution is no longer dominated by Brittle and Ductile.

This should be interpreted as a **balanced-class evaluation / class-prior sensitivity experiment**, not as a replacement for evaluation on the dataset's natural class distribution. Equal class counts make overall accuracy and macro-F1 easier to interpret without the majority classes dominating the aggregate score. They do not, by themselves, establish performance under every possible real-world prevalence of Mixed fractures, nor do they test a change in image acquisition conditions or morphology.

The effect of augmentation was not consistent across architectures. TinyViT-21M decreased by 5.00 percentage points in test accuracy and by 0.0512 in macro-F1. ResNet-18 increased by 1.67 percentage points in accuracy and by 0.0174 in macro-F1. These results do not support a general claim that augmentation improves performance; instead, they suggest that its effect may depend on the architecture and the augmentation policy.

**Table 4. Held-out test confusion matrices.** Rows are true labels; columns are predictions. Class order is Brittle, Ductile, Mixed.

| Model and setup | Brittle | Ductile | Mixed |
|:--|--:|--:|--:|
| TinyViT-21M, no augmentation — Brittle | 20 | 0 | 0 |
| TinyViT-21M, no augmentation — Ductile | 0 | 17 | 3 |
| TinyViT-21M, no augmentation — Mixed | 3 | 2 | 15 |
| TinyViT-21M, augmented — Brittle | 20 | 0 | 0 |
| TinyViT-21M, augmented — Ductile | 0 | 15 | 5 |
| TinyViT-21M, augmented — Mixed | 4 | 2 | 14 |
| ResNet-18, no augmentation — Brittle | 19 | 0 | 1 |
| ResNet-18, no augmentation — Ductile | 1 | 17 | 2 |
| ResNet-18, no augmentation — Mixed | 0 | 4 | 16 |
| ResNet-18, augmented — Brittle | 19 | 0 | 1 |
| ResNet-18, augmented — Ductile | 0 | 18 | 2 |
| ResNet-18, augmented — Mixed | 0 | 4 | 16 |

The Mixed class remains the most challenging class in these test results. Its recall was 75% for non-augmented TinyViT-21M and 70% after augmentation. For ResNet-18, Mixed recall was 80% in both configurations. The dominant errors for TinyViT-21M include Mixed images predicted as Brittle or Ductile, while ResNet-18's Mixed errors were predominantly Mixed → Ductile. This pattern is consistent with possible overlap in the visual features of mixed fracture surfaces, but the confusion matrices alone cannot establish the cause.

**Table 5. Validation performance for the same comparison.**

| Model | Training setup | Accuracy | Macro-F1 |
|:--|:--|--:|--:|
| TinyViT-21M | No augmentation; class weights | 95.24% | 0.9521 |
| TinyViT-21M | Augmented | 98.41% | 0.9841 |
| ResNet-18 | No augmentation; class weights | 92.06% | 0.9219 |
| ResNet-18 | Augmented | 92.06% | 0.9214 |

The validation results do not track the held-out test results uniformly. Augmented TinyViT-21M has substantially higher validation performance but lower test performance than its non-augmented counterpart. ResNet-18 has identical validation accuracy in both configurations, with almost unchanged macro-F1. This validation–test divergence makes it important to avoid selecting an augmentation strategy using validation accuracy alone.

The validation confusion matrices (same class order) were:

| Model and setup | Brittle | Ductile | Mixed |
|:--|--:|--:|--:|
| TinyViT-21M, no augmentation — Brittle | 21 | 0 | 0 |
| TinyViT-21M, no augmentation — Ductile | 0 | 21 | 0 |
| TinyViT-21M, no augmentation — Mixed | 0 | 3 | 18 |
| TinyViT-21M, augmented — Brittle | 21 | 0 | 0 |
| TinyViT-21M, augmented — Ductile | 0 | 21 | 0 |
| TinyViT-21M, augmented — Mixed | 0 | 1 | 20 |
| ResNet-18, no augmentation — Brittle | 19 | 0 | 2 |
| ResNet-18, no augmentation — Ductile | 0 | 19 | 2 |
| ResNet-18, no augmentation — Mixed | 0 | 1 | 20 |
| ResNet-18, augmented — Brittle | 20 | 0 | 1 |
| ResNet-18, augmented — Ductile | 0 | 19 | 2 |
| ResNet-18, augmented — Mixed | 0 | 2 | 19 |

**Interpretation and limitations.** The two architectures have similar overall test accuracy without augmentation, but this does not demonstrate that they learn equivalent representations or have the same performance ceiling. Their class-wise error patterns differ. The balanced evaluation is intentional: it increases the relative representation of Mixed cases compared with the original dataset and reduces the relative contribution of Brittle and Ductile cases. It therefore probes class-wise recognition under a changed class prior. It does not mean that the evaluation contains more unique Mixed morphologies than the original dataset; the number of distinct held-out Mixed images remains limited to 20 in the test subset and 21 in validation.

Likewise, the observed architecture-dependent response to augmentation is preliminary: the available results do not isolate whether the cause is the transformation type or strength, random seed, checkpoint selection, or interactions with the model architecture. Each test set contains only 20 examples per class, so one prediction changes per-class recall by five percentage points and total accuracy by approximately 1.67 percentage points. The results should therefore be treated as descriptive rather than conclusive. For a fuller assessment, report both (a) performance on a naturally distributed held-out test set and (b) performance on a balanced test set, keeping the two protocols clearly labelled. If the intended deployment prevalence of Mixed fractures is known, additionally report prevalence-adjusted metrics or evaluate on a test set sampled to that target prevalence.

The next experiments should:

1. Document the exact augmentation operations, probabilities and magnitudes used in this comparison.
2. Run augmentation ablations (individual transformations and combinations) for both architectures, keeping the split, optimization settings and evaluation protocol fixed.
3. Repeat each configuration across multiple random seeds and report mean and standard deviation for accuracy, macro-F1 and per-class recall.
4. Use specimen- or group-aware splits where reliable specimen identifiers are available; filename-derived groups should not be assumed to represent physical specimens without verification.
5. Inspect misclassified Mixed images and compare feature embeddings or saliency maps to investigate possible morphological overlap with Brittle and Ductile classes.
6. Keep the reserved test set out of augmentation selection and hyperparameter tuning; use it only for final evaluation.


**## 4. Discussion**



Fine-tuning improved classification performance across both TinyViT architectures. TinyViT-21M achieved the highest validation accuracy of **\*\*99.29%\*\***, compared with 97.86% for ResNet-18 and 98.93% for TinyViT-11M.



On the independent test set, TinyViT-21M achieved **\*\*96.06% accuracy and 0.8921 macro-F1\*\***. Classification remained accurate for Brittle and Ductile fractures, with F1 scores of 0.9829 and 0.9754, respectively. However, performance on Mixed fractures decreased substantially, with an F1 score of 0.7179.



Of the 11 test errors, six involved Mixed fractures classified as Brittle or Ductile. Errors were concentrated in 100Cr6 and 316L samples, indicating differences in classification performance across materials.



These results demonstrate the effectiveness of transfer learning for SEM fracture classification while highlighting the difficulty of recognizing mixed fracture morphologies across different samples.

The additional augmentation comparison in Section 3.4 qualifies this conclusion: augmentation did not improve held-out test performance consistently across the two architectures. TinyViT-21M performed worse on the small held-out test subset after augmentation, whereas ResNet-18 improved slightly. Mixed remained the most difficult class in both models. Because the validation and test trends diverged for augmented TinyViT-21M, and each class contributes only 20 test images in this comparison, repeated-seed experiments and augmentation ablations are needed before drawing a general conclusion about augmentation effectiveness.



**## 5. Optional Mixed-class augmentation**



Run \`python augmentation.py --data-dir .\` to augment the Mixed training images

until their count matches the largest training class. With the current split,

this adds **\*\*609\*\*** generated images to the 102 Mixed training images. The

augmentation uses right-angle rotations, flips and small brightness/contrast

changes.



The original images and \`dataset_split.csv\` are left untouched. Generated

images are saved under \`Mixed/augmented/\`, and

\`dataset_split_augmented.csv\` contains the original split plus the generated

training rows. Only the Mixed training rows are augmented; validation and test

rows are unchanged. The training notebooks use the shared

\`prepare_training_split\` helper to create or validate this augmented manifest

before loading their datasets.



Use \`--target-count N\` to choose another training count and \`--seed N\` to

control the random variants. The command refuses to overwrite an existing

augmentation output or manifest.



**## 6. Shared training and prediction**



Install the Python dependencies with \`pip install -r requirements.txt\`. Choose

a PyTorch and torchvision build that matches your platform; the supplied

requirements intentionally do not pin platform-specific PyTorch wheels.



Use [modelling.ipynb]\(./modelling.ipynb) for the combined Train, Validate, and

Test workflow. It supports ResNet-18, DeiT-Tiny, TinyViT-11M, and TinyViT-21M,

with head-only training or selective fine-tuning. Training writes checkpoints

under \`results/shared_pipeline/\` and refuses to overwrite an existing

checkpoint. The model-specific notebooks remain available for the original

experiments; their saved outputs predate augmentation and should be refreshed

by rerunning the notebook before interpreting results.



Validation runs against the original \`dataset_split.csv\`. Final test evaluation

is opt-in and also reads that original split; generated Mixed images are used

for training only and must not be added to validation or test.



**## 7. Repository structure**



The notebooks are designed to run from the repository root (\`SEM Image Database\` in the local project) with the following existing structure:



\`\`\`text

augmentation.py

requirements.txt

sem_fracture/

    data.py

    models.py

    training.py

    prediction.py

    pipeline.py

Brittle/                       # SEM image class folders

Ductile/

Mixed/

Labels.csv

Licence.txt

dataset_split.csv

modelling.ipynb

train_resnet18.ipynb

train_resnet18_finetune.ipynb

deit_tiny.ipynb

deit_tiny_finetune.ipynb

tinyvit_11m.ipynb

tinyvit_11m_finetune.ipynb

tinyvit_21m.ipynb

tinyvit_21m_finetune.ipynb

tests/

results/

    resnet18_best.pth

    resnet18_validation_analysis.png

    resnet18_finetune/

    deit_tiny/

    deit_tiny_finetune/

    tinyvit_11m/

    tinyvit_11m_finetune/

    tinyvit_21m/

    tinyvit_21m_finetune/

README.md

\`\`\`



**## References**



1\. Campari, A., Konert, F., Razavi, N., Sobol, O., and Alvaro, A. **\*\*Hydrogen-assisted cracking: A deep learning approach for fractographic analysis.\*\*** *\*Computational Materials Science\** **\*\*262\*\*** (2026), 114366. [doi:10.1016/j.commatsci.2025.114366]\(https\://doi.org/10.1016/j.commatsci.2025.114366).

2\. Campari, A. **\*\*SEM Image dataset.\*\*** Zenodo (2025). [doi:10.5281/zenodo.15510590]\(https\://doi.org/10.5281/zenodo.15510590).
