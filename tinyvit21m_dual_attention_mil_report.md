# Dual-Attention Multiple-Instance Learning on Frozen TinyViT-21M Features for Mixed Fracture Classification

**Experimental report | 25 November 2026**  
**Status:** Completed  
**Reference model:** Fine-tuned TinyViT-21M (`tiny_vit_21m_224.in1k`)  
**Dataset:** SEM fracture images associated with Campari et al. [1, 2]

## Abstract

A dual-attention multiple-instance learning (MIL) classifier was evaluated to address the low sensitivity of an existing three-class scanning electron microscopy (SEM) fracture classifier to the minority Mixed class. The fine-tuned TinyViT-21M encoder was frozen and applied to nine overlapping regional crops and one global view per image. Two learned attention branches aggregated the regional features into brittle- and ductile-evidence scores, which were combined to classify Brittle, Ductile, and Mixed fractures. The MIL head introduced 149,124 trainable parameters while leaving all 20,623,299 TinyViT parameters unchanged. On a 279-image test set, Mixed recall improved from **0.7000 to 0.8000**, but Mixed precision decreased from **0.7368 to 0.6154** and Mixed F1 decreased from **0.7179 to 0.6957**. Overall test accuracy decreased from **96.06% to 94.98%**, and macro-F1 from **0.8921 to 0.8811**. The models exhibited partially complementary errors, but improved sensitivity to Mixed morphology came at the expense of additional Ductile-to-Mixed false positives. The evaluated frozen-feature MIL formulation did not outperform the fine-tuned whole-image classifier.

## 1. Research question

The preceding whole-image TinyViT-21M classifier performed strongly on Brittle and Ductile fracture surfaces but correctly identified only 14 of 20 Mixed test images. The working hypothesis was that an image-level representation could obscure localized minority morphology within a heterogeneous fracture surface. The present experiment tested whether region-specific aggregation of features from the existing TinyViT encoder could improve minority-class recognition without modifying the backbone.

The primary comparison concerned three-class accuracy and macro-F1, with particular attention to Mixed precision, recall, and F1. A secondary objective was to assess whether separate brittle- and ductile-attention branches learned different spatial emphases.

## 2. Materials and methods

### 2.1 Dataset and partitioning

The dataset comprised 1,961 labeled SEM images. The original `dataset_split.csv` was preserved; no images were reassigned or added.

| Class | Training | Validation | Test | Total |
|:--|--:|--:|--:|--:|
| Brittle | 589 | 117 | 116 | 822 |
| Ductile | 711 | 142 | 143 | 996 |
| Mixed | 102 | 21 | 20 | 143 |
| **Total** | **1,402** | **280** | **279** | **1,961** |

Mixed accounted for 7.3% of all images. Mixed samples originated from 41 filename-derived training groups, five validation groups, and seven test groups. Groups were disjoint between partitions, although the filename prefixes have not been independently verified as physical specimen identifiers.

### 2.2 Frozen TinyViT feature encoder

The feature encoder was the previously trained `tiny_vit_21m_224.in1k` checkpoint, obtained after training a three-class head followed by selective fine-tuning of the last TinyViT stage and classifier. The selected fine-tuning checkpoint was from epoch 3 of the restarted run (seed 43). All **20,623,299** TinyViT parameters were frozen for this experiment. The original classification layer was bypassed during feature extraction, yielding a 576-dimensional embedding per view.

Grayscale SEM images were converted to three channels and normalized with the same ImageNet statistics as the reference classifier. Each original image was bilinearly resized to a 448 × 448 working view. Nine overlapping 224 × 224 crops were extracted at horizontal and vertical offsets of 0, 112, and 224 pixels, corresponding to 50% overlap. A separate 224 × 224 global image was included as a tenth view. Consequently, every image was represented by a **10 × 576** feature matrix.

For training, feature caches were generated for four fixed flip combinations: unchanged, horizontal, vertical, and horizontal-plus-vertical. A random variant was chosen per training example. Validation and test inputs received no augmentation. The experiment used the existing unaugmented split, not additional synthetically generated Mixed images.

### 2.3 Dual-attention MIL formulation

Two independent attention branches received the same ten feature vectors. Each branch consisted of a 576→128 linear projection, `tanh` activation, and a 128→1 linear projection. Softmax normalization over the ten views generated class-specific regional attention weights. Attention-weighted sums were fed to separate linear heads producing brittle- and ductile-evidence logits. The new MIL module contained **149,124 trainable parameters**.

Training labels were re-expressed as two evidence-presence targets:

| Image label | Brittle target | Ductile target |
|:--|--:|--:|
| Brittle | 1 | 0 |
| Ductile | 0 | 1 |
| Mixed | 1 | 1 |

Let `b` and `d` be sigmoid-transformed brittle and ductile evidence scores. The unnormalized class scores were defined as:

- Brittle: `b × (1 − d)`
- Ductile: `(1 − b) × d`
- Mixed: `b × d`

These were normalized to sum to one before selecting the highest-scoring class. This decision rule assumes at least one of the two mechanisms is present and omits a neither-mechanism category. The resulting scores were used for classification but were not independently probability-calibrated.

Binary cross-entropy with logits was averaged over the two outputs, then weighted by the original image-level class weights: Brittle **0.7934**, Ductile **0.6573**, and Mixed **4.5817**. Thus, the minority-class weighting was retained while the output formulation changed from three mutually exclusive classes to two binary evidence labels.

### 2.4 Training and selection

The MIL head was optimized with AdamW, learning rate `1e-4`, weight decay `1e-4`, batch size 16, seed 43, and gradient clipping at norm 1.0. Training was limited to 30 epochs with early stopping patience of four epochs. The selected checkpoint maximized **validation macro-F1**. The best checkpoint occurred at **epoch 8**, with training terminating after epoch 12. Only MIL-head parameters were updated; cached TinyViT embeddings remained fixed.

The reference whole-image model was reproduced using the global-view embeddings and its original classifier. Both models were assessed on identical validation and test images. The primary MIL test result used the selected checkpoint and the original decision rule, without post hoc threshold adjustment.

## 3. Results

### 3.1 Validation

| Metric | Fine-tuned TinyViT-21M | Dual-attention MIL |
|:--|--:|--:|
| Correct predictions | **278/280** | 276/280 |
| Accuracy | **0.9929** | 0.9857 |
| Macro-F1 | **0.9810** | 0.9645 |
| Mixed precision | **1.0000** | 0.8696 |
| Mixed recall | 0.9048 | **0.9524** |
| Mixed F1 | **0.9500** | 0.9091 |

Confusion matrices use actual classes as rows and predicted classes as columns, ordered Brittle, Ductile, Mixed.

**Reference TinyViT, validation**

| Actual / predicted | Brittle | Ductile | Mixed |
|:--|--:|--:|--:|
| Brittle | 117 | 0 | 0 |
| Ductile | 0 | 142 | 0 |
| Mixed | 0 | 2 | 19 |

**Dual-attention MIL, validation**

| Actual / predicted | Brittle | Ductile | Mixed |
|:--|--:|--:|--:|
| Brittle | 116 | 0 | 1 |
| Ductile | 0 | 140 | 2 |
| Mixed | 1 | 0 | 20 |

MIL corrected both Mixed images missed by the reference classifier (`316L_AR_04_15.png` and `mX65_H2_26_29.png`), but introduced four other errors. The gain in Mixed recall therefore did not translate into improved macro-F1 or Mixed F1.

### 3.2 Test

| Metric | Fine-tuned TinyViT-21M | Dual-attention MIL | Change (MIL − reference) |
|:--|--:|--:|--:|
| Correct predictions | **268/279** | 265/279 | −3 |
| Accuracy | **0.9606** | 0.9498 | −0.0108 |
| Macro-F1 | **0.8921** | 0.8811 | −0.0110 |
| Mixed precision | **0.7368** | 0.6154 | −0.1214 |
| Mixed recall | 0.7000 | **0.8000** | +0.1000 |
| Mixed F1 | **0.7179** | 0.6957 | −0.0222 |

**Reference TinyViT, test**

| Actual / predicted | Brittle | Ductile | Mixed |
|:--|--:|--:|--:|
| Brittle | 115 | 0 | 1 |
| Ductile | 0 | 139 | 4 |
| Mixed | 3 | 3 | 14 |

**Dual-attention MIL, test**

| Actual / predicted | Brittle | Ductile | Mixed |
|:--|--:|--:|--:|
| Brittle | 116 | 0 | 0 |
| Ductile | 0 | 133 | 10 |
| Mixed | 3 | 1 | 16 |

Mixed recall increased from **14/20** to **16/20**, but Ductile-to-Mixed errors increased from four to ten. Brittle recall improved from 115/116 to 116/116; Ductile recall declined from 139/143 to 133/143.

### 3.3 Paired error analysis

The paired test audit identified **259 images correct under both models**, **six reference-model errors corrected by MIL**, **nine new MIL errors**, and **five errors shared by both**. Three of the six corrected errors involved Mixed samples (`100Cr6_AR_44_22.png`, `316L_H2_13_10.png`, and `X65_H2_28_18.png`). One Mixed image classified correctly by the reference model (`100Cr6_AR_33_28.png`) became a new Brittle prediction. This produces a net gain of two correctly classified Mixed images.

Residual Mixed false negatives included `100Cr6_AR_33_1.png`, `100Cr6_AR_62_4.png`, and `X65_AIR_10_20.png`. Multiple new Ductile-to-Mixed errors occurred in the `X65_AIR_08` filename-derived group. The error pattern suggests that the greater sensitivity of the dual-evidence formulation was accompanied by weaker rejection of false brittle evidence in otherwise Ductile-labeled images; this interpretation is not equivalent to confirming the physical morphology of those regions.

### 3.4 Regional attention and evidence analysis

Regional attention maps were inspected for four validation cases, including two corrections and two newly introduced errors. The brittle and ductile branches frequently assigned different weights to local crops and the global view. For `mX65_H2_26_29.png` (Mixed, recovered by MIL), the brittle branch concentrated on upper local crops while ductile attention assigned a weight of **0.789** to the global view. Conversely, `mX65_H2_26_22.png` (Mixed, misclassified as Brittle by MIL) had a ductile global-view attention weight of **0.654**, demonstrating that attention intensity by itself did not determine a correct mechanism classification.

The mean validation evidence scores were:

| Actual class | Mean brittle evidence | Mean ductile evidence |
|:--|--:|--:|
| Brittle | 0.9848 | 0.0503 |
| Ductile | 0.1183 | 0.9861 |
| Mixed | 0.8993 | 0.9335 |

The classes were broadly separated in the two-evidence space, although errors appeared where one mechanism's evidence was moderate rather than negligible. These attention maps indicate learned model emphasis, **not validated pixel-level segmentation** or proof that the emphasized crops physically contain the corresponding fracture mechanism.

### 3.5 Exploratory threshold sensitivity

A validation-only sensitivity analysis tested stricter evidence thresholds for assigning Mixed. At the original threshold of **0.50**, Mixed precision/recall/F1 were **0.8696/0.9524/0.9091**. At **0.65**, the corresponding values were **1.0000/0.8571/0.9231**. The stricter threshold removed false Mixed predictions on validation but missed additional genuine Mixed images; its Mixed F1 remained below the reference model's **0.9500**.

Threshold analysis followed inspection of the model errors and test results. It is presented **only as a diagnostic**, not as an independently tested improvement. The reported primary test performance is for the original threshold and epoch-8 checkpoint.

## 4. Discussion

The dual-attention MIL model demonstrated that regionally aggregated features can recover some Mixed fracture examples missed by a whole-image classifier. Its increased sensitivity did not translate into better balanced classification: test Mixed F1 decreased by **0.0222**, test macro-F1 by **0.0110**, and accuracy by **1.08 percentage points**. The reference fine-tuned TinyViT-21M therefore remains the stronger standalone classifier under the reported criteria.

There are several plausible explanations for the discrepancy between Mixed recall and precision. The frozen encoder had been optimized for image-level three-class discrimination, not explicitly for recognizing local brittle and ductile components. Attention branches were trained with image-level labels only; a Mixed image was never accompanied by regional mechanism annotations. The two-output evidence-to-class rule also makes the presence of moderate evidence for the second mechanism consequential for a Mixed decision. The present design does not distinguish these potential sources of error causally.

The partially complementary errors are a finding of the paired comparison, but they do not establish that an ensemble of the two models would improve performance. Such an ensemble would require a separately defined method and independent assessment.

## 5. Limitations

- **Small minority evaluation cohorts:** 21 Mixed validation images from five filename-derived groups and 20 Mixed test images from seven groups limit precision and stability of class-specific estimates.
- **Unverified physical grouping:** Filename-derived groups were disjoint, but are not confirmed as independent specimens or acquisition sessions.
- **Weak spatial supervision:** No crop-level or pixel-level ground truth was available for validating localization of brittle or ductile regions.
- **Frozen feature encoder:** Regional representations were extracted from a backbone previously trained for whole-image classification; local features were not fine-tuned jointly with MIL.
- **Resizing and scale:** Local views were obtained after resampling to 448 × 448, potentially altering fine texture and apparent scale.
- **Single MIL training run:** No multiple-seed replication, group-aware cross-validation, uncertainty intervals, or external-dataset test was performed.
- **Post-test exploration:** Attention/error diagnostics and threshold sensitivity were examined after the test set had been used. Additional model selection based on these observations would require a new independent evaluation before confirmatory claims.
- **Evidence-score calibration:** The normalized two-evidence output was not assessed for probabilistic calibration.

## 6. Conclusion

A frozen TinyViT-21M encoder with a lightweight dual-attention MIL head increased Mixed fracture test recall from **70% to 80%** but introduced sufficient false Mixed predictions to reduce Mixed F1, macro-F1, and overall accuracy. This experiment does **not** support superior classification performance for the evaluated MIL implementation. It does establish partial complementarity between global and regional predictions, motivating a future, independently validated study of morphology-aware feature learning rather than further tuning against the present test set.

## 7. Reproducibility record

| Item | Value / artifact |
|:--|:--|
| Data split | `dataset_split.csv` (1,402 / 280 / 279) |
| Base architecture | `tiny_vit_21m_224.in1k` |
| Source checkpoint | `results/tinyvit_21m_finetune/checkpoints/tinyvit_21m_finetuned_best.pth` |
| Source fine-tuning checkpoint | Epoch 3 (seed 43) |
| MIL experiment notebook | `tinyvit_21m_dual_attention_mil.ipynb` (local) |
| Feature representation | 10 views × 576 dimensions per image |
| New trainable parameters | 149,124 |
| Optimization | AdamW; LR `1e-4`; weight decay `1e-4`; batch 16; gradient clip 1.0 |
| Maximum epochs / patience | 30 / 4 |
| Best MIL epoch / stop epoch | 8 / 12 |
| MIL checkpoint | `results/tinyvit_21m_dual_attention_mil/checkpoints/tinyvit21m_dual_attention_mil_best.pth` (local) |
| Local result files | `training_history.json`, `validation_prediction_audit.csv`, `test_comparison.json`, `test_prediction_audit.csv`, `attention_case_study.csv`, `validation_evidence_scores.csv`, `threshold_sensitivity.csv`, and diagnostic plots |

The present document records the experimentally observed results. The locally generated notebook, checkpoint, feature caches, and figures **are not included in this Markdown document** and must be committed or archived separately if full execution-level reproducibility is required.

## References

1. Campari, A., Konert, F., Razavi, N., Sobol, O., and Alvaro, A. *Hydrogen-assisted cracking: A deep learning approach for fractographic analysis*. **Computational Materials Science** 262 (2026), 114366. https://doi.org/10.1016/j.commatsci.2025.114366
2. Campari, A. *SEM Image dataset*. Zenodo (2025). https://doi.org/10.5281/zenodo.15510590
