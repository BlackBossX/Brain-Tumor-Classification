# Step-by-Step Workflow Roadmap

This maps directly onto the brief's Part A–E structure. Treat each box as one
notebook section. Don't skip the order — later steps depend on artifacts
(fitted objects, splits) created in earlier ones.

## Part A — Data understanding & preprocessing

**Step 1: Load & inspect**
- `pd.read_csv(...)`, `.shape`, `.dtypes`, `.head()`
- Identify numeric vs categorical columns (see file 01)
- State: `patient_id` is the identifier, drop it — it's a row label with no
  predictive signal, and keeping it risks the model latching onto
  meaningless ID patterns (overfitting risk).

**Step 2: Missing values**
- Order of operations matters here:
  1. Fix dirty markers first: `"Unknown"` → `NaN`, `"?"` → `NaN`,
     normalize `gender` casing (see file 01).
  2. THEN run `df.isna().sum()` to get true counts — report this table.
  3. Explain numeric dtype check (even though in this file all numeric
     columns already load cleanly — show you checked).
  4. **Split your data into train/val/test BEFORE you fit any imputer**
     (see Part C Step 1 below — yes, do that step conceptually now, even
     if you write the code in sequence).
  5. Fit a `SimpleImputer` (median for numeric — robust to skew/outliers in
     things like `ki67_index`; most-frequent or a constant `"Missing"`
     category for categorical) on **training data only**.
  6. `.transform()` (not `.fit_transform()`) on val and test.

  Why fit on train only? Because the median/mode of the *whole* dataset
  contains information from the val/test rows. If you use it to fill gaps
  in training data, your training process indirectly "knows" something
  about the data it'll later be judged on — this is data leakage, and it
  makes your validation/test scores overly optimistic. The leaked
  information has nowhere to go in real deployment (you won't have her
  held-out labels to compute a fresh median from).

**Step 3: Encoding and scaling**
- One-hot encode nominal columns; ordinal-encode the genuinely ordinal ones
  (see file 01 for which is which); binary-map the Yes/No columns.
- Target encoding: for a 3-class softmax output, use **integer labels**
  (0,1,2) with `sparse_categorical_crossentropy`, OR **one-hot** labels with
  `categorical_crossentropy`. Pick one and be explicit about why it
  constrains your loss choice (see file 04 for the mechanics).
- Scale numeric features with `StandardScaler` (zero mean, unit variance) —
  fit on train only, again. Explain: gradient descent moves faster and more
  stably when features share a similar scale; otherwise a feature like
  `mri_intensity` (0–255) would dominate the gradient compared to
  `headache_severity` (0–10) purely because of units, not real importance.

**Step 4: Class balance**
- Report the table from file 01 (42/34/23 split).
- Consequence for (a) training: the model may become biased toward
  predicting the majority class (Glioma) since it minimizes loss faster
  that way.
- Consequence for (b) evaluation: plain accuracy becomes misleading — a
  model that just always predicts Glioma would get ~42% accuracy without
  learning anything real.
- Mitigation: class weights in the loss function (`class_weight` parameter)
  or oversampling the minority classes (e.g. SMOTE) — pick ONE and justify.

## Part B — Model design

**Step 1: Architecture**
- Input units = number of columns *after* encoding (count this for real in
  your notebook — it will be more than 27 because one-hot expands columns).
- Hidden layers: with ~9,000 rows and a modest number of features, 2 hidden
  layers (e.g. 64 → 32 units) is a reasonable, justifiable starting point —
  not so large that you'll obviously overfit a dataset this size, not so
  small that it can't learn the patterns.
- Output units = 3 (one per class).
- Hidden activation: ReLU (standard, avoids vanishing gradients).
- Output activation: **softmax** — because you have 3 mutually-exclusive
  classes, and softmax converts raw scores into a valid probability
  distribution over them (sums to 1). The number of classes determines
  this: binary problems use sigmoid + 1 output unit; multi-class uses
  softmax + N output units.
- Loss: `categorical_crossentropy` (one-hot labels) or
  `sparse_categorical_crossentropy` (integer labels) — must match your
  Part A Step 3 label encoding choice exactly, or training will error out
  or silently misbehave.

**Step 2: Regularisation**
- Overfitting = the model memorizes patterns specific to the training rows
  (including noise) rather than learning patterns that generalize. You'll
  see it as training accuracy climbing while validation accuracy stalls or
  drops.
- Add at least one of: Dropout (randomly zero out a fraction of neurons
  each training step, forcing redundancy), L2 weight decay (penalizes
  large weights, keeping the model simpler), early stopping (halt training
  once validation loss stops improving), batch normalization.
- **Given your madam's explicit overfitting warning, I'd add at least two:
  Dropout + Early Stopping is a strong, easy-to-justify combo.** See file 03.
- Print `model.summary()` for the architecture diagram + trainable param
  count (Keras gives you this for free).

## Part C — Splitting, training, tuning

**Step 1: Evaluation protocol**
- Typical split: 70% train / 15% validation / 15% test, or 80/10/10 — pick
  one and justify based on dataset size (9,000 rows is plenty for either).
- **Stratify by `tumor_type`** in `train_test_split(..., stratify=y)` so
  each split keeps the same 42/34/23 class ratio — otherwise you risk a
  test set that's accidentally skewed.
- Roles: train = where weights get updated; validation = where you check
  generalization *during* development and pick hyperparameters; test =
  touched exactly once, at the very end, to report the final unbiased
  estimate of real-world performance.
- Test must be set aside before tuning: if you peek at test performance
  while still adjusting hyperparameters, you are — even unintentionally —
  tuning your model to fit the test set, which defeats its entire purpose
  as an unbiased estimate. This is the same root issue as data leakage in
  Part A, just at a different stage of the pipeline.

**Step 2: Training setup**
- Optimiser: Adam is the standard safe default (adaptive learning rate,
  works well out-of-the-box on tabular data).
- Learning rate: start at `0.001` (Adam's common default) — justify as "a
  well-tested standard starting point, then tuned in Step 3."
- Batch size: 32 is a typical default for a dataset this size — large
  enough for stable gradient estimates, small enough to fit in memory
  easily and allow many weight updates per epoch.
- Train, plot loss & accuracy curves (train vs val, per epoch).
- Interpret: cite the actual shape of YOUR curves — e.g. "training loss
  keeps falling past epoch 15 while validation loss flattens/rises ⇒
  overfitting starting around epoch 15" (your specific numbers will differ
  — read your own plot, don't guess at this in advance).

**Step 3: Hyperparameter tuning**
- Vary number of layers, units per layer, learning rate, dropout rate,
  batch size — present as a comparison table (rows = configurations,
  columns = val accuracy/loss).
- Pick your **best model using validation performance only.** Test set has
  not been touched yet at this point.

## Part D — Final evaluation (test set touched exactly once, here)

**Step 1: Core metrics** — accuracy, per-class precision/recall/F1, macro-F1
(treats all classes equally — useful here since Pituitary is the minority
class you care about not ignoring), weighted-F1 (accounts for class size —
useful for "overall" performance given the real-world class distribution).
Macro is more informative here specifically *because* of the imbalance —
weighted can mask poor performance on the minority class.

**Step 2: Confusion matrix** — produce it, then reason about *which two
classes get confused* using the data dictionary: e.g. Glioma and Meningioma
can occur in overlapping `tumor_location` categories or have overlapping
`tumor_size_mm`/`edema_grade` distributions, which would make them harder
for the model to separate than, say, Pituitary tumours (which are
anatomically distinct — `Sellar` location is a strong, almost exclusive,
signal for Pituitary).

**Step 3: Confidence** — pick 2 high-confidence and 2 low-confidence
predictions (look at `model.predict()` probability outputs, not just the
argmax class), and discuss what's different about their feature values
(e.g. low-confidence cases might have several missing values that got
imputed, or feature values that sit near a decision boundary between two
classes).

## Part E — Critical analysis

**Step 1: Baseline** — train a `RandomForestClassifier` or
`LogisticRegression` on the *same* preprocessed splits, compare in a table.
On tabular data like this, it's common (and fine to report honestly) for a
random forest to match or beat a small MLP — the discussion question
("was the NN worth the complexity?") wants an honest evidence-based answer,
not a forced "yes."

**Step 2: Feature insight** — use `sklearn.inspection.permutation_importance`
on your trained model, or manually ablate (drop one feature, retrain or
re-evaluate, measure performance drop). Likely uninformative candidates
given the data dictionary: `region` and `ethnicity` have no obvious
biological link to tumour type — but confirm this empirically rather than
asserting it; the data may surprise you.

**Step 3: Limitations & ethics** — two limitations (e.g. synthetic data
may not reflect real clinical feature correlations; single train/test split
rather than cross-validation means some performance variance is
unmeasured). One deployment risk (false negatives on a rarer tumour type
delaying real treatment) and one safeguard (mandatory clinician review of
any model prediction before action — model as decision support only, not
autonomous diagnosis).
