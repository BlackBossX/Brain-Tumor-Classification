# The Big Picture — What This Assignment Actually Wants

## What is a neural network, in one paragraph?

A feed-forward neural network (MLP = multilayer perceptron) is a function that takes
your 27 patient features as numbers, multiplies them by a bunch of learned weights,
passes the result through some non-linear "activation" functions, and outputs three
numbers — one probability for Glioma, one for Meningioma, one for Pituitary — that
sum to 1. "Training" means adjusting all those weights, bit by bit, so the network's
guesses get closer to the true labels in your training data. That's it. Everything
else in this assignment is about doing that *correctly and honestly*, not leaking
information, and proving you understand why each step matters.

## What is the examiner actually testing?

Read the rubric structure again (Parts A–E). Notice what it is **not** asking:
it never says "get 95% accuracy." It is asking you to demonstrate a *correct
process*. A model with 70% accuracy but a flawless, leak-free, well-justified
pipeline will score higher than a model with 90% accuracy built on a pipeline
that secretly cheated (e.g. scaled using statistics from data it later got
graded on). This matters enormously for your project given what your madam said
about testing on her own held-out data — see file 03 for why.

## The five parts, and how they connect

```
Part A: Data understanding & preprocessing
   |  -> produces: cleaned dataframe, fitted imputers/encoders/scaler,
   |     train/val/test splits
   v
Part B: Model design
   |  -> produces: an architecture (on paper / in code), justified by
   |     Part A's findings (e.g. number of input units = number of
   |     features AFTER encoding, not before)
   v
Part C: Splitting, training, tuning
   |  -> produces: a trained model, training curves, a hyperparameter
   |     comparison table, ONE selected "best" model (chosen using
   |     validation data only)
   v
Part D: Final evaluation
   |  -> uses test set EXACTLY ONCE, reports metrics + confusion matrix
   |     + confidence examples
   v
Part E: Critical analysis
   -> baseline comparison, feature importance, limitations & ethics
```

Notice the one-way arrow. You are not allowed to go back up the chain after
looking at test results. If your test accuracy looks bad and you go back and
change your architecture, you have broken the entire methodology — and madam
testing on her own data is effectively a *second, even stricter* version of
this same rule (see file 03).

## Why does the order matter so much?

Because the entire assignment is structured around one core idea in ML:

> **Anything you learn from data must only be learned from training data,
> and then applied unchanged elsewhere.**

This single idea explains almost every instruction in the PDF:
- Why you fit the imputer on train only (Part A Step 2)
- Why you fit the scaler on train only (Part A Step 3)
- Why test is set aside before tuning (Part C Step 1)
- Why test is used only once (Part D)
- Why your `predict.py` must reuse *saved* preprocessing objects instead of
  recalculating anything (Deliverables, Step 3-4)

Once this idea clicks, the whole assignment stops looking like 20 disconnected
tasks and starts looking like one consistent rule applied 20 times.

## What "submit your model so the examiner can test it" really means

Your madam will take her own dataset (same 27 columns + patient_id, no
tumor_type, or tumor_type hidden for her own checking), feed it through your
`predict.py`, and expect predicted classes back — **without editing your
code.** This means:

- Your `predict.py` cannot assume any specific row count, order, or that
  every column has zero missing values.
- Every transformation (impute → encode → scale) must come from objects you
  saved during training (e.g. a fitted `SimpleImputer`, `OneHotEncoder`,
  `StandardScaler`), not recomputed from her new data.
- If her data has a category your training data never saw (e.g. some new
  string in `tumor_location`), your pipeline must not crash.

This is exactly why "fit on train only, apply elsewhere" is not just an
academic nicety — it is the literal mechanism that makes your submission
work on unseen data at all.

## What you will actually produce, concretely

1. One Jupyter notebook, run top to bottom, with a fixed `random_state`/seed,
   that reproduces every number and plot in your report.
2. One PDF report (max 12 pages) answering each lettered/numbered question
   in the brief — tables and plots, not just code dumps.
3. One `.h5` model file + the fitted preprocessing pipeline (scaler, imputer,
   encoder objects saved, e.g. with `pickle` or `joblib`) + one `predict.py`
   that chains: load raw CSV → apply saved preprocessing → load `.h5` model →
   output predicted class per row.

## How to use the other files

- **01_your_data_explained.md** — the specific quirks in *your* CSV, so Part A
  is grounded in fact rather than guesswork.
- **02_workflow_roadmap.md** — a step-by-step execution order, mapped to the
  exact sections of the brief.
- **03_overfitting_playbook.md** — read this before you train anything. It's
  the single highest-leverage file given what madam said.
- **04_keras_concepts_for_beginners.md** — the vocabulary (epochs, batches,
  dropout, softmax, categorical cross-entropy, etc.) explained from zero.
