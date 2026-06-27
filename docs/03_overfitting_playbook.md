# The Overfitting Playbook

Your madam specifically warned: she will test your submitted model on *her
own* held-out data, and an overfitted model will score lower. This file is
about taking that seriously and structurally, not just adding one dropout
layer and hoping.

## First, the concept, precisely

A model overfits when it has effectively "memorized" quirks of your
training rows — including noise, coincidental correlations, and even things
like patient ID patterns — rather than learning the general relationship
between symptoms/scans and tumour type that would hold on *any* similar
patient, including ones madam's dataset contains that you've never seen.

The tell-tale sign: **training performance keeps improving while validation
performance stalls or gets worse.** The gap between the two is your
overfitting signal.

There are actually **two separate risks** in your situation, and most
students only guard against the first:

1. **Overfitting within your own train/val/test split** — the "classic"
   ML overfitting taught in courses. Caught by your loss/accuracy curves
   and validation metrics.
2. **Overfitting to your own dataset's idiosyncrasies that madam's dataset
   won't share** — a subtler, assignment-specific risk. Even a model that
   looks perfectly well-fit on YOUR test set can still perform worse on
   HER data if your *preprocessing pipeline itself* baked in assumptions
   that don't hold for her data (e.g. you hardcoded "if region == 'Rural'"
   logic, or your imputer's fitted statistics happen to be unusual to your
   sample, or you accidentally treated a column as having only the
   categories seen in your training data).

Risk #2 is why the "fit transformations on train only, save them, reuse
unchanged" rule isn't just about your internal test set — it's about
making your whole pipeline portable to a dataset it's never seen, which is
exactly what will happen.

## Checklist: structural safeguards against risk #1 (classic overfitting)

- [ ] **Dropout** between hidden layers (e.g. `Dropout(0.3)`) — randomly
      disables neurons during training so the network can't rely too
      heavily on any single neuron/feature combination.
- [ ] **Early stopping** monitoring `val_loss`, with `patience` of a few
      epochs and `restore_best_weights=True` — stops training at the
      point of best generalization rather than training until train loss
      is near-zero.
- [ ] **L2 weight regularization** (optional addition) — penalizes large
      weights in the loss function, discouraging the network from fitting
      noise with extreme weight values.
- [ ] **Keep the architecture proportionate to the data** — for ~9,000
      rows and ~30-40 features (post-encoding), a network with millions of
      parameters is asking for trouble. Two hidden layers in the
      32–128 unit range each is a sane, defensible range. Bigger isn't
      automatically better and the brief explicitly wants you to justify
      size with reference to the data.
- [ ] **Look at your actual loss curves, not just the final number.**
      A model whose val_loss is still decreasing when training stops is
      probably underfit; one where val_loss has clearly turned upward
      while train_loss keeps falling is overfit; one where both flatten
      together near the end is well-fit. Say which one yours is, citing
      the epoch where the divergence starts (if any).

## Checklist: structural safeguards against risk #2 (pipeline portability)

This is the part most students miss, and it's the part most relevant to
"madam will test on her own data."

- [ ] **Every fitted object is saved, not recomputed.** Your `SimpleImputer`,
      your encoder(s), and your `StandardScaler` must be fit exactly once
      (on your training split) and then saved to disk (e.g.
      `joblib.dump(scaler, 'scaler.pkl')`). Your `predict.py` loads these
      saved objects — it never calls `.fit()` or `.fit_transform()` again.
- [ ] **Handle unseen categories gracefully.** If madam's data has a row
      with, say, a `tumor_location` value your training data never saw
      (unlikely here, but the principle generalizes), your encoder should
      not crash. `OneHotEncoder(handle_unknown='ignore')` is the relevant
      sklearn option — set this deliberately and mention in your report
      that you did.
- [ ] **Don't hardcode any number you computed from your data.** No
      `df['bmi'].fillna(27.3)` with a number typed in by hand — always
      reference the *saved, fitted* imputer object instead. The brief
      explicitly calls this out: "do not hardcode values from your
      training run by hand."
- [ ] **`predict.py` must run on raw, uncleaned input in the exact same
      column format as `brain_tumor_data.csv`,** including patient_id and
      the same missing-value markers (`"Unknown"`, `"?"`, blanks) — because
      that's the format madam's held-out file will arrive in. Your script
      needs its own copy of the "fix `Unknown`/`?` to NaN, normalize
      gender casing" logic at the top, before applying the saved imputer.
- [ ] **Test this yourself before submitting**: take a random 50-row
      sample of your own data, pretend it's "madam's held-out set"
      (drop the tumor_type column), run it through `predict.py` exactly as
      written, and confirm it produces sensible output with zero manual
      intervention. If it breaks here, it'll break for her too.

## Why this combination specifically matters for your grade

Madam framed this as a marks-affecting risk, which tells you the rubric
likely rewards:
- A validation/test gap that's small and explained, not hand-waved away.
- Explicit evidence in your report that you used regularization
  *deliberately*, with reasoning, not just because "everyone adds dropout."
- A `predict.py` that visibly does NOT recompute statistics from new data —
  this is checkable just by reading your code, and is an easy place to lose
  marks if you get it wrong even with a great model.

## A simple gut-check before you submit

Ask yourself: "If I deleted my training CSV right now, could `predict.py`
still correctly process a brand new file?" If the honest answer involves
"well, it would re-fit the scaler on whatever's there" — that's a leak, fix
it. If the honest answer is "yes, because everything it needs is in the
saved `.pkl`/`.h5` files," you're in good shape.
