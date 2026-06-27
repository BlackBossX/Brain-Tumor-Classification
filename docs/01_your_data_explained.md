# Your Actual Dataset, Explained

I opened `brain_tumor_data.csv` directly (not just the dictionary) and checked
every column's real raw values. Here's exactly what's in it, so your Part A
answers are based on fact, not assumption.

## Shape and types

- 9,000 rows, 29 columns (matches the dictionary).
- `patient_id`: unique per row (e.g. `PT100001`). This is the identifier
  column — **exclude it from modelling.** Why? It carries no biological
  signal about tumour type; it's just a row label. If you accidentally left
  it in as a feature, a neural net might find some meaningless pattern in
  the ID numbers and "leak" toward overfitting — a textbook example of a
  spurious feature.

- When you load with plain `pd.read_csv()`, pandas *already* auto-detects
  empty cells in fully-numeric columns as `NaN` and gives correct dtypes
  (`int64`/`float64`) for every numeric column: `age, bmi, tumor_size_mm,
  tumor_growth_rate, headache_severity, mri_intensity, ct_density,
  edema_grade, ki67_index, bp_systolic, bp_diastolic, wbc_count, crp_level`.
  So in this particular file you will **not** find a numeric column stuck as
  `object`/string dtype because of a marker like `"N/A"` mixed into numbers.

  **But the assignment explicitly asks you to check for and explain this.**
  Show the check anyway (e.g. `df.dtypes`, plus an explicit scan for stray
  string tokens inside numeric columns) — demonstrating the check is part of
  the marks, even where the answer turns out to be "none found here."

## The real missing-value markers in your data

Two categorical columns use a **non-standard missing marker** instead of a
truly empty cell:

| Column | Marker used | Count |
|---|---|---|
| `alcohol_consumption` | the literal string `"Unknown"` | 381 rows |
| `genetic_marker_status` | the literal string `"?"` | 247 rows |

Everything else missing is a genuinely blank CSV cell, which pandas reads as
real `NaN` automatically.

**What to do:** replace `"Unknown"` and `"?"` with `np.nan` explicitly, e.g.

```python
df['alcohol_consumption'] = df['alcohol_consumption'].replace('Unknown', np.nan)
df['genetic_marker_status'] = df['genetic_marker_status'].replace('?', np.nan)
```

Then your `df.isna().sum()` will show the true missing count per column,
which the assignment asks you to report.

## Missing value counts per column (true counts, after fixing the markers above)

| Column | Missing count |
|---|---|
| patient_id | 0 |
| age | 0 |
| gender | 0 |
| ethnicity | 281 |
| region | 0 |
| bmi | 436 |
| smoking_status | 381 |
| alcohol_consumption | 3,804 *(381 "Unknown" + 3,423 truly blank — see note)* |
| family_history | 356 |
| tumor_size_mm | 0 |
| tumor_location | 0 |
| tumor_growth_rate | 540 |
| headache_severity | 457 |
| nausea / vision_problems / seizures / memory_loss / balance_issues | 0 |
| mri_intensity | 356 |
| ct_density | 434 |
| edema_grade | 283 |
| contrast_enhancement | 1,136 |
| ki67_index | 664 |
| bp_systolic | 0 |
| bp_diastolic | 197 |
| wbc_count | 253 |
| crp_level | 445 |
| genetic_marker_status | 247 *(all from the "?" marker)* |

Note: `alcohol_consumption`'s 3,804 missing count already includes the 381
"Unknown" rows once you convert them to `NaN` — pandas' default read already
treated a large chunk of that column as blank too. Re-run `isna().sum()`
yourself after the fix to confirm the exact figure in your own notebook; the
number above is what I measured directly from your file.

## The "dirty categorical" issue: `gender`

Your `gender` column literally contains four distinct strings:
`'Male', 'Female', 'male', 'female'`. This is a classic data-cleaning trap —
if you one-hot encode *before* fixing the casing, you'll get **four**
columns instead of two, and the model will treat `'Male'` and `'male'` as
unrelated categories. Fix with:

```python
df['gender'] = df['gender'].str.capitalize()
```

(Always check `.unique()` on every categorical column before encoding —
this is the general lesson, not just a gender-specific fix.)

## Categorical columns and how to think about ordinal vs nominal

The assignment explicitly wants you to justify one-hot vs ordinal per
column. Here's the actual breakdown of your data:

**Truly ordinal (has a natural order) — ordinal/integer encode:**
- `alcohol_consumption`: None < Moderate < Heavy
- `edema_grade`: already numeric 0,1,2,3 in your data — no encoding needed
- `contrast_enhancement`: None < Mild < Moderate < Strong

**Nominal (no natural order) — one-hot encode:**
- `gender`, `ethnicity`, `region`, `smoking_status`, `family_history`,
  `tumor_location`, `genetic_marker_status`, and the Yes/No symptom columns
  (`nausea`, `vision_problems`, `seizures`, `memory_loss`, `balance_issues`)

For binary Yes/No columns specifically, you don't need full one-hot (which
would give 2 redundant columns) — just map `Yes→1, No→0`. That's
mathematically identical to ordinal/one-hot for a 2-category column but
saves you a column.

## Class balance

Measured directly from your CSV:

| Class | Count | Proportion |
|---|---|---|
| Glioma | 3,802 | 42.2% |
| Meningioma | 3,089 | 34.3% |
| Pituitary | 2,109 | 23.4% |

This is **moderate imbalance**, not severe (nothing like 95/5). Worth
naming in your report, and worth stratifying your train/val/test split by
this column so all three splits keep roughly this same ratio.

## A note on `tumor_growth_rate` and similar continuous medical features

Some of these (`ki67_index`, `crp_level`, `tumor_growth_rate`) have skewed
ranges (e.g. `ki67_index` 0.1–60). This is worth mentioning when you discuss
why scaling matters — gradient-based training works best when input
features are on comparable scales, and proliferation markers like `ki67`
have a very different numeric range than, say, `headache_severity` (0–10).
