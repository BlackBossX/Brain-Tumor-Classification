# Neural Network & Keras Concepts, From Zero

You said you've never trained one before. This file explains the vocabulary
you'll run into, in the order you'll meet it, with no assumed prior
knowledge.

## 1. What is a neuron / layer, mechanically?

A single neuron does: take some numbers in, multiply each by a weight,
add them up, add one more number (the "bias"), then squash the result
through an activation function. A "layer" is just a bunch of neurons doing
this in parallel, all looking at the same inputs.

```
inputs (your 27+ features) -> [Hidden Layer 1: 64 neurons] ->
[Hidden Layer 2: 32 neurons] -> [Output Layer: 3 neurons] -> probabilities
```

"Feed-forward" just means information flows in one direction, input to
output, with no loops.

## 2. Activation functions

- **ReLU** (Rectified Linear Unit): outputs the input if positive, else 0.
  Used in hidden layers because it's simple, fast, and avoids a problem
  called "vanishing gradients" that older activation functions (sigmoid,
  tanh) suffer from in deeper networks.
- **Softmax**: used only in the output layer for multi-class problems.
  Takes the raw output numbers ("logits") and converts them into
  probabilities that sum to 1 across all classes. E.g. raw outputs
  `[2.0, 0.5, -1.0]` become something like `[0.78, 0.18, 0.04]`.

## 3. Loss function

The loss function measures "how wrong" the model's prediction is for one
example, as a single number. Training = adjusting weights to make this
number smaller, on average, across all training examples.

- **Categorical cross-entropy**: the standard loss for multi-class
  classification. Roughly: it punishes the model heavily for being
  confidently wrong, and rewards being confidently right, in a way that's
  mathematically well-behaved for gradient descent.
- `categorical_crossentropy` expects your labels as **one-hot vectors**
  (e.g. Glioma = `[1,0,0]`).
- `sparse_categorical_crossentropy` expects your labels as plain
  **integers** (e.g. Glioma = `0`). Functionally equivalent math, different
  expected input format — pick based on how you encoded your target in
  Part A.

## 4. Optimiser, learning rate, epochs, batch size

- **Optimiser** = the algorithm that decides *how* to adjust weights given
  the loss. **Adam** is the standard modern default — it adapts the step
  size per-parameter automatically, which usually trains faster and more
  reliably than plain gradient descent without much tuning.
- **Learning rate** = how big a step the optimiser takes each update. Too
  high → training is unstable/diverges. Too low → training is painfully
  slow. `0.001` is Adam's well-tested default starting point.
- **Epoch** = one full pass through the entire training dataset.
- **Batch size** = how many training examples the model looks at before
  updating its weights once. With batch size 32 and, say, 6,000 training
  rows, one epoch = ~188 weight updates.

## 5. Overfitting & regularisation (see file 03 for the full playbook)

- **Dropout**: during training, randomly "turn off" a fraction of neurons
  on each forward pass (e.g. `Dropout(0.3)` turns off ~30%). Forces the
  network to not rely on any single neuron, which improves generalization.
  Dropout is automatically disabled during evaluation/prediction — Keras
  handles this for you.
- **Early stopping**: a callback that watches validation loss during
  training and stops once it hasn't improved for a set number of epochs
  ("patience"), optionally rolling back to the best-performing weights seen
  so far.
- **L2 regularization (weight decay)**: adds a penalty to the loss based on
  how large the weights are, discouraging the network from fitting noise
  with extreme weight values.

## 6. Keras code skeleton (just to anchor the vocabulary — write your own
final version, with your own comments, per the academic integrity rules)

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax'),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',  # if labels are integers 0/1/2
    metrics=['accuracy'],
)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
)
```

`history.history` gives you the per-epoch train/val loss and accuracy,
which is what you plot for Part C Step 2.

## 7. Saving the model and pipeline (for the deliverables)

```python
model.save('studentID_model.h5')

import joblib
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(imputer_numeric, 'imputer_numeric.pkl')
joblib.dump(imputer_categorical, 'imputer_categorical.pkl')
joblib.dump(encoder, 'encoder.pkl')
```

Your `predict.py` then does, conceptually:

```python
df = pd.read_csv(new_data_path)
# 1. clean markers ("Unknown"->NaN, "?"->NaN, gender casing)
# 2. df_num = imputer_numeric.transform(df[numeric_cols])
# 3. df_cat = imputer_categorical.transform(df[categorical_cols])
# 4. encoded = encoder.transform(df_cat)
# 5. X = scaler.transform(concat(df_num, encoded))
# 6. model = keras.models.load_model('studentID_model.h5')
# 7. probs = model.predict(X)
# 8. predicted_class = label_decoder[probs.argmax(axis=1)]
```

This is the same flow as training, minus any `.fit()` calls — every
transformation step uses an object that was already fitted and saved.

## 8. Quick glossary

| Term | One-line meaning |
|---|---|
| Weights | The learned numbers that determine each neuron's behaviour |
| Bias | An extra learned number added per neuron, shifts the activation |
| Forward pass | Computing predictions by pushing inputs through the network |
| Backpropagation | The algorithm that computes how to adjust weights to reduce loss |
| Gradient | The direction/amount to nudge each weight to reduce loss |
| Logits | Raw, pre-softmax output numbers |
| Trainable parameters | Total count of all weights + biases the model learns |
| Stratify | Splitting data so each subset keeps the same class proportions |
