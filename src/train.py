from pathlib import Path
import json, random
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models"
NUM_WORDS, MAX_LEN, SEED = 10000, 100, 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
dataset = load_dataset("dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset")
df = dataset["train"].to_pandas()
text_col = "text_es" if "text_es" in df.columns else next(c for c in df.columns if "spanish" in str(c).lower() or "espanol" in str(c).lower())
df = df[["labels", text_col]].dropna(); df.columns = ["label", "message"]
df["label"] = df["label"].astype(str).str.lower().str.strip(); df = df[df.label.isin(["ham", "spam"])]
X, y = df.message.astype(str).values, df.label.map({"ham": 0, "spam": 1}).values
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=.30, random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=.50, random_state=SEED, stratify=y_tmp)
tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>"); tokenizer.fit_on_texts(X_train)
encode = lambda values: pad_sequences(tokenizer.texts_to_sequences(values), maxlen=MAX_LEN, padding="post", truncating="post")
# Ignorar los ceros de padding; de lo contrario la GRU procesa hasta 100
# posiciones y los ceros posteriores pueden borrar la información del mensaje.
model = Sequential([Input(shape=(MAX_LEN,)), Embedding(NUM_WORDS, 128, mask_zero=True), GRU(64), Dropout(.4), Dense(32, activation="relu"), Dropout(.3), Dense(1, activation="sigmoid")])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(encode(X_train), y_train, validation_data=(encode(X_val), y_val), epochs=15, batch_size=32, callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])
print("Test:", model.evaluate(encode(X_test), y_test, verbose=0))
OUT.mkdir(exist_ok=True); model.save(OUT / "modelo_spam_gru.keras"); (OUT / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
print(f"Artefactos guardados en {OUT}")
