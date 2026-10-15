from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
ROOT = Path(__file__).resolve().parents[1]; MODEL_PATH = ROOT/"models/modelo_spam_gru.keras"; TOKENIZER_PATH = ROOT/"models/tokenizer.json"; MAX_LEN = 100
def load_artifacts(model_path=MODEL_PATH, tokenizer_path=TOKENIZER_PATH):
    if not model_path.exists(): raise FileNotFoundError(f"No existe el modelo: {model_path}")
    if not tokenizer_path.exists(): raise FileNotFoundError(f"No existe el tokenizador: {tokenizer_path}. Ejecuta `python src/train.py`.")
    return load_model(model_path, compile=False), tokenizer_from_json(tokenizer_path.read_text(encoding="utf-8"))
def predict(messages, threshold=.5):
    model, tokenizer = load_artifacts(); messages = [str(x) for x in messages]
    values = pad_sequences(tokenizer.texts_to_sequences(messages), maxlen=MAX_LEN, padding="post", truncating="post")
    probabilities = np.asarray(model.predict(values, verbose=0)).ravel()
    return [{"mensaje": m, "prediccion": "SPAM" if p >= threshold else "NORMAL", "probabilidad_spam": float(p)} for m,p in zip(messages, probabilities)]
