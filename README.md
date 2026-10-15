## Clasificador de spam GRU

Instala las dependencias con `pip install -r requirements.txt`.

Si falta `models/tokenizer.json`, entrena y guarda los artefactos con:

```bash
python src/train.py
```

Inicia la aplicación:

```bash
streamlit run app/streamlit_app.py
```

El modelo y el tokenizador deben permanecer juntos en `models/`; el tokenizador es parte del modelo de inferencia porque convierte cada mensaje en las secuencias que espera la GRU.
