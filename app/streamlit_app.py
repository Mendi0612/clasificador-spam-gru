from pathlib import Path
import sys
import streamlit as st
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.predict import predict, load_artifacts
st.set_page_config(page_title="Clasificador de spam", page_icon="📩")
st.title("Clasificador de SMS: spam o normal")
try: load_artifacts()
except FileNotFoundError as error: st.error(str(error)); st.stop()
message = st.text_area("Escribe un mensaje", height=140)
if st.button("Clasificar", type="primary"):
    if not message.strip(): st.warning("Escribe un mensaje para clasificar.")
    else:
        result = predict([message])[0]; st.metric("Resultado", result["prediccion"])
        st.write(f"Probabilidad de spam: **{result['probabilidad_spam']:.1%}**")
