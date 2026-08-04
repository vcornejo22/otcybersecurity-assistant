"""IEC 62443 Assistant — Streamlit Frontend."""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="IEC 62443 Assistant",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Asistente IEC 62443")
st.markdown("Consultá la norma de ciberseguridad industrial en lenguaje natural.")
st.divider()

# ── Sidebar ──────────────────────────────────────────
with st.sidebar:
    st.header("📋 Información")

    try:
        health = requests.get(f"{API_URL}/api/health", timeout=5).json()
        st.success(f"API: {health.get('status', '?')} v{health.get('version', '?')}")

        if health.get("rag_loaded"):
            st.info(
                f"📚 {health.get('documents_count', 0)} documentos\n"
                f"🧩 {health.get('chunks_count', 0)} chunks"
            )
        else:
            st.warning("RAG no cargado — ejecutá `make ingest` primero")
    except Exception:
        st.error("API no disponible")

    st.divider()

    if st.button("🗑️ Limpiar Chat", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Chat history ─────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 💬 Chat")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

with col2:
    st.markdown("### 📄 Fuentes consultadas")
    if st.session_state.messages:
        last = st.session_state.messages[-1]
        if last["role"] == "assistant" and "sources" in last:
            for src in last["sources"]:
                with st.expander(f"📄 {src.get('document', '?')} — {src.get('relevance', 0):.0%}"):
                    st.caption(src.get("chunk", "")[:500])

# ── Input ────────────────────────────────────────────
if prompt := st.chat_input("Tu pregunta sobre IEC 62443..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("🔍 Consultando..."):
        try:
            resp = requests.post(
                f"{API_URL}/api/query",
                json={"question": prompt, "top_k": 3, "temperature": 0.3},
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data.get("sources", []),
                    }
                )
            else:
                err = resp.json().get("error", {}).get("message", resp.text)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"❌ Error: {err}",
                        "sources": [],
                    }
                )
        except Exception as e:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"❌ No se pudo conectar a la API: {e}",
                    "sources": [],
                }
            )

    st.rerun()

st.divider()
st.markdown(
    "<div style='text-align: center; color: #666;'>🛡️ IEC 62443 Assistant — TFI UFASTA</div>",
    unsafe_allow_html=True,
)
