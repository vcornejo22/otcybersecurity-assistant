"""OT Cybersecurity Assistant — Streamlit Frontend."""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")
AUTH_HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

st.set_page_config(
    page_title="OT Cybersecurity Assistant",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Asistente de Ciberseguridad Industrial")
st.markdown("Consultá normativas de ciberseguridad industrial (OT/ICS) en lenguaje natural.")
st.divider()

# ── Session state defaults ─────────────────────────
for key in (
    "messages", "top_k", "temperature",
    "enable_multi_query", "enable_thinking",
):
    if key == "messages":
        st.session_state.setdefault(key, [])
    elif key == "top_k":
        st.session_state.setdefault(key, 3)
    elif key == "temperature":
        st.session_state.setdefault(key, 0.3)
    else:
        st.session_state.setdefault(key, False)

# ── Sidebar ──────────────────────────────────────────
with st.sidebar:
    st.header("📋 Información")

    try:
        health = requests.get(f"{API_URL}/api/health", timeout=5).json()
        st.success(f"API: {health.get('status', '?')} v{health.get('version', '?')}")

        if not API_KEY:
            st.warning("⚠️ API_KEY no configurada — las consultas fallarán con 401")

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
    st.subheader("⚙️ Configuración")

    st.slider("TOP_K_DEFAULT", 1, 10, key="top_k",
              help="Cantidad de fragmentos relevantes a recuperar")
    st.slider("Temperatura", 0.0, 1.0, 0.1, key="temperature",
              help="Controla la creatividad de la respuesta (0 = preciso, 1 = creativo)")
    st.toggle("MultiQuery (expansión de consulta con LLM)", key="enable_multi_query",
              help="Activa una llamada extra al LLM para reescribir la pregunta en 3 variantes")
    st.toggle("Thinking (razonamiento del modelo Qwen3)", key="enable_thinking",
              help="Activa el razonamiento interno del modelo (más lento, puede cortar por length)")

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
                title = f"📄 {src.get('filename', '?')} — pág. {src.get('page_number', '?')}"
                with st.expander(title):
                    st.caption(src.get("excerpt", "")[:500])

# ── Input ────────────────────────────────────────────
if prompt := st.chat_input("Tu pregunta sobre ciberseguridad industrial..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("🔍 Consultando..."):
        try:
            resp = requests.post(
                f"{API_URL}/api/query",
                json={
                    "question": prompt,
                    "top_k": st.session_state.top_k,
                    "temperature": st.session_state.temperature,
                    "enable_multi_query": st.session_state.enable_multi_query,
                    "enable_thinking": st.session_state.enable_thinking,
                },
                headers=AUTH_HEADERS,
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
    "<div style='text-align: center; color: #666;'>"
    "🛡️ Asistente de Ciberseguridad Industrial — TFI UFASTA"
    "</div>",
    unsafe_allow_html=True,
)
