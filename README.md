# 🛡️ OT Cybersecurity Assistant

Asistente conversacional RAG para el estándar IEC 62443 de ciberseguridad industrial. Consultá la norma en lenguaje natural con respuestas fundamentadas y citas a las fuentes.

## 🚀 Quick Start

```bash
# 1. Clonar y configurar
cp .env.example .env
# Editar .env con tu LLM_API_KEY del proveedor

# 2. Levantar servicios
make up

# 3. Ingerir documentos IEC 62443
make ingest

# 4. Probar
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{"question":"¿Qué es un security level en IEC 62443?"}'
```

Frontend: http://localhost:8501  
API docs: http://localhost:8000/docs

## 📋 Makefile

| Comando | Descripción |
|---------|-------------|
| `make build` | Instalar dependencias con uv |
| `make run` | API local con hot-reload |
| `make test` | Ejecutar tests |
| `make lint` | Ruff check |
| `make format` | Ruff format |
| `make up` | Docker Compose (api + chroma + frontend) |
| `make down` | Detener servicios |
| `make rebuild` | Reconstruir imágenes sin caché |
| `make logs` | Logs del API en Docker |
| `make ingest` | Ingerir PDFs a ChromaDB |
| `make clean` | Limpiar cachés y venv |

## 🏗️ Arquitectura

```
Usuario → Streamlit (8501) → FastAPI (8000) → ChromaDB (8001) → LLM API (OpenAI-compatible)
                                    ↓
                              RAG Pipeline
                         (ingest → retrieve → generate)
```

- **FastAPI**: API REST con endpoints `/api/health` y `/api/query`
- **ChromaDB**: Vector store HTTP (chromadb/chroma)
- **Streamlit**: Frontend de chat
- **LLM API (OpenAI-compatible)**: Embeddings (ej: `qwen3-embedding`) + LLM (ej: `qwen3.6`)
- **LangChain**: Orquestación del retrieval (MMR + Ensemble; MultiQuery opcional vía `ENABLE_MULTI_QUERY`)

## ⚙️ Configuración

Copiar `.env.example` a `.env`:

| Variable | Descripción | Default |
|----------|-------------|---------|
| `LLM_API_KEY` | API key del proveedor LLM (OpenAI-compatible) | — |
| `LLM_BASE_URL` | URL base del proveedor LLM | `https://api.openai.com/v1` |
| `LLM_MODEL` | Modelo de generación | `qwen3.7-max` |
| `LLM_MODEL_EMBEDDING` | Modelo de embeddings | `qwen3-embedding` |
| `LLM_MAX_TOKENS` | Tope de tokens de la respuesta | `1024` |
| `LLM_ENABLE_THINKING` | Razonamiento Qwen3 (`true` más lento, puede cortar por length) | `false` |
| `ENABLE_MULTI_QUERY` | Expansión de query con LLM (agrega 1 llamada extra) | `false` |
| `ENABLE_QUERY_TRANSLATION` | Traduce la query al inglés antes de buscar (mejora recuperación cross-lingüe) | `true` |
| `CHROMA_HOST` | Host de ChromaDB | `localhost` |
| `CHROMA_PORT` | Puerto de ChromaDB | `8001` |
| `CHROMA_COLLECTION` | Colección en ChromaDB | `iec62443_docs` |
| `API_KEY` | API key para autenticar clientes (backend y frontend) | — |
| `JWT_SECRET` | Secreto para firmar tokens JWT | — |

## 🔐 Autenticación y credenciales

`/api/query` exige un header `Authorization: Bearer <token>`. Hay dos formas de obtener credenciales:

### Opción 1 — API Key (la más simple)

1. Definí `API_KEY` en tu `.env` (por ejemplo `API_KEY=sk-mi-clave-secreta`).
2. Usala directamente como token:

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Authorization: Bearer sk-mi-clave-secreta" \
  -H "Content-Type: application/json" \
  -d '{"question":"¿Qué es un security level en IEC 62443?"}'
```

El frontend Streamlit lee la misma variable `API_KEY` del entorno (en Docker se la pasa `docker-compose.yml`), así que basta con tenerla en `.env` y hacer `make up`.

### Opción 2 — Token JWT (expira a las 24 h)

1. Pedile un token al endpoint de login (el `password` es tu `API_KEY`):

```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"user":"mi-usuario","password":"sk-mi-clave-secreta"}'
```

2. La respuesta trae el JWT; usalo como Bearer:

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Authorization: Bearer <token-jwt>" \
  -H "Content-Type: application/json" \
  -d '{"question":"¿Qué es un security level en IEC 62443?"}'
```

> ⚠️ En producción usá un `JWT_SECRET` fuerte y aleatorio (nunca el default) y una `API_KEY` distinta por entorno.

## 🧪 Tests

```bash
make test          # pytest
make lint          # ruff check
make format        # ruff format
```

CI/CD con GitHub Actions: lint → test → build en cada push a `main`.

## 📄 Licencia

MIT
