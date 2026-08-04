# Análisis STRIDE — IEC 62443 Assistant

## Arquitectura objetivo

```
Cliente ──HTTPS──▶ FastAPI ──httpx──▶ Nan Builders API
                      │                    (embeddings + LLM)
                      ├──▶ ChromaDB
                      └──▶ Prometheus metrics
```

Superficie de ataque: API HTTP (puerto 8000), ChromaDB HTTP (puerto 8001), Streamlit (puerto 8501).

---

## S — Spoofing (Suplantación de identidad)

| Amenaza | Severidad | Mitigación |
|---------|-----------|------------|
| Cliente no autenticado consume la API | Alta | `POST /api/query` requiere `Authorization: Bearer` (API Key o JWT) |
| API Key robada en tránsito | Media | HTTPS (TLS 1.3) en producción. `HMAC.compare_digest()` para timing-safe verification |
| JWT forjado | Alta | Firmado con `HS256` + `JWT_SECRET`. Expiración de 24 h. Validación de `iat` y `exp` |
| Suplantación del servidor ChromaDB | Baja | ChromaDB en red interna Docker. No expuesto al exterior |

## T — Tampering (Modificación de datos)

| Amenaza | Severidad | Mitigación |
|---------|-----------|------------|
| Modificación de consultas en tránsito | Media | HTTPS. CSP headers (`Content-Security-Policy`) |
| Inyección de prompts maliciosos | Media | `max_length=500` en `QueryRequest`. `field_validator` sanitiza input |
| Manipulación de respuestas del LLM | Baja | TLS en conexión a Nan Builders API |

## R — Repudiation (Repudio)

| Amenaza | Severidad | Mitigación |
|---------|-----------|------------|
| Usuario niega haber hecho una consulta | Baja | Logging estructurado con `request_id` único por request. Timestamps UTC |
| Operador niega cambios en configuración | — | Fuera de alcance del TFI (no hay panel admin) |

## I — Information Disclosure (Fuga de información)

| Amenaza | Severidad | Mitigación |
|---------|-----------|------------|
| Fuga de API Key en logs | Alta | `Authorization` header nunca se loguea. Tokens hasheados con SHA-256 si se necesita tracing |
| Exposición de chunks completos | Media | El RAG devuelve excerpts, no chunks enteros. `excerpt` truncado en `Source` |
| Stack traces en errores 500 | Media | Exception handlers capturan y devuelven `internal_server_error` genérico |
| ChromaDB expuesto | Alta | Puerto 8001 no bindeado al host en producción. Solo red interna Docker |
| Métricas Prometheus públicas | Baja | `/api/metrics` requiere API Key (admin scope). Sin autenticación en TFI, protegido en docs |

## D — Denial of Service (Denegación de servicio)

| Amenaza | Severidad | Mitigación |
|---------|-----------|------------|
| Flood de consultas al endpoint | Alta | Rate limiting: 30 req/min por IP (`slowapi`). Header `Retry-After` en respuesta 429 |
| Consultas que agotan tokens del LLM | Media | `max_length=500` en input. Timeout de 60 s en httpx hacia Nan Builders |
| PDFs maliciosos en ingesta | Baja | `pymupdf.FileDataError` capturado. `IngestionError` en input inválido |
| Ataque de fuerza bruta al login | Media | Rate limiting aplica a `/api/auth/login` (mismo límite global) |

## E — Elevation of Privilege (Elevación de privilegios)

| Amenaza | Severidad | Mitigación |
|---------|-----------|------------|
| Acceso a `/api/metrics` sin autorización | Baja | Documentado que requiere API Key. En TFI no se fuerza (métrica pública aceptable) |
| JWT con scope elevado | Baja | Scope `"query"` hardcodeado en `create_jwt()`. Sin endpoint admin en TFI |
| Bypass de rate limit cambiando IP | Baja | Aceptado para el alcance del TFI. En producción: API Key-based rate limiting |

---

## Headers de seguridad implementados

| Header | Valor |
|--------|-------|
| `X-Content-Type-Options` | `nosniff` |
| `X-Frame-Options` | `DENY` |
| `X-XSS-Protection` | `1; mode=block` |
| `Strict-Transport-Security` | `max-age=31536000; includeSubDomains` |
| `Content-Security-Policy` | `default-src 'self'` |
| `Referrer-Policy` | `no-referrer` |
| `X-Request-ID` | UUID único por request |

---

## Riesgos residuales aceptados

- ChromaDB sin autenticación interna (red Docker aislada, puerto no expuesto).
- Rate limiting por IP (bypasseable con VPN/proxy). Aceptable para demo TFI.
- `/api/metrics` sin autenticación forzada (requiere API Key documentada, no enforceada en código).
- Sin WAF ni DDoS protection (fuera de alcance del TFI).
