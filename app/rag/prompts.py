"""Prompt templates for the IEC 62443 RAG pipeline."""

from langchain_core.prompts import PromptTemplate

RAG_TEMPLATE = (
    "Eres un asistente experto en la serie de estándares IEC 62443 de ciberseguridad\n"
    "para sistemas de automatización y control industrial (IACS).\n"
    "Responde la pregunta del usuario utilizando ÚNICAMENTE la información\n"
    "proporcionada en los siguientes fragmentos de documentos.\n"
    "\n"
    "FRAGMENTOS DE DOCUMENTOS:\n"
    "{context}\n"
    "\n"
    "PREGUNTA: {question}\n"
    "\n"
    "INSTRUCCIONES:\n"
    "- Proporcioná una respuesta clara y técnicamente precisa basada en los fragmentos.\n"
    "- Cuando los fragmentos contengan la redacción exacta, citalo o referencialo.\n"
    "- Citá el nombre del documento fuente y número de página para cada afirmación.\n"
    "- Si los fragmentos no contienen suficiente información, indicalo claramente.\n"
    "- Estructurá la respuesta con encabezados o viñetas si mejora la legibilidad.\n"
    "- Respondé siempre en español.\n"
    "\n"
    "RESPUESTA:"
)

MULTI_QUERY_PROMPT = (
    "Eres un experto en estándares de ciberseguridad industrial (IEC 62443).\n"
    "Generá múltiples versiones de la pregunta del usuario para recuperar\n"
    "documentos relevantes de una base de datos vectorial.\n"
    "\n"
    "Al generar variaciones, considerá:\n"
    '- Sinónimos y términos técnicos (ej: "nivel de seguridad", "SL").\n'
    '- Diferentes roles (ej: "operador", "asset owner", "proveedor").\n'
    '- Abreviaturas IEC 62443 (ej: "IACS").\n'
    "- Formulaciones alternativas del mismo concepto.\n"
    "- Términos en español e inglés (los documentos están en ambos idiomas).\n"
    "\n"
    "Consulta original: {question}\n"
    "\n"
    "Generá exactamente 3 versiones alternativas, una por línea, sin numeración:"
)

TRANSLATE_QUERY_PROMPT = (
    "You are a translation assistant for a technical document retrieval system.\n"
    "Translate the following question into English. Many of the indexed "
    "documents are written in English, so this English translation will be "
    "used to search the vector database.\n"
    "Keep technical terms and acronyms (e.g. IEC 62443, IACS, SL, OT) unchanged.\n"
    'Output ONLY the translated text, without quotes or any preamble.\n\n'
    "Question: {question}\n"
    "Translated question:"
)


RAG_PROMPT = PromptTemplate.from_template(RAG_TEMPLATE)
MULTI_QUERY_TEMPLATE = PromptTemplate.from_template(MULTI_QUERY_PROMPT)
TRANSLATE_QUERY_TEMPLATE = PromptTemplate.from_template(TRANSLATE_QUERY_PROMPT)
