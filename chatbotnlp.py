# Este doc utiliza tecnicas de NLP convencionales para analizar una pregunta y generar un clasificador del tipo de respuesta
# en funcion a un patron.

import spacy
import sys

from spacy.matcher import Matcher

# Carga el modelo de lenguaje en español de SpaCy
nlp = spacy.load("es_core_news_sm")

# Define patrones para identificar preguntas sobre créditos y deudas
credit_patterns = [
    [{"LOWER": {"IN": ["sacar", "obtener", "necesitar"]}}, {
        "LOWER": {"IN": ["crédito", "credito"]}}],
    [{"LOWER": {"IN": ["querer"]}}, {"LOWER": {"IN": ["sacar", "obtener"]}}, {
        "LOWER": {"IN": ["crédito", "credito", "prestamo", "préstamo"]}}],
    [{"LOWER": "querer"}, {"LOWER": "averigurar"}, {"LOWER": {"IN": ["sacar", "obtener"]}}, {
        "LOWER": {"IN": ["crédito", "credito", "prestamo", "préstamo"]}}],
]

debt_patterns = [
    [{"LOWER": {"IN": ["una", "un"]}}, {
        "LOWER": {"IN": ["deuda", "crédito"]}}, {"LOWER": "pendiente"}],

    [{"LOWER": {"IN": ["pagar", "liquidar", "conocer"]}},
        {"LOWER": {"IN": ["pagar", "deuda", "crédito"]}}],

    [{"LOWER": "deuda"}, {"LOWER": "pendiente"}, {"LOWER": "pagar"}],

    # [{"LOWER": "necesitar"}, {
    #     "LOWER":  {"IN": ["pagar", "liquidar", "conocer"]}}, {"LOWER": "deuda"}],
]

split_patterns = [
    [{"LOWER": {"IN": ["como", "cuanto"]}}, {
        "LOWER": {"IN": ["sería", "es"]}}, {"LOWER": {"IN": ["las", "la"]}}, {"LOWER": "cuotas"}],
    [{"LOWER": {"IN": ["como", "cuanto"]}}, {
        "LOWER": {"IN": ["ser"]}}, {"LOWER": {"IN": ["el"]}}, {"LOWER": "cuota"}],
]

# Inicializa el matcher con los patrones
matcher = Matcher(nlp.vocab)
matcher.add("credito", credit_patterns)
matcher.add("deuda", debt_patterns)
matcher.add("cuotas", split_patterns)

# Función para determinar el tipo de pregunta

# doc = nlp("A complex-example,!")
# print([token.text for token in doc])


def determine_intent(text):

    text = text.lower()
    texto = texto_2_lemmas(text)

    doc = nlp(texto)

    matches = matcher(doc)
    print([token.text for token in doc])
    print([token.lemma_ for token in doc])

    # doc = nlp("Hello, world! Hello world!")
    # matches = matcher(doc)
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        span = doc[start: end]  # The matched span
        print(match_id, string_id, start, end, span.text)

    if any([match[0] == nlp.vocab.strings["credito"] for match in matches]):
        return "Credito"
    elif any([match[0] == nlp.vocab.strings["deuda"] for match in matches]):
        return "Deuda"
    elif any([match[0] == nlp.vocab.strings["cuotas"] for match in matches]):
        return "Cuotas"
    else:
        return "no_identificado"


def texto_2_lemmas(text):
    text = text.lower()
    doc = nlp(text)

    output = []
    for token in doc:
        if not token.is_stop:
            output.append(token.lemma_)

    text = ' '.join(output)
    return(text)


# Ejemplos de preguntas
questions = [
    "¿Cómo puedo obtener un crédito?",
    "Quiero sacar un crédito, ¿cuáles son los requisitos?",
    "Tengo una deuda pendiente, ¿cómo puedo pagarla?",
    "Necesito liquidar mi deuda, ¿cuál es el proceso?",
    "¿Cuáles son las tasas de interés para los créditos?",
    "Como sería las cuotas, y más o menos mi sueldo en mano es de 320 mil pesos más o menos",
    "Hola quería averiguar para sacar un préstamo"
]

# Determina la intención de cada pregunta
for question in questions:
    intent = determine_intent(question)
    print(f"Pregunta: {question}\nTipo de Intención: {intent}\n")

# question = None

# while True:
#     if not question:
#         question = input("Prompt: ")
#     if question in ['quit', 'q', 'exit']:
#         sys.exit()

#     intent = determine_intent(question)
#     print(f"Pregunta: {question}\nTipo de Intención: {intent}\n")
#     question = None
