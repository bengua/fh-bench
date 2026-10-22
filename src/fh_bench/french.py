"""The French the bench needs: prompts that ask a model to answer in French, and the words it looks for.

French on purpose: what is measured is the model's French, so the
questions and the expected words have to be French.
"""

REFUSAL_MARKERS = ("je ne peux pas", "désolé", "je ne suis pas en mesure")

WORDS = frozenset({"le", "la", "les", "un", "une", "des", "du", "de", "et", "est", "que", "qui", "dans", "pour", "pas", "sur",
                   "avec", "au", "aux", "ce", "cette", "son", "ses", "ne", "se", "en", "il", "elle", "nous", "vous", "ils", "sont"})

SAMPLE = "Le ciel est bleu parce que la lumière est diffusée dans l'air"

AGENT_ANSWER = "La fenêtre de déploiement est jeudi à 02:30, jamais pendant une session de jeu."

# What a model answers to each of CASES when it does well (used by the tests).
GOOD_ANSWERS = {"ciel": "La lumière du Soleil est diffusée par l'atmosphère, surtout le bleu.", "Traduis": "La réunion est reportée à jeudi à 15 h.",
                "OUI ou NON": "NON", "riment": "saison, raison, prison"}

# (prompt, kind, expected). groups: every group needs one of its words; starts: the answer opens with it; commas: at least n.
CASES = (
    ("Explique en deux phrases pourquoi le ciel est bleu.", "groups", (("rayleigh", "diffus", "lumière"),)),
    ("Traduis en français : 'The meeting is moved to Thursday at 3 pm.'", "groups", (("jeudi",), ("15", "3"))),
    ("Réponds par OUI ou NON : Paris est-elle la capitale de l'Allemagne ?", "starts", "NON"),
    ("Donne trois mots français qui riment avec 'maison', séparés par des virgules.", "commas", 2),
)


def judged(text: str, kind: str, expected) -> bool:
    low = text.lower()
    if kind == "groups":
        return all(any(w in low for w in group) for group in expected)
    if kind == "starts":
        return text.strip().upper().startswith(expected)
    return text.count(",") >= expected
