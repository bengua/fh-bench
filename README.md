# fh-bench

[![Licence : MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/OWNERTEST/fh-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNERTEST/fh-bench/actions/workflows/ci.yml)

**Mesurer ce qu'on peut vraiment demander à un modèle local — et l'afficher sur une seule carte.**

`fh-bench` interroge un ou plusieurs serveurs compatibles avec l'API OpenAI (Ollama, llama.cpp,
vLLM, LiteLLM…), fait passer à chaque modèle une batterie d'épreuves notées **par du code**, pas
par un autre modèle, puis imprime un tableau : ce que chaque modèle sait faire, à moitié, ou pas du tout.

## Fonctionnalités

Chaque épreuve est jugée par un programme qui vérifie le résultat :

| Épreuve | Ce qui est vérifié |
|---|---|
| `tool_call` | l'outil attendu est vraiment appelé, avec les bons arguments |
| `json_schema` | la réponse est un JSON qui respecte le schéma demandé |
| `reasoning` | la réponse à un petit problème est la bonne |
| `code` | le programme écrit passe ses tests (exécuté dans un interpréteur isolé, sans environnement, CPU et mémoire bridés) |
| `long_context` | une information cachée est retrouvée dans un texte de 4k à 128k tokens |
| `french` | la réponse est en français correct |
| `instruction` | les consignes de forme sont respectées |
| `vision` | la couleur d'une image est bien lue (modèles avec vision seulement) |
| `fim` | complétion de code « au milieu » (serveurs llama.cpp) |
| `speed` | débit de génération en tokens par seconde |
| `agentic` | optionnel : le modèle travaille seul sur un petit dépôt, jugé par ses tests |

Autres points :

- **Plusieurs façons de servir un même modèle** : les serveurs d'un même `group` sont comparés ; le
  premier passe toute la batterie, les autres une version courte (vitesse, outils, JSON).
- **Respect de la machine** : un jeu qui tourne sur la carte graphique n'est jamais interrompu, et une
  commande optionnelle remet la machine dans son état habituel à la fin.
- **Résultats en JSON**, relisibles par `fh-bench map`, ou émis comme événements vers votre propre outil.

## Installation depuis zéro

Il faut Python 3.11 ou plus récent, et un serveur de modèles qui tourne déjà.

```sh
git clone https://github.com/OWNERTEST/fh-bench
cd fh-bench
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
fh-bench --help
```

Aucune carte graphique n'est nécessaire pour installer ni pour lancer les tests (`pip install -e '.[dev]'`
puis `./verify.sh`).

## Utilisation

Mesurer un seul serveur, par exemple Ollama :

```sh
fh-bench run --base-url http://127.0.0.1:11434/v1 --model qwen3:8b --context 32768
fh-bench map
```

Les commandes :

```text
fh-bench list                 les serveurs à mesurer, regroupés par modèle
fh-bench run                  lance la batterie (--tasks tool_call,code  --no-variants  --servers a,b  --out DIR)
fh-bench map [DIR]            imprime le tableau des résultats
fh-bench emit [DIR]           réémet les résultats enregistrés comme événements
fh-bench import-turnstile     convertit une carte de slots turnstile en fichier fh-bench.toml
```

Lecture du tableau : `✓` réussi (score ≥ 0,8), `~` partiel (≥ 0,5), `✗` échec, `—` sans objet.

## Configuration

### Pointer vers n'importe quel serveur compatible OpenAI

Il suffit de donner l'adresse du serveur, en option ou par variable d'environnement :

| Serveur | `FH_BENCH_BASE_URL` | `FH_BENCH_MODEL` |
|---|---|---|
| Ollama | `http://127.0.0.1:11434/v1` | `qwen3:8b` |
| llama.cpp (`llama-server`) | `http://127.0.0.1:8080` | (inutile) |
| vLLM | `http://127.0.0.1:8000/v1` | `Qwen/Qwen3-8B` |
| LiteLLM | `http://127.0.0.1:4000/v1` | le nom du modèle dans LiteLLM |

```sh
export FH_BENCH_BASE_URL=http://127.0.0.1:8000/v1
export FH_BENCH_MODEL=Qwen/Qwen3-8B
export FH_BENCH_API_KEY=...        # seulement si le serveur demande un jeton
fh-bench run
```

### Variables d'environnement

| Variable | Rôle | Défaut |
|---|---|---|
| `FH_BENCH_BASE_URL` | adresse du serveur (un seul serveur) | — |
| `FH_BENCH_MODEL` | nom du modèle | celui du serveur |
| `FH_BENCH_NAME` | libellé dans les résultats | le nom du modèle, sinon `default` |
| `FH_BENCH_CONTEXT` | fenêtre de contexte du serveur, en tokens | `8192` |
| `FH_BENCH_VISION` | `1` si le modèle accepte les images | désactivé |
| `FH_BENCH_API_KEY` | jeton envoyé en `Authorization: Bearer` | aucun |
| `FH_BENCH_CONFIG` | fichier de configuration TOML | `./fh-bench.toml`, puis `~/.config/fh-bench/config.toml` |
| `FH_BENCH_RESULTS` | dossier des résultats | `~/.local/share/fh-bench/results` |
| `FH_BENCH_EVENT_CMD` | commande qui reçoit chaque événement (avec `--emit`) | sortie standard, une ligne JSON par événement |
| `FH_BENCH_OPENFOX` | adresse d'un harnais d'agent OpenFox pour l'épreuve `agentic` | vide : épreuve ignorée |
| `FH_BENCH_OPENFOX_TOKEN_FILE` | fichier contenant le jeton de ce harnais | `~/.config/fh-bench/openfox-token` |

### Plusieurs serveurs

Pour comparer plusieurs serveurs, décrivez-les dans un fichier TOML : voir
[`examples/fh-bench.example.toml`](examples/fh-bench.example.toml). Un serveur peut y déclarer une
commande `load` (démarrer ou changer de modèle avant la mesure) et le fichier une commande `after`
(remettre la machine en état à la fin). Les jetons n'y figurent jamais : `api_key_env` donne seulement
le **nom** de la variable d'environnement qui les contient.

## Pourquoi ce projet

Les classements publics disent ce qu'un modèle vaut en moyenne, pas ce qu'il fera chez vous, sur votre
matériel, avec votre façon de le servir. `fh-bench` répond à une question simple : *à ce modèle-là, sur
cette machine, qu'est-ce que je peux confier ?* — mesuré, pas supposé.

## Licence

MIT — voir [LICENSE](LICENSE).

## English

**Measure what you can actually ask a local model — and print it as one map.**

`fh-bench` talks to any OpenAI-compatible server (Ollama, llama.cpp, vLLM, LiteLLM…), runs each model
through a battery of tasks **checked by code** (a tool really called, JSON that validates, a correct answer,
generated code whose tests pass in a sandboxed interpreter, a needle found at 4k–128k tokens, a colour read
from an image, fill-in-the-middle, speed, and an optional agentic task), and prints a table of results.

```sh
git clone https://github.com/OWNERTEST/fh-bench
cd fh-bench && python3 -m venv .venv && . .venv/bin/activate && pip install -e .

# any OpenAI-compatible server
export FH_BENCH_BASE_URL=http://127.0.0.1:11434/v1   # Ollama; llama.cpp: http://127.0.0.1:8080, vLLM: :8000/v1, LiteLLM: :4000/v1
export FH_BENCH_MODEL=qwen3:8b
fh-bench run
fh-bench map
```

Optional: `FH_BENCH_API_KEY` (bearer token), `FH_BENCH_CONTEXT`, `FH_BENCH_VISION=1`, `FH_BENCH_RESULTS`,
`FH_BENCH_EVENT_CMD`, and `FH_BENCH_OPENFOX` for the agentic task (skipped when unset). Several servers are
described in a TOML file — see [`examples/fh-bench.example.toml`](examples/fh-bench.example.toml).
No GPU is needed to install or test (`pip install -e '.[dev]' && ./verify.sh`). License: MIT.

---

Un projet [FreedzHost](https://freedzhost.cloud) — souveraineté numérique, IA locale, homelab.
A [FreedzHost](https://freedzhost.cloud) project.
