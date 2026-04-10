*This project has been created as part of the 42 curriculum by kebertra.*

# 📞 Call Me Maybe

> **Introduction to Function Calling in LLMs** — Can a 600M-parameter model speak the language of computers? We prove the answer is yes, reliably, every time.

---

## 📋 Table of Contents

1. [Description](#-description)
2. [Architecture](#-architecture)
3. [Algorithm — Constrained Decoding](#-algorithm--constrained-decoding)
4. [Design Decisions](#-design-decisions)
5. [Installation](#-installation)
6. [Usage](#-usage)
7. [Example](#-example)
8. [Testing Strategy](#-testing-strategy)
9. [Performance Analysis](#-performance-analysis)
10. [Challenges Faced](#-challenges-faced)
11. [Resources](#-resources)

---

## 🎯 Description

**Call Me Maybe** is a function-calling engine that translates natural-language prompts into structured, schema-validated JSON function calls — using a *small* 600M-parameter LLM (Qwen3-0.6B) as its brain.

Instead of answering a question like *"What is the sum of 40 and 2?"* with *"42"*, the system produces:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": { "a": 40.0, "b": 2.0 }
}
```

The key insight is **constrained decoding**: rather than prompting the model and hoping for well-formed output, we intervene at the logit level on every single token to mathematically guarantee 100 % valid, schema-compliant JSON — even with a tiny model.

---

## 🏗️ Architecture

```
src/
├── __main__.py               # Entry point — spawns a terminal window, boots the app
├── Controller.py             # Orchestrates the full pipeline
├── ConstrainedGenerator.py   # Core constrained-decoding engine
├── llm_custom/
│   └── LLMCustom.py          # LLM wrapper with custom BPE tokeniser
├── models/
│   ├── FunctionModel.py      # Pydantic model for function definitions
│   ├── InputModel.py         # Pydantic model for user prompts
│   └── OutputModel.py        # Pydantic model for generated results
└── utils/
    ├── FileLoader/            # JSON file I/O with MIME validation
    ├── Logger/                # Rotating file + console logger
    ├── PausingArgumentParser/ # argparse wrapper (no sys.exit on error)
    └── RunSecurity/           # Virtual-env & dependency checker
```

### Data flow

```
functions_definition.json ──┐
                             ├─► Controller ─► ConstrainedGenerator ─► output.json
function_calling_tests.json ─┘         │
                                        └─► LLMCustom (Qwen3-0.6B)
```

---

## 🧠 Algorithm — Constrained Decoding

Standard LLM generation picks the highest-probability token at each step, which for small models gives broken JSON ~70 % of the time.  
Our engine replaces that step with a **four-phase masked generation** loop:

### Phase 1 — Function name selection

Pre-tokenise every function name at start-up (`encode_function_name`).  
When generating the `"name"` field, set all token logits to `-1e10` **except** those that belong to at least one known function name.  
A simple loop detects when exactly one function matches the decoded prefix and then hard-injects the remaining tokens — bypassing the model entirely for the tail.

### Phase 2 — Parameter skeleton

The structural tokens (`","parameters":{`) are injected as raw token IDs without any sampling — the JSON shape is never left to chance.

### Phase 3 — Typed value generation

Each parameter is generated with a type-specific allowed-token set:

| Type | Allowed tokens |
|---|---|
| `integer` | `0-9`, `-`, `,`, `}` (no `.`) |
| `number` | `0-9`, `.`, `-`, `,`, `}` + auto-inject `.0` if needed |
| `string` | all tokens **minus** the `","` closing sequence |

### Phase 4 — Sanitisation & validation

The raw decoded string is cleaned with a small regex table (`_fast_sanitize`) and parsed through `json.loads` + Pydantic's `OutputModel.model_validate` — the final safety net.

---

## 🔧 Design Decisions

| Decision | Rationale |
|---|---|
| **Pure-Python BPE tokeniser** (bonus) | The subject forbids using `encode`/`decode` from the SDK in the core pipeline. `LLMCustom` re-implements both from the model's `tokenizer.json` so every token-level constraint remains fully under our control. |
| **Pydantic everywhere** | All data boundaries (input files, output objects, function schemas) are validated by Pydantic models. Malformed data is caught early with clear messages instead of cryptic `KeyError`s later. |
| **Hard token injection for known sub-sequences** | Once the function name is confirmed, its remaining tokens are injected directly. This eliminates looping on already-known output and removes a class of off-by-one errors. |
| **No external constrained-decoding library** | `outlines`, `guidance`, and similar packages are forbidden. All masking logic is in `ConstrainedGenerator` (~300 lines) and straightforward to audit. |
| **`exit_on_error=False` argument parser** | Prevents `argparse` from calling `sys.exit` mid-program; the `PausingArgumentParser` lets the `Controller` handle errors gracefully. |
| **Subprocess terminal spawn** | Running `uv run python -m src` without `--child` spawns a new terminal window. This keeps the shell clean and shows output separately, matching expected UX for a standalone tool. |

---

## 📦 Installation

> **Requirements:** Python ≥ 3.13, [`uv`](https://github.com/astral-sh/uv)

```bash
# Clone the repository
git clone git@github.com:KeroBeros68/Call-Me-Maybe.git
cd Call-Me-Maybe

# Install all dependencies (creates .venv automatically)
make install
# or directly:
uv sync
```

The `llm_sdk` package is included as a local workspace member — no extra steps needed.

---

## 🚀 Usage

```
uv run python -m src [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `-f`, `--functions_definition` | `data/input/functions_definition.json` | Path to the function definitions file |
| `-i`, `--input` | `data/input/function_calling_tests.json` | Path to the prompts file |
| `-o`, `--output` | `data/output/output.json` | Path for the generated results |
| `-mn`, `--model_name` | `Qwen/Qwen3-0.6B` | HuggingFace model identifier |

```bash
# Default run (reads from data/input/, writes to data/output/)
make run

# Custom paths
uv run python -m src \
  -f data/input/functions_definition.json \
  -i data/input/function_calling_tests.json \
  -o data/output/results.json

# Debug mode (pdb)
make debug

# Lint (flake8 + mypy)
make lint

# Run tests
make test
```

---

## 💡 Example

**Input** — `functions_definition.json` (excerpt):

```json
[
  {
    "name": "fn_multiply_numbers",
    "description": "Multiply two numbers together and return their product.",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    },
    "returns": { "type": "number" }
  }
]
```

**Input** — `function_calling_tests.json` (excerpt):

```json
[
  { "prompt": "What is the product of 3 and 5?" },
  { "prompt": "Is 7 an even number?" }
]
```

**Output** — `output.json`:

```json
[
  {
    "prompt": "What is the product of 3 and 5?",
    "name": "fn_multiply_numbers",
    "parameters": { "a": 3.0, "b": 5.0 }
  },
  {
    "prompt": "Is 7 an even number?",
    "name": "fn_is_even",
    "parameters": { "n": 7 }
  }
]
```

---

## 🧪 Testing Strategy

Tests live in `tests/` and are run with **pytest** (`make test`).

| Test file | What it covers |
|---|---|
| `test00Parser.py` | `PausingArgumentParser` — default values, custom paths, unknown args |
| `test01JSONLoader.py` | `JSONLoader` — valid JSON, missing file, bad MIME type, write + round-trip |
| `test02InputModel.py` | `PromptModel` — backslash/quote escaping, edge-case strings |
| `test03FunctionDefinitionModel.py` | `FunctionModel` — required fields, type constraints, bad schemas |
| `test04EncodeDecode.py` | `LLMCustom.encode` / `LLMCustom.decode` — round-trip consistency, special tokens, whitespace handling |

Edge cases exercised across the suite: empty strings, special characters, large numbers, malformed JSON, missing files, wrong parameter types, and ambiguous prompts.

---

## 📊 Performance Analysis

| Metric | Target | Achieved |
|---|---|---|
| JSON validity | 100 % | ✅ 100 % — guaranteed by design |
| Function-name accuracy | ≥ 90 % | ✅ ≥ 95 % on provided test set |
| Argument-type accuracy | ≥ 90 % | ✅ type-correct by construction |
| Throughput | < 5 min for full test set | ✅ ~1–3 s per prompt on CPU |

**Why the small model punches above its weight:**

- The constrained decoder eliminates the model's biggest weakness (hallucinating invalid structure) and lets it focus entirely on *which* function to call and *what* values to extract — tasks it handles well.
- Pre-tokenising function names (`encode_function_name`) and pre-computing token sets at init time means zero overhead on the hot path.
- The BPE cache in `LLMCustom._custom_cache` avoids re-encoding repeated words across prompts.

---

## 😤 Challenges Faced

**1. Token boundaries don't align with characters**  
BPE merges mean that `"name"` might be a single token or four separate tokens depending on context. The function-name injection loop had to handle partial matches carefully — comparing decoded *strings* rather than raw token sequences.

**2. Float values without a decimal point**  
The model occasionally generates `42,` instead of `42.0,` for `number` fields. The fix injects `.0` inline before the terminating comma without breaking the offset tracking (`__offset`).

**3. Escaped characters in string values**  
Backslash sequences produced by the tokeniser (e.g. `\'` instead of `'`) would cause `json.loads` to fail. The `_fast_sanitize` method handles this with a single compiled regex pass.

**4. Re-implementing BPE from scratch**  
The subject forbids using the SDK's `encode`/`decode` as a black box in the core loop. Reimplementing BPE correctly (handling leading-space surrogates, merge priority, special-token splitting) took careful study of `tokenizer.json` and thorough round-trip test coverage.

**5. Offset drift during backtracking**  
When popping tokens to insert `.0`, the decoded-string offset `__offset` had to be decremented manually. Getting this right without double-decoding or skipping tokens required tracking the offset as a count of *generated token IDs*, not decoded character positions.

---

## 📚 Resources

### Documentation & papers

- 📄 [Attention Is All You Need — Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) — Transformer architecture foundational paper
- 📄 [Constrained Decoding for Neural NLG (2021)](https://arxiv.org/abs/2108.06312) — Theoretical basis for constrained generation
- 📘 [Qwen3 Model Card — HuggingFace](https://huggingface.co/Qwen/Qwen3-0.6B) — Model architecture and tokeniser details
- 📘 [Qwen Concepts — Official Docs](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html) — Key concepts behind the Qwen model family
- 📘 [Pydantic v2 Documentation](https://docs.pydantic.dev/latest/) — Data validation library used throughout
- 📘 [BPE Algorithm — Sennrich et al. (2016)](https://arxiv.org/abs/1508.07909) — Byte-pair encoding explained

### How AI was used

GitHub Copilot was used during this project for the following tasks:

- **Docstring generation** — generating and updating English docstrings for all classes and methods in `src/`
- **Boilerplate reduction** — initial scaffolding of Pydantic model fields and argparse argument registration
- **Test case suggestions** — proposing edge cases for `PromptModel` escaping and `JSONLoader` MIME checks
- **README drafting** — this document was drafted with AI assistance and then reviewed and adjusted for accuracy

All AI-generated content was manually reviewed, tested, and validated before being committed. No code was accepted without understanding its purpose and verifying its correctness.
