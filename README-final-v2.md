# AI Support Ticket Triage System

A multi-agent AI system that automatically classifies incoming customer support tickets, retrieves similar past resolutions using RAG, drafts a suggested reply with an open-source LLM, and decides whether it can be auto-sent or needs human review.

## Architecture

```
Incoming Ticket
      ↓
Classifier Agent        → predicts department/queue (TF-IDF + Logistic Regression)
      ↓
Retriever Agent         → finds similar past resolutions, filtered by predicted queue (embeddings + ChromaDB)
      ↓
Resolution Agent        → drafts a grounded reply using retrieved context (LLM via OpenRouter)
      ↓
Escalation Agent        → decides: auto-send, or flag for human review (LLM via OpenRouter)
      ↓
Output: Suggested reply + escalation decision
```

All four agents are orchestrated as a **LangGraph** state graph, with the classifier's prediction feeding directly into the retriever's search filter — so retrieval is scoped to the correct department instead of searching the whole database.

## Dataset

[Multilingual Customer Support Tickets](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets) (Kaggle), filtered to English (~11,900 rows). Fields used: `body` (ticket text), `answer` (real agent resolution), `queue` (department — classification target), `priority` (escalation signal).

## Results

- Classifier: **~38-44% accuracy**, macro F1 improved from 0.31 → 0.35 after applying `class_weight='balanced'` to address class imbalance across 10 queue categories (ranging from 168 to 3,412 examples).
- Retrieval: verified qualitatively — semantically similar tickets (e.g., "charged twice" ↔ "duplicate charge") are correctly retrieved even without shared keywords.
- End-to-end pipeline tested on a held-out sample of tickets, producing grounded (non-hallucinated) draft replies and reasoned escalation decisions.

## Setup

Requires a `.env` file with:
```
OPENROUTER_API_KEY=your_key_here
```

```bash
make install    # install dependencies
make run        # execute the full notebook (data loading, training, RAG, agents, LangGraph)
make test       # run automated smoke tests against the notebook
```

## Project Structure

```
ai-support-triage/
├── data/                     # dataset (not committed)
├── notebooks/
│   └── 001_eda.ipynb         # full pipeline: EDA, classifier, RAG, agents, LangGraph
├── tests/
│   └── test_notebook.py      # executes the notebook end-to-end and checks outputs
├── requirements.txt
└── Makefile
```

---

## Key Concepts & Skills Demonstrated

This section is a deliberate part of the README — it's the explanation I can give in an interview for *why* each piece exists, not just what it's called.

### Classification: TF-IDF + Logistic Regression
The first stage predicts which department (`queue`) a ticket belongs to. **TF-IDF (Term Frequency–Inverse Document Frequency)** turns text into numbers by scoring each word based on how distinctive it is to a specific document, relative to how common it is overall — so a word like "refund" scores high in billing tickets, while filler words like "the" score near zero everywhere. **Logistic Regression** was chosen as the model itself because it's fast, interpretable, and a sensible baseline to try before reaching for anything heavier. Since the dataset's categories were imbalanced (168 to 3,412 examples per queue), I used `class_weight='balanced'`, which penalizes mistakes on rare categories more heavily — trading a small amount of raw accuracy for meaningfully fairer performance across all categories (macro F1 improved from 0.31 to 0.35).

### Vectorization
A general term for turning text (or any raw data) into numbers a model can actually process. This project uses **two different kinds** for two different purposes:
- **TF-IDF vectors** for classification — good at capturing *which distinct words* appear, ideal when categories have genuinely different vocabulary
- **Embeddings** (below) for retrieval — good at capturing *meaning*, needed when wording differs but the underlying concept is the same

### Embeddings & MiniLM (`all-MiniLM-L6-v2`)
An embedding model converts a sentence into a fixed-length list of numbers (384 numbers, for this model) that represents its *meaning* — two sentences with similar meaning end up with mathematically similar number-lists, even if they share almost no exact words (e.g., "charged twice" and "duplicate charge"). MiniLM specifically was chosen because it's small and fast enough to run on a normal laptop CPU with no GPU, while still being good enough quality for similarity search — a deliberate speed/quality tradeoff suited to a project like this.

### RAG (Retrieval-Augmented Generation)
Instead of asking an LLM to answer purely from its training memory (which risks generic or hallucinated answers), RAG first retrieves real, relevant reference material and feeds it to the LLM as grounding context before it generates a response. Here, that means: embed the incoming ticket, search for the most similar *already-resolved* tickets, and hand their real resolutions to the LLM so its drafted reply is based on genuine precedent, not invention.

### ChromaDB (Vector Database)
A local, zero-setup database purpose-built for storing embeddings and efficiently searching "which stored vectors are closest to this new one." Chosen over cloud options like Pinecone specifically because it needs no account, no server, and no cost — appropriate for a solo/portfolio project, while remaining conceptually swappable to a cloud vector DB for production scale.

### LLMs, OpenRouter, and API Aggregators
An **LLM (Large Language Model)** is a neural network trained on huge amounts of text to generate language — the same category of model as GPT-4 or Claude. Training one from scratch requires trillions of words and millions of dollars in compute, which is why real projects (including this one) use an existing pre-trained LLM rather than building one. **OpenRouter** is an API aggregator — a service that hosts many different open-source LLMs behind one unified, OpenAI-compatible API, so a single API key and a single request format can call many different models without separate integrations for each. This project calls an open-source model through OpenRouter's free tier, with a retry-with-backoff wrapper to gracefully handle the rate limits that come with shared free-tier access.

### Prompt Engineering (Structured Output)
Rather than letting the LLM respond in free-form prose, the escalation agent is explicitly instructed to respond in a fixed format (`Decision: [AUTO-SEND or ESCALATE]` / `Reason: ...`). This makes the model's output predictable and easy to parse in code, instead of needing to interpret an open-ended paragraph.

### Multi-Agent Orchestration with LangGraph
LangGraph structures a pipeline as a **graph**: each processing step (classifier, retriever, resolution agent, escalation agent) is a **node**, and **edges** define what runs next. A shared **state** (a `TypedDict` acting like a clipboard) is passed along and updated by each node — so the retriever can read the classifier's predicted queue, the resolution agent can read the retriever's results, and so on. This was chosen over a plain sequential script because a graph structure cleanly supports conditional branching and retry loops (e.g., "if the draft reply isn't confident, loop back to retrieval") in a way a straight-line script can't express without turning into tangled if/else logic.

## Known Limitations & Future Work

- Classifier accuracy is limited by real-world overlap in ticket vocabulary across queues (e.g., "software bug" appears across multiple departments) — a known characteristic of this dataset, not a modeling bug, confirmed by manually inspecting sample tickets per category before modeling.
- Currently uses a free-tier open-source model via OpenRouter; designed to swap cleanly to Hermes or a locally-hosted model via Ollama.
- A knowledge graph (e.g., NetworkX/Neo4j) capturing relationships *between* ticket types, and a Streamlit demo UI, are natural next extensions.
