# Challenges & Design Decisions

A behind-the-scenes look at the engineering decisions, dead ends, and fixes that shaped this project — kept separate from the README so the reasoning has room to breathe.

---

## Challenges Faced

### 1. The first dataset turned out to have no real signal
I started with Kaggle's "Customer Support Ticket Dataset," but a baseline classifier came back at ~21% accuracy on 5 balanced classes — essentially random guessing. Instead of assuming a better model would fix it, I read raw text samples across categories and found the actual problem: an unfilled template placeholder (`{product_purchased}`) and the exact same boilerplate sentence repeated verbatim across different labeled categories. The dataset's text simply didn't correlate with its own labels. I switched to the "Multilingual Customer Support Tickets" dataset, which had real, varied ticket text, and re-validated the same way (reading samples per category) before committing to it.

**Takeaway:** a low baseline score is a prompt to inspect the data itself before reaching for a fancier model.

### 2. Class imbalance was hiding poor performance
The real dataset's `queue` categories ranged from 168 to 3,412 examples — a ~20x spread. A first-pass Logistic Regression model hit 44% overall accuracy, which looked reasonable, until I checked precision/recall *per category* and found it was essentially never predicting the smallest classes (0.00 recall on one category). Applying `class_weight='balanced'` traded a few points of raw accuracy (44%→38%) for a meaningfully fairer model across all categories, measured properly with macro F1 (0.31→0.35).

**Takeaway:** overall accuracy can hide systematic failure on minority classes; per-class metrics are what actually reveal it.

### 3. Small, real-world data issues
A single row with a missing `body` field crashed the TF-IDF vectorizer. Checked the actual count first (1 out of 11,923) before deciding how to handle it — dropping one row was a proportionate fix, not something that needed a more elaborate imputation strategy.

### 4. Free-tier LLM access is a moving target
Getting a working, free LLM connection took several attempts:
- A specific Hermes model turned out to be paid, not free
- A free DeepSeek model was deprecated from the free tier between when I found it and when I used it
- A free Gemma model hit a shared rate limit from other users

Rather than keep guessing model names from documentation or search results, I queried OpenRouter's live model list directly via their API to see what was actually free *right now*, and picked a model against explicit criteria: general-purpose (not a narrow domain fine-tune), documented, and free of unusual usage restrictions. I also added a retry-with-backoff wrapper around every LLM call, so a transient rate limit causes a short pause and a retry instead of crashing the whole pipeline.

**Takeaway:** free-tier availability for hosted models changes without notice; designing for graceful retries mattered more than picking the "perfect" model up front.

### 5. Database batch limits
Inserting all ~11,900 ticket embeddings into ChromaDB in a single call hit a batch-size limit. Fixed by chunking the insert into batches of 5,000 using simple list slicing — a pattern that generalizes to any API or database with a request-size cap.

### 6. The classifier's output wasn't actually connected to anything
Early on, I had a working classifier and a working retrieval step, but they weren't linked — retrieval searched the entire knowledge base regardless of what the classifier predicted. This was only truly fixed when wiring the LangGraph pipeline: the retriever node explicitly reads the classifier's predicted queue from the shared state and uses it to filter the vector search, so retrieval is scoped to the correct department instead of searching everything.

**Takeaway:** having two working components doesn't mean they're actually integrated — the connection between them needs to be deliberately built and tested.

---

## Design Decisions & Why

| Decision | Chosen | Alternative considered | Reasoning |
|---|---|---|---|
| Classification target | `queue` (department) | `type` (IT-ops category) | Matches the "route the ticket" framing of the project more directly |
| Classifier | Logistic Regression + TF-IDF | A fine-tuned transformer | Fast, interpretable baseline; proven sufficient before adding complexity |
| Class imbalance handling | `class_weight='balanced'` | Manually merging small categories | A general-purpose fix that doesn't need re-tuning as data changes |
| Embedding model | `all-MiniLM-L6-v2` | `all-mpnet-base-v2` | Runs comfortably on CPU with no GPU; good enough quality for this use case |
| Vector store | ChromaDB (local, persistent) | Pinecone / Weaviate | No account or server needed; appropriate scale for a single-developer project |
| LLM access | OpenRouter (hosted API) | Local model via Ollama | No hardware dependency; let me focus on agent logic rather than infrastructure |
| LLM model | Open-source model selected via live availability check | A fixed model chosen upfront from documentation | Free-tier model availability changes frequently; checking live availability avoided repeated dead ends |
| Orchestration | LangGraph | Sequential function calls | Supports conditional branching and retry paths cleanly, which a linear script would need tangled conditionals to express |

---

## What Each Core Piece Actually Does

**TF-IDF** scores words by how distinctive they are to one document versus the whole collection — used for classification, where categories have genuinely different vocabulary.

**Embeddings** (via MiniLM) convert text into numbers representing *meaning*, so differently-worded but semantically similar tickets can still be matched — used for retrieval.

**RAG (Retrieval-Augmented Generation)** grounds the LLM's drafted replies in real, retrieved past resolutions instead of letting it generate purely from memory, reducing the risk of confident-sounding but fabricated answers.

**ChromaDB** stores those embeddings and efficiently finds the closest matches to a new query.

**OpenRouter** is an API aggregator — it hosts many different LLMs behind one unified, OpenAI-compatible interface, so a single integration can call many different underlying models.

**LangGraph** structures the whole pipeline as a graph: each agent is a node, edges define what runs next, and a shared state object carries information (ticket text, predicted queue, retrieved context, draft reply, escalation decision) between them — enabling the classifier's output to directly inform the retriever's search, and leaving room for conditional logic like retries.

---

## Concepts I Had to Actually Understand Along the Way

### Precision vs. recall (not just "accuracy")
Overall accuracy hid a real problem until I looked at precision and recall per category. Two categories made the distinction click:
- **Technical Support** (the largest category, 683 examples in one test batch): high recall (0.73) but low precision (0.42). The model was *eager* to guess this label — it caught most of the real Technical Support tickets, but also wrongly slapped that label on plenty of tickets that weren't actually Technical Support.
- **Human Resources** (a small category, 41 examples): precision 1.00, recall 0.12 — almost the opposite behavior. When the model *did* predict Human Resources, it was always right, but it was too cautious to say so most of the time, defaulting to a bigger category instead.

In short: **precision** asks "when the model says X, how often is it actually X?" and **recall** asks "of all the real X's, how many did the model actually catch?" A model can look fine on one and be quietly failing on the other — which is exactly what overall accuracy was hiding.

### `stratify=y` in the train/test split
With imbalanced categories (168 to 3,412 examples), a plain random split could, by chance, put almost all of a rare category's examples into one side of the split — leaving the model with too few (or zero) examples of that category to learn from, or to be fairly tested on. `stratify=y` forces both the train and test sets to preserve the same category proportions as the full dataset, removing that risk entirely.

### Macro average vs. weighted average
These two numbers, computed from the exact same per-category results, can tell very different stories:
- **Weighted average** scales each category's score by how many examples it has — so large categories dominate the number.
- **Macro average** treats every category equally, regardless of size.

The gap between them (weighted F1 0.41 vs. macro F1 0.31 in an early test) was the actual signal that the model was performing well on big categories and poorly on small ones — a gap that a single "accuracy" number would never reveal.

### Classification vs. clustering (a terminology mix-up worth clarifying)
These sound similar but are different problems. **Classification** predicts a *known* label — we tell the model in advance what the categories are (Billing, Technical Support, etc.) and it learns to assign one. **Clustering** finds groups in data *without* being given labels upfront — useful when you don't know the categories in advance. Everything in this project (the queue classifier) is classification; no clustering was used.

### Why the classifier was trained from scratch, but the embedding model and LLM weren't
This came down to the *scope* of what each model needed to learn:
- The classifier only needed to learn one narrow, well-defined pattern — "which words associate with which of these 10 specific categories" — learnable from a few thousand labeled examples in seconds on a laptop.
- The embedding model (MiniLM) and the LLM both needed to learn something far broader first — general sentence meaning, or general language itself — which requires enormous datasets and compute that aren't practical to reproduce for a single project. Both were used pre-trained, exactly as intended: the valuable skill is knowing how to use an existing foundation model effectively (via embeddings for retrieval, via prompting for generation), not retraining one from zero.

---

## Honest Limitations

- Classifier accuracy (~38-44%) reflects genuine vocabulary overlap across departments in the real-world data (confirmed by manually reading samples across categories), not an unoptimized model.
- The project currently uses a free-tier open-source LLM rather than a specific named model, chosen for live availability and reliability rather than brand.
- Retrieval uses metadata filtering by department as a lightweight substitute for a full knowledge graph; a graph capturing relationships *between* issue types (e.g., via NetworkX) is a natural next extension.
- No dedicated UI yet — the pipeline runs from a notebook/CLI rather than a demo-ready interface.
