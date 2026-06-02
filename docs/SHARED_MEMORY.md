# Sharing Repo Memory Across the Team

There are two different kinds of memory in this project.

## 1. Repo Memory for AI Coding Assistants

`AGENTS.md` is the shared, committed memory file for this repo. Team members who
open the repository with an AI coding assistant can get the same project context:
architecture, conventions, commands, and expected chatbot behavior.

When the team learns something durable about the project, update `AGENTS.md` and
commit it.

Good examples:
- The source of truth for configuration.
- The intended retrieval flow.
- How HITL decisions should work.
- Commands that should be used for setup, ingestion, running, and evaluation.
- Team coding conventions.

Avoid adding:
- Personal notes.
- Secrets.
- Temporary debugging state.
- Large copied datasets or generated database files.

## 2. Runtime Memory for the Chatbot

The chatbot's domain context comes from the ChromaDB collection created during
ingestion.

By default, each developer has a local copy:

```env
CHROMA_PERSIST_DIR=./data/chroma_db
```

To make every teammate get the same chatbot context, use one of these patterns.

### Option A: Same Dataset, Local Ingestion

This is simplest for a capstone/demo project.

1. Keep the same `data/fraud_oracle.csv` version across the team.
2. Each teammate runs:

```bash
python ingestion/ingest.py
```

3. The generated ChromaDB files stay local and uncommitted.

### Option B: Shared Chroma Storage

Use this if the team needs one live shared knowledge base.

1. Put ChromaDB persistence on a shared mounted volume, or run a shared Chroma
   service if the deployment supports it.
2. Set each teammate's `.env` to the same shared path or service settings.
3. Restrict who can reset/reingest the collection.

### Option C: Seeded Export

Use this when you want reproducible demos without every person embedding from
scratch.

1. One maintainer ingests the approved dataset.
2. Package the generated vector store as a release artifact, not a normal git
   commit.
3. Teammates download/extract it into `data/chroma_db`.

## Recommended Team Practice

- Commit `AGENTS.md` and this guide.
- Commit `.env.example`, not `.env`.
- Do not commit `data/chroma_db`.
- Document dataset changes in PRs.
- Re-run evaluation after meaningful prompt, retrieval, dataset, or threshold
  changes.

