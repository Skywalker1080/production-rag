# Production RAG

End-to-end production-grade RAG system, built simply and hardened iteratively sprint by sprint.

## Architecture Pillars (agreed)

1. **Indexing**: ingestion and preprocessing of data toward chunks and embeddings.
2. **Storage**: choice of storage engine, its trade-offs, and storage structure.
3. **Retrieval Engine**: core retrieval of relevant information under agreed SLAs for the LLM.
4. **Synthesis**: combining retrieved information into context for the LLM; retries and all LLM work live here.
5. **Evaluation**: continuous assessment of system quality, including system-level evals and LLM tracing.

## Phase 1 Scope (Indexing only)

- Ingest from txt, md, html, pdf files plus http(s) URLs (directories
  explicitly out of scope).
- Normalize each input, wrap it in a Document, format to one markdown
  shape, chunk for retrieval.
- Document Class is user-designed (see issue #1).

## Language

**Indexing**:
The pillar that turns raw inputs into normalized Documents.
_Avoid_: ingestion engine, preprocessing pipeline

**Storage**:
The pillar that decides where and how processed data lives.
_Avoid_: database, vector store (no engine chosen)

**Retrieval Engine**:
The pillar that fetches relevant information under SLAs.
_Avoid_: retriever, search

**Synthesis**:
The pillar that builds LLM context from retrieved information, including retries.
_Avoid_: generation, prompting

**Evaluation**:
The pillar that measures system quality via evals and tracing.
_Avoid_: testing, monitoring

**Document**:
The normalized wrapper every ingestor returns.
_Avoid_: chunk, embedding, record (no shape decided)

**Normalization**:
The cleaning applied so different formats share one Document contract.
_Avoid_: parsing, chunking, embedding

**Heading path**:
The breadcrumb of headings above a chunk (e.g. Leave Policy > Casual Leave).
_Avoid_: section trail, outline

**Table bundle**:
A table kept atomic together with its explaining paragraph above and footnotes below.
_Avoid_: table chunk

## Agreed decisions

- **Metadata groups**: provenance (where/when) + structural/positional + categorization/semantic are in; access control/governance is out of scope. Exact per-stage metadata fields pending.
- **Document home**: Document lives in indexing until vector-DB ingest.
- **Execution model**: pre-retrieval work is programmatic flow; LangGraph orchestrates retrieval → synthesis → eval only.
- **Domain**: company policy handbook (single-tenant; RBAC stays out). Chunking models policy shape: heading paths, atomic tables bundled with above/below context, version/date filtering, repeated header/footer suppression.

## Undecided (explicitly not assumed)

- Chunking strategy, embedding model, storage engine, retrieval SLAs, LLM choice, eval metrics, exact per-stage metadata fields.
- Anything not listed above has no decision yet.
