# Plan 0002: HTML Extractor (indexing, html path)

Status: Accepted. Source of truth for issue #3. Builds on 0001's
dispatcher + `Extractor` Protocol; no dispatcher redesign.

## 1. Objective

Second format normalizer: `.html` upload → cleaned `Document` under the
same §3 contract as 0001 (visible text only, no markup/scripts/styles,
malformed input repaired never crashed).

## 2. Scope

In: `HtmlExtractor`, `text/html` routing, bs4+html5lib repair, Unstructured
pass to the shared markdown shape, provenance metadata, tests.
Out: table fidelity (best-effort flatten now, tracked in #8), image/OCR,
markdownify Alternatives (Unstructured locked), md/pdf normalizers.

## 3. Contract

```python
class HtmlExtractor:
    kinds: ClassVar[tuple[str, ...]] = ("html",)
    def extract(self, path, raw, mime, encoding=None) -> Document: ...
```

Same provenance subset as 0001 §3.1 (`detected_mime=text/html`).
Same errors; same `stage("html_normalize", component="indexing")`
logging. Immutability boundary (§3.3) unchanged.

## 4. Pipeline steps

1. **Route**: dispatcher `identify()` maps `text/html` → `html` kind
   (today all `text/*` → `text`; this is the one dispatcher change:
   exact-match html before the `text/` prefix rule). Registry gains
   `"html": HtmlExtractor()`. `.htm`/`.html` extension hints added.
2. **Repair + strip**: bs4 with `html5lib` parser (repairs broken trees
   on the fly — no regex stripping, per user decision). Drop
   `script`/`style`/`noscript`; keep visible text order.
3. **Markdown shape**: lazy `unstructured` import INSIDE the extractor
   (`partition_html`), so nltk/torch cost never touches dispatcher or
   txt paths. Elements → shared markdown string downstream systems see.
4. **Metadata + Document**: identical to txt (sha256, size, mime,
   encoding used).

## 5. Dependencies (`uv add`)

- `beautifulsoup4`, `html5lib` (light, import at module level).
- `unstructured` (heavy transitive: nltk/torch — lazy import only,
  install-time cost accepted per user omni-channel call).

## 6. Files

- `src/production_rag/indexing/html.py` — `HtmlExtractor`.
- `src/production_rag/indexing/dispatcher.py` — html kind mapping +
  registry entry + extension hints (small, registry-only diff).
- `tests/fixtures/` (new) — `sample_page.html` (well-formed),
  `sample_malformed.html` (nested/broken tags).
- `tests/test_html_extractor.py` — cases below.

## 7. Tests (stdlib unittest)

- Well-formed page → Document, headings/paragraphs preserved in order,
  zero angle brackets in content.
- Malformed nesting → repairs, no crash, no text loss (assert key
  sentences from both sides of the break present).
- `script`/`style` bodies absent from content.
- Table → best-effort text present (fidelity explicitly NOT asserted;
  #8 owns it).
- `.html` extension with plain-text body still ingests (sniff-tolerant).
- Registry conformance extends to `HtmlExtractor`.
- Metadata `detected_mime` is `text/html`.

## 8. Acceptance (issue #3)

- [ ] Reads an `.html` file and returns a cleaned `Document`.
- [ ] Markup/scripts/styles removed; visible text preserved.
- [ ] Malformed HTML does not crash the pipeline.
