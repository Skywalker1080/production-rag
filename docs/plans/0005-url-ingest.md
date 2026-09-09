# Plan 0005: URL Ingest (indexing, fetch-then-route)

Status: Accepted. Source of truth for issue #10. No new extractor: a
URL is just another byte source that rejoins identify → route →
extract, so fetched HTML resumes the html path and fetched PDFs the
pdf path. Content sniff decides, not Content-Type.

## 1. Objective

`ingest_url(url) -> Document`: fetch, cap, sniff, route, normalize —
with network failures typed and loud.

## 2. Scope

In: scheme gate, timeout, redirect bound, streaming size cap,
FetchError taxonomy, sniff-based routing incl. extension hint from the
URL path, source-URL provenance, CLI support, mocked tests.
Out: auth/proxy support, JS rendering (static markup only — no browser
engine in Phase-1), SSRF egress policy beyond scheme gating (recorded
risk, not enforced).

## 3. Contract

```python
ingest_url(url: str, encoding: str | None = None) -> Document
```

- Raises: `FetchError` (bad scheme, DNS/timeout/HTTP error, oversize),
  plus the usual `EmptyFileError`/`FileTooLargeError`/
  `UnsupportedFormatError` from the shared path.
- Provenance: `file_name` = URL path basename (or host if bare),
  `file_path` = full URL, `detected_mime` = sniffed (not the header).

## 4. Pipeline steps (dispatcher)

1. **Gate**: `urlparse` scheme must be http/https, else `FetchError`
   (no `file://`, no bare paths — those go to `ingest()`).
2. **Fetch**: `requests.get(stream=True, timeout=10)`,
   `raise_for_status()`. Read at most `MAX_BYTES + 1` bytes off the
   stream; crossing the cap → `FileTooLargeError`. Empty body →
   `EmptyFileError`. All inside `stage("fetch", component="indexing")`.
3. **Rejoin**: `identify(raw, url_path_basename)` → guards → `route()`
   → extractor. Identical downstream to file ingest.
4. **SSRF note (accepted risk)**: scheme gating only. Private-range
   egress blocking is future hardening, not this plan.

## 5. Dependencies

- None (`requests` already direct). Timeout/cap constants in
  dispatcher (` FETCH_TIMEOUT_S = 10`).

## 6. Files

- `src/production_rag/indexing/errors.py` — `FetchError(IndexingError)`
  carrying url + reason.
- `src/production_rag/indexing/dispatcher.py` — `ingest_url()`,
  shared `_finalize()` for the guard-route-extract tail used by both
  `ingest()` and `ingest_url()` (no duplicated shape).
- `ingest.py` — URL-looking first arg routes to `ingest_url()`.
- `tests/test_url_ingest.py` — mocked `requests.get` cases below.

## 7. Tests (stdlib unittest, network mocked at the seam)

- HTML bytes → cleaned Document, no tags (rejoins html path).
- PDF bytes (sample.pdf) → ordered text (rejoins pdf path).
- HTTP 404 → `FetchError`.
- Timeout/DNS error → `FetchError` chained.
- `file:///etc/passwd` → `FetchError`, no request attempted.
- Stream past cap → `FileTooLargeError`.
- Registry/conformance untouched (no new extractor).

## 8. Acceptance (issue #10)

- [ ] http/https only, redirects bounded, timeout enforced.
- [ ] Oversized bodies rejected before buffering past the cap.
- [ ] Network failures fail loudly as FetchError.
- [ ] Fetched bytes route by sniff.
