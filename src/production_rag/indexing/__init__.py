"""Indexing pillar: file upload -> Document.

Planned stages (issue #7, not yet implemented):
- identifier: detect format, enforce empty / size guards
- normalizers: one per accepted format (txt, md, html, pdf)
- md_formatter: single Unstructured pass to the shared markdown contract
"""
