"""Generate committed PDF fixtures (dev-only; reportlab never in runtime).

Regenerate: uv run python tests/fixtures/make_pdf_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen.canvas import Canvas  # type: ignore[import-untyped]

HERE = Path(__file__).parent


def _text_pdf(path: Path, pages: list[str]) -> None:
    canvas = Canvas(str(path))
    for page in pages:
        canvas.drawString(72, 720, page)
        canvas.showPage()
    canvas.save()


def main() -> None:
    _text_pdf(
        HERE / "sample.pdf",
        ["First page carries the opening sentence.", "Second page closes it."],
    )

    # Mixed: real text page + blank page (blank yields no text).
    _text_pdf(HERE / "sample_mixed.pdf", ["Only this page has words."])
    writer = PdfWriter()
    reader = PdfReader(str(HERE / "sample_mixed.pdf"))
    writer.add_page(reader.pages[0])
    writer.add_blank_page(width=200, height=200)
    with open(HERE / "sample_mixed.pdf", "wb") as f:
        writer.write(f)

    # Scanned: blank pages only, extract_text finds nothing.
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    with open(HERE / "sample_scanned.pdf", "wb") as f:
        writer.write(f)

    # Locked: user-password encrypted, no decrypt attempts at runtime.
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt("secret")
    with open(HERE / "sample_locked.pdf", "wb") as f:
        writer.write(f)


if __name__ == "__main__":
    main()
