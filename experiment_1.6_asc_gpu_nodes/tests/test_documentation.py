from docx import Document
from markdown_it import MarkdownIt

from scripts.render_plan import render


def test_word_document_preserves_headings_lists_code_and_tables(tmp_path):
    source = tmp_path / "plan.md"
    source.write_text("# Plan\n\nA **ranker**.\n\n| Task | Size |\n| --- | --- |\n| L | `64` |\n\n"
                      "- Training\n- Evaluation\n\n```python\nprint(1)\n```\n")
    target = tmp_path / "plan.docx"
    render(source, target)
    document = Document(target)
    assert document.paragraphs[0].text == "Plan"
    assert any(paragraph.text == "A ranker." for paragraph in document.paragraphs)
    assert any("print(1)" in paragraph.text for paragraph in document.paragraphs)
    assert [cell.text for cell in document.tables[0].rows[1].cells] == ["L", "64"]
    assert len(document.tables) == sum(token.type == "table_open" for token in
                                       MarkdownIt("commonmark").enable("table").parse(source.read_text()))