"""Built-in documentation dialog with Markdown + KaTeX rendering."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

_DOCS_DIR = Path(__file__).parent / 'docs'

_DOC_FILES = [
    '01_getting_started.md',
    '02_map_types.md',
    '03_coordinates.md',
    '04_radio_horizon.md',
    '05_link_profile.md',
    '06_advanced.md',
]

_H2 = 2
_H3 = 3


def _render_markdown(md_text: str) -> str:
    """Convert Markdown text to HTML with dollarmath support."""
    from markdown_it import MarkdownIt
    from mdit_py_plugins.dollarmath import dollarmath_plugin

    md = MarkdownIt('commonmark', {'html': True, 'typographer': True})
    md.enable('table')
    md.enable('strikethrough')
    dollarmath_plugin(md, double_inline=True)
    return md.render(md_text)


def _inject_heading_ids(html: str) -> str:
    """Add id attributes to h1-h4 tags for TOC anchor navigation."""

    def _slugify(text: str) -> str:
        slug = text.lower().strip()
        slug = re.sub(r'[^\w\s-]', '', slug)
        return re.sub(r'[\s]+', '-', slug)

    counter: dict[str, int] = {}

    def _replacer(m: re.Match) -> str:
        tag = m.group(1)
        attrs = m.group(2) or ''
        content = m.group(3)
        slug = _slugify(content)
        count = counter.get(slug, 0)
        counter[slug] = count + 1
        if count > 0:
            slug = f'{slug}-{count}'
        return f'<{tag}{attrs} id="{slug}">{content}</{tag}>'

    return re.sub(
        r'<(h[1-4])([^>]*)>(.*?)</\1>',
        _replacer,
        html,
        flags=re.DOTALL,
    )


def _extract_headings(html: str) -> list[tuple[int, str, str]]:
    """Extract (level, id, text) tuples from heading tags."""
    results = []
    for m in re.finditer(r'<h([1-4])[^>]*id="([^"]*)"[^>]*>(.*?)</h\1>', html):
        level = int(m.group(1))
        hid = m.group(2)
        text = re.sub(r'<[^>]+>', '', m.group(3)).strip()
        results.append((level, hid, text))
    return results


def _build_full_html(body_html: str) -> str:
    """Wrap body HTML with full document including KaTeX and CSS."""
    docs_url = _DOCS_DIR.as_uri()
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="{docs_url}/katex/katex.min.css">
<link rel="stylesheet" href="{docs_url}/style.css">
</head>
<body>
<div class="content">
{body_html}
</div>
<script src="{docs_url}/katex/katex.min.js"></script>
<script src="{docs_url}/katex/auto-render.min.js"></script>
<script>
document.addEventListener("DOMContentLoaded", function() {{
    renderMathInElement(document.body, {{
        delimiters: [
            {{left: "$$", right: "$$", display: true}},
            {{left: "$", right: "$", display: false}}
        ],
        throwOnError: false
    }});
}});
</script>
</body>
</html>"""


class HelpDialog(QDialog):
    """Non-modal documentation browser with TOC and search."""

    _TITLE = 'SK24mapper — Документация'
    _SEARCH_PLACEHOLDER = 'Поиск по документации… (Ctrl+F)'

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self._TITLE)
        self.setMinimumSize(950, 620)
        self.showMaximized()
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
        )

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._toc_tree = QTreeWidget()
        self._toc_tree.setHeaderHidden(True)
        self._toc_tree.setMinimumWidth(220)
        self._toc_tree.setMaximumWidth(320)
        self._toc_tree.setStyleSheet(
            'QTreeWidget { padding-top: 40px; }QTreeView::item { padding: 4px 0; }'
        )
        self._toc_tree.itemClicked.connect(self._on_toc_clicked)
        root.addWidget(self._toc_tree)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        search_bar = QWidget()
        search_bar_layout = QHBoxLayout(search_bar)
        search_bar_layout.setContentsMargins(8, 6, 8, 6)
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText(self._SEARCH_PLACEHOLDER)
        self._search_input.setClearButtonEnabled(True)
        self._search_input.textChanged.connect(self._on_search_changed)
        search_bar_layout.addWidget(self._search_input)
        search_bar.setFixedHeight(44)
        right_layout.addWidget(search_bar)

        self._web = QWebEngineView()
        right_layout.addWidget(self._web)

        root.addWidget(right_panel, 1)

        find_shortcut = QShortcut(QKeySequence('Ctrl+F'), self)
        find_shortcut.activated.connect(self._focus_search)
        esc_shortcut = QShortcut(QKeySequence('Escape'), self)
        esc_shortcut.activated.connect(self._on_escape)

        self._section_anchors: list[str] = []
        self._load_docs()

    def _load_docs(self) -> None:
        """Load all markdown files, render HTML, and populate TOC."""
        combined_html_parts: list[str] = []

        for fname in _DOC_FILES:
            fpath = _DOCS_DIR / fname
            if not fpath.exists():
                logger.warning('Doc file missing: %s', fpath)
                continue
            md_text = fpath.read_text(encoding='utf-8')
            section_html = _render_markdown(md_text)
            section_html = _inject_heading_ids(section_html)
            combined_html_parts.append(section_html)

        full_body = '\n<hr>\n'.join(combined_html_parts)
        full_html = _build_full_html(full_body)

        base_url = QUrl.fromLocalFile(str(_DOCS_DIR) + '/')
        self._web.setHtml(full_html, base_url)

        self._build_toc(full_body)

    def _build_toc(self, html: str) -> None:
        """Parse headings and build QTreeWidget."""
        self._toc_tree.clear()
        self._section_anchors.clear()
        headings = _extract_headings(html)

        stack: list[QTreeWidgetItem] = []
        for level, hid, text in headings:
            item = QTreeWidgetItem([text])
            item.setData(0, Qt.ItemDataRole.UserRole, hid)
            self._section_anchors.append(hid)

            if level == 1:
                self._toc_tree.addTopLevelItem(item)
                stack = [item]
            elif level == _H2:
                while len(stack) > 1:
                    stack.pop()
                if stack:
                    stack[-1].addChild(item)
                else:
                    self._toc_tree.addTopLevelItem(item)
                stack.append(item)
            elif level == _H3:
                while len(stack) > _H2:
                    stack.pop()
                if stack:
                    stack[-1].addChild(item)
                else:
                    self._toc_tree.addTopLevelItem(item)
                stack.append(item)
            elif stack:
                stack[-1].addChild(item)
            else:
                self._toc_tree.addTopLevelItem(item)

        for i in range(self._toc_tree.topLevelItemCount()):
            top_item = self._toc_tree.topLevelItem(i)
            if top_item is not None:
                top_item.setExpanded(True)

    @Slot(QTreeWidgetItem, int)
    def _on_toc_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        """Scroll to the heading anchor when TOC item is clicked."""
        anchor = item.data(0, Qt.ItemDataRole.UserRole)
        if anchor:
            js = (
                f'document.getElementById("{anchor}")'
                '?.scrollIntoView({behavior:"smooth",block:"start"});'
            )
            self._web.page().runJavaScript(js)

    @Slot(str)
    def _on_search_changed(self, text: str) -> None:
        """Search text in the web view."""
        if text:
            self._web.findText(text)
        else:
            self._web.findText('')

    @Slot()
    def _focus_search(self) -> None:
        """Focus the search input."""
        self._search_input.setFocus()
        self._search_input.selectAll()

    @Slot()
    def _on_escape(self) -> None:
        """Handle Escape: clear search or close dialog."""
        if self._search_input.text():
            self._search_input.clear()
        else:
            self.close()
