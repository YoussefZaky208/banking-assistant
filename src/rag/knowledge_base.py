"""Load the bank's documents and cut them into chunks (one chunk = one section)."""
import re
from dataclasses import dataclass
from pathlib import Path

_SECTION = re.compile(r"^## (\d+)\. (.+)$", re.M)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str        # "cards#5@en"  (unique)
    section_id: str      # "cards#5"     (same section in both languages)
    doc_id: str
    doc_title: str
    section_title: str
    lang: str            # "en" or "ar"
    body: str

    @property
    def text(self) -> str:
        """What we search over: titles + body, so a chunk carries its own context."""
        return f"{self.doc_title} - {self.section_title}\n{self.body}"

    @property
    def citation(self) -> str:
        number = self.section_id.split("#")[1]
        return f"{self.doc_title}, §{number}"


def _front_matter(text: str) -> dict:
    block = text.split("---")[1]
    return dict(line.split(": ", 1) for line in block.strip().splitlines())


def load_chunks(kb_dir="knowledge_base") -> list[Chunk]:
    chunks = []
    for path in sorted(Path(kb_dir).glob("*/*.md")):
        text = path.read_text(encoding="utf-8")
        meta = _front_matter(text)
        content = text.split("---", 2)[2]
        heads = list(_SECTION.finditer(content))
        for i, head in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(content)
            body = content[head.end():end].strip()
            section_id = f"{meta['doc_id']}#{head.group(1)}"
            chunks.append(Chunk(
                chunk_id=f"{section_id}@{meta['lang']}", section_id=section_id,
                doc_id=meta["doc_id"], doc_title=meta["title"],
                section_title=head.group(2).strip(), lang=meta["lang"], body=body))
    return chunks


def detect_lang(text: str) -> str:
    """'ar' if most letters are Arabic, else 'en'."""
    arabic = len(re.findall(r"[؀-ۿ]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return "ar" if arabic > latin else "en"
