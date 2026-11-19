import re
from dataclasses import dataclass, field
from functools import lru_cache

from transformers import AutoTokenizer

from config.chunker import get_settings

CHUNKER_VERSION = "2"

HEADING_SEPARATOR = " › "

FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*(?:\{\s*#([\w-]+)\s*\})?\s*$")

PROSE_SEPARATORS = [(r"\n", "\n"), (r"(?<=[.!?])\s+", " "), (r" +", " ")]
CODE_SEPARATORS = [(r"\n\s*\n", "\n\n"), (r"\n", "\n"), (r" +", " ")]


@dataclass
class Section:
    heading_path: list[str]
    anchor: str | None
    lines: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    source_path: str
    heading_path: list[str]
    anchor: str | None
    chunk_index: int
    text: str
    token_count: int

    @property
    def heading(self) -> str:
        return HEADING_SEPARATOR.join(self.heading_path)

    @property
    def embed_text(self) -> str:
        return build_embed_text(self.heading_path, self.text)


@lru_cache
def get_tokenizer():
    return AutoTokenizer.from_pretrained(get_settings().EMBEDDING_MODEL)


def count_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False, verbose=False))


def build_embed_text(heading_path: list[str], body: str) -> str:
    header = HEADING_SEPARATOR.join(heading_path)
    return f"{header}\n{body}" if header else body


def split_sections(lines: list[str]) -> list[Section]:
    sections = [Section(heading_path=[], anchor=None)]
    open_headings: list[tuple[int, str]] = []
    in_fence = False

    for line in lines:
        if FENCE_RE.match(line):
            in_fence = not in_fence
        heading = None if in_fence else HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            while open_headings and open_headings[-1][0] >= level:
                open_headings.pop()
            open_headings.append((level, heading.group(2)))
            sections.append(Section([text for _, text in open_headings], heading.group(3)))
        else:
            sections[-1].lines.append(line)

    return [s for s in sections if any(line.strip() for line in s.lines)]


def split_blocks(lines: list[str]) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False

    def flush():
        if current:
            blocks.append("\n".join(current))
            current.clear()

    for line in lines:
        if FENCE_RE.match(line):
            if not in_fence:
                flush()
            current.append(line)
            if in_fence:
                flush()
            in_fence = not in_fence
        elif in_fence or line.strip():
            current.append(line)
        else:
            flush()
    flush()
    return blocks


def split_to_fit(text: str, budget: int, separators: list[tuple[str, str]]) -> list[str]:
    if count_tokens(text) <= budget:
        return [text]
    if not separators:
        return [text[i:i + budget] for i in range(0, len(text), budget)]

    pattern, joiner = separators[0]
    parts = [part for part in re.split(pattern, text) if part.strip()]
    if len(parts) == 1:
        return split_to_fit(text, budget, separators[1:])

    pieces: list[str] = []
    current = ""
    for part in parts:
        for piece in split_to_fit(part, budget, separators[1:]):
            candidate = f"{current}{joiner}{piece}" if current else piece
            if count_tokens(candidate) <= budget:
                current = candidate
            else:
                if current:
                    pieces.append(current)
                current = piece
    if current:
        pieces.append(current)
    return pieces


def split_code_block(block: str, budget: int) -> list[str]:
    lines = block.split("\n")
    opening = lines[0]
    has_closing = len(lines) > 1 and FENCE_RE.match(lines[-1])
    closing = lines[-1] if has_closing else "```"
    inner = "\n".join(lines[1:-1] if has_closing else lines[1:])
    inner_budget = max(budget - count_tokens(f"{opening}\n{closing}"), 1)
    return [f"{opening}\n{piece}\n{closing}" for piece in split_to_fit(inner, inner_budget, CODE_SEPARATORS)]


def chunk_section(section: Section, max_tokens: int) -> list[str]:
    header = HEADING_SEPARATOR.join(section.heading_path)
    budget = max(max_tokens - count_tokens(header), max_tokens // 2)

    pieces: list[str] = []
    for block in split_blocks(section.lines):
        if count_tokens(block) <= budget:
            pieces.append(block)
        elif FENCE_RE.match(block):
            pieces.extend(split_code_block(block, budget))
        else:
            pieces.extend(split_to_fit(block, budget, PROSE_SEPARATORS))

    bodies: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for piece in pieces:
        tokens = count_tokens(piece)
        if current and current_tokens + tokens > budget:
            bodies.append("\n\n".join(current))
            current, current_tokens = [], 0
        current.append(piece)
        current_tokens += tokens
    if current:
        bodies.append("\n\n".join(current))
    return bodies


def chunk_document(source_path: str, lines: list[str], max_tokens: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in split_sections(lines):
        for body in chunk_section(section, max_tokens):
            chunks.append(Chunk(
                source_path=source_path,
                heading_path=section.heading_path,
                anchor=section.anchor,
                chunk_index=len(chunks),
                text=body,
                token_count=count_tokens(build_embed_text(section.heading_path, body)),
            ))
    return chunks
