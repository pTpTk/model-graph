"""Keep generated result indexes in the repository's single README."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def update_index(name, markdown, directory):
    """Replace one generated section, resolving its links from the root README."""
    def link(match):
        target = match[1]
        if target.startswith(('https://', 'http://', '#', '/')):
            return match[0]
        return '](' + (directory / target).resolve().relative_to(ROOT).as_posix() + ')'
    markdown = re.sub(r'\]\(([^)]+)\)', link, markdown)
    markdown = re.sub(r'^(#+) ', r'#\1 ', markdown, flags=re.MULTILINE)
    start, end = f'<!-- {name}:start -->', f'<!-- {name}:end -->'
    section = f'{start}\n{markdown.rstrip()}\n{end}'
    path = ROOT / 'README.md'
    text = path.read_text()
    if start in text:
        text = re.sub(re.escape(start) + r'.*?' + re.escape(end), lambda _: section, text, flags=re.DOTALL)
    else:
        text = text.rstrip() + '\n\n' + section + '\n'
    path.write_text(text)
