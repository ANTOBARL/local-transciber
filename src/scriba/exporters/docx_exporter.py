"""Microsoft Word (.docx) export written with the standard library (WordprocessingML in a ZIP)."""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from scriba.exporters.render import interruption_note, speaker_blocks
from scriba.models import Transcript
from scriba.utils.time import format_clock

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
</Relationships>"""

DOCUMENT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

STYLES = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W_NS}">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>
<w:sz w:val="22"/><w:lang w:val="it-IT"/></w:rPr></w:rPrDefault>
<w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="276" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/>
<w:pPr><w:spacing w:after="240"/></w:pPr><w:rPr><w:b/><w:sz w:val="40"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Meta"><w:name w:val="Meta"/><w:basedOn w:val="Normal"/>
<w:pPr><w:spacing w:after="40"/></w:pPr><w:rPr><w:color w:val="595959"/><w:sz w:val="20"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Note"><w:name w:val="Note"/><w:basedOn w:val="Normal"/>
<w:pPr><w:shd w:val="clear" w:color="auto" w:fill="FDECEA"/><w:spacing w:before="200" w:after="200"/></w:pPr>
<w:rPr><w:b/><w:color w:val="B42318"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="TurnHeader"><w:name w:val="Turn header"/><w:basedOn w:val="Normal"/>
<w:pPr><w:keepNext/><w:spacing w:before="200" w:after="40"/></w:pPr><w:rPr><w:sz w:val="20"/></w:rPr></w:style>
</w:styles>"""


def _run(text: str, bold: bool = False, color: str | None = None) -> str:
    props = ("<w:b/>" if bold else "") + (f'<w:color w:val="{color}"/>' if color else "")
    rpr = f"<w:rPr>{props}</w:rPr>" if props else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def _paragraph(runs: str, style: str | None = None) -> str:
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{ppr}{runs}</w:p>"


def render_document_xml(transcript: Transcript) -> str:
    body: list[str] = [_paragraph(_run(Path(transcript.source_file).name), "Title")]
    meta = [
        ("Lingua / Language", transcript.language),
        ("Durata / Duration", format_clock(transcript.duration)),
        ("Modello / Model", transcript.model),
        ("Creato / Created", f"{transcript.created_at:%Y-%m-%d %H:%M} UTC"),
    ]
    if transcript.speakers:
        meta.append(("Speaker", ", ".join(transcript.speaker_label(s) or s for s in transcript.speakers)))
    body += [_paragraph(_run(f"{k}: ", bold=True) + _run(v), "Meta") for k, v in meta]

    note = interruption_note(transcript)
    if note:
        body.append(_paragraph(_run(note), "Note"))

    if not transcript.has_timestamps and not transcript.speakers:
        body += [_paragraph(_run(line)) for line in transcript.text.splitlines() if line.strip()] or [_paragraph("")]
    else:
        for block in speaker_blocks(transcript):
            header = _run(format_clock(block.start), color="6366F1")
            if block.speaker:
                header += _run("  " + block.speaker, bold=True)
            body.append(_paragraph(header, "TurnHeader"))
            body.append(_paragraph(_run(block.text)))

    if note:
        body.append(_paragraph(_run(note), "Note"))

    section = ('<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
               '<w:pgMar w:top="1418" w:right="1418" w:bottom="1418" w:left="1418" w:header="709" w:footer="709"'
               ' w:gutter="0"/></w:sectPr>')
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<w:document xmlns:w="{W_NS}"><w:body>{"".join(body)}{section}</w:body></w:document>')


def _core_properties(transcript: Transcript) -> str:
    created = transcript.created_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        f"<dc:title>{escape(Path(transcript.source_file).name)}</dc:title><dc:creator>Scriba</dc:creator>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>'
        "</cp:coreProperties>"
    )


def export_docx(transcript: Transcript, path: Path) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES)
        zf.writestr("_rels/.rels", ROOT_RELS)
        zf.writestr("word/_rels/document.xml.rels", DOCUMENT_RELS)
        zf.writestr("word/styles.xml", STYLES)
        zf.writestr("word/document.xml", render_document_xml(transcript))
        zf.writestr("docProps/core.xml", _core_properties(transcript))
    return path
