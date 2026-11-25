// Markdown report (the subset used in ml/reports/*/README.md) → thesis-style .docx.
// Usage: npm install docx && node tools/md2docx.js ml/reports/foodphoto/README.md ml/reports/foodphoto/report.docx
// Layout: A4, Times New Roman 14 pt, 1.5 line spacing, 1.25 cm first-line indent, numbered tables
// ("Таблиця N — …" above) and figures ("Рисунок N — …" below), page numbers top right. The
// "Зміст" section is skipped: the thesis has one table of contents.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType,
  AlignmentType, BorderStyle, ShadingType, LevelFormat, Header, PageNumber, HeadingLevel,
  VerticalAlign,
} = require("docx");

const [src, out] = process.argv.slice(2);
const baseDir = path.dirname(src);
const lines = fs.readFileSync(src, "utf8").replace(/\r/g, "").split("\n");

const FONT = "Times New Roman";
const BODY = 28;            // 14 pt
const SMALL = 24;           // 12 pt in tables
const CODE_FONT = "Courier New";
const PAGE_W = 11906, MARGIN_L = 1701, MARGIN_R = 850, MARGIN_TB = 1134;
const TEXT_W = PAGE_W - MARGIN_L - MARGIN_R;   // twips
const INDENT = 709;                            // 1.25 cm

// ------------------------------------------------------------------ inline
function inline(text, base = {}) {
  const runs = [];
  const re = /(\*\*([^*]+)\*\*|\*([^*]+)\*|`([^`]+)`|\[([^\]]+)\]\(([^)]+)\))/g;
  let last = 0, m;
  const push = (t, extra = {}) => {
    if (t) runs.push(new TextRun({ text: t, font: FONT, size: base.size || BODY, ...base, ...extra }));
  };
  while ((m = re.exec(text))) {
    push(text.slice(last, m.index));
    if (m[2] !== undefined) push(m[2], { bold: true });
    else if (m[3] !== undefined) push(m[3], { italics: true });
    else if (m[4] !== undefined) push(m[4], { font: CODE_FONT, size: (base.size || BODY) - 4 });
    else push(m[5]);
    last = re.lastIndex;
  }
  push(text.slice(last));
  return runs;
}

const body = (children, opts = {}) => new Paragraph({
  children, alignment: AlignmentType.JUSTIFIED,
  indent: { firstLine: INDENT }, spacing: { line: 360, before: 0, after: 0 }, ...opts,
});

// ------------------------------------------------------------------ images
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

function image(rel) {
  const file = path.join(baseDir, rel);
  const { w, h } = pngSize(file);
  const maxW = Math.round(TEXT_W / 1440 * 96);       // px at 96 dpi
  const maxH = Math.round(8.2 * 96);
  let width = maxW, height = Math.round(h * maxW / w);
  if (height > maxH) { height = maxH; width = Math.round(w * maxH / h); }
  return new Paragraph({
    alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width, height } })],
  });
}

// ------------------------------------------------------------------ tables
let tableNo = 0;
const border = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const borders = { top: border, bottom: border, left: border, right: border };

function splitRow(line) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(c => c.trim());
}

function table(rows, indentTw) {
  const header = splitRow(rows[0]);
  const align = splitRow(rows[1]).map(a => a.endsWith(":") ? (a.startsWith(":") ? "c" : "r") : "l");
  const data = rows.slice(2).map(splitRow);
  const n = header.length;
  const width = TEXT_W - indentTw;
  const plain = s => s.replace(/\*\*|`/g, "");
  const lens = [...Array(n).keys()].map(i => Math.max(
    plain(header[i]).length * 0.9, ...data.map(r => plain(r[i] || "").length), 4));
  const capped = lens.map(l => Math.min(l, 60));
  const total = capped.reduce((a, b) => a + b, 0);
  let widths = capped.map(l => Math.max(900, Math.round(width * l / total)));
  const scale = width / widths.reduce((a, b) => a + b, 0);
  widths = widths.map(w => Math.floor(w * scale));
  widths[n - 1] += width - widths.reduce((a, b) => a + b, 0);

  const together = data.length <= 15;   // small tables stay on one page
  const cell = (text, i, head, keep) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    shading: head ? { fill: "F2F2F2", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({
      alignment: head ? AlignmentType.CENTER
        : align[i] === "r" ? AlignmentType.RIGHT : align[i] === "c" ? AlignmentType.CENTER : AlignmentType.LEFT,
      spacing: { line: 240, before: 0, after: 0 }, keepNext: keep,
      children: inline(text, { size: SMALL, bold: head || undefined }),
    })],
  });
  return new Table({
    width: { size: width, type: WidthType.DXA }, columnWidths: widths,
    indent: indentTw ? { size: indentTw, type: WidthType.DXA } : undefined,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, i, true, true)) }),
      ...data.map((r, k) => new TableRow({ cantSplit: true,
        children: header.map((_, i) => cell(r[i] || "", i, false, together && k < data.length - 1)) })),
    ],
  });
}

// ------------------------------------------------------------------ blocks
const children = [];
let listInstance = 0;
let i = 0;
let paragraph = [];
let listCtx = null;          // {kind:'bullet'|'number', instance}
const afterTable = () => children.push(new Paragraph({ spacing: { before: 0, after: 120 }, children: [] }));

function flushParagraph() {
  if (!paragraph.length) return;
  const text = paragraph.join(" ");
  paragraph = [];
  const cap = text.match(/^\*(Таблиця|Рисунок)( \d+)? — (.+)\*$/);
  if (cap && cap[1] === "Таблиця") {
    tableNo += 1;
    children.push(new Paragraph({
      keepNext: true, spacing: { line: 360, before: 120, after: 60 },
      children: inline(`Таблиця ${tableNo} — ${cap[3]}`),
    }));
  } else if (cap) {
    children.push(new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { line: 276, before: 0, after: 200 },
      children: inline(`${cap[1]}${cap[2]} — ${cap[3]}`, { size: BODY - 2 }),
    }));
  } else {
    children.push(body(inline(text)));
  }
}

const listRe = /^(\s*)([-*]|\d+\.)\s+(.*)$/;

function listItem(level, kind, text, instance) {
  return new Paragraph({
    numbering: { reference: kind === "number" ? "numbers" : "bullets", level, instance },
    alignment: AlignmentType.JUSTIFIED, spacing: { line: 360, before: 0, after: 0 },
    children: inline(text),
  });
}

let skipping = false;   // the "Зміст" section: Word builds its own table of contents
while (i < lines.length) {
  const line = lines[i];

  if (/^## /.test(line)) skipping = /^## Зміст/.test(line);
  if (skipping) { i++; continue; }

  if (line.startsWith("```")) {
    flushParagraph(); listCtx = null;
    i++;
    while (i < lines.length && !lines[i].startsWith("```")) {
      children.push(new Paragraph({
        spacing: { line: 240, before: 0, after: 0 }, indent: { left: INDENT },
        children: [new TextRun({ text: lines[i] || " ", font: CODE_FONT, size: 18 })],
      }));
      i++;
    }
    children.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
    i++; continue;
  }

  const h = line.match(/^(#{1,3}) (.+)$/);
  if (h) {
    flushParagraph(); listCtx = null;
    const level = h[1].length;
    children.push(new Paragraph({
      heading: [HeadingLevel.TITLE, HeadingLevel.HEADING_1, HeadingLevel.HEADING_2][level - 1],
      children: inline(h[2], { size: level === 1 ? 32 : BODY, bold: true, color: "000000" }),
    }));
    i++; continue;
  }

  const img = line.match(/^!\[[^\]]*\]\(([^)]+)\)$/);
  if (img) { flushParagraph(); listCtx = null; children.push(image(img[1])); i++; continue; }

  if (/^\s*\|/.test(line)) {
    flushParagraph();
    const indentChars = line.length - line.trimStart().length;
    const rows = [];
    while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(lines[i++]);
    children.push(table(rows, indentChars ? INDENT : 0));
    afterTable();
    continue;
  }

  const li = line.match(listRe);
  if (li) {
    flushParagraph();
    const indent = li[1].length;
    const kind = /\d/.test(li[2]) ? "number" : "bullet";
    const level = indent === 0 ? 0 : indent <= 3 ? 1 : 2;
    if (!listCtx || (level === 0 && listCtx.kind !== kind)) listCtx = { kind, instance: ++listInstance, sub: {} };
    let instance = listCtx.instance;
    if (level > 0) {              // nested lists keep their own counter per parent item
      listCtx.sub[level] = listCtx.sub[level] || ++listInstance;
      instance = kind === listCtx.kind ? listCtx.instance : listCtx.sub[level];
    } else {
      listCtx.sub = {};
    }
    let text = li[3];
    i++;
    while (i < lines.length && lines[i].trim() && !listRe.test(lines[i]) && /^\s+\S/.test(lines[i])
           && !/^\s*\|/.test(lines[i])) {
      text += " " + lines[i].trim();
      i++;
    }
    children.push(listItem(level, kind, text, instance));
    continue;
  }

  if (!line.trim()) {
    flushParagraph();
    // a blank line inside a list: the list goes on if the next line is indented or another item
    const next = lines.slice(i + 1).find(l => l.trim());
    if (listCtx && next && !/^\s+\S/.test(next) && !listRe.test(next)) listCtx = null;
    i++; continue;
  }

  if (listCtx && /^\s+\S/.test(line)) {           // a paragraph that belongs to a list item
    const level = line.length - line.trimStart().length <= 3 ? 1 : 2;
    let text = line.trim();
    i++;
    while (i < lines.length && lines[i].trim() && !listRe.test(lines[i]) && !/^\s*\|/.test(lines[i])) {
      text += " " + lines[i].trim(); i++;
    }
    const cap = text.match(/^\*Таблиця — (.+)\*$/);
    if (cap) {
      tableNo += 1;
      children.push(new Paragraph({ keepNext: true, indent: { left: INDENT }, spacing: { line: 360, before: 120, after: 60 },
        children: inline(`Таблиця ${tableNo} — ${cap[1]}`) }));
    } else {
      children.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: { left: INDENT * level },
        spacing: { line: 360 }, children: inline(text) }));
    }
    continue;
  }

  listCtx = null;
  paragraph.push(line.trim());
  i++;
}
flushParagraph();

// ------------------------------------------------------------------ document
const bulletLevels = [0, 1, 2].map(level => ({
  level, format: LevelFormat.BULLET, text: ["–", "•", "◦"][level], alignment: AlignmentType.LEFT,
  style: { paragraph: { indent: { left: INDENT * (level + 1) + 283, hanging: 283 } }, run: { font: FONT } },
}));
const numberLevels = [0, 1, 2].map(level => ({
  level, format: LevelFormat.DECIMAL, text: `%${level + 1}.`, alignment: AlignmentType.LEFT,
  style: { paragraph: { indent: { left: INDENT * (level + 1) + 397, hanging: 397 } } },
}));

const doc = new Document({
  creator: "Kolos",
  title: "Розпізнавання страв за фотографією",
  styles: {
    default: { document: { run: { font: FONT, size: BODY } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 32, bold: true, color: "000000" },
        paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 0, after: 240, line: 360 }, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: BODY, bold: true, color: "000000" },
        paragraph: { indent: { firstLine: INDENT }, spacing: { before: 360, after: 180, line: 360 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: BODY, bold: true, color: "000000" },
        paragraph: { indent: { firstLine: INDENT }, spacing: { before: 240, after: 120, line: 360 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: bulletLevels },
    { reference: "numbers", levels: numberLevels },
  ] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 },
      margin: { top: MARGIN_TB, bottom: MARGIN_TB, left: MARGIN_L, right: MARGIN_R, header: 567 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: BODY })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => { fs.writeFileSync(out, buf); console.log(`wrote ${out}: ${children.length} blocks, ${tableNo} tables`); });
