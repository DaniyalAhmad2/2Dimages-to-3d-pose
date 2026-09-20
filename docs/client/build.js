// Renders content.json -> Pose3D-Getting-Started.docx (A4, compact, client-facing).
// Content model: { title, subtitle, intro, sections: [{ heading, blocks: [block] }] }
// block: { p: "text" } | { bullets: ["..."] } | { steps: ["..."] } | { sub: "sub-heading" }
// Inline markup inside any text: **bold**, `code`, and http(s) URLs become links.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  LevelFormat, BorderStyle, ExternalHyperlink,
} = require("docx");

const content = JSON.parse(fs.readFileSync(path.join(__dirname, "content.json"), "utf8"));
const FONT = "Calibri";
const BODY = 21;      // half-points: 10.5 pt
const SMALL = 18;
const INK = "1F2937";
const ACCENT = "1D4ED8";

function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|https?:\/\/[^\s)]+)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, size: BODY, color: INK, ...base }));
    const tok = m[0];
    if (tok.startsWith("**")) out.push(new TextRun({ text: tok.slice(2, -2), bold: true, font: FONT, size: BODY, color: INK, ...base }));
    else if (tok.startsWith("`")) out.push(new TextRun({ text: tok.slice(1, -1), font: "Consolas", size: BODY - 1, color: "111827", shading: { type: "clear", fill: "EEF2F7" }, ...base }));
    else out.push(new ExternalHyperlink({ link: tok, children: [new TextRun({ text: tok, font: FONT, size: BODY, color: ACCENT, underline: {}, ...base })] }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, size: BODY, color: INK, ...base }));
  return out;
}

const numbering = {
  config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 400, hanging: 240 } }, run: { font: FONT, size: BODY, color: INK } } }] },
  ],
};
// one numbered list per steps block, so every list restarts at 1
let stepLists = 0;
function stepsRef() {
  const ref = `steps${++stepLists}`;
  numbering.config.push({ reference: ref, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 440, hanging: 300 } }, run: { font: FONT, size: BODY, color: INK } } }] });
  return ref;
}

const children = [];
children.push(new Paragraph({ children: [new TextRun({ text: content.title, font: FONT, size: 40, bold: true, color: "111827" })], spacing: { after: 60 } }));
children.push(new Paragraph({ children: [new TextRun({ text: content.subtitle, font: FONT, size: SMALL + 1, color: "4B5563" })],
  spacing: { after: 160 }, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "CBD5E1", space: 6 } } }));
if (content.intro) children.push(new Paragraph({ children: runs(content.intro), spacing: { after: 140, line: 264 } }));

for (const section of content.sections) {
  children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: section.heading, font: FONT, size: 26, bold: true, color: ACCENT })],
    spacing: { before: 220, after: 80 }, keepNext: true }));
  for (const block of section.blocks) {
    if (block.sub) {
      children.push(new Paragraph({ children: [new TextRun({ text: block.sub, font: FONT, size: BODY + 1, bold: true, color: "111827" })], spacing: { before: 120, after: 40 }, keepNext: true }));
    } else if (block.p !== undefined) {
      children.push(new Paragraph({ children: runs(block.p), spacing: { after: 90, line: 264 } }));
    } else if (block.bullets) {
      for (const item of block.bullets) children.push(new Paragraph({ children: runs(item), numbering: { reference: "bullets", level: 0 }, spacing: { after: 50, line: 259 } }));
    } else if (block.steps) {
      const ref = stepsRef();
      for (const item of block.steps) children.push(new Paragraph({ children: runs(item), numbering: { reference: ref, level: 0 }, spacing: { after: 50, line: 259 } }));
    }
  }
}

const doc = new Document({
  creator: "Pose3D",
  title: content.title,
  numbering,
  styles: { default: { document: { run: { font: FONT, size: BODY, color: INK } } } },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1000, right: 1080, bottom: 1000, left: 1080 } } },
    children,
  }],
});

const out = path.join(__dirname, "Pose3D-Getting-Started.docx");
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(out, buf); console.log("wrote", out, buf.length, "bytes"); });
