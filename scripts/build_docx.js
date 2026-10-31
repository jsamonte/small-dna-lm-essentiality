// Convert the project's Markdown (a known subset) to .docx.
// Usage: node md2docx.js <input.md> <output.docx> [--line-numbers]
const fs = require("fs");
const path = require("path");
const D = require("docx");

const [, , INPUT, OUTPUT, ...flags] = process.argv;
const LINE_NUMBERS = flags.includes("--line-numbers");
// --label "Manuscript · Topic · Author · Date" prints an identifying line
// under the title, so a printed copy cannot be confused with another document.
const labelIdx = flags.indexOf("--label");
const LABEL = labelIdx >= 0 ? flags[labelIdx + 1] : null;
const BASE = path.dirname(path.resolve(INPUT));

const SERIF = "Times New Roman";
const CONTENT_DXA = 9360; // 6.5in at 1in margins

const ENTITIES = {
  "&alpha;": "α",
  "&plusmn;": "±",
  "&Phi;": "Φ",
  "&phi;": "φ",
  "&rho;": "ρ",
  "&times;": "×",
  "&mdash;": "—",
  "&amp;": "&",
};

function entities(s) {
  for (const [k, v] of Object.entries(ENTITIES)) s = s.split(k).join(v);
  return s;
}

// Build-machinery notes that belong in the repo, not in a document a
// collaborator reads or edits.
const BUILD_NOTES = [
  /^\*Methods, Results and Limitations are generated/,
  /^\*Reproduce every number above/,
];

// ---- inline: **bold**, *italic*, `code`, nestable ----
function runs(text, baseOpts) {
  const opts = baseOpts || {};
  const out = [];
  const re = /(\*\*[\s\S]+?\*\*|\*[^*\n]+?\*|`[^`]+`)/g;
  let last = 0;
  let m;
  const push = (t, extra) => {
    if (!t) return;
    out.push(new D.TextRun(Object.assign({ text: t, font: SERIF }, opts, extra)));
  };
  while ((m = re.exec(text)) !== null) {
    push(text.slice(last, m.index));
    const tok = m[0];
    // recurse so `code` inside *italics* (and vice versa) is still marked up
    if (tok.startsWith("**"))
      out.push(...runs(tok.slice(2, -2), Object.assign({}, opts, { bold: true })));
    else if (tok.startsWith("`"))
      push(tok.slice(1, -1), { font: "Consolas", size: 19 });
    else
      out.push(...runs(tok.slice(1, -1), Object.assign({}, opts, { italics: true })));
    last = m.index + tok.length;
  }
  push(text.slice(last));
  return out.length ? out : [new D.TextRun({ text: "", font: SERIF })];
}

function para(text, extra) {
  return new D.Paragraph(
    Object.assign({ children: runs(text), spacing: { after: 160, line: 276 } }, extra)
  );
}

// ---- tables ----
function splitRow(line) {
  let s = line.trim();
  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|")) s = s.slice(0, -1);
  return s.split("|").map((c) => c.trim());
}

function buildTable(lines) {
  const header = splitRow(lines[0]);
  const body = lines.slice(2).map(splitRow);
  const n = header.length;
  const hasHeader = header.some((c) => c.length > 0);

  let widths;
  if (n === 2) widths = [Math.round(CONTENT_DXA * 0.42), CONTENT_DXA - Math.round(CONTENT_DXA * 0.42)];
  else {
    const w = Math.floor(CONTENT_DXA / n);
    widths = Array(n).fill(w);
    widths[n - 1] = CONTENT_DXA - w * (n - 1);
  }

  const mkCell = (txt, i, isHead) =>
    new D.TableCell({
      width: { size: widths[i], type: D.WidthType.DXA },
      shading: isHead
        ? { type: D.ShadingType.CLEAR, fill: "EEEEEE", color: "auto" }
        : undefined,
      margins: { top: 60, bottom: 60, left: 110, right: 110 },
      children: [
        new D.Paragraph({
          children: runs(txt, isHead ? { bold: true } : undefined),
          spacing: { after: 0, line: 240 },
        }),
      ],
    });

  const rows = [];
  if (hasHeader) {
    rows.push(
      new D.TableRow({
        tableHeader: true,
        children: header.map((c, i) => mkCell(c, i, true)),
      })
    );
  }
  for (const r of body) {
    const cells = [];
    for (let i = 0; i < n; i++) cells.push(mkCell(r[i] === undefined ? "" : r[i], i, false));
    rows.push(new D.TableRow({ children: cells }));
  }

  return new D.Table({
    columnWidths: widths,
    width: { size: CONTENT_DXA, type: D.WidthType.DXA },
    rows,
  });
}

// ---- block parse ----
function convert(md) {
  const lines = entities(md).replace(/\r\n/g, "\n").split("\n");
  const out = [];
  let i = 0;

  const isTable = (k) =>
    lines[k] !== undefined &&
    lines[k].trim().startsWith("|") &&
    lines[k + 1] !== undefined &&
    /^\s*\|[\s:|-]+\|\s*$/.test(lines[k + 1]);

  while (i < lines.length) {
    const line = lines[i];
    const t = line.trim();

    if (t === "") {
      i++;
      continue;
    }

    // A build note may be wrapped over several source lines, so skip the
    // whole paragraph, not just the line that matched.
    if (BUILD_NOTES.some((r) => r.test(t))) {
      while (i < lines.length && lines[i].trim() !== "") i++;
      continue;
    }

    // headings
    const h = /^(#{1,4})\s+(.*)$/.exec(t);
    if (h) {
      const lvl = h[1].length;
      const txt = h[2];
      if (lvl === 1) {
        out.push(
          new D.Paragraph({
            children: runs(txt, { bold: true, size: 34 }),
            spacing: { after: LABEL ? 100 : 300, line: 276 },
            alignment: D.AlignmentType.CENTER,
          })
        );
        if (LABEL) {
          out.push(
            new D.Paragraph({
              alignment: D.AlignmentType.CENTER,
              spacing: { after: 340 },
              border: {
                bottom: {
                  style: D.BorderStyle.SINGLE,
                  size: 6,
                  color: "AAAAAA",
                  space: 10,
                },
              },
              children: [
                new D.TextRun({
                  text: LABEL,
                  font: SERIF,
                  size: 19,
                  color: "555555",
                }),
              ],
            })
          );
        }
      } else {
        out.push(
          new D.Paragraph({
            heading:
              lvl === 2 ? D.HeadingLevel.HEADING_1 : lvl === 3 ? D.HeadingLevel.HEADING_2 : D.HeadingLevel.HEADING_3,
            children: runs(txt, { bold: true, size: lvl === 2 ? 28 : 24, color: "000000" }),
            spacing: { before: 320, after: 140 },
          })
        );
      }
      i++;
      continue;
    }

    // image
    const img = /^!\[[^\]]*\]\(([^)]+)\)$/.exec(t);
    if (img) {
      const p = path.resolve(BASE, img[1]);
      if (fs.existsSync(p)) {
        const data = fs.readFileSync(p);
        // Read the real pixel size out of the PNG header (IHDR at byte 16)
        // rather than assuming an aspect ratio, so a figure of any shape
        // keeps its proportions.
        const pxW = data.readUInt32BE(16);
        const pxH = data.readUInt32BE(20);
        const w = 620;
        out.push(
          new D.Paragraph({
            alignment: D.AlignmentType.CENTER,
            spacing: { before: 200, after: 120 },
            children: [
              new D.ImageRun({
                type: "png",
                data,
                transformation: { width: w, height: Math.round((w * pxH) / pxW) },
              }),
            ],
          })
        );
      }
      i++;
      continue;
    }

    // table
    if (isTable(i)) {
      const block = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) block.push(lines[i]), i++;
      out.push(buildTable(block));
      out.push(new D.Paragraph({ text: "", spacing: { after: 160 } }));
      continue;
    }

    // ordered list item
    const ol = /^(\d+)\.\s+(.*)$/.exec(t);
    if (ol) {
      const parts = [ol[2]];
      i++;
      while (i < lines.length && lines[i].trim() !== "" && !/^(\d+)\.\s+/.test(lines[i].trim()) && !lines[i].trim().startsWith("#")) {
        parts.push(lines[i].trim());
        i++;
      }
      out.push(
        new D.Paragraph({
          children: runs(ol[1] + ". " + parts.join(" ")),
          spacing: { after: 160, line: 276 },
          indent: { left: 420, hanging: 420 },
        })
      );
      continue;
    }

    // paragraph: join wrapped lines
    const parts = [t];
    i++;
    while (i < lines.length) {
      const nx = lines[i].trim();
      if (nx === "" || nx.startsWith("#") || nx.startsWith("|") || nx.startsWith("![") || /^\d+\.\s+/.test(nx)) break;
      parts.push(nx);
      i++;
    }
    out.push(para(parts.join(" ")));
  }

  return out;
}

const children = convert(fs.readFileSync(INPUT, "utf8"));

const sectionProps = {
  page: {
    size: { width: 12240, height: 15840 }, // US Letter
    margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
  },
};
if (LINE_NUMBERS) {
  sectionProps.lineNumbers = { countBy: 5, restart: D.LineNumberRestartFormat.NEW_PAGE, distance: 360 };
}

const doc = new D.Document({
  styles: {
    default: {
      document: { run: { font: SERIF, size: 23 } },
      heading1: { run: { font: SERIF, color: "000000" } },
      heading2: { run: { font: SERIF, color: "000000" } },
      heading3: { run: { font: SERIF, color: "000000" } },
    },
  },
  sections: [
    {
      properties: sectionProps,
      footers: {
        default: new D.Footer({
          children: [
            new D.Paragraph({
              alignment: D.AlignmentType.CENTER,
              children: [new D.TextRun({ children: [D.PageNumber.CURRENT], font: SERIF, size: 19 })],
            }),
          ],
        }),
      },
      children,
    },
  ],
});

D.Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUTPUT, buf);
  console.log("wrote " + OUTPUT + " (" + buf.length + " bytes, " + children.length + " blocks)");
});
