/**
 * The small part of Markdown the assistant writes: paragraphs, bullet and numbered lists, fenced code,
 * inline code, bold, italics and links. Builds React elements, never HTML, so nothing it says can run.
 */

import { Fragment, type ReactNode } from "react";
import { openExternal } from "../../lib/open-external";

type Block =
  | { kind: "p"; text: string }
  | { kind: "h"; text: string }
  | { kind: "code"; lang: string; text: string }
  | { kind: "ul" | "ol"; items: string[] };

export function parseBlocks(src: string): Block[] {
  const lines = src.replace(/\r\n/g, "\n").split("\n");
  const blocks: Block[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const fence = line.match(/^\s*```(\w*)/);
    if (fence) {
      const body: string[] = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) body.push(lines[i++]);
      i++;
      blocks.push({ kind: "code", lang: fence[1], text: body.join("\n") });
      continue;
    }
    if (!line.trim()) { i++; continue; }
    const heading = line.match(/^#{1,6}\s+(.*)/);
    if (heading) { blocks.push({ kind: "h", text: heading[1] }); i++; continue; }
    const bullet = /^\s*[-*]\s+/;
    const numbered = /^\s*\d+[.)]\s+/;
    if (bullet.test(line) || numbered.test(line)) {
      const kind = bullet.test(line) ? "ul" : "ol";
      const re = kind === "ul" ? bullet : numbered;
      const items: string[] = [];
      while (i < lines.length && (re.test(lines[i]) || (/^\s{2,}\S/.test(lines[i]) && items.length))) {
        if (re.test(lines[i])) items.push(lines[i].replace(re, ""));
        else items[items.length - 1] += ` ${lines[i].trim()}`;
        i++;
      }
      blocks.push({ kind, items });
      continue;
    }
    const para: string[] = [];
    while (i < lines.length && lines[i].trim() && !/^\s*```/.test(lines[i]) && !bullet.test(lines[i]) && !numbered.test(lines[i]) && !/^#{1,6}\s/.test(lines[i])) para.push(lines[i++]);
    blocks.push({ kind: "p", text: para.join(" ") });
  }
  return blocks;
}

/** Inline code, bold, italics and links */
export function inline(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*\s][^*]*\*)|(\[[^\]]+\]\((https?:\/\/[^)\s]+)\))/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const tok = m[0];
    if (m[1]) out.push(<code key={k++} className="px-1 py-px rounded bg-surface-sunken font-mono text-[0.92em]">{tok.slice(1, -1)}</code>);
    else if (m[2]) out.push(<strong key={k++} className="font-semibold text-fg">{tok.slice(2, -2)}</strong>);
    else if (m[3]) out.push(<em key={k++}>{tok.slice(1, -1)}</em>);
    else if (m[4]) {
      const label = tok.slice(1, tok.indexOf("]"));
      const href = m[5];
      out.push(<a key={k++} href={href} className="text-accent-text underline" onClick={(e) => { e.preventDefault(); void openExternal(href); }}>{label}</a>);
    }
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export default function Markdown({ text }: { text: string }) {
  return (
    <div className="space-y-2">
      {parseBlocks(text).map((b, i) => {
        switch (b.kind) {
          case "h": return <div key={i} className="font-semibold text-fg">{inline(b.text)}</div>;
          case "p": return <p key={i}>{inline(b.text)}</p>;
          case "code": return (
            <pre key={i} className="p-2 rounded-md bg-surface-sunken border border-line font-mono text-[11px] leading-snug overflow-x-auto whitespace-pre">{b.text}</pre>
          );
          case "ul": return <ul key={i} className="list-disc pl-4 space-y-0.5">{b.items.map((it, j) => <li key={j}>{inline(it)}</li>)}</ul>;
          case "ol": return <ol key={i} className="list-decimal pl-4 space-y-0.5">{b.items.map((it, j) => <li key={j}>{inline(it)}</li>)}</ol>;
          default: return <Fragment key={i} />;
        }
      })}
    </div>
  );
}
