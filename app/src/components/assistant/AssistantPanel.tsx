import { useEffect, useRef, useState } from "react";
import { useAssistantStore, type ChatItem } from "../../store/assistant-store";
import { MODELS, saveKey, deleteKey } from "../../lib/assistant/api";
import { appState } from "../../lib/assistant/prompt";
import { openExternal } from "../../lib/open-external";
import Button from "../ui/Button";
import Select from "../ui/Select";
import Markdown from "./Markdown";

const SUGGESTIONS = [
  "Explain this model, layer by layer",
  "Why is this shape wrong?",
  "Where do most of the parameters go?",
  "How did my last run go, and what should I try next?",
];

/** The assistant: a chat that can read the model, its runs and its data */
export default function AssistantPanel() {
  const key = useAssistantStore((s) => s.key);
  const refreshKey = useAssistantStore((s) => s.refreshKey);
  const [settingsOpen, setSettingsOpen] = useState(false);

  useEffect(() => { void refreshKey(); }, [refreshKey]);

  const showSettings = settingsOpen || (key !== null && !key.configured);
  return (
    <div className="h-full flex flex-col min-h-0" style={{ background: "var(--bg-secondary)" }}>
      <Header settingsOpen={showSettings} onToggleSettings={() => setSettingsOpen((o) => !o)} canCloseSettings={!!key?.configured} />
      {key === null ? null : showSettings ? <Settings onDone={() => setSettingsOpen(false)} /> : <Chat />}
    </div>
  );
}

function Header({ settingsOpen, onToggleSettings, canCloseSettings }: { settingsOpen: boolean; onToggleSettings: () => void; canCloseSettings: boolean }) {
  const reset = useAssistantStore((s) => s.reset);
  const hasItems = useAssistantStore((s) => s.items.length > 0);
  const model = useAssistantStore((s) => MODELS.find((m) => m.id === s.model)?.label ?? s.model);
  return (
    <div className="flex items-center gap-1 px-3 h-10 flex-shrink-0" style={{ borderBottom: "1px solid var(--border)" }}>
      <span className="text-[11px] text-fg-muted flex-1 truncate">{settingsOpen ? "Settings" : model}</span>
      {!settingsOpen && hasItems && (
        <Button variant="ghost" size="sm" onClick={reset} title="Start a new conversation">New chat</Button>
      )}
      {(!settingsOpen || canCloseSettings) && (
        <Button variant="ghost" size="sm" onClick={onToggleSettings} aria-pressed={settingsOpen} title="Assistant settings">
          {settingsOpen ? "Done" : "Settings"}
        </Button>
      )}
    </div>
  );
}

function Settings({ onDone }: { onDone: () => void }) {
  const key = useAssistantStore((s) => s.key);
  const refreshKey = useAssistantStore((s) => s.refreshKey);
  const model = useAssistantStore((s) => s.model);
  const setModel = useAssistantStore((s) => s.setModel);
  const [value, setValue] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      await saveKey(value);
      setValue("");
      await refreshKey();
      onDone();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  const remove = async () => {
    try { await deleteKey(); } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    await refreshKey();
  };

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-5 text-xs text-fg-secondary leading-relaxed">
      <section className="space-y-2">
        <h3 className="text-[11px] font-semibold uppercase tracking-wider text-fg-muted">Anthropic API key</h3>
        {key?.configured ? (
          <p>
            {key.source === "environment"
              ? "Using the key from the ANTHROPIC_API_KEY environment variable."
              : "A key is saved in your system keychain."}
          </p>
        ) : (
          <p>
            The assistant uses Claude through your own Anthropic API key, billed to your account.{" "}
            <a href="https://console.anthropic.com/settings/keys" className="text-accent-text underline"
              onClick={(e) => { e.preventDefault(); void openExternal("https://console.anthropic.com/settings/keys"); }}>
              Get a key
            </a>.
          </p>
        )}
        <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); void save(); }}>
          <input
            type="password"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder={key?.configured ? "Replace with a new key" : "sk-ant-..."}
            aria-label="Anthropic API key"
            autoComplete="off"
            spellCheck={false}
            className="flex-1 min-w-0 h-8 px-2.5 rounded-lg bg-surface-sunken border border-line text-fg font-mono text-[11px] focus:outline-none focus:border-accent"
          />
          <Button variant="primary" type="submit" disabled={!value.trim() || saving}>{saving ? "Saving" : "Save"}</Button>
        </form>
        {error && <p role="alert" className="text-danger">{error}</p>}
        <p className="text-fg-muted">
          The key is stored in your system keychain, and only the app's native side reads it. It's never shown again or saved in project files.
        </p>
        {key?.configured && key.source === "keychain" && (
          <Button variant="danger" size="sm" onClick={remove}>Remove key</Button>
        )}
      </section>

      <section className="space-y-2">
        <h3 className="text-[11px] font-semibold uppercase tracking-wider text-fg-muted">Model</h3>
        <Select value={model} onChange={setModel} options={MODELS.map((m) => ({ value: m.id, label: m.label }))} aria-label="Model" />
        <p className="text-fg-muted">{MODELS.find((m) => m.id === model)?.note}</p>
      </section>

      <section className="space-y-2">
        <h3 className="text-[11px] font-semibold uppercase tracking-wider text-fg-muted">What's sent</h3>
        <p>
          Each message sends your question with the model's layers and settings, shape problems, training settings and a summary of recent runs.
          When the assistant asks for more, the app sends the generated code, run curves, predicted classes and dataset statistics.
          Dataset contents such as images and text stay on this computer.
        </p>
      </section>
    </div>
  );
}

function Chat() {
  const items = useAssistantStore((s) => s.items);
  const busy = useAssistantStore((s) => s.busy);
  const send = useAssistantStore((s) => s.send);
  const scroller = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [items, busy]);

  return (
    <>
      <div ref={scroller} className="flex-1 overflow-y-auto px-3 py-3 space-y-3 min-h-0" aria-live="polite">
        {items.length === 0 ? (
          <div className="space-y-3 pt-2">
            <p className="text-xs text-fg-secondary leading-relaxed">
              Ask about the model on the canvas, a shape error, or how a run went. The assistant can read the graph, its code, runs, predictions and dataset statistics.
            </p>
            <div className="flex flex-col gap-1.5">
              {SUGGESTIONS.map((s) => (
                <button key={s} onClick={() => void send(s)}
                  className="text-left text-xs px-3 py-2 rounded-lg border border-line text-fg-secondary hover:bg-surface-raised hover:text-fg transition-colors">
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : items.map((item) => <Item key={item.id} item={item} />)}
        {busy && <div className="text-[11px] text-fg-muted animate-pulse">Thinking…</div>}
      </div>
      <Composer />
    </>
  );
}

function Item({ item }: { item: ChatItem }) {
  switch (item.kind) {
    case "user":
      return (
        <div className="flex justify-end">
          <div className="max-w-[85%] px-3 py-2 rounded-xl text-xs leading-relaxed whitespace-pre-wrap bg-accent-subtle text-fg">{item.text}</div>
        </div>
      );
    case "assistant":
      return <div className="text-xs leading-relaxed text-fg-secondary"><Markdown text={item.text} /></div>;
    case "tools":
      return (
        <ul className="space-y-0.5">
          {item.tools.map((t) => (
            <li key={t.id} className="flex items-center gap-1.5 text-[11px] text-fg-muted" title={t.detail}>
              <span aria-hidden="true" className={t.status === "error" ? "text-danger" : t.status === "done" ? "text-success" : "animate-pulse"}>
                {t.status === "error" ? "✕" : t.status === "done" ? "✓" : "•"}
              </span>
              <span>{t.label}{t.status === "error" && t.detail ? `: ${t.detail}` : ""}</span>
            </li>
          ))}
        </ul>
      );
    case "notice":
      return (
        <div role={item.tone === "error" ? "alert" : undefined}
          className={`text-[11px] leading-relaxed px-3 py-2 rounded-lg border ${item.tone === "error" ? "bg-danger-subtle border-danger text-danger" : "border-line text-fg-muted"}`}>
          {item.text}
        </div>
      );
  }
}

function Composer() {
  const busy = useAssistantStore((s) => s.busy);
  const send = useAssistantStore((s) => s.send);
  const stop = useAssistantStore((s) => s.stop);
  const draft = useAssistantStore((s) => s.draft);
  const setDraft = useAssistantStore((s) => s.setDraft);
  const [showSent, setShowSent] = useState(false);
  const input = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { input.current?.focus(); }, [draft]);

  const submit = () => {
    if (busy || !draft.trim()) return;
    const text = draft;
    setDraft("");
    void send(text);
  };

  return (
    <div className="flex-shrink-0 p-3 space-y-2" style={{ borderTop: "1px solid var(--border)" }}>
      <textarea
        ref={input}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); submit(); }
        }}
        rows={3}
        placeholder="Ask about your model…"
        aria-label="Message the assistant"
        className="w-full resize-none px-2.5 py-2 rounded-lg bg-surface-sunken border border-line text-xs text-fg leading-relaxed focus:outline-none focus:border-accent"
      />
      <div className="flex items-center gap-2">
        <button onClick={() => setShowSent((s) => !s)} className="text-[11px] text-fg-muted hover:text-fg-secondary underline-offset-2 hover:underline" aria-expanded={showSent}>
          {showSent ? "Hide what's sent" : "See what's sent"}
        </button>
        <span className="flex-1" />
        {busy
          ? <Button size="sm" onClick={stop}>Stop</Button>
          : <Button variant="primary" size="sm" onClick={submit} disabled={!draft.trim()}>Send</Button>}
      </div>
      {showSent && (
        <pre className="max-h-48 overflow-auto p-2 rounded-md bg-surface-sunken border border-line font-mono text-[10px] leading-snug text-fg-muted whitespace-pre-wrap">
          {appState()}
        </pre>
      )}
    </div>
  );
}
