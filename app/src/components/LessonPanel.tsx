import { useEffect, useMemo } from "react";
import { useLessonStore, lessonRuns } from "../store/lesson-store";
import { useGraphStore } from "../store/graph-store";
import { useTrainingStore } from "../store/training-store";
import { useRunStore } from "../store/run-store";
import { findLesson } from "../lib/lessons/catalog";
import { evaluate } from "../lib/lessons/checks";
import { useChangeSetStore } from "../lib/assistant/change-set";
import { askAssistant } from "../store/assistant-store";
import Button, { IconButton } from "./ui/Button";

/** The lesson in progress, floating at the top left of the canvas: the goal, the current step, its checklist and hints */
export default function LessonPanel({ onShowLessons }: { onShowLessons: () => void }) {
  const active = useLessonStore((s) => s.active);
  const collapsed = useLessonStore((s) => s.collapsed);
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);
  const config = useTrainingStore((s) => s.config);
  const runs = useRunStore((s) => s.runs);
  // The assistant's pending changes are only a preview: a step they'd complete counts once they're applied
  const previewing = useChangeSetStore((s) => s.base !== null);

  const lesson = active ? findLesson(active.lessonId) : undefined;
  const step = lesson && active ? lesson.steps[active.stepIndex] : undefined;
  const done = !!(step && active?.done.includes(step.id));

  const result = useMemo(() => {
    if (!active || !step) return null;
    return evaluate(step.check, {
      // Shape errors are worked out here rather than read from the canvas, which updates them a moment later
      graph: { nodes, edges },
      config,
      runs: lessonRuns(runs, active.startedAt),
      confirmed: active.confirmed,
      stepId: step.id,
    });
  }, [active, step, nodes, edges, config, runs]);

  useEffect(() => {
    if (step && result?.passed && !done && !previewing) useLessonStore.getState().markDone(step.id);
  }, [step, result?.passed, done, previewing]);

  if (!active || !lesson || !step) return null;
  const { exit, next, confirmStep, showHint, setCollapsed } = useLessonStore.getState();
  const stepCount = lesson.steps.length;

  if (collapsed) {
    return (
      <button type="button" onClick={() => setCollapsed(false)}
        className="pointer-events-auto absolute top-3 left-3 z-30 flex items-center gap-2 h-8 pl-2.5 pr-3 rounded-full bg-surface border border-line text-[11px] text-fg-secondary hover:text-fg"
        style={{ boxShadow: "var(--node-shadow-hover)" }}
        aria-label={`Show lesson: ${lesson.title}`}>
        <BookIcon />
        <span className="font-semibold text-fg">{lesson.title}</span>
        <span className="tabular-nums">{active.finished ? "Done" : `${active.stepIndex + 1}/${stepCount}`}</span>
      </button>
    );
  }

  return (
    <section aria-label={`Lesson: ${lesson.title}`}
      className="absolute top-3 left-3 z-30 w-[320px] max-h-[calc(100%-24px)] overflow-y-auto rounded-xl bg-surface border border-line text-xs"
      style={{ boxShadow: "var(--node-shadow-hover)" }}>
      <header className="flex items-start gap-2 px-3.5 pt-3 pb-2">
        <span className="mt-0.5 text-accent-text"><BookIcon /></span>
        <div className="min-w-0 flex-1">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-fg-muted">Lesson</div>
          <h2 className="text-[13px] font-semibold text-fg leading-snug">{lesson.title}</h2>
        </div>
        <IconButton size="sm" label="Minimize lesson" onClick={() => setCollapsed(true)}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><line x1="5" y1="12" x2="19" y2="12" /></svg>
        </IconButton>
        <IconButton size="sm" label="Leave lesson" onClick={exit}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
        </IconButton>
      </header>

      <p className="px-3.5 pb-2.5 text-fg-secondary leading-relaxed">{lesson.goal}</p>

      <ol className="flex gap-1 px-3.5 pb-3" aria-label="Steps">
        {lesson.steps.map((s, i) => {
          const state = active.done.includes(s.id) ? "done" : i === active.stepIndex ? "current" : "todo";
          return (
            <li key={s.id} className={`h-1 flex-1 rounded-full ${state === "done" ? "bg-success" : state === "current" ? "bg-accent" : "bg-surface-raised"}`}
              aria-label={`Step ${i + 1}: ${state === "done" ? "done" : state === "current" ? "current" : "to do"}`} />
          );
        })}
      </ol>

      {active.finished ? (
        <div className="px-3.5 pb-3.5">
          <div className="rounded-lg p-3 bg-success-subtle">
            <div className="font-semibold text-fg mb-1">Lesson complete</div>
            <p className="text-fg-secondary leading-relaxed">
              You've finished every step. The model and its runs are yours to keep changing, or save the project to come back to it.
            </p>
          </div>
          <div className="flex justify-end gap-2 mt-3">
            <Button size="sm" onClick={exit}>Close</Button>
            <Button size="sm" variant="primary" onClick={() => { exit(); onShowLessons(); }}>More lessons</Button>
          </div>
        </div>
      ) : (
        <div className="px-3.5 pb-3.5 border-t border-line pt-3">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-fg-muted mb-1 tabular-nums">Step {active.stepIndex + 1} of {stepCount}</div>
          <p className="text-fg leading-relaxed">{step.instruction}</p>

          {result && result.parts.length > 0 && (
            <ul className="mt-2.5 space-y-1" aria-label="Checklist">
              {result.parts.map((p) => (
                <li key={p.label} className={`flex items-center gap-2 ${p.passed || done ? "text-fg" : "text-fg-muted"}`}>
                  <CheckMark on={p.passed || done} />
                  {p.label}
                </li>
              ))}
            </ul>
          )}

          {done ? (
            <div className="mt-3 rounded-lg p-3 bg-success-subtle" role="status">
              <div className="flex items-center gap-1.5 font-semibold text-fg mb-1"><CheckMark on /> Done</div>
              <p className="text-fg-secondary leading-relaxed">{step.explain}</p>
            </div>
          ) : (
            active.hintsShown > 0 && (
              <ul className="mt-3 space-y-1.5" aria-label="Hints" aria-live="polite">
                {step.hints.slice(0, active.hintsShown).map((h, i) => (
                  <li key={i} className="rounded-lg px-2.5 py-2 bg-surface-raised text-fg-secondary leading-relaxed">
                    <span className="font-semibold text-fg">Hint {i + 1}.</span> {h}
                  </li>
                ))}
              </ul>
            )
          )}

          <div className="flex items-center justify-end gap-2 mt-3">
            {!done && active.hintsShown < step.hints.length && (
              <Button size="sm" variant="ghost" onClick={showHint}>
                {active.hintsShown === 0 ? "Show a hint" : "Another hint"}
              </Button>
            )}
            <Button size="sm" variant="ghost" className="mr-auto" onClick={() => askAssistant(tutorQuestion(done, active.stepIndex))}>
              Ask the tutor
            </Button>
            {!done && step.check.kind === "manual" && (
              <Button size="sm" variant="primary" onClick={confirmStep}>{step.check.button ?? "Done"}</Button>
            )}
            {done && (
              <Button size="sm" variant="primary" onClick={next}>{active.stepIndex + 1 >= stepCount ? "Finish lesson" : "Next step"}</Button>
            )}
          </div>
        </div>
      )}
    </section>
  );
}

/** A starting question for the assistant, which the learner can edit before sending */
function tutorQuestion(done: boolean, stepIndex: number): string {
  return done
    ? `Why did that work? Can you explain what step ${stepIndex + 1} showed?`
    : `I'm stuck on step ${stepIndex + 1}. Can you help me work it out without just giving me the answer?`;
}

function CheckMark({ on }: { on: boolean }) {
  return on ? (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--success)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" aria-label="done" role="img" className="flex-shrink-0">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  ) : (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-label="not yet" role="img" className="flex-shrink-0">
      <circle cx="12" cy="12" r="8" />
    </svg>
  );
}

export function BookIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M2 4h6a4 4 0 0 1 4 4v12a3 3 0 0 0-3-3H2z" /><path d="M22 4h-6a4 4 0 0 0-4 4v12a3 3 0 0 1 3-3h7z" />
    </svg>
  );
}
