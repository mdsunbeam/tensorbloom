import Dialog from "./ui/Dialog";
import Button from "./ui/Button";
import { LESSONS } from "../lib/lessons/catalog";
import { useLessonStore } from "../store/lesson-store";

/** The list of guided lessons. Picking one replaces the canvas with its starting model. */
export default function LessonsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const completed = useLessonStore((s) => s.completed);
  const activeId = useLessonStore((s) => s.active?.lessonId);

  const start = async (id: string) => {
    onClose();
    await useLessonStore.getState().start(id);
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      width={620}
      title="Lessons"
      description="Short, hands-on lessons. Each one starts from a real model, gives you a goal, and checks your work as you go."
      footer={<Button onClick={onClose}>Close</Button>}
    >
      <ol className="p-4 grid grid-cols-2 gap-2.5">
        {LESSONS.map((l, i) => {
          const isDone = completed.includes(l.id);
          return (
            <li key={l.id}>
              <button type="button" onClick={() => void start(l.id)}
                className="w-full h-full text-left rounded-lg border border-line p-3 hover:bg-surface-raised hover:border-accent transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent">
                <div className="flex items-center gap-2 mb-1">
                  <span className="flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-semibold bg-accent-subtle text-accent-text tabular-nums">{i + 1}</span>
                  <span className="text-xs font-semibold text-fg leading-snug">{l.title}</span>
                </div>
                <p className="text-[11px] leading-relaxed text-fg-secondary mb-2">{l.summary}</p>
                <div className="flex items-center gap-2 text-[10px] text-fg-muted">
                  <span className="capitalize">{l.level}</span>
                  <span aria-hidden="true">·</span>
                  <span>About {l.minutes} min</span>
                  {activeId === l.id ? (
                    <span className="ml-auto font-semibold text-accent-text">In progress</span>
                  ) : isDone ? (
                    <span className="ml-auto font-semibold text-success">Done</span>
                  ) : null}
                </div>
              </button>
            </li>
          );
        })}
      </ol>
    </Dialog>
  );
}
