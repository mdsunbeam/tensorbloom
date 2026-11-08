# Lessons

TensorBloom ships with guided lessons: short, hands-on exercises that start from a real model, give the learner a goal, and check their work as they go. Open them from **Help > Lessons**, from the welcome card on an empty canvas, or from the Getting Started guide.

While a lesson is open, a panel at the top left of the canvas shows the goal, the current step, a checklist, and hints. A step is done when its check passes; the panel then explains what just happened, and **Next step** moves on.

## The built-in lessons

| # | Lesson | Starts from | Teaches |
|---|--------|-------------|---------|
| 1 | Train your first model | Simple MLP on MNIST | Training, epochs, loss, validation accuracy, Try it |
| 2 | Fix a shape error | Simple MLP with a broken `fc2` | Tensor shapes, `in_features` and `out_features` |
| 3 | Find a good learning rate | Simple MLP | Learning rate too big, too small, and about right |
| 4 | Stop a model from overfitting | Wide MLP without dropout, on FashionMNIST | Overfitting, dropout, weight decay |
| 5 | Add BatchNorm and compare runs | LeNet-5 with sigmoid activations | Batch normalization, comparing runs |
| 6 | Why does a CNN beat an MLP? | Simple MLP on CIFAR-10 | Convolution, pooling, spatial structure |
| 7 | Why ReLU? Train a deep network | Six-layer sigmoid MLP, plain SGD | Activation functions, vanishing gradients |
| 8 | Shrink nanoGPT and see what it costs | A 4-block nanoGPT | Transformers, model size, validation loss |

Every lesson trains in a few minutes or less on a laptop CPU. Each one's claims (the sigmoid network stays near 10%, BatchNorm fixes it, and so on) were checked against real training runs.

## The lesson format

Lessons are plain data, defined in `app/src/lib/lessons/catalog.ts` with the types in `app/src/lib/lessons/types.ts`. A lesson survives `JSON.stringify` unchanged, so the same definition can drive the lesson panel, the tests, and the AI assistant when it tutors a lesson.

```ts
{
  id: "fix-shape-error",
  title: "Fix a shape error",
  summary: "One sentence for the lesson list.",
  goal: "What the learner is trying to do, shown at the top of the panel.",
  level: "beginner",            // or "intermediate"
  minutes: 5,
  concepts: ["tensor shapes", "in_features and out_features"],
  start: {
    template: "Simple MLP",      // a template from templates.ts
    dataset: "FashionMNIST",     // optional: swaps the template's dataset
    training: { epochs: 2 },     // optional: training settings after the template's defaults
    edits: [                     // optional: changes to the template's graph, in order
      { op: "set_param", target: { label: "fc2" }, param: "in_features", value: 500 },
    ],
  },
  steps: [
    {
      id: "fix",
      instruction: "One layer can't take the data it's given. Find it and fix it.",
      check: { kind: "all", checks: [
        { kind: "no_shape_errors", label: "No shape errors" },
        { kind: "connected", label: "Every layer is connected" },
      ] },
      hints: ["A nudge.", "More specific.", "The answer."],
      explain: "Shown when the step passes: what happened and why.",
      solution: [{ op: "set_param", target: { label: "fc2" }, param: "in_features", value: 512 }],
    },
  ],
}
```

### Picking layers

Edits and checks pick layers with a selector: `{ label: "fc2" }`, `{ moduleType: "Conv2d" }`, or both. Labels are the names shown on the canvas.

### Edits

Edits build a lesson's starting graph, and describe a step's solution.

| Edit | What it does |
|------|--------------|
| `set_param` | Sets a parameter on every matching layer |
| `remove_layer` | Removes every matching layer and connects its input to whatever it fed |
| `replace_layer` | Swaps the module of every matching layer, keeping its connections |
| `insert_after` | Adds a layer right after the one matching layer, taking over everything it fed |
| `set_training` | Sets a training setting, such as `optimizer.lr` or `epochs` |

### Checks

| Check | Passes when |
|-------|-------------|
| `no_shape_errors` | No layer has a shape error |
| `connected` | Every layer has an input and an output, and data reaches the loss |
| `layer_count` | The number of matching layers is within `min` and `max` |
| `param` | A parameter on every matching layer compares as given (`eq`, `ne`, `lt`, `lte`, `gt`, `gte`) |
| `followed_by` | Every layer matching `select` feeds straight into one matching `next` |
| `training` | A training setting compares as given |
| `run` | A run since the lesson started meets every condition in `where` |
| `manual` | The learner presses the step's button, for steps about looking at something |
| `all`, `any` | All, or any, of `checks` pass |

A `run` check's conditions can look at a metric (`bestValAcc`, `bestValLoss`, `trainValAccGap`, and others), compare it with the lesson's earlier runs (`than: "first"`, `"previous"` or `"every_previous"`), look at the run's training settings, or check the graph it trained with `graph`. Prefer comparisons with earlier runs over fixed numbers: training results vary between machines, but "the BatchNorm run beats the plain one" holds everywhere.

Give a check a `label` to show it as a line of the step's checklist. Leave the label out when the line would give the answer away.

A step stays done once it passes, even if later edits would make its check fail again.

## For the assistant

`app/src/lib/lessons/checks.ts` has what a tutor needs:

- `evaluate(check, context)` checks a step against a graph, the training settings, and the lesson's runs.
- `describeCheck(check)` puts a check in words, such as "in_features on fc2 is 512". It can give the answer away, so the panel shows labels instead.

`lessonContext()` in `app/src/store/lesson-store.ts` returns the lesson in progress as plain data: the lesson's goal and concepts, the current step's instruction, what it checks, whether it passes, and which hints the learner has already seen and which are left. A tutor should work through the remaining hints in order before it gives an answer.

## Adding a lesson

1. Add it to `LESSONS` in `catalog.ts`.
2. Give every step that changes the graph or settings a `solution`.
3. Run `npm test` in `app/`. The tests build each lesson's starting graph, check that it's connected and has no shape errors (apart from lessons about fixing them), and walk through every step to make sure none passes before its work is done and each passes once it is.
4. Train it for real in the app, and make sure what the `explain` text says actually happens.
