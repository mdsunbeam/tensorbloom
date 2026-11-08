# The Assistant

The assistant is a chat panel that can read the model on your canvas, its generated code, your training runs and your dataset's statistics. Ask it why a shape is wrong, what a layer does, where the parameters go, or why a run's validation loss went up.

Open it with the **Assistant** tab at the top of the right panel, or press **Ctrl+J**. Shape errors in the Properties panel also have an **Explain** button that opens the assistant with the question ready.

## Setting it up

The assistant uses Claude through your own Anthropic API key, billed to your Anthropic account.

1. Create a key at [console.anthropic.com](https://console.anthropic.com/settings/keys).
2. Open the assistant. The first time, it shows its settings. Paste the key and press **Save**.

The key is kept in your operating system's keychain (Keychain on macOS, Credential Manager on Windows, the Secret Service on Linux). Only the app's native side reads it; the interface never sees it again, and it's never written to project files. **Settings > Remove key** deletes it. When developing, the app also uses the `ANTHROPIC_API_KEY` environment variable if no key is saved.

**Settings** also picks the model: Claude Opus 5.5 (the default), Claude Sonnet 5.5 (faster and cheaper) or Claude Haiku 5.5 (fastest and cheapest).

## What it can do

The assistant can look at:

- The graph: every layer, its parameters, connections and output shapes
- Shape problems and the app's one-click fixes
- The layer catalog: every layer's parameters and what they mean
- The generated PyTorch code
- Runs: settings, per-epoch curves and results, and what changed between two runs
- A trained run's predictions on validation samples, including the mistakes (needs Python)
- Dataset statistics: sizes, sample shape, classes, input mean and range (needs Python)

And it can change the model when you ask: add, remove and connect layers, change their settings, apply shape fixes, group layers, tidy the layout, and change training settings. Try "add dropout after each Linear layer" or "make fc2 smaller and fix the shapes".

## Reviewing its changes

The assistant's changes show on the canvas as a preview while it works: added layers and connections in green, changed layers in amber, and removed ones as faded red outlines. A bar at the bottom of the canvas, and a card above the message box, list the changes and any new shape errors.

- **Apply** keeps them. All of them together are one undo step, so **Ctrl+Z** takes them back.
- **Discard** puts the model and training settings back exactly as they were.

While a preview is showing, the canvas can't be edited and undo is off. The assistant checks every layer type and setting against the catalog, and fixes the shape errors its changes cause before it hands them over. Opening another project or template keeps pending changes as an undo step.

## What's sent

Each message sends your question along with a snapshot of the app: the layers and their settings, shape problems, the selected layer, the training settings and a summary of recent runs. **See what's sent** under the message box shows that snapshot exactly. When the assistant asks for more, the app also sends the generated code, run curves, predicted classes and dataset statistics.

Dataset contents such as images and text stay on your computer.
