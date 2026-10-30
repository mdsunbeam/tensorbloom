/**
 * Plain-language hints for the errors people hit most often while training.
 * The raw message is still shown; the hint says what to try.
 */

const RULES: Array<{ match: RegExp; hint: (m: RegExpMatchArray) => string }> = [
  {
    match: /mat1 and mat2 shapes cannot be multiplied \((\d+)x(\d+) and (\d+)x(\d+)\)/,
    hint: (m) => `A Linear layer gets ${m[2]} features but its in_features is ${m[3]}. Set in_features to ${m[2]}, or check the layers before it on the canvas.`,
  },
  {
    match: /out of memory/i,
    hint: () => "The GPU ran out of memory. Try a smaller batch size, turn on mixed precision, or make the model smaller.",
  },
  {
    match: /Target -?\d+ is out of bounds|index \d+ is out of bounds for dimension/,
    hint: () => "The labels go higher than the number of outputs. Make the last layer's out_features equal to the number of classes.",
  },
  {
    match: /Expected input batch_size \((\d+)\) to match target batch_size \((\d+)\)/,
    hint: () => "The model's output and the labels have different batch sizes, usually because a Flatten or reshape merged the batch dimension. Check the shapes on the canvas.",
  },
  {
    match: /size of tensor a \((\d+)\) must match the size of tensor b \((\d+)\)/,
    hint: (m) => `Two tensors that are combined (added, compared to a target, …) have sizes ${m[1]} and ${m[2]}. Check the shapes going into the loss and any Add or Concat layers.`,
  },
  {
    match: /Given groups=\d+, weight of size \[[^\]]+\], expected input\[[^\]]+\] to have (\d+) channels, but got (\d+) channels/,
    hint: (m) => `A convolution expects ${m[1]} input channels but gets ${m[2]}. Set its in_channels to ${m[2]}.`,
  },
  {
    match: /No module named '([\w.]+)'/,
    hint: (m) => `The Python package "${m[1].split(".")[0]}" isn't installed in TensorBloom's Python environment. Install it with pip, then try again.`,
  },
  {
    match: /HTTP Error|URLError|urlopen error|getaddrinfo|Name or service not known|Temporary failure in name resolution|ConnectionError|timed out/i,
    hint: () => "The dataset couldn't be downloaded. Check your internet connection and try again.",
  },
  {
    match: /No such file or directory|FileNotFoundError/,
    hint: () => "A file the run needs wasn't found. Check the dataset path in the Data node.",
  },
  {
    match: /Sidecar (is )?not running|sidecar exited|Broken pipe/i,
    hint: () => "The Python process stopped. Restart TensorBloom and try again.",
  },
];

export function explainTrainingError(message: string): string | null {
  for (const rule of RULES) {
    const m = message.match(rule.match);
    if (m) return rule.hint(m);
  }
  return null;
}
