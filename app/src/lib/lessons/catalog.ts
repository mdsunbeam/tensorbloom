/**
 * The built-in lessons. Each one is plain data in the format described in types.ts and
 * docs/guides/lessons.md. Keep thresholds relative ("beat your first run") where training
 * numbers vary between machines, and absolute only where any reasonable run clears them.
 * Lessons use a constant learning rate (no scheduler) so runs are easy to compare, and must
 * train in a few minutes on a laptop CPU.
 */

import type { Lesson, LessonCheck } from "./types";

const valid: LessonCheck = {
  kind: "all",
  checks: [
    { kind: "no_shape_errors", label: "No shape errors" },
    { kind: "connected", label: "Every layer is connected" },
  ],
};

export const LESSONS: Lesson[] = [
  // ───────────────────────────────────────────────────────────────────────────
  {
    id: "first-run",
    title: "Train your first model",
    summary: "Train a small network to read handwritten digits, then see what it gets wrong.",
    goal: "Train a model that reads handwritten digits correctly at least 97% of the time.",
    level: "beginner",
    minutes: 5,
    concepts: ["training", "epoch", "loss", "validation accuracy"],
    start: { template: "Simple MLP", training: { scheduler: "None", epochs: 3 } },
    steps: [
      {
        id: "train",
        instruction: "This model takes a 28×28 image of a digit and guesses which digit it is. Press Train in the toolbar (or Ctrl+Enter) and watch the Charts tab.",
        check: { kind: "run", label: "A training run has finished" },
        hints: [
          "The Train button is at the right end of the toolbar.",
          "The first run downloads MNIST, which takes a moment. The status in the toolbar says what it's doing.",
        ],
        explain: "Each epoch is one pass over the 48,000 training images. Loss measures how wrong the guesses are, so it should fall. Validation accuracy is measured on 12,000 images the model never trains on, so it tells you how well the model works on new digits.",
      },
      {
        id: "accuracy",
        instruction: "Get validation accuracy to 97% or more. If your run is already there, you're done with this step.",
        check: { kind: "run", where: [{ metric: "bestValAcc", op: "gte", value: 0.97 }], label: "Validation accuracy of 97% or more" },
        hints: [
          "More passes over the data usually help a model this small.",
          "Raise Epochs in the Training tab (top of the right panel) and train again.",
          "Set Epochs to 5 and press Train.",
        ],
        explain: "Simple models get most digits right quickly. The last few percent are the hard, messy ones, which is where better architectures earn their keep.",
        solution: [{ op: "set_training", field: "epochs", value: 5 }],
      },
      {
        id: "mistakes",
        instruction: "Press Try it in the run summary, and switch to Mistakes. Look at a few: would you have read them correctly?",
        check: { kind: "manual", button: "I've looked at the mistakes" },
        hints: ["Try it is in the summary card at the top of the Charts tab, once a run has finished."],
        explain: "Many mistakes are digits a person would hesitate over too. Others are clear to you but not to the model, which only sees pixels in a row: it flattens the image first, so it has no idea which pixels are next to each other. The convolution lesson fixes exactly that.",
      },
    ],
  },

  // ───────────────────────────────────────────────────────────────────────────
  {
    id: "fix-shape-error",
    title: "Fix a shape error",
    summary: "Find and fix a layer that can't take the shape it's given, then change the model without breaking it.",
    goal: "Make every layer agree on the shape of the data passing between them.",
    level: "beginner",
    minutes: 5,
    concepts: ["tensor shapes", "in_features and out_features", "shape errors"],
    start: {
      template: "Simple MLP",
      training: { scheduler: "None", epochs: 2 },
      edits: [{ op: "set_param", target: { label: "fc2" }, param: "in_features", value: 500 }],
    },
    steps: [
      {
        id: "fix",
        instruction: "One layer in this model can't take the data it's given. Find it and fix it.",
        check: { kind: "all", checks: [valid, { kind: "layer_count", select: { moduleType: "Linear" }, min: 3 }] },
        hints: [
          "A red edge marks where the problem is, and the toolbar counts problems.",
          "Hover the red edge into fc2, or select fc2: it says what it gets and what it expects.",
          "fc1 puts out 512 numbers, but fc2's in_features is 500. Set fc2's in_features to 512, or press the fix in Properties.",
        ],
        explain: "A Linear layer's in_features must match the size of what comes in, which is the out_features of the Linear before it. The ReLU and Dropout in between don't change the size.",
        solution: [{ op: "set_param", target: { label: "fc2" }, param: "in_features", value: 512 }],
      },
      {
        id: "widen",
        instruction: "Make the first hidden layer wider: set fc1's out_features to 1024. Something else will break. Fix that too.",
        check: {
          kind: "all",
          checks: [
            { kind: "param", select: { label: "fc1" }, param: "out_features", op: "eq", value: 1024, label: "fc1 has 1024 outputs" },
            valid,
          ],
        },
        hints: [
          "Changing what one layer puts out changes what the next one has to take in.",
          "fc2 still expects 512 inputs.",
          "Set fc2's in_features to 1024.",
        ],
        explain: "Sizes ripple forward: change one layer's output and every layer up to the next Linear sees the new size. Wider layers can learn more, at the cost of more parameters (fc1 now has about 800,000).",
        solution: [
          { op: "set_param", target: { label: "fc1" }, param: "out_features", value: 1024 },
          { op: "set_param", target: { label: "fc2" }, param: "in_features", value: 1024 },
        ],
      },
      {
        id: "train",
        instruction: "Train the wider model to check it really works.",
        check: {
          kind: "run",
          where: [{ graph: { kind: "param", select: { label: "fc1" }, param: "out_features", op: "eq", value: 1024 } }],
          label: "The wider model has trained",
        },
        hints: ["Press Train (Ctrl+Enter)."],
        explain: "Shape checking happens before training, so a graph without shape errors trains without the dimension errors you'd otherwise only see when PyTorch runs it.",
      },
    ],
  },

  // ───────────────────────────────────────────────────────────────────────────
  {
    id: "learning-rate",
    title: "Find a good learning rate",
    summary: "Train with a learning rate that's far too big, then far too small, then find one that works.",
    goal: "See what the learning rate does, and pick one that beats both extremes.",
    level: "beginner",
    minutes: 8,
    concepts: ["learning rate", "optimizer", "divergence"],
    start: { template: "Simple MLP", training: { scheduler: "None", epochs: 2 } },
    steps: [
      {
        id: "too-big",
        instruction: "In the Training panel, set the learning rate to 1 and train.",
        check: { kind: "run", where: [{ training: "optimizer.lr", op: "gte", value: 0.5 }], label: "Trained with a learning rate of 0.5 or more" },
        hints: [
          "The learning rate is in the Training tab, at the top of the right panel.",
          "Set Learning rate to 1, then press Train.",
        ],
        explain: "The learning rate is how big a step the optimizer takes each time. Too big and every step overshoots, so the loss jumps around or climbs and accuracy sits near 10%, which is guessing.",
        solution: [{ op: "set_training", field: "optimizer.lr", value: 1 }],
      },
      {
        id: "too-small",
        instruction: "Now set it to 0.000001 and train again.",
        check: { kind: "run", where: [{ training: "optimizer.lr", op: "lte", value: 0.000005 }], label: "Trained with a learning rate of 0.000005 or less" },
        hints: ["Type 0.000001 (or 1e-6) into Learning rate, then press Train."],
        explain: "Tiny steps are safe but slow: the loss falls smoothly, but in the same two epochs it ends far behind where a good learning rate gets. It would catch up eventually, after many more epochs than you want to wait for.",
        solution: [{ op: "set_training", field: "optimizer.lr", value: 0.000001 }],
      },
      {
        id: "just-right",
        instruction: "Find a learning rate that beats both runs on validation accuracy.",
        check: { kind: "run", where: [{ metric: "bestValAcc", op: "gt", than: "every_previous" }], label: "A run that beats every earlier one" },
        hints: [
          "Good values are usually somewhere between the two you tried.",
          "For Adam, values around 0.001 are a common starting point.",
          "Set Learning rate to 0.001 and train.",
        ],
        explain: "There's a wide range of learning rates that work, and a cliff on either side. When a new model won't train, the learning rate is the first thing to check. The Runs tab shows all three runs on one chart.",
        solution: [{ op: "set_training", field: "optimizer.lr", value: 0.001 }],
      },
    ],
  },

  // ───────────────────────────────────────────────────────────────────────────
  {
    id: "overfitting",
    title: "Stop a model from overfitting",
    summary: "Watch a model memorize its training data, then use dropout or weight decay to make it generalize.",
    goal: "Close the gap between how the model does on its training data and on data it hasn't seen.",
    level: "intermediate",
    minutes: 12,
    concepts: ["overfitting", "generalization", "dropout", "weight decay"],
    start: {
      template: "Simple MLP",
      dataset: "FashionMNIST",
      training: { scheduler: "None", epochs: 15 },
      edits: [
        { op: "remove_layer", target: { moduleType: "Dropout" } },
        { op: "set_param", target: { label: "fc1" }, param: "out_features", value: 1024 },
        { op: "set_param", target: { label: "fc2" }, param: "in_features", value: 1024 },
      ],
    },
    steps: [
      {
        id: "baseline",
        instruction: "This model classifies photos of clothing. Train it, then compare the train and validation curves in Charts.",
        check: { kind: "run", label: "A training run has finished" },
        hints: ["Press Train. It takes 15 epochs, so give it a minute or two."],
        explain: "Training loss keeps falling all the way, but validation loss stops improving after about 10 epochs and starts to climb, ending at about twice the training loss. The model is memorizing the training images instead of learning what makes a shirt a shirt. That's overfitting.",
      },
      {
        id: "regularize",
        instruction: "Change the model or its settings so that it overfits less, and train again. Get a smaller gap between training and validation loss than the first run.",
        check: {
          kind: "run",
          where: [{ metric: "trainValLossGap", op: "lt", than: "first" }],
          label: "A smaller train/validation gap than the first run",
        },
        hints: [
          "Two common fixes: randomly switch off some neurons during training, or penalize large weights.",
          "Add a Dropout layer after relu1 and after relu2, or set Weight decay in the Training panel.",
          "Add a Dropout with p = 0.3 right after relu1 and another after relu2, then train.",
        ],
        explain: "Dropout makes the network work without any one neuron, so it can't lean on a few memorized details. Weight decay pulls the weights toward zero, which keeps the model simpler. Either way the model fits its training data less tightly, and validation loss stays down instead of climbing.",
        solution: [
          { op: "insert_after", target: { label: "relu1" }, module: "Dropout", label: "drop1", params: { p: 0.3 } },
          { op: "insert_after", target: { label: "relu2" }, module: "Dropout", label: "drop2", params: { p: 0.3 } },
        ],
      },
      {
        id: "compare",
        instruction: "Open the Runs tab and compare the two runs' validation loss curves, epoch by epoch.",
        check: { kind: "manual", button: "I've compared them" },
        hints: ["The Runs tab is next to Charts at the bottom. Both runs are picked for comparison already."],
        explain: "The first run's validation loss climbs in the last epochs while the regularized run's holds steady. Validation accuracy ends up about the same here, and that's common: regularization's main job is to make the numbers you measure on held-out data trustworthy, so the model does about as well on data it has never seen.",
      },
    ],
  },

  // ───────────────────────────────────────────────────────────────────────────
  {
    id: "batchnorm",
    title: "Add BatchNorm and compare runs",
    summary: "Put batch normalization into a classic CNN and see how much faster it learns.",
    goal: "Make LeNet-5 learn faster by normalizing between layers.",
    level: "intermediate",
    minutes: 10,
    concepts: ["batch normalization", "convolution", "comparing runs"],
    start: {
      template: "LeNet-5",
      training: { scheduler: "None", epochs: 3 },
      edits: [{ op: "replace_layer", target: { moduleType: "Tanh" }, module: "Sigmoid" }],
    },
    steps: [
      {
        id: "baseline",
        instruction: "This is LeNet-5, an early convolutional network, with sigmoid activations. Train it once as it is.",
        check: { kind: "run", label: "A training run has finished" },
        hints: ["Press Train."],
        explain: "Validation accuracy probably stayed near 10%, which is guessing. Sigmoid squashes values into 0 to 1, and as signals pass through several layers they drift into ranges where learning is very slow.",
      },
      {
        id: "add-bn",
        instruction: "Add a BatchNorm2d right after each Conv2d, so each convolution feeds a BatchNorm2d.",
        check: {
          kind: "all",
          checks: [
            { kind: "followed_by", select: { moduleType: "Conv2d" }, next: { moduleType: "BatchNorm2d" }, label: "Each Conv2d feeds a BatchNorm2d" },
            valid,
          ],
        },
        hints: [
          "Press Ctrl+K and search for BatchNorm2d, then connect it between the convolution and its activation.",
          "num_features must equal the number of channels coming in, which is the Conv2d's out_channels.",
          "conv1 has 6 output channels and conv2 has 16, so use BatchNorm2d(6) after conv1 and BatchNorm2d(16) after conv2.",
        ],
        explain: "BatchNorm rescales each channel to a steady range, batch by batch, before the activation sees it. It needs one setting: how many channels there are.",
        solution: [
          { op: "insert_after", target: { label: "conv1" }, module: "BatchNorm2d", label: "bn1", params: { num_features: 6 } },
          { op: "insert_after", target: { label: "conv2" }, module: "BatchNorm2d", label: "bn2", params: { num_features: 16 } },
        ],
      },
      {
        id: "train-bn",
        instruction: "Train the model with BatchNorm and beat the first run's validation accuracy.",
        check: {
          kind: "run",
          where: [
            { graph: { kind: "layer_count", select: { moduleType: "BatchNorm2d" }, min: 2 } },
            { metric: "bestValAcc", op: "gt", than: "first" },
          ],
          label: "The BatchNorm model beats the first run",
        },
        hints: ["Press Train. If it doesn't beat the first run, check that each BatchNorm2d sits between a Conv2d and its activation."],
        explain: "Same layers, same optimizer, same three epochs, and the BatchNorm model goes from guessing to getting most digits right. Normalizing between layers keeps the signals in a range where every layer can learn, which is why almost every modern CNN uses it.",
      },
      {
        id: "compare",
        instruction: "In the Runs tab, compare the two runs' validation accuracy epoch by epoch.",
        check: { kind: "manual", button: "I've compared them" },
        hints: ["The Runs tab overlays both runs on the same chart."],
        explain: "The plain run stays near 10%, which is guessing, for all three epochs. The BatchNorm run starts slowly too, then takes off. Faster, steadier training is the main thing BatchNorm buys you, and it matters most in models that would otherwise barely train.",
      },
    ],
  },

  // ───────────────────────────────────────────────────────────────────────────
  {
    id: "cnn-vs-mlp",
    title: "Why does a CNN beat an MLP?",
    summary: "Train a plain network on color photos, then add convolutions and see the difference.",
    goal: "Beat a fully connected network on CIFAR-10 by adding convolution layers.",
    level: "intermediate",
    minutes: 15,
    concepts: ["convolution", "pooling", "spatial structure", "parameter sharing"],
    start: {
      template: "Simple MLP",
      dataset: "CIFAR10",
      training: { scheduler: "None", epochs: 5 },
      edits: [{ op: "set_param", target: { label: "fc1" }, param: "in_features", value: 3072 }],
    },
    steps: [
      {
        id: "baseline",
        instruction: "This network classifies small color photos into 10 classes, like cats, ships and trucks. Train it as it is.",
        check: { kind: "run", label: "A training run has finished" },
        hints: ["Press Train. The first run downloads CIFAR-10, which is about 170 MB."],
        explain: "It does better than guessing (10%), but not by as much as on digits. Flatten turns the 32×32×3 photo into 3,072 numbers in a row, so the network has to learn separately that a cat's ear is a cat's ear in every position.",
      },
      {
        id: "add-conv",
        instruction: "Before Flatten, add a Conv2d, a ReLU and a MaxPool2d. Then fix whatever the new shapes break.",
        check: {
          kind: "all",
          checks: [
            { kind: "layer_count", select: { moduleType: "Conv2d" }, min: 1, label: "At least one Conv2d" },
            valid,
          ],
        },
        hints: [
          "Add the layers between the Data node and flatten: Conv2d, then ReLU, then MaxPool2d.",
          "The images have 3 channels, so in_channels is 3. With kernel_size 3 and padding 1 the image stays 32×32, and MaxPool2d(2) halves it to 16×16.",
          "Use Conv2d(in_channels 3, out_channels 32, kernel_size 3, padding 1), ReLU, MaxPool2d(kernel_size 2). Then fc1 gets 32 × 16 × 16 = 8192 inputs, so set its in_features to 8192.",
        ],
        explain: "A convolution slides the same small filter over every position, so a pattern it learns in one place works everywhere. Pooling shrinks the image so later layers see bigger areas.",
        solution: [
          { op: "insert_after", target: { moduleType: "Data" }, module: "Conv2d", label: "conv1", params: { in_channels: 3, out_channels: 32, kernel_size: 3, padding: 1 } },
          { op: "insert_after", target: { label: "conv1" }, module: "ReLU", label: "conv_relu" },
          { op: "insert_after", target: { label: "conv_relu" }, module: "MaxPool2d", label: "pool1", params: { kernel_size: 2 } },
          { op: "set_param", target: { label: "fc1" }, param: "in_features", value: 8192 },
        ],
      },
      {
        id: "train-conv",
        instruction: "Train the convolutional model and beat the first run's validation accuracy.",
        check: {
          kind: "run",
          where: [
            { graph: { kind: "layer_count", select: { moduleType: "Conv2d" }, min: 1 } },
            { metric: "bestValAcc", op: "gt", than: "first" },
          ],
          label: "The CNN beats the MLP",
        },
        hints: ["Press Train. If it's not better, check that the Conv2d comes before Flatten."],
        explain: "Even one convolution layer helps, because it builds in two facts about images: nearby pixels belong together, and a pattern means the same thing wherever it appears. Stack more conv layers and the gap grows, which is what LeNet-5 and ResNet-18 do.",
      },
    ],
  },

  // ───────────────────────────────────────────────────────────────────────────
  {
    id: "activations",
    title: "Why ReLU? Train a deep network",
    summary: "A deep network with sigmoid activations barely learns. Swap in ReLU and watch it take off.",
    goal: "Get a six-layer network to learn by changing its activation functions.",
    level: "intermediate",
    minutes: 8,
    concepts: ["activation functions", "vanishing gradients", "depth"],
    start: {
      template: "Simple MLP",
      training: { scheduler: "None", epochs: 3, "optimizer.type": "SGD", "optimizer.lr": 0.05, "optimizer.momentum": 0 },
      edits: [
        { op: "remove_layer", target: { moduleType: "Dropout" } },
        { op: "insert_after", target: { label: "relu2" }, module: "Linear", label: "fc3a", params: { in_features: 256, out_features: 256 } },
        { op: "insert_after", target: { label: "fc3a" }, module: "ReLU", label: "relu3" },
        { op: "insert_after", target: { label: "relu3" }, module: "Linear", label: "fc3b", params: { in_features: 256, out_features: 256 } },
        { op: "insert_after", target: { label: "fc3b" }, module: "ReLU", label: "relu4" },
        { op: "insert_after", target: { label: "relu4" }, module: "Linear", label: "fc3c", params: { in_features: 256, out_features: 256 } },
        { op: "insert_after", target: { label: "fc3c" }, module: "ReLU", label: "relu5" },
        { op: "replace_layer", target: { moduleType: "ReLU" }, module: "Sigmoid" },
      ],
    },
    steps: [
      {
        id: "baseline",
        instruction: "This network has six Linear layers with a Sigmoid after each hidden one. Train it with plain SGD.",
        check: { kind: "run", label: "A training run has finished" },
        hints: ["Press Train."],
        explain: "The loss hardly moves at first, and accuracy can sit near 10%. Sigmoid's slope is at most 0.25, so the signal that tells early layers how to change shrinks at every layer on the way back. With six layers, the first ones barely learn. That's the vanishing gradient problem.",
      },
      {
        id: "swap",
        instruction: "Replace every Sigmoid with a ReLU.",
        check: {
          kind: "all",
          checks: [
            { kind: "layer_count", select: { moduleType: "Sigmoid" }, max: 0, label: "No Sigmoid layers left" },
            { kind: "layer_count", select: { moduleType: "ReLU" }, min: 5, label: "Five ReLU layers" },
            valid,
          ],
        },
        hints: [
          "Delete a Sigmoid, add a ReLU (Ctrl+K), and connect it where the Sigmoid was.",
          "There are five Sigmoids, one after each Linear except the last.",
        ],
        explain: "ReLU passes positive values through unchanged, so its slope is 1 wherever it's active and the learning signal reaches the early layers at full strength.",
        solution: [{ op: "replace_layer", target: { moduleType: "Sigmoid" }, module: "ReLU" }],
      },
      {
        id: "train-relu",
        instruction: "Train the ReLU network and beat the sigmoid network's validation accuracy.",
        check: {
          kind: "run",
          where: [
            { graph: { kind: "layer_count", select: { moduleType: "Sigmoid" }, max: 0 } },
            { metric: "bestValAcc", op: "gt", than: "first" },
          ],
          label: "The ReLU network beats the sigmoid one",
        },
        hints: ["Press Train."],
        explain: "Same depth, same optimizer, same data, and only the activation changed. Switching to ReLU is a big part of why deep networks became trainable around 2010. Other fixes, like BatchNorm and skip connections, came later and attack the same problem.",
      },
    ],
  },

];

export function findLesson(id: string): Lesson | undefined {
  return LESSONS.find((l) => l.id === id);
}
