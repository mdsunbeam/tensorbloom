"""Turns dataset samples and model outputs into the small JSON payloads used by Try it.

Why this file exists
--------------------
A PyTorch tensor cannot be sent to the UI directly: it is not JSON, and a full
language-model output would be enormous. The sidecar therefore describes each
tensor with a small tagged dictionary such as:

    {"kind": "image",  "png": "...", "width": 32, "height": 32}
    {"kind": "values", "values": [0.1, 0.2], "shape": [64, 65], "total": 4160}
    {"kind": "text",   "text": "the cat sat", "token_count": 12, "truncated": False}

This module owns those descriptions. Keeping them out of the sidecar has two
benefits: the presentation rules can be tested without loading a model, and new
formats (audio, segmentation, a real tokenizer) can add an encoder without editing
the training and prediction code.

What was moved from tensorbloom_sidecar.py
------------------------------------------
The image, values, class, reconstruction, label, and dataset-attribute helpers are
moved from the sidecar's old predict helpers, and the old describe() branches are
split into the output and target encoders below. Their JSON shapes are unchanged.
The text decoder, dataset semantics, and encoder pipeline are new.
"""
from __future__ import annotations

import base64
import io
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, TypedDict

import torch


# Keep one prediction card readable even if the model was trained on long sequences.
MAX_TEXT_TOKENS = 512
MAX_TEXT_CHARS = 4000
VALUES_LIMIT = 12

# Token ids produced by the built-in text datasets are 64-bit integers. Restricting
# the text input encoder to integer dtypes keeps ordinary float tensors numeric.
_TOKEN_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)


# ---------------------------------------------------------------------------
# Wire contracts
#
# These TypedDicts document the exact JSON the UI receives. They are also mirrored
# by the discriminated union in app/src/components/TryModelDialog.tsx.
# ---------------------------------------------------------------------------

class ImagePayload(TypedDict):
    kind: Literal["image"]
    png: str
    width: int
    height: int


class ValuesPayload(TypedDict):
    kind: Literal["values"]
    values: list[float]
    shape: list[int]
    total: int


class TextPayload(TypedDict):
    kind: Literal["text"]
    text: str
    token_count: int
    truncated: bool


class ClassGuess(TypedDict):
    index: int
    prob: float


class ClassesPayload(TypedDict):
    kind: Literal["classes"]
    top: list[ClassGuess]


class ClassTarget(TypedDict):
    kind: Literal["class"]
    index: int


Payload = ImagePayload | ValuesPayload | TextPayload
SampleValue = Payload | ClassesPayload | ClassTarget


class SamplePayload(TypedDict):
    input: Payload
    target: SampleValue | None
    output: SampleValue | None
    correct: bool | None


# ---------------------------------------------------------------------------
# Decoding token ids back to text
#
# The dataset knows how to turn text into ids for training. To show a prediction we
# need the reverse mapping. This is *not* the model's Embedding layer: an embedding
# maps ids to learned vectors, while this decoder maps ids back to words or letters.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecodedText:
    text: str
    token_count: int
    truncated: bool


class TokenDecoder(Protocol):
    """Turns token ids into text.

    A Hugging Face tokenizer needs merge and spacing rules that a plain join cannot
    reproduce, so encoders depend on this small interface rather than on the shape
    of a vocabulary dictionary.
    """

    @property
    def vocab_size(self) -> int: ...

    def decode(self, ids: Sequence[int]) -> DecodedText: ...


@dataclass(frozen=True)
class VocabDecoder:
    """Decodes the simple word and character vocabularies built by the datasets."""

    itos: tuple[str, ...]
    join: str = " "
    pad_id: int | None = None

    @classmethod
    def from_vocab(cls, vocab: Mapping[str, int]) -> "VocabDecoder":
        """Build an id-to-token table from the dictionaries used during training."""
        if not vocab:
            return cls(())
        # Ids are normally 0..n with no gaps, but using max() keeps the decoder safe
        # if a dataset ever reserves a sparse range. Unassigned slots stay as <i>.
        size = max(vocab.values()) + 1
        itos = [f"<{i}>" for i in range(size)]
        for token, idx in vocab.items():
            if 0 <= idx < size:
                itos[idx] = token
        return cls(tuple(itos), " ", vocab.get("<pad>"))

    @classmethod
    def from_chars(cls, chars: Sequence[str]) -> "VocabDecoder":
        """Character vocabularies join with no separator and have no padding id.

        In Tiny Shakespeare, id 0 is a real character (usually a newline), not
        <pad>. Treating it as padding would silently delete real text.
        """
        return cls(tuple(chars), "", None)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def decode(self, ids: Sequence[int]) -> DecodedText:
        seq = [int(i) for i in ids]

        # Word datasets pad every sequence to max_length. Remove only trailing pads:
        # they carry no text, and an internal pad would indicate an unusual batch.
        if self.pad_id is not None:
            while seq and seq[-1] == self.pad_id:
                seq.pop()

        token_count = len(seq)
        truncated = token_count > MAX_TEXT_TOKENS
        if truncated:
            seq = seq[:MAX_TEXT_TOKENS]

        # An id outside the vocabulary should never crash a prediction card. Show it
        # explicitly so the mismatch is visible instead of silently dropping data.
        tokens = [self.itos[i] if 0 <= i < len(self.itos) else f"<id {i}>" for i in seq]
        text = self.join.join(tokens)

        # Character-level text cannot be cut at a token boundary if it is unusually
        # long; cap the string as a second safety net.
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS].rstrip()
            truncated = True

        return DecodedText(text=text, token_count=token_count, truncated=truncated)


@dataclass(frozen=True)
class DatasetSemantics:
    """Presentation metadata that a dataset can attach for the Try it view.

    Today this carries a text decoder and optional class names. Datasets that do not
    attach it keep the old behavior: images show as pictures and everything else as
    numbers.
    """

    decoder: TokenDecoder | None = None
    class_names: tuple[str, ...] | None = None


EMPTY_SEMANTICS = DatasetSemantics()


def dataset_attr(ds, name: str):
    """An attribute of the dataset under any random_split or Subset wrappers.

    Training wraps datasets in random_split and _limit_samples wraps them in Subset.
    Both expose the original dataset as `.dataset`, so this walks down until the
    requested attribute is found.
    """
    while ds is not None:
        if hasattr(ds, name):
            return getattr(ds, name)
        ds = getattr(ds, "dataset", None)
    return None


def dataset_semantics(ds) -> DatasetSemantics:
    """The semantics attached to a dataset, with a fallback for image datasets."""
    attached = dataset_attr(ds, "semantics")
    classes = dataset_attr(ds, "classes")
    class_names = tuple(str(c) for c in classes) if classes else None
    if isinstance(attached, DatasetSemantics):
        # A dataset may attach a decoder but still rely on its existing .classes
        # attribute, as CIFAR and ImageFolder do. Merge the two sources.
        if attached.class_names is None and class_names is not None:
            return replace(attached, class_names=class_names)
        return attached
    return DatasetSemantics(class_names=class_names) if class_names is not None else EMPTY_SEMANTICS


# ---------------------------------------------------------------------------
# Basic payloads
#
# image_payload and values_payload are moved from the sidecar with the same output.
# ---------------------------------------------------------------------------

def is_image(x: torch.Tensor) -> bool:
    """Whether a tensor looks like a CHW image the card can draw as a PNG.

    Conditions, checked in order:
    - It is a torch tensor. Dataset items can hold non-tensor values such as PIL
      images, ints, or lists, and those must not reach the image path.
    - It has exactly three dimensions, so the shape reads as [C, H, W]. Token id
      sequences are 1-D and tabular feature rows are 2-D, so neither matches.
    - It has one or three channels, the grayscale and RGB layouts the PNG encoder
      knows how to turn into an array.
    - Height and width are at least 4. A tiny 3-D feature map or embedding is not a
      useful picture and is more likely to be another kind of data.
    """
    return (torch.is_tensor(x) and x.dim() == 3 and x.shape[0] in (1, 3)
            and x.shape[1] >= 4 and x.shape[2] >= 4)


def image_payload(x: torch.Tensor, normalization: Any | None) -> ImagePayload:
    """A [C, H, W] tensor as a small PNG, undoing the dataset's normalization."""
    from PIL import Image

    img = x.detach().float().cpu()
    # The card should show the image as it looked before Normalize was applied.
    if normalization is not None:
        img = img * torch.tensor(normalization.std).view(-1, 1, 1) + torch.tensor(normalization.mean).view(-1, 1, 1)
    elif img.min() < 0 or img.max() > 1:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    arr = (img.clamp(0, 1) * 255).round().to(torch.uint8)
    pil = Image.fromarray(arr[0].numpy() if arr.shape[0] == 1 else arr.permute(1, 2, 0).numpy())
    pil.thumbnail((160, 160))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return {"kind": "image", "png": base64.b64encode(buf.getvalue()).decode("ascii"),
            "width": pil.width, "height": pil.height}


def values_payload(t: torch.Tensor, limit: int = VALUES_LIMIT) -> ValuesPayload:
    """The first few numbers, plus the full shape and element count.

    The UI uses `shape` in the tooltip and `total` for the "… N more" suffix, so the
    card can show a 4,160-element tensor without sending 4,160 numbers.
    """
    # detach(): payloads never need gradients.
    # float(): integer token ids and float logits print through the same path.
    # cpu(): .tolist() only works on host memory, and building JSON should not
    #        read values back from a GPU or MPS device.
    # reshape(-1): flatten [C, H, W] or [T, V] into one list; `shape` keeps the
    #        original dimensions for the tooltip.
    flat = t.detach().float().cpu().reshape(-1)
    return {"kind": "values", "values": [round(v, 4) for v in flat[:limit].tolist()],
            "shape": list(t.shape), "total": flat.numel()}


def text_payload(decoded: DecodedText) -> TextPayload:
    return {"kind": "text", "text": decoded.text, "token_count": decoded.token_count,
            "truncated": decoded.truncated}


def _visual_payload(x: torch.Tensor, normalization: Any | None) -> Payload:
    """The old image-or-values choice, used for reconstructed inputs and outputs."""
    return image_payload(x, normalization) if is_image(x) else values_payload(x)


def _label_of(y) -> int | None:
    """The class index a target holds, or None when it isn't a single class.

    A language-model target has many ids, so it returns None here and is handled by
    the text target encoder instead. A scalar integer is a class index.
    """
    if isinstance(y, int):
        return y
    if torch.is_tensor(y) and y.numel() == 1 and not y.is_floating_point():
        return int(y.item())
    return None


def _token_ids_target(y) -> torch.Tensor | None:
    """A target that holds token ids, as opposed to one class index or a float target."""
    # A single-element target is a class index (or a scalar regression value), never a
    # token sequence. Keeping that check here prevents the two interpretations from
    # overlapping.
    if y is None or (torch.is_tensor(y) and y.numel() == 1):
        return None
    try:
        t = y if torch.is_tensor(y) else torch.as_tensor(y)
    except (TypeError, ValueError):
        return None
    if t.dtype not in _TOKEN_DTYPES or t.numel() <= 1:
        return None
    return t


# ---------------------------------------------------------------------------
# Encoder pipeline
#
# Each role has an ordered list of encoders. The first one whose applies() returns
# True builds the payload. The last encoder in each list is the numeric fallback, so
# a new format never breaks an existing model.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SampleContext:
    """The tensors and dataset metadata one encoder may look at."""

    x: torch.Tensor
    out: torch.Tensor | None = None
    target: Any | None = None
    semantics: DatasetSemantics = EMPTY_SEMANTICS
    is_reconstruction: bool = False
    normalization: Any | None = None
    # Set after the output is encoded. The target encoders need to know both the
    # payload kind and which encoder produced it: a reconstruction leaves its target
    # empty, while a normal numeric output keeps one.
    output_kind: str | None = None
    output_encoder: str | None = None


@dataclass(frozen=True)
class EncodedPayload:
    payload: SampleValue | None
    correct: bool | None = None
    # The name of the encoder that produced this payload. The target stage uses it to
    # distinguish a reconstruction output from an ordinary numeric output.
    encoder: str | None = None


@dataclass(frozen=True)
class PayloadEncoder:
    name: str
    applies: Callable[[SampleContext], bool]
    encode: Callable[[SampleContext], EncodedPayload]


# --- Inputs ---------------------------------------------------------------
# Old behavior: an image becomes a PNG, everything else becomes values.
# New behavior: integer token ids with a decoder become text first.

def _input_image_applies(ctx: SampleContext) -> bool:
    return is_image(ctx.x)


def _encode_input_image(ctx: SampleContext) -> EncodedPayload:
    return EncodedPayload(image_payload(ctx.x, ctx.normalization))


def _input_text_applies(ctx: SampleContext) -> bool:
    # A decoder alone is not enough: the tensor must actually contain ids. This keeps
    # float feature vectors from a text dataset's custom preprocessing numeric.
    return ctx.semantics.decoder is not None and ctx.x.dtype in _TOKEN_DTYPES


def _encode_input_text(ctx: SampleContext) -> EncodedPayload:
    decoder = ctx.semantics.decoder
    if decoder is None:  # defensive: applies() already checked
        return EncodedPayload(values_payload(ctx.x))
    ids = ctx.x.detach().cpu().reshape(-1).tolist()
    return EncodedPayload(text_payload(decoder.decode(ids)))


def _input_values_applies(_ctx: SampleContext) -> bool:
    # The total fallback: every prediction card must have an input payload.
    return True


def _encode_input_values(ctx: SampleContext) -> EncodedPayload:
    return EncodedPayload(values_payload(ctx.x))


# --- Outputs --------------------------------------------------------------
# Order matters. It mirrors the old describe() chain:
#   classes -> [new text logits] -> reconstruction -> values

def _output_classes_applies(ctx: SampleContext) -> bool:
    out = ctx.out
    return (not ctx.is_reconstruction and out is not None
            and out.dim() == 1 and out.numel() > 1)


def _encode_output_classes(ctx: SampleContext) -> EncodedPayload:
    """A 1-D class vector becomes the top three guesses and a right/wrong verdict."""
    out = ctx.out
    if out is None:  # defensive
        return EncodedPayload(None)
    probs = torch.softmax(out, dim=0)
    top = probs.topk(min(3, probs.numel()))
    payload: ClassesPayload = {
        "kind": "classes",
        "top": [{"index": int(i), "prob": round(float(p), 4)}
                for p, i in zip(top.values, top.indices)],
    }
    label = _label_of(ctx.target) if ctx.target is not None else None
    correct = int(top.indices[0]) == label if label is not None else None
    return EncodedPayload(payload, correct)


def _output_text_applies(ctx: SampleContext) -> bool:
    """Language-model logits: one score per vocabulary token at every position.

    The guards keep this narrow:
    - 2-D output, because a class vector is 1-D.
    - Last dimension equals the decoder's vocabulary size.
    - Number of positions equals the number of input tokens.
    - The target is not a single class index, which would indicate a classifier.
    """
    out = ctx.out
    decoder = ctx.semantics.decoder
    if out is None or decoder is None or decoder.vocab_size <= 1:
        return False
    if out.dim() != 2 or out.shape[-1] != decoder.vocab_size:
        return False
    if out.shape[0] != ctx.x.reshape(-1).numel():
        return False
    if ctx.target is None:
        return True
    if _label_of(ctx.target) is not None:
        return False
    tokens = _token_ids_target(ctx.target)
    return tokens is not None and tokens.numel() == out.shape[0]


def _encode_output_text(ctx: SampleContext) -> EncodedPayload:
    """Keep the model's best token at each position and decode the whole sequence."""
    out, decoder = ctx.out, ctx.semantics.decoder
    if out is None or decoder is None:  # defensive
        return EncodedPayload(values_payload(out) if out is not None else None)
    # argmax is deterministic: the Try it card shows the top-1 prediction, not a sample.
    ids = out.detach().argmax(dim=-1).cpu().tolist()
    return EncodedPayload(text_payload(decoder.decode(ids)))


def _output_reconstruction_applies(ctx: SampleContext) -> bool:
    # Same condition as the old reconstruction branch: the output has exactly as many
    # elements as the input, so it can be shown in the input's shape.
    return (ctx.is_reconstruction and ctx.out is not None
            and ctx.out.numel() == ctx.x.numel())


def _encode_output_reconstruction(ctx: SampleContext) -> EncodedPayload:
    out = ctx.out
    if out is None:  # defensive
        return EncodedPayload(None)
    return EncodedPayload(_visual_payload(out.reshape(ctx.x.shape), ctx.normalization))


def _output_values_applies(ctx: SampleContext) -> bool:
    # The total fallback for any output no other encoder recognized.
    return ctx.out is not None


def _encode_output_values(ctx: SampleContext) -> EncodedPayload:
    out = ctx.out
    return EncodedPayload(values_payload(out) if out is not None else None)


# --- Targets --------------------------------------------------------------
# The old describe() treated targets differently by branch:
#   - classes: a class index only if the target is one class
#   - reconstruction: no target at all
#   - everything else: a class index if possible, otherwise numbers
# The conditions below preserve that behavior and add token text for text output.

def _target_class_applies(ctx: SampleContext) -> bool:
    # Reconstruction samples have never shown a target, whether they were shown as an
    # image or as numbers. A scalar label excludes the text and value encoders by
    # running first.
    return ctx.output_encoder != "reconstruction" and _label_of(ctx.target) is not None


def _encode_target_class(ctx: SampleContext) -> EncodedPayload:
    label = _label_of(ctx.target)
    if label is None:  # defensive
        return EncodedPayload(None)
    return EncodedPayload({"kind": "class", "index": label})


def _target_text_applies(ctx: SampleContext) -> bool:
    # Only decode a target as text when the text encoder produced the output. This
    # prevents per-token class labels from being mistaken for words in a future
    # token-classification model.
    return (ctx.output_encoder == "text" and ctx.semantics.decoder is not None
            and _token_ids_target(ctx.target) is not None)


def _encode_target_text(ctx: SampleContext) -> EncodedPayload:
    decoder = ctx.semantics.decoder
    tokens = _token_ids_target(ctx.target)
    if decoder is None or tokens is None:  # defensive
        return EncodedPayload(None)
    return EncodedPayload(text_payload(decoder.decode(tokens.cpu().reshape(-1).tolist())))


def _target_values_applies(ctx: SampleContext) -> bool:
    # Matches the old else branch: numbers for a non-class target. Reconstruction is
    # excluded because its target has always been left empty, even when the
    # reconstructed output itself is numeric.
    return (ctx.output_kind in ("values", "text")
            and ctx.output_encoder != "reconstruction"
            and ctx.target is not None)


def _encode_target_values(ctx: SampleContext) -> EncodedPayload:
    return EncodedPayload(values_payload(torch.as_tensor(ctx.target)))


# ---------------------------------------------------------------------------
# Registries and orchestration
# ---------------------------------------------------------------------------

INPUT_ENCODERS: tuple[PayloadEncoder, ...] = (
    PayloadEncoder("image", _input_image_applies, _encode_input_image),
    PayloadEncoder("text", _input_text_applies, _encode_input_text),
    PayloadEncoder("values", _input_values_applies, _encode_input_values),
)

OUTPUT_ENCODERS: tuple[PayloadEncoder, ...] = (
    PayloadEncoder("classes", _output_classes_applies, _encode_output_classes),
    PayloadEncoder("text", _output_text_applies, _encode_output_text),
    PayloadEncoder("reconstruction", _output_reconstruction_applies, _encode_output_reconstruction),
    PayloadEncoder("values", _output_values_applies, _encode_output_values),
)

TARGET_ENCODERS: tuple[PayloadEncoder, ...] = (
    PayloadEncoder("class", _target_class_applies, _encode_target_class),
    PayloadEncoder("text", _target_text_applies, _encode_target_text),
    PayloadEncoder("values", _target_values_applies, _encode_target_values),
)

_ENCODERS = {"input": INPUT_ENCODERS, "output": OUTPUT_ENCODERS, "target": TARGET_ENCODERS}


def _encode(role: str, ctx: SampleContext) -> EncodedPayload:
    for encoder in _ENCODERS[role]:
        if encoder.applies(ctx):
            result = encoder.encode(ctx)
            # Tag the result with the encoder name so later stages can tell a
            # reconstruction output from an ordinary numeric output.
            return EncodedPayload(result.payload, result.correct, encoder.name)
    return EncodedPayload(None)


def describe_sample(x: torch.Tensor, out: torch.Tensor | None, y,
                    *, semantics: DatasetSemantics, is_reconstruction: bool,
                    normalization: Any | None = None) -> SamplePayload:
    """Describe one prediction with the payload shape the Try it view renders.

    The output is encoded first because the target rules depend on what kind of
    output was produced: a text output decodes a token target, a class output
    carries a class target, and a reconstruction output carries no target.
    """
    base = SampleContext(x=x, semantics=semantics, normalization=normalization)

    input_enc = _encode("input", base)
    output_enc = _encode("output", replace(base, out=out, target=y,
                                           is_reconstruction=is_reconstruction))
    output_kind = output_enc.payload["kind"] if output_enc.payload is not None else None
    target_enc = _encode("target", replace(base, out=out, target=y,
                                           is_reconstruction=is_reconstruction,
                                           output_kind=output_kind,
                                           output_encoder=output_enc.encoder))

    # Values is a total fallback for inputs and outputs, but this keeps the types
    # honest if an encoder is ever changed.
    input_payload = input_enc.payload if input_enc.payload is not None else values_payload(x)

    return {
        "input": input_payload,
        "target": target_enc.payload,
        "output": output_enc.payload,
        "correct": output_enc.correct,
    }
