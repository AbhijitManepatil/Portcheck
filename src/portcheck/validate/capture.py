"""``capture_run`` and capture directory I/O (spec sections 7.5 and 8.5).

``torch`` and ``transformers`` are imported lazily inside ``capture_run`` so that
everything else in this module (and the rest of the package) works without them.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file

from portcheck.config import Config, load_config
from portcheck.envelope import InvalidInputError, MissingDependencyError, NotFoundError
from portcheck.fingerprint import env_fingerprint
from portcheck.paths import (
    create_run_dir,
    get_root,
    is_run_id,
    read_json,
    rel_to_root,
    resolve_in_root,
    runs_dir,
    write_json,
)
from portcheck.schemas import BenchRow, Capture
from portcheck.validate.workload import LoadedWorkload, load_workload

CAPTURE_JSON = "capture.json"
CAPTURE_TENSORS = "capture.safetensors"
# Plain greedy decoding, whatever the model's own generation_config says.
GREEDY = {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0}
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


# -------------------------------------------------------------------------- directory


def write_capture(
    directory: Path, capture: Capture, last_logits: np.ndarray, generated_ids: np.ndarray
) -> None:
    write_json(directory / CAPTURE_JSON, capture.model_dump(mode="json"))
    save_file(
        {
            "last_logits": np.ascontiguousarray(last_logits, dtype=np.float32),
            "generated_ids": np.ascontiguousarray(generated_ids, dtype=np.int64),
        },
        str(directory / CAPTURE_TENSORS),
    )


def resolve_capture(ref: str) -> tuple[str, Path]:
    """Find a capture by run id or by a path to a capture directory inside the root.

    Returns ``(name, directory)`` where name is the run id, or the root-relative path.
    """
    if not isinstance(ref, str) or not ref.strip():
        raise InvalidInputError("A capture must be named by a run id or a directory path.")
    ref = ref.strip()
    if is_run_id(ref):
        if not is_run_id(ref, "capture"):
            raise InvalidInputError(f"'{ref}' is not a capture run id.")
        directory = resolve_in_root(f".portcheck/runs/{ref}")
        if (directory / CAPTURE_JSON).is_file():
            return ref, directory
        raise NotFoundError(f"No capture with run id '{ref}' under .portcheck/runs.")
    directory = resolve_in_root(ref)
    if not (directory / CAPTURE_JSON).is_file():
        raise NotFoundError(
            f"'{ref}' is neither a capture run id nor a directory with {CAPTURE_JSON}."
        )
    name = directory.name if directory.parent == runs_dir() and is_run_id(directory.name) else None
    return name or rel_to_root(directory), directory


def load_capture(ref: str) -> tuple[str, Capture, Path]:
    name, directory = resolve_capture(ref)
    capture = Capture.model_validate(read_json(directory / CAPTURE_JSON))
    return name, capture, directory


def load_tensors(directory: Path) -> dict[str, np.ndarray]:
    path = directory / CAPTURE_TENSORS
    if not path.is_file():
        raise NotFoundError(f"Capture '{directory.name}' has no {CAPTURE_TENSORS}.")
    tensors = load_file(str(path))
    for key in ("last_logits", "generated_ids"):
        if key not in tensors:
            raise InvalidInputError(f"Capture '{directory.name}' is missing the tensor '{key}'.")
        if tensors[key].ndim != 2 or 0 in tensors[key].shape:
            raise InvalidInputError(
                f"Capture '{directory.name}': tensor '{key}' must be a non-empty 2-D array, "
                f"but has shape {tuple(tensors[key].shape)}."
            )
    if tensors["last_logits"].shape[0] != tensors["generated_ids"].shape[0]:
        raise InvalidInputError(
            f"Capture '{directory.name}': the two tensors have different numbers of prompts."
        )
    return tensors


# ---------------------------------------------------------------------------- capture


def _import_runtime() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise MissingDependencyError(
            f"capture needs torch and transformers ({exc.name or exc} is not installed). "
            "Install the `run` extra from your checkout of this repo with: pip install '.[run]' "
            "(install the PyTorch build that matches this machine's GPU first)."
        ) from exc
    return torch, transformers


def _load_model(transformers: Any, model_ref: str, revision: str | None, dtype: Any) -> Any:
    loader = transformers.AutoModelForCausalLM.from_pretrained
    try:
        major = int(str(transformers.__version__).split(".")[0])
    except ValueError:
        major = 0
    # transformers 5 renamed `torch_dtype` to `dtype` and warns on the old name.
    key = "dtype" if major >= 5 else "torch_dtype"
    return loader(model_ref, revision=revision, **{key: dtype})


def _eos_ids(model: Any) -> set[int]:
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        eos = getattr(model.config, "eos_token_id", None)
    if eos is None:
        return set()
    return {int(e) for e in eos} if isinstance(eos, (list, tuple)) else {int(eos)}


def _count_new_tokens(rows: np.ndarray, eos: set[int]) -> int:
    """New tokens per row, counted up to and including the first end-of-sequence id."""
    total = 0
    for row in rows:
        length = len(row)
        for index, token in enumerate(row):
            if int(token) in eos:
                length = index + 1
                break
        total += length
    return total


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def capture_run(
    workload: str, label: str, inject_fault: float | None = None, config: Config | None = None
) -> tuple[str, Capture, list[str]]:
    """Run the workload on this machine and write a capture directory.

    Returns ``(run_id, capture, warnings)``. This is the only function in the package
    that may use the network (the model download done by transformers for a hub id).
    """
    if not isinstance(label, str) or not _LABEL_RE.match(label):
        raise InvalidInputError(
            "label must be 1 to 64 characters: letters, digits, '.', '_' or '-', "
            "starting with a letter or digit."
        )
    if inject_fault is not None and not (
        isinstance(inject_fault, (int, float)) and inject_fault > 0
    ):
        raise InvalidInputError("inject_fault must be a positive standard deviation.")

    loaded: LoadedWorkload = load_workload(workload)
    wl = loaded.workload
    config = config or load_config(get_root())
    torch, transformers = _import_runtime()
    warnings: list[str] = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(wl.seed)
    dtype_name = wl.dtype
    if device == "cpu" and dtype_name != "float32":
        warnings.append(f"dtype {dtype_name} is not used on cpu; ran in float32 instead.")
        dtype_name = "float32"
    dtype = getattr(torch, dtype_name)

    try:
        model = _load_model(transformers, loaded.model_ref, wl.revision, dtype)
    except (OSError, ValueError) as exc:
        raise InvalidInputError(f"Could not load model '{wl.model}': {exc}") from exc
    model = model.to(device).eval()

    tokenizer = None
    pad_id = 0
    if loaded.prompt_form == "text":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            loaded.model_ref, revision=wl.revision
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        prompts = [tokenizer(text)["input_ids"] for text in loaded.texts]
        if any(len(p) == 0 for p in prompts):
            raise InvalidInputError("A text prompt tokenised to zero tokens.")
    else:
        prompts = loaded.input_ids
        vocab = int(model.config.vocab_size)
        if any(tok >= vocab for p in prompts for tok in p):
            raise InvalidInputError(
                f"A prompt has a token id outside the model's vocabulary ({vocab})."
            )
    limit = getattr(model.config, "max_position_embeddings", None)
    if isinstance(limit, int) and limit > 0:
        longest = max(len(p) for p in prompts)
        if longest + wl.max_new_tokens > limit:
            raise InvalidInputError(
                f"The longest prompt ({longest} tokens) plus max_new_tokens ({wl.max_new_tokens}) "
                f"exceeds the model's context length ({limit}). Shorten the prompt or lower "
                "max_new_tokens."
            )
    eos = _eos_ids(model)
    gen_pad_id = pad_id if tokenizer is not None else (min(eos) if eos else 0)

    if inject_fault is not None:
        embedding = model.get_input_embeddings().weight
        generator = torch.Generator(device="cpu").manual_seed(wl.seed)
        noise = torch.randn(embedding.shape, generator=generator) * float(inject_fault)
        with torch.no_grad():
            embedding.add_(noise.to(device=embedding.device, dtype=embedding.dtype))

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    num_prompts = len(prompts)
    logits_rows: list[np.ndarray] = []
    generated = np.full((num_prompts, wl.max_new_tokens), -1, dtype=np.int64)

    with torch.inference_mode():
        # Parity pass: batch size 1, one forward pass then greedy generation per prompt.
        for index, ids in enumerate(prompts):
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            mask = torch.ones_like(input_ids)
            out = model(input_ids=input_ids, attention_mask=mask)
            logits_rows.append(out.logits[0, -1].to(torch.float32).cpu().numpy())
            seq = model.generate(
                input_ids=input_ids,
                attention_mask=mask,
                **GREEDY,
                max_new_tokens=wl.max_new_tokens,
                pad_token_id=gen_pad_id,
            )
            new = seq[0, input_ids.shape[1] :].cpu().numpy().astype(np.int64)
            generated[index, : len(new)] = new

        # Bench pass: cycle through the prompts for each batch size.
        bench: list[BenchRow] = []
        warmup, measured = config.bench.warmup_batches, config.bench.measured_batches
        for batch_size in wl.batch_sizes:
            seconds: list[float] = []
            tokens = 0
            for batch_index in range(warmup + measured):
                start_at = batch_index * batch_size
                batch = [prompts[(start_at + k) % num_prompts] for k in range(batch_size)]
                width = max(len(p) for p in batch)
                ids_t = torch.full((batch_size, width), pad_id, dtype=torch.long)
                mask_t = torch.zeros((batch_size, width), dtype=torch.long)
                for row, p in enumerate(batch):  # left-pad
                    ids_t[row, width - len(p) :] = torch.tensor(p, dtype=torch.long)
                    mask_t[row, width - len(p) :] = 1
                ids_t, mask_t = ids_t.to(device), mask_t.to(device)
                sync()
                started = time.perf_counter()
                seq = model.generate(
                    input_ids=ids_t,
                    attention_mask=mask_t,
                    **GREEDY,
                    max_new_tokens=wl.max_new_tokens,
                    pad_token_id=gen_pad_id,
                )
                sync()
                elapsed = time.perf_counter() - started
                if batch_index >= warmup:
                    seconds.append(elapsed)
                    tokens += _count_new_tokens(seq[:, width:].cpu().numpy(), eos)
            total = float(sum(seconds))
            bench.append(
                BenchRow(
                    batch_size=batch_size,
                    tokens_generated=tokens,
                    seconds=_round(total),
                    tokens_per_second=_round(tokens / total if total > 0 else 0.0, 3),
                    latency_p50_s=_round(np.percentile(seconds, 50)),
                    latency_p95_s=_round(np.percentile(seconds, 95)),
                )
            )

    peak = int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
    capture = Capture(
        label=label,
        workload=wl.model_dump(mode="json"),
        workload_sha256=loaded.sha256,
        fingerprint=env_fingerprint(),
        num_prompts=num_prompts,
        fault_injected=inject_fault is not None,
        bench=bench,
        peak_memory_bytes=peak,
    )
    run_id, directory = create_run_dir("capture")
    write_capture(directory, capture, np.stack(logits_rows), generated)
    return run_id, capture, warnings
