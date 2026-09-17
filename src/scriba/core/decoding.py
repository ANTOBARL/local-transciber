"""Chunk decoding with a confidence score.

qwen-asr's `_infer_asr` returns only text. To tell real speech from room chatter we also need how sure
the model was: the mean log-probability of the tokens it generated. Clear speech scores around -0.1;
background chatter, where the model is guessing, usually below -0.3.

Implemented for the transformers backend (greedy decoding: the chosen token is the arg-max) and for
vLLM (per-token log-probabilities). Any other model object falls back to `_infer_asr` without a score.
"""

from __future__ import annotations

from typing import Any

from scriba.utils.logging import get_logger

log = get_logger("decoding")

EOS_TOKEN_IDS = (151645, 151643)


def decode(model: Any, contexts: list[str], wavs: list, languages: list) -> tuple[list[str], list[float | None]]:
    """Raw model outputs and their mean token log-probability (None when unavailable)."""
    backend = getattr(model, "backend", None)
    try:
        if backend == "transformers" and hasattr(model, "processor"):
            return _decode_transformers(model, contexts, wavs, languages)
        if backend == "vllm" and getattr(model, "sampling_params", None) is not None:
            return _decode_vllm(model, contexts, wavs, languages)
    except (TypeError, AttributeError) as exc:  # an upstream API change: keep transcribing without scores
        log.warning("Confidence scores unavailable (%s); decoding without them", exc)
    return list(model._infer_asr(contexts, wavs, languages)), [None] * len(wavs)


def _decode_transformers(model, contexts, wavs, languages):
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    class ChosenLogProb(LogitsProcessor):
        """Records log p(arg-max token) at every step; leaves the scores untouched."""

        def __init__(self) -> None:
            self.steps: list[torch.Tensor] = []

        def __call__(self, input_ids, scores):
            self.steps.append(torch.log_softmax(scores.float(), dim=-1).max(dim=-1).values)
            return scores

    prompts = [model._build_text_prompt(context=c, force_language=fl) for c, fl in zip(contexts, languages)]
    batch = model.max_inference_batch_size
    batch = len(prompts) if batch is None or batch < 0 else batch
    texts: list[str] = []
    scores: list[float | None] = []
    for i in range(0, len(prompts), batch):
        inputs = model.processor(text=prompts[i:i + batch], audio=wavs[i:i + batch], return_tensors="pt",
                                 padding=True)
        inputs = inputs.to(model.model.device).to(model.model.dtype)
        recorder = ChosenLogProb()
        with torch.no_grad():
            out = model.model.generate(**inputs, max_new_tokens=model.max_new_tokens,
                                       logits_processor=LogitsProcessorList([recorder]))
        new = out.sequences[:, inputs["input_ids"].shape[1]:]
        steps = torch.stack(recorder.steps, dim=1).cpu() if recorder.steps else None
        texts.extend(model.processor.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False))
        for k, ids in enumerate(new.tolist()):
            n = next((j for j, t in enumerate(ids) if t in EOS_TOKEN_IDS), len(ids))
            scores.append(float(steps[k, :n].mean()) if steps is not None and n > 0 else None)
    return texts, scores


def _decode_vllm(model, contexts, wavs, languages):
    params = model.sampling_params.clone()
    params.logprobs = 0
    requests = [{"prompt": model._build_text_prompt(context=c, force_language=fl), "multi_modal_data": {"audio": [w]}}
                for c, w, fl in zip(contexts, wavs, languages)]
    outputs = model.model.generate(requests, sampling_params=params, use_tqdm=False)
    texts, scores = [], []
    for o in outputs:
        best = o.outputs[0]
        texts.append(best.text)
        n = len(best.token_ids)
        cumulative = getattr(best, "cumulative_logprob", None)
        scores.append(float(cumulative) / n if cumulative is not None and n else None)
    return texts, scores
