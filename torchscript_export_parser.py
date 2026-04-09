"""
produces three files:
  encoder.pt: (input_ids, attn_mask) -> encoder_hidden
  decoder_first.pt: first decode step, no past KV
              (dec_ids, enc_hidden, enc_mask) -> (logits, self_past, cross_past)
  decoder_step.pt: subsequent decode steps
              (dec_ids, enc_hidden, enc_mask, self_past, cross_past) -> (logits, self_past)

flan-t5-small dims:
  num_layers = 8, num_heads = 6, d_kv = 64, d_model = 512
"""

import os
import contextlib
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = "./scibowl_parser"
OUTPUT_DIR = "./scibowl_parser_ts"

NUM_LAYERS = 8
NUM_HEADS  = 6
D_KV       = 64
D_MODEL    = 512


# trace safety
def _simple_bidirectional_mask(config, inputs_embeds, attention_mask, **kwargs):
    extended = attention_mask[:, None, None, :].to(inputs_embeds.dtype)
    return (1.0 - extended) * torch.finfo(inputs_embeds.dtype).min


def _simple_causal_mask(config, inputs_embeds, attention_mask,
                         past_key_values_length=0, **kwargs):
    batch, q_len = inputs_embeds.shape[:2]
    kv_len = q_len + past_key_values_length
    return torch.zeros(batch, 1, q_len, kv_len,
                       dtype=inputs_embeds.dtype, device=inputs_embeds.device)


@contextlib.contextmanager
def _patch_t5_masks():
    import transformers.models.t5.modeling_t5 as t5_mod
    saved = {}
    for name, fn in [("create_bidirectional_mask", _simple_bidirectional_mask),
                     ("create_causal_mask",         _simple_causal_mask)]:
        if hasattr(t5_mod, name):
            saved[name] = getattr(t5_mod, name)
            setattr(t5_mod, name, fn)
    try:
        yield
    finally:
        for name, orig in saved.items():
            setattr(t5_mod, name, orig)


#cache helpers; apparently newer transformers use these instead of tuples??
def _make_cache(self_past, cross_past, num_layers):
    """
    Build an EncoderDecoderCache from two separate flat tensors.
      self_past  : (num_layers * 2, H, self_seq_len, D)  — grows each step
      cross_past : (num_layers * 2, H, src_len,      D)  — constant (encoder KV)
    """
    from transformers.cache_utils import DynamicCache, EncoderDecoderCache
    self_cache  = DynamicCache()
    cross_cache = DynamicCache()
    for i in range(num_layers):
        self_cache.update(self_past[i * 2], self_past[i * 2 + 1], layer_idx=i)
        cross_cache.update(cross_past[i * 2], cross_past[i * 2 + 1], layer_idx=i)
    return EncoderDecoderCache(self_cache, cross_cache)


def _extract_self_past(cache, num_layers):
    """Extract self-attention KV → (num_layers * 2, H, self_seq_len, D)."""
    return torch.stack([
        t for i in range(num_layers)
        for t in (cache.self_attention_cache.layers[i].keys,
                  cache.self_attention_cache.layers[i].values)
    ])


def _extract_cross_past(cache, num_layers):
    """Extract cross-attention KV → (num_layers * 2, H, src_len, D).  Static after step 1."""
    return torch.stack([
        t for i in range(num_layers)
        for t in (cache.cross_attention_cache.layers[i].keys,
                  cache.cross_attention_cache.layers[i].values)
    ])

class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class DecoderFirstWrapper(torch.nn.Module):
    """
    First decode step — no prior KV cache.
    Returns (logits, self_past, cross_past).
    cross_past is static (encoder KV); self_past grows each subsequent step.
    """
    def __init__(self, decoder, lm_head, num_layers: int):
        super().__init__()
        self.decoder    = decoder
        self.lm_head    = lm_head
        self.num_layers = num_layers

    def forward(
        self,
        decoder_input_ids:      torch.Tensor,
        encoder_hidden_states:  torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ):
        out = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=None,
            use_cache=True,
        )
        logits     = self.lm_head(out.last_hidden_state)
        self_past  = _extract_self_past(out.past_key_values, self.num_layers)
        cross_past = _extract_cross_past(out.past_key_values, self.num_layers)
        return logits, self_past, cross_past


class DecoderStepWrapper(torch.nn.Module):
    """
    Subsequent decode steps.
    Takes (dec_ids, enc_hidden, enc_mask, self_past, cross_past).
    Returns (logits, new_self_past).  cross_past is unchanged — keep it in Rust.
    """
    def __init__(self, decoder, lm_head, num_layers: int):
        super().__init__()
        self.decoder    = decoder
        self.lm_head    = lm_head
        self.num_layers = num_layers

    def forward(
        self,
        decoder_input_ids:      torch.Tensor,
        encoder_hidden_states:  torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        self_past:              torch.Tensor,
        cross_past:             torch.Tensor,
    ):
        cache = _make_cache(self_past, cross_past, self.num_layers)
        out = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
        logits    = self.lm_head(out.last_hidden_state)
        self_past = _extract_self_past(out.past_key_values, self.num_layers)
        return logits, self_past

def export():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    model.eval()

    dummy_input = tokenizer("hello world", return_tensors="pt", max_length=16, truncation=True)
    dummy_ids   = dummy_input["input_ids"]
    dummy_mask  = dummy_input["attention_mask"]

    enc_wrapper = EncoderWrapper(model.encoder)
    enc_wrapper.eval()

    print("Tracing encoder...")
    with _patch_t5_masks():
        traced_encoder = torch.jit.trace(enc_wrapper, (dummy_ids, dummy_mask))
    traced_encoder.save(f"{OUTPUT_DIR}/encoder.pt")
    print(f"  Saved {OUTPUT_DIR}/encoder.pt")

    with torch.no_grad():
        with _patch_t5_masks():
            dummy_enc_hidden = enc_wrapper(dummy_ids, dummy_mask)

    dec_first = DecoderFirstWrapper(model.decoder, model.lm_head, NUM_LAYERS)
    dec_first.eval()

    dummy_dec_ids = torch.zeros((1, 1), dtype=torch.long)  # T5 BOS token

    print("Tracing decoder_first...")
    with _patch_t5_masks():
        traced_dec_first = torch.jit.trace(
            dec_first,
            (dummy_dec_ids, dummy_enc_hidden, dummy_mask),
        )
    traced_dec_first.save(f"{OUTPUT_DIR}/decoder_first.pt")
    print(f"  Saved {OUTPUT_DIR}/decoder_first.pt")

    # get real self_past/cross_past shapes from stuff in first step
    with torch.no_grad():
        with _patch_t5_masks():
            _, dummy_self_past, dummy_cross_past = dec_first(
                dummy_dec_ids, dummy_enc_hidden, dummy_mask
            )

    dec_step = DecoderStepWrapper(model.decoder, model.lm_head, NUM_LAYERS)
    dec_step.eval()

    dummy_dec_ids_2 = torch.zeros((1, 1), dtype=torch.long)

    print("Tracing decoder_step...")
    with _patch_t5_masks():
        traced_dec_step = torch.jit.trace(
            dec_step,
            (dummy_dec_ids_2, dummy_enc_hidden, dummy_mask,
             dummy_self_past, dummy_cross_past),
        )
    traced_dec_step.save(f"{OUTPUT_DIR}/decoder_step.pt")
    print(f"  Saved {OUTPUT_DIR}/decoder_step.pt")

    print()
    for f in sorted(Path(OUTPUT_DIR).glob("*.pt")):
        print(f"  {f.name}  {f.stat().st_size / 1e6:.1f} MB")


def verify():
    import json as _json

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model_ref = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    model_ref.eval()

    encoder   = torch.jit.load(f"{OUTPUT_DIR}/encoder.pt")
    dec_first = torch.jit.load(f"{OUTPUT_DIR}/decoder_first.pt")
    dec_step  = torch.jit.load(f"{OUTPUT_DIR}/decoder_step.pt")

    with open("training_data.json") as f:
        samples = _json.load(f)

    for i, sample in enumerate(samples[:3]):
        prompt = sample["input"]
        expected = sample["output"]

        inputs    = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]

        with torch.no_grad():
            ref_ids = model_ref.generate(input_ids=input_ids,
                                         attention_mask=attn_mask,
                                         max_new_tokens=512)
        ref_out = tokenizer.decode(ref_ids[0], skip_special_tokens=True)

        enc_hidden = encoder(input_ids, attn_mask)
        dec_ids = torch.zeros((1, 1), dtype=torch.long)
        logits, self_past, cross_past = dec_first(dec_ids, enc_hidden, attn_mask)

        EOS = 1
        generated = []
        next_token = int(logits[0, -1].argmax())
        while next_token != EOS and len(generated) < 512:
            generated.append(next_token)
            dec_ids = torch.tensor([[next_token]], dtype=torch.long)
            logits, self_past = dec_step(dec_ids, enc_hidden, attn_mask, self_past, cross_past)
            next_token = int(logits[0, -1].argmax())
        ts_out = tokenizer.decode(generated, skip_special_tokens=True)

        print(f"\n--- Sample {i} ---")
        print(f"Expected : {expected[:100]}")
        print(f"Base model: {ref_out[:100]}")
        print(f"TorchScript: {ts_out[:100]}")


if __name__ == "__main__":
    export()
    verify()
