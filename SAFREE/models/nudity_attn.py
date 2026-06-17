"""
Nudity cross-attention suppression (training-free).

A different lever than SAFREE's text-embedding projection: instead of editing the text
vector once before the UNet, this down-weights the *cross-attention* that image regions
pay to nudity-associated tokens, at every UNet cross-attention layer and every denoising
step. So it suppresses where/how nudity is spatially realized, not just the prompt vector.

Per-token suppression weights w in [0,1] (1 = keep, 0 = fully suppress that token's
attention) are derived from how much each token embedding lies in the nudity subspace P_c.

Important: suppression is applied only to the conditional CFG branch(es) (branch index >=
suppress_from_branch), never the unconditional branch -- otherwise the reweighting cancels
in the guidance difference (uncond + s*(cond - uncond)) and has no effect.
"""
import torch
from diffusers.models.attention_processor import Attention, AttnProcessor


class NudityAttnProcessor:
    def __init__(self, token_weights, heads, suppress_from_branch=1):
        self.token_weights = token_weights          # [num_tokens], in [0,1]
        self.heads = heads
        self.suppress_from_branch = suppress_from_branch

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kwargs):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        is_cross = encoder_hidden_states is not None
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        kv = encoder_hidden_states if is_cross else hidden_states
        if attn.norm_cross and is_cross:
            kv = attn.norm_encoder_hidden_states(kv)
        key = attn.to_k(kv)
        value = attn.to_v(kv)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attn_probs = attn.get_attention_scores(query, key, attention_mask)  # [B*heads, q, tokens]

        if is_cross and self.token_weights is not None and attn_probs.shape[-1] == self.token_weights.shape[-1]:
            bh = attn_probs.shape[0]
            w = self.token_weights.to(dtype=attn_probs.dtype, device=attn_probs.device)  # [tokens]
            branch = (torch.arange(bh, device=attn_probs.device) // self.heads)           # [B*heads]
            suppress = (branch >= self.suppress_from_branch).view(bh, 1, 1)               # bool mask
            full = torch.where(suppress, w.view(1, 1, -1), torch.ones_like(w).view(1, 1, -1))
            attn_probs = attn_probs * full
            attn_probs = attn_probs / attn_probs.sum(-1, keepdim=True).clamp(min=1e-8)

        hidden_states = torch.bmm(attn_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(1, 2).view(b, c, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def register_nudity_attn(unet, token_weights, suppress_from_branch=1):
    """Install the suppression processor on every cross-attention (attn2) module."""
    for name, module in unet.named_modules():
        if isinstance(module, Attention) and name.endswith("attn2"):
            module.set_processor(NudityAttnProcessor(token_weights, module.heads, suppress_from_branch))


def clear_nudity_attn(unet):
    """Restore default attention processors on cross-attention modules."""
    for name, module in unet.named_modules():
        if isinstance(module, Attention) and name.endswith("attn2"):
            module.set_processor(AttnProcessor())
