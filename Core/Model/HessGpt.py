# Core/Model/HessGpt.py — Naylis v1
"""
NaylisGPT — ~200M paramètres
  vocab_size  : 49152  (cosmo2-tokenizer)
  embed_dim   : 768
  num_heads   : 12
  num_layers  : 18
  n_kv_heads  : 4      (GQA 3:1)
  rel_rank    : 32     (canaux Naylis par head)
  max_seq_len : 1024
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Attention'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'TransformerBlock'))

from attention import RMSNorm, KVCache
from transformer_block import NaylisBlock


class NaylisGPT(nn.Module):
    def __init__(
        self,
        vocab_size            : int   = 49_152,
        embed_dim             : int   = 768,
        num_heads             : int   = 12,
        num_layers            : int   = 12,
        max_seq_len           : int   = 1024,
        dropout               : float = 0.0,
        use_rope              : bool  = True,
        use_yarn              : bool  = False,
        yarn_scale            : float = 1.0,
        yarn_original_max_len : int   = 1024,
        use_swiglu            : bool  = True,
        n_kv_heads            : Optional[int]   = 4,
        use_qk_norm           : bool  = True,
        soft_cap              : Optional[float] = None,
        use_flash_attn        : bool  = True,
        rel_rank              : int   = 32,
    ):
        super().__init__()

        # ── Validations ──────────────────────────────────────────
        assert vocab_size > 0
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) % num_heads ({num_heads}) != 0"
        if n_kv_heads is not None:
            assert num_heads % n_kv_heads == 0, \
                f"num_heads ({num_heads}) % n_kv_heads ({n_kv_heads}) != 0"

        # ── Attributs ────────────────────────────────────────────
        self.vocab_size            = vocab_size
        self.embed_dim             = embed_dim
        self.num_heads             = num_heads
        self.num_layers            = num_layers
        self.max_seq_len           = max_seq_len
        self.n_kv_heads            = n_kv_heads
        self.rel_rank              = rel_rank

        # ── Embeddings ───────────────────────────────────────────
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.dropout          = nn.Dropout(dropout)

        # ── Blocs Naylis ─────────────────────────────────────────
        self.blocks = nn.ModuleList([
            NaylisBlock(
                embed_dim             = embed_dim,
                num_heads             = num_heads,
                dropout               = dropout,
                use_rope              = use_rope,
                max_seq_len           = max_seq_len,
                use_yarn              = use_yarn,
                yarn_scale            = yarn_scale,
                yarn_original_max_len = yarn_original_max_len,
                use_swiglu            = use_swiglu,
                n_kv_heads            = n_kv_heads,
                use_qk_norm           = use_qk_norm,
                use_flash_attn        = use_flash_attn,
                soft_cap              = soft_cap,
                rel_rank              = rel_rank,
            )
            for _ in range(num_layers)
        ])

        # ── Norm finale + head ───────────────────────────────────
        self.ln_final    = RMSNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying token_embeddings ↔ output_head
        self.output_head.weight = self.token_embeddings.weight

        # ── Init ─────────────────────────────────────────────────
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    # ─────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids    : torch.Tensor,
        targets      : Optional[torch.Tensor]    = None,
        pad_token_id : Optional[int]             = None,
        past_kv      : Optional[List[KVCache]]   = None,
        use_kv_cache : bool                      = False,
        cu_seqlens_q : Optional[torch.Tensor]    = None,
        cu_seqlens_k : Optional[torch.Tensor]    = None,
        max_seqlen_q : Optional[int]             = None,
        max_seqlen_k : Optional[int]             = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[KVCache]]]:

        B, S = input_ids.shape
        x    = self.dropout(self.token_embeddings(input_ids))

        # BF16 sur GPU
        if x.device.type == 'cuda' and x.dtype == torch.float32:
            x = x.to(torch.bfloat16)

        new_past_kv: Optional[List[KVCache]] = [] if use_kv_cache else None

        for i, block in enumerate(self.blocks):
            layer_past = past_kv[i] if past_kv is not None else None
            x, new_kv  = block(
                x,
                past_kv      = layer_past,
                use_kv_cache = use_kv_cache,
                cu_seqlens_q = cu_seqlens_q,
                cu_seqlens_k = cu_seqlens_k,
                max_seqlen_q = max_seqlen_q,
                max_seqlen_k = max_seqlen_k,
            )
            if use_kv_cache:
                new_past_kv.append(new_kv)

        x      = self.ln_final(x)
        logits = self.output_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index = -100,
            )

        return logits, loss, new_past_kv

    # ─────────────────────────────────────────────────────────────
    # Génération — KV Cache + top_k + top_p
    # ─────────────────────────────────────────────────────────────
    def generate(
        self,
        input_ids     : torch.Tensor,
        max_new_tokens: int                          = 50,
        temperature   : float                        = 1.0,
        top_k         : Optional[int]                = None,
        top_p         : Optional[float]              = None,
        eos_token_id  : Optional[int | List[int]]    = None,  # accepte int ou liste
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()

        # Normalise eos_token_id en set pour le test d'appartenance O(1)
        if eos_token_id is None:
            eos_ids = None
        elif isinstance(eos_token_id, int):
            eos_ids = {eos_token_id}
        else:
            eos_ids = set(eos_token_id)

        with torch.no_grad():
            if input_ids.size(1) > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]

            prefill_logits, _, past_kv = self.forward(input_ids, use_kv_cache=True)
            next_logits = prefill_logits[:, -1, :]

            for _ in range(max_new_tokens):
                if temperature == 0.0:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = next_logits / temperature
                    if top_k is not None:
                        k_         = min(top_k, logits.size(-1))
                        topk_v, _  = torch.topk(logits, k_)
                        logits     = logits.masked_fill(logits < topk_v[:, [-1]], float('-inf'))
                    if top_p is not None and top_p < 1.0:
                        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
                        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        remove    = (cum_probs - F.softmax(sorted_logits, dim=-1)) >= top_p
                        sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
                        logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)
                    next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)

                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Arrêt si le token généré est dans eos_ids (int ou liste)
                if eos_ids is not None and next_token.item() in eos_ids:
                    break

                decode_logits, _, past_kv = self.forward(
                    next_token, past_kv=past_kv, use_kv_cache=True)
                next_logits = decode_logits[:, -1, :]

        if was_training:
            self.train()
        return input_ids

    # ─────────────────────────────────────────────────────────────
    # Utilitaires
    # ─────────────────────────────────────────────────────────────
    def count_parameters(self) -> dict:
        total   = sum(p.numel() for p in self.parameters())
        naylis  = sum(
            p.numel()
            for b in self.blocks
            for name, p in b.attention.named_parameters()
            if 'rel_q_proj' in name or 'rel_k_proj' in name or 'graph_scale' in name
        )
        return {
            'total_M'    : round(total   / 1e6, 2),
            'naylis_K'   : round(naylis  / 1e3, 1),
            'naylis_pct' : f'{naylis / total * 100:.2f}%',
        }

    def get_config(self) -> dict:
        return {
            'vocab_size'  : self.vocab_size,
            'embed_dim'   : self.embed_dim,
            'num_heads'   : self.num_heads,
            'num_layers'  : self.num_layers,
            'max_seq_len' : self.max_seq_len,
            'n_kv_heads'  : self.n_kv_heads,
            'rel_rank'    : self.rel_rank,
        }

    def resize_token_embeddings(self, new_vocab_size: int):
        if new_vocab_size == self.vocab_size:
            return
        old_emb = self.token_embeddings
        # Nouveau embedding — copie les poids existants, init normal pour les nouveaux tokens
        new_emb = nn.Embedding(new_vocab_size, self.embed_dim)
        nn.init.normal_(new_emb.weight, mean=0.0, std=0.02)
        n = min(old_emb.num_embeddings, new_vocab_size)
        with torch.no_grad():
            new_emb.weight.data[:n] = old_emb.weight.data[:n]
        self.token_embeddings = new_emb
        # output_head recréé PUIS tied — ne pas appliquer _init_weights après
        self.output_head        = nn.Linear(self.embed_dim, new_vocab_size, bias=False)
        self.output_head.weight = self.token_embeddings.weight   # weight tying préservé
        self.vocab_size         = new_vocab_size
