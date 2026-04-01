#!/usr/bin/env python3
"""
sft.py — Naylis v1
===================
SFT full (tous paramètres) de NaylisGPT sur un mix de datasets curatés.

DATASETS :
  - Smol-Magpie-Ultra     (50k) : cœur du dialogue
  - Smol-Constraints      (10k) : IFEval / instructions
  - OpenHermes 2.5        (20k) : polyvalence
  - Self-OSS-Starcoder2   (10k) : logique de code

SPÉCIFICITÉS vs pretrain :
  - max_seq_len 1024 (aligné pretrain)
  - Loss masquée : on ne calcule la loss que sur les tokens assistant
  - Optimiseur : Muon+MARS (blocs) + AdamW (embed/norms)
  - Scheduler : WSD (warmup=5%, stable, cosine decay 20%)
  - Checkpoint : reprise automatique depuis ./Model/naylis_sft.pt
  - Pretrain chargé depuis ./Model/naylis_pretrain.pt

FORMAT ChatML attendu :
  <|im_start|>user
  ...
  <|im_end|>
  <|im_start|>assistant
  ...
  <|im_end|>

USAGE :
  python sft.py
  python sft.py --no-compile
  python sft.py --epochs 2 --lr 1.5e-4
"""

import os
os.environ["TORCHINDUCTOR_CACHE_DIR"]      = "./CompileCache"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.makedirs("./CompileCache", exist_ok=True)

import sys
import time
import math
import json
import gc
import traceback
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Optional, List, Tuple

torch.set_float32_matmul_precision('high')

# ── Paths Core ────────────────────────────────────────────────────────────────
_root = os.path.dirname(__file__)
sys.path.append(os.path.join(_root, 'Core', 'Model'))
sys.path.append(os.path.join(_root, 'Core', 'Attention'))
sys.path.append(os.path.join(_root, 'Core', 'FeedForward'))
sys.path.append(os.path.join(_root, 'Core', 'TransformerBlock'))

from HessGpt import NaylisGPT
from attention import KVCache

# ── Args ──────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--no-compile',   action='store_true')
    p.add_argument('--compile-mode', default='default',
                   choices=['default', 'reduce-overhead', 'max-autotune'])
    p.add_argument('--epochs',       type=int,   default=1)
    p.add_argument('--lr',           type=float, default=1.5e-4)
    p.add_argument('--batch-size',   type=int,   default=4)
    p.add_argument('--grad-acc',     type=int,   default=8)
    return p.parse_args()

ARGS   = get_args()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    # Modèle — mêmes hyperparams que pretrain
    'vocab_size'            : None,         # rempli après tokenizer
    'embed_dim'             : 768,
    'num_heads'             : 12,
    'num_layers'            : 18,
    'max_seq_len'           : 1024,         # aligné avec le pretrain
    'dropout'               : 0.0,
    'use_rope'              : True,
    'use_yarn'              : False,
    'yarn_scale'            : 1.0,
    'yarn_original_max_len' : 1024,         # longueur pretrain de référence
    'use_swiglu'            : True,
    'n_kv_heads'            : 4,
    'use_qk_norm'           : True,
    'soft_cap'              : None,
    'use_flash_attn'        : True,
    'rel_rank'              : 32,
    # Training
    'batch_size'            : ARGS.batch_size,
    'gradient_accumulation' : ARGS.grad_acc,
    'max_grad_norm'         : 1.0,
    'learning_rate'         : ARGS.lr,
    'weight_decay'          : 0.05,
    'adam_beta1'            : 0.9,
    'adam_beta2'            : 0.95,
    'adam_eps'              : 1e-8,
    'num_epochs'            : ARGS.epochs,  # 1 epoch suffit sur 60K samples
    # Data — mix de 4 datasets (90k total)
    'datasets'              : {
        'HuggingFaceTB/smol-magpie-ultra'                : 50_000,   # cœur du dialogue
        'HuggingFaceTB/smol-constraints'                 : 10_000,   # IFEval / instructions
        'teknium/OpenHermes-2.5'                         : 20_000,   # polyvalence
        'bigcode/self-oss-instruct-sc2-exec-filter-50k'  : 10_000,   # logique de code
    },
    'val_split_ratio'       : 0.02,         # 2% → validation
    'warmup_ratio'          : 0.05,
    'decay_ratio'           : 0.20,
    'min_lr_ratio'          : 0.1,
    # Validation
    'validate_every_steps'  : 200,
    'val_batches'           : 40,
    'save_every_steps'      : 500,
    # Checkpoint
    'pretrain_ckpt'         : './Model/naylis_pretrain.pt',
    'sft_ckpt'              : './Model/naylis_sft.pt',
    # Compile
    'use_compile'           : not ARGS.no_compile,
    'compile_mode'          : ARGS.compile_mode,
    # DataLoader
    'num_workers'           : 1,
    # Tokens spéciaux ChatML
    'im_start'              : '<|im_start|>',
    'im_end'                : '<|im_end|>',
    'role_user'             : 'user',
    'role_assistant'        : 'assistant',
}

print('=' * 70)
print('  Naylis v1 — SFT (Magpie-Ultra / Constraints / OpenHermes / Starcoder2)')
print('=' * 70)
if DEVICE == 'cuda':
    print(f'  GPU  : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB')
    cap = torch.cuda.get_device_capability()
    print(f'  SM   : {cap[0]}{cap[1]}')
print(f'  embed={CONFIG["embed_dim"]}  layers={CONFIG["num_layers"]}  '
      f'heads={CONFIG["num_heads"]}  kv={CONFIG["n_kv_heads"]}  '
      f'max_seq={CONFIG["max_seq_len"]}  YaRN={CONFIG["use_yarn"]}')


# ── Tokenizer ─────────────────────────────────────────────────────────────────
print('\nTokenizer...')
tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/cosmo2-tokenizer')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
CONFIG['vocab_size'] = len(tokenizer)
EOS_ID = tokenizer.eos_token_id
print(f'  vocab={len(tokenizer)}  eos={EOS_ID}')

# Tokens ChatML — natifs du cosmo2-tokenizer, toujours présents
IM_START_ID = tokenizer.convert_tokens_to_ids(CONFIG['im_start'])
IM_END_ID   = tokenizer.convert_tokens_to_ids(CONFIG['im_end'])
print(f'  im_start_id={IM_START_ID}  im_end_id={IM_END_ID}')

# IDs pour les rôles (texte brut, tokenisé séparément)
def encode_no_special(text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)

USER_IDS      = encode_no_special(CONFIG['role_user'])
ASSISTANT_IDS = encode_no_special(CONFIG['role_assistant'])
NEWLINE_IDS   = encode_no_special('\n')


# ── Tokenisation ChatML avec masque de loss ───────────────────────────────────
def tokenize_conversation(messages: List[dict]) -> Tuple[List[int], List[int]]:
    """
    Convertit une liste de messages en tokens + masque.
    Masque = 0 sur les tokens user/système, 1 sur les tokens assistant.
    Retourne (input_ids, loss_mask) — même longueur.

    Format attendu par message : {'role': str, 'content': str}
    """
    input_ids  : List[int] = []
    loss_mask  : List[int] = []

    for msg in messages:
        role    = msg.get('role', 'user')
        content = msg.get('content', '')

        # Header : <|im_start|>role\n
        header = [IM_START_ID] + encode_no_special(role) + NEWLINE_IDS
        # Body   : content<|im_end|>\n
        body   = encode_no_special(content) + [IM_END_ID] + NEWLINE_IDS

        is_assistant = (role == 'assistant')

        # Header → jamais dans la loss
        input_ids += header
        loss_mask += [0] * len(header)

        # Body → dans la loss seulement pour l'assistant
        input_ids += body
        loss_mask += [int(is_assistant)] * len(body)

    return input_ids, loss_mask


# ── Dataset SFT ───────────────────────────────────────────────────────────────
class SFTDataset(Dataset):
    """
    Charge smol-smoltalk, tokenise et tronque à max_seq_len.
    Garde le masque de loss pour ne backprop que sur les réponses assistant.
    """
    def __init__(
        self,
        examples  : List[dict],
        max_seq_len: int,
    ):
        self.max_seq_len = max_seq_len
        self.data: List[Tuple[List[int], List[int]]] = []

        skipped = 0
        for ex in examples:
            messages = ex.get('messages', [])
            if not messages:
                skipped += 1
                continue

            ids, mask = tokenize_conversation(messages)

            # Tronquer
            if len(ids) > max_seq_len + 1:
                ids  = ids[:max_seq_len + 1]
                mask = mask[:max_seq_len + 1]

            # Ignorer les exemples sans tokens assistant dans la fenêtre
            if sum(mask) == 0:
                skipped += 1
                continue

            self.data.append((ids, mask))

        if skipped:
            print(f'  ⚠️  {skipped} exemples ignorés (vides ou 0 token assistant)')

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        ids, mask = self.data[idx]
        ids  = torch.tensor(ids,  dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.long)
        return ids, mask


def sft_collate_fn(batch, pad_id: int, max_seq_len: int):
    """
    Padding à droite jusqu'à max_seq_len.
    Retourne x, y, loss_weight où loss_weight masque les tokens non-assistant
    et les positions padding.
    """
    ids_list, mask_list = zip(*batch)
    B     = len(ids_list)
    # +1 pour le décalage x → y
    L_max = min(max(len(t) for t in ids_list), max_seq_len + 1)

    x_pad = torch.full((B, L_max), pad_id, dtype=torch.long)
    y_pad = torch.full((B, L_max), -100,   dtype=torch.long)   # -100 = ignore_index
    w_pad = torch.zeros(B, L_max, dtype=torch.float)

    for i, (ids, mask) in enumerate(zip(ids_list, mask_list)):
        L = min(len(ids), L_max)
        x_pad[i, :L] = ids[:L]

        # y = décalage de 1 (prochaine token)
        if L > 1:
            y_tgt = ids[1:L]
            m_tgt = mask[1:L]
            # Appliquer le masque : remplacer les positions non-assistant par -100
            y_tgt = torch.where(m_tgt.bool(), y_tgt, torch.full_like(y_tgt, -100))
            y_pad[i, :L - 1] = y_tgt
            w_pad[i, :L - 1] = m_tgt.float()

    # x = tous sauf le dernier token ; y/w = tous sauf le premier
    return x_pad[:, :-1], y_pad[:, :-1], w_pad[:, :-1]


# ── Normalisation formats datasets ────────────────────────────────────────────
def normalize_example(ex: dict, ds_name: str) -> Optional[dict]:
    """
    Unifie les formats de datasets vers {'messages': [{'role':..., 'content':...}]}.

    - smol-magpie-ultra, smol-constraints, self-oss-starcoder2 : champ 'messages' natif
    - OpenHermes 2.5 : champ 'conversations' avec clés from/value (ShareGPT)
    """
    # Format natif ChatML — déjà bon
    if 'messages' in ex and ex['messages']:
        return ex

    # Format ShareGPT (OpenHermes 2.5)
    if 'conversations' in ex and ex['conversations']:
        role_map = {'human': 'user', 'gpt': 'assistant', 'system': 'system',
                    'human_turn': 'user', 'model_turn': 'assistant'}
        msgs = []
        for turn in ex['conversations']:
            role    = role_map.get(turn.get('from', '').lower(), 'user')
            content = turn.get('value', '') or turn.get('content', '')
            if content:
                msgs.append({'role': role, 'content': content})
        if msgs:
            return {'messages': msgs}

    return None


# ── Chargement multi-datasets SFT ─────────────────────────────────────────────
def load_sft_datasets() -> Tuple[List[dict], List[dict]]:
    from datasets import load_dataset

    print(f'\n  Chargement datasets SFT...')
    all_examples: List[dict] = []

    for ds_name, n_samples in CONFIG['datasets'].items():
        print(f'  → {ds_name} ({n_samples:,} exemples)...')
        try:
            ds    = load_dataset(ds_name, split='train')
            total = len(ds)
            n     = min(n_samples, total)
            ds    = ds.shuffle().select(range(n))

            added, skipped = 0, 0
            for ex in ds:
                normed = normalize_example(dict(ex), ds_name)
                if normed:
                    all_examples.append(normed)
                    added += 1
                else:
                    skipped += 1

            print(f'     {added:,} ajoutés  |  {skipped} ignorés  '
                  f'(total dispo : {total:,})')
        except Exception as e:
            print(f'  ⚠️  Erreur {ds_name} : {e}')

    total = len(all_examples)
    print(f'\n  Total combiné  : {total:,} exemples')

    # Split train / val
    val_n          = max(1, int(total * CONFIG['val_split_ratio']))
    train_n        = total - val_n
    train_examples = all_examples[:train_n]
    val_examples   = all_examples[train_n:]

    print(f'  Train          : {train_n:,}')
    print(f'  Val            : {val_n:,}')
    return train_examples, val_examples


# ── WSD Scheduler ─────────────────────────────────────────────────────────────
class WSDScheduler:
    def __init__(self, optimizers, max_lr, total_steps,
                 warmup_ratio=0.05, decay_ratio=0.20, min_lr_ratio=0.1):
        self.optimizers   = optimizers if isinstance(optimizers, list) else [optimizers]
        self.max_lr       = max_lr
        self.min_lr       = max_lr * min_lr_ratio
        self.total_steps  = total_steps
        self.warmup_steps = max(1, int(total_steps * warmup_ratio))
        self.decay_steps  = max(1, int(total_steps * decay_ratio))
        self.stable_steps = max(0, total_steps - self.warmup_steps - self.decay_steps)
        self.current_step = 0

    def get_lr(self) -> float:
        s = self.current_step
        if s < self.warmup_steps:
            return self.max_lr * (s / self.warmup_steps)
        elif s < self.warmup_steps + self.stable_steps:
            return self.max_lr
        else:
            d = s - self.warmup_steps - self.stable_steps
            p = min(d / max(self.decay_steps, 1), 1.0)
            return self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * p))

    def step(self) -> float:
        lr = self.get_lr()
        self.current_step += 1
        for opt in self.optimizers:
            for pg in opt.param_groups:
                pg['lr'] = lr * 5.0 if pg.get('is_muon', False) else lr
        return lr

    def get_last_lr(self): return [self.get_lr()]
    def state_dict(self):  return {'current_step': self.current_step}
    def load_state_dict(self, sd): self.current_step = sd.get('current_step', 0)


# ── Muon + MARS-M (identique pretrain) ───────────────────────────────────────
def _zeropower_via_newtonschulz5(G, steps: int = 5):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + 1e-7)
    if G.size(0) > G.size(1): X = X.T
    for _ in range(steps):
        A = X @ X.T; B = b * A + c * (A @ A); X = a * X + B @ X
    if G.size(0) > G.size(1): X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=3, weight_decay=0.0, use_mars=True, mars_gamma=0.025):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                     ns_steps=ns_steps, weight_decay=weight_decay,
                                     use_mars=use_mars, mars_gamma=mars_gamma))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom, nest = group['lr'], group['momentum'], group['nesterov']
            ns, wd        = group['ns_steps'], group['weight_decay']
            use_mars, mg  = group.get('use_mars', True), group.get('mars_gamma', 0.025)
            for p in group['params']:
                if p.grad is None or p.grad.ndim < 2: continue
                g     = p.grad
                state = self.state[p]
                if use_mars:
                    if 'prev_grad' not in state:
                        state['prev_grad'] = torch.zeros_like(g)
                    prev = state['prev_grad']
                    c_t  = torch.clamp(
                        (mg / (1. - mg)) * (g.norm() + 1e-8) / (prev.norm() + 1e-8),
                        max=1.0)
                    g = g + c_t * (g - prev)
                    state['prev_grad'].copy_(p.grad)
                if 'buf' not in state:
                    state['buf'] = torch.zeros_like(g)
                buf = state['buf']
                buf.mul_(mom).add_(g)
                g   = (g + mom * buf) if nest else buf
                g   = _zeropower_via_newtonschulz5(g, steps=ns)
                g   = g * max(g.size(0), g.size(1)) ** 0.5
                if wd: p.mul_(1. - lr * wd)
                p.add_(g, alpha=-lr)


def configure_optimizers(model, lr: float, weight_decay: float, betas, eps):
    EXCLUDE = {'token_embeddings.weight', 'output_head.weight'}
    muon_params, adamw_decay, adamw_nodecay = [], [], []
    for pn, p in model.named_parameters():
        if not p.requires_grad: continue
        if pn in EXCLUDE:
            adamw_nodecay.append(p); continue
        if p.ndim >= 2 and 'norm' not in pn.lower() and 'embed' not in pn.lower():
            muon_params.append(p)
        elif p.ndim >= 2:
            adamw_decay.append(p)
        else:
            adamw_nodecay.append(p)

    muon_opt = Muon(
        [{'params': muon_params, 'is_muon': True}],
        lr=lr * 5.0, momentum=0.95, weight_decay=weight_decay,
    )
    adamw_opt = torch.optim.AdamW(
        [{'params': adamw_decay,   'weight_decay': weight_decay},
         {'params': adamw_nodecay, 'weight_decay': 0.0}],
        lr=lr, betas=betas, eps=eps,
    )
    muon_param_count  = sum(p.numel() for p in muon_params)
    adamw_param_count = sum(p.numel() for p in adamw_decay) + \
                        sum(p.numel() for p in adamw_nodecay)
    print(f'  Muon+MARS  : {muon_param_count / 1e6:.2f}M params  lr={lr * 5.0:.2e}')
    print(f'  AdamW      : {adamw_param_count / 1e6:.2f}M params  lr={lr:.2e}')
    return muon_opt, adamw_opt


# ── Checkpoint ────────────────────────────────────────────────────────────────
class CheckpointManager:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    def save(self, model, optimizers, scheduler, metadata: dict):
        m         = model._orig_mod if hasattr(model, '_orig_mod') else model
        muon, adamw = optimizers
        cp = {
            'model_state_dict'    : m.state_dict(),
            'muon_state_dict'     : muon.state_dict(),
            'adamw_state_dict'    : adamw.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        info_path = self.path.replace('.pt', '_info.json')
        info = {**metadata, 'last_save': datetime.now().isoformat(), 'config': CONFIG}
        tmp_json = info_path + '.tmp'
        with open(tmp_json, 'w') as f:
            json.dump(info, f, indent=2, default=str)
        tmp_pt = self.path + '.tmp'
        torch.save(cp, tmp_pt)
        os.replace(tmp_pt, self.path)
        os.replace(tmp_json, info_path)
        print(f'  💾 SAVE  step={metadata["global_step"]:,}  [{self.path}]')

    def load(self) -> Optional[dict]:
        if not os.path.exists(self.path):
            return None
        print(f'\nCheckpoint SFT trouvé : {self.path}')
        cp        = torch.load(self.path, map_location='cpu', weights_only=False)
        info_path = self.path.replace('.pt', '_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
            for k in ('global_step', 'current_epoch', 'total_training_time'):
                cp[k] = info.get(k, 0)
        else:
            cp.update({'global_step': 0, 'current_epoch': 1, 'total_training_time': 0.0})
        return cp


# ── Validation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, val_loader, max_batches: int = 40) -> Tuple[float, float]:
    """
    Calcule la val loss en ne comptant que les tokens assistant (même masque que train).
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0
    ae  = (DEVICE == 'cuda')
    adt = torch.bfloat16 if ae else torch.float32
    try:
        for i, (x, y, w) in enumerate(val_loader):
            if i >= max_batches: break
            x, y, w = x.to(DEVICE), y.to(DEVICE), w.to(DEVICE)
            with torch.amp.autocast(DEVICE, dtype=adt, enabled=ae):
                logits, _, _ = model(x)
                # Loss pondérée par le masque assistant
                loss = F.cross_entropy(
                    logits.reshape(-1, CONFIG['vocab_size']),
                    y.reshape(-1),
                    ignore_index=-100,
                    reduction='none',
                )
                # Moyenne sur les tokens assistant uniquement
                w_flat = w.reshape(-1)
                n_tok  = w_flat.sum().clamp(min=1)
                loss   = (loss * w_flat).sum() / n_tok
            total_loss   += loss.item()
            total_tokens += 1
    finally:
        model.train()
    avg = total_loss / max(total_tokens, 1)
    return math.exp(min(avg, 10)), avg


# ── Training loop ─────────────────────────────────────────────────────────────
def train_epoch(
    model,
    train_loader,
    val_loader,
    optimizers,
    scheduler,
    ckpt_mgr,
    history,
    global_step: int,
    total_time : float,
    epoch      : int,
    total_steps: int,
) -> Tuple[int, float]:

    muon_opt, adamw_opt = optimizers
    ae  = (DEVICE == 'cuda')
    adt = torch.bfloat16 if ae else torch.float32

    model.train()
    acc_steps   = 0
    epoch_loss  = 0.0
    epoch_toks  = 0
    valid_steps = 0
    t0          = time.time()

    pbar = tqdm(
        total        = len(train_loader),
        desc         = f'  E{epoch}/{CONFIG["num_epochs"]}',
        unit         = 'it',
        dynamic_ncols= True,
        colour       = 'green',
    )

    muon_opt.zero_grad(set_to_none=True)
    adamw_opt.zero_grad(set_to_none=True)
    lr = scheduler.get_lr()   # lr initial dès le 1er batch

    try:
        for step_in_epoch, (x, y, w) in enumerate(train_loader):
            x, y, w = x.to(DEVICE), y.to(DEVICE), w.to(DEVICE)

            try:
                with torch.amp.autocast(DEVICE, dtype=adt, enabled=ae):
                    logits, _, _ = model(x)
                    loss_raw = F.cross_entropy(
                        logits.reshape(-1, CONFIG['vocab_size']),
                        y.reshape(-1),
                        ignore_index=-100,
                        reduction='none',
                    )
                    w_flat  = w.reshape(-1)
                    n_tok   = w_flat.sum().clamp(min=1)
                    loss    = (loss_raw * w_flat).sum() / n_tok

                # Normaliser par gradient_accumulation
                (loss / CONFIG['gradient_accumulation']).backward()
                acc_steps += 1

                epoch_loss += loss.detach().item()
                epoch_toks += int(n_tok.item())
                valid_steps += 1

            except torch.cuda.OutOfMemoryError:
                pbar.write(f'\n  OOM — skip batch {step_in_epoch}')
                torch.cuda.empty_cache()
                muon_opt.zero_grad(set_to_none=True)
                adamw_opt.zero_grad(set_to_none=True)
                acc_steps = 0
                gc.collect()
                model.train()
                pbar.update(1)
                continue

            if acc_steps >= CONFIG['gradient_accumulation']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
                muon_opt.step()
                adamw_opt.step()
                lr = scheduler.step()
                muon_opt.zero_grad(set_to_none=True)
                adamw_opt.zero_grad(set_to_none=True)
                acc_steps = 0
                global_step += 1

                # Validation
                if global_step % CONFIG['validate_every_steps'] == 0:
                    ppl_v, vloss = validate(model, val_loader, CONFIG['val_batches'])
                    pbar.write(f'  [val  step={global_step:,}] '
                               f'loss={vloss:.4f}  ppl={ppl_v:.2f}')
                    history.setdefault('validations', []).append({
                        'step': global_step, 'val_loss': vloss, 'val_ppl': ppl_v})

                # Checkpoint
                if global_step % CONFIG['save_every_steps'] == 0:
                    elapsed = time.time() - t0
                    ckpt_mgr.save(model, optimizers, scheduler, {
                        'global_step': global_step,
                        'current_epoch': epoch,
                        'total_training_time': total_time + elapsed,
                    })

                # Signal Naylis — toutes les 100 steps
                if global_step % 100 == 0:
                    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
                    scales = [b.attention.graph_scale.detach().abs().mean().item()
                              for b in raw.blocks]
                    avg_s = sum(scales) / len(scales)
                    g_max = max(scales)
                    g_min = min(scales)
                    pbar.write(
                        f'  [naylis step={global_step:,}] '
                        f'|graph_scale| avg={avg_s:.5f}  min={g_min:.5f}  max={g_max:.5f}'
                    )

            # Postfix mis à jour à chaque batch (lr visible dès le début)
            avg_loss = epoch_loss / max(valid_steps, 1)
            ppl      = math.exp(min(avg_loss, 10))
            pbar.set_postfix(
                step =f'{global_step}/{total_steps}',
                loss =f'{avg_loss:.4f}',
                ppl  =f'{ppl:.1f}',
                lr   =f'{lr:.2e}',
            )
            pbar.update(1)

    except KeyboardInterrupt:
        pbar.close()
        elapsed = time.time() - t0
        total_time += elapsed
        ckpt_mgr.save(model, optimizers, scheduler, {
            'global_step': global_step, 'current_epoch': epoch,
            'total_training_time': total_time,
        })
        raise

    pbar.close()
    elapsed = time.time() - t0
    total_time += elapsed
    avg_loss = epoch_loss / max(valid_steps, 1)
    print(f'\n  Epoch {epoch} terminée | loss={avg_loss:.4f} | '
          f'ppl={math.exp(min(avg_loss, 10)):.2f} | {elapsed / 60:.1f}min')

    history.setdefault('epochs', []).append({
        'epoch': epoch, 'loss': avg_loss,
        'time_sec': elapsed, 'global_step': global_step,
    })
    return global_step, total_time


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print('\n' + '=' * 70)
    print('  CHARGEMENT DATASET')
    print('=' * 70)

    train_examples, val_examples = load_sft_datasets()

    print('\n  Tokenisation...')
    t0 = time.time()
    train_ds = SFTDataset(train_examples, CONFIG['max_seq_len'])
    val_ds   = SFTDataset(val_examples,   CONFIG['max_seq_len'])
    print(f'  Train : {len(train_ds):,} séquences  ({time.time() - t0:.1f}s)')
    print(f'  Val   : {len(val_ds):,} séquences')

    del train_examples, val_examples
    gc.collect()

    _collate = lambda b: sft_collate_fn(b, EOS_ID, CONFIG['max_seq_len'])
    train_loader = DataLoader(
        train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
        num_workers=CONFIG['num_workers'], collate_fn=_collate,
        pin_memory=(DEVICE == 'cuda'), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
        num_workers=0, collate_fn=_collate,
        pin_memory=(DEVICE == 'cuda'),
    )

    batches_per_epoch = len(train_loader)
    steps_per_epoch   = math.ceil(batches_per_epoch / CONFIG['gradient_accumulation'])
    total_steps       = steps_per_epoch * CONFIG['num_epochs']
    print(f'  Batches/epoch : {batches_per_epoch:,}')
    print(f'  Steps/epoch   : {steps_per_epoch:,}')
    print(f'  Total steps   : {total_steps:,}')

    print('\n' + '=' * 70)
    print('  CRÉATION MODÈLE')
    print('=' * 70)

    ckpt_mgr = CheckpointManager(CONFIG['sft_ckpt'])

    model = NaylisGPT(
        vocab_size            = CONFIG['vocab_size'],
        embed_dim             = CONFIG['embed_dim'],
        num_heads             = CONFIG['num_heads'],
        num_layers            = CONFIG['num_layers'],
        max_seq_len           = CONFIG['max_seq_len'],
        dropout               = CONFIG['dropout'],
        use_rope              = CONFIG['use_rope'],
        use_yarn              = CONFIG['use_yarn'],
        yarn_scale            = CONFIG['yarn_scale'],
        yarn_original_max_len = CONFIG['yarn_original_max_len'],
        use_swiglu            = CONFIG['use_swiglu'],
        n_kv_heads            = CONFIG['n_kv_heads'],
        use_qk_norm           = CONFIG['use_qk_norm'],
        soft_cap              = CONFIG['soft_cap'],
        use_flash_attn        = CONFIG['use_flash_attn'],
        rel_rank              = CONFIG['rel_rank'],
    ).to(DEVICE)

    # ── Resize embeddings si nécessaire (tokens ChatML ajoutés) ───────────────
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    if CONFIG['vocab_size'] != raw_model.vocab_size:
        print(f'  Resize embeddings : {raw_model.vocab_size} → {CONFIG["vocab_size"]}')
        raw_model.resize_token_embeddings(CONFIG['vocab_size'])

    # ── Charger le pretrain ───────────────────────────────────────────────────
    global_step   = 0
    current_epoch = 1
    total_time    = 0.0
    cp = ckpt_mgr.load()

    if cp is not None:
        # Reprendre depuis un checkpoint SFT existant
        print('\n  Reprise SFT...')
        raw_model.load_state_dict(cp['model_state_dict'], strict=False)
        global_step   = cp.get('global_step', 0)
        current_epoch = cp.get('current_epoch', 1)
        total_time    = cp.get('total_training_time', 0.0)
        if current_epoch > CONFIG['num_epochs']:
            print('  ✅ SFT déjà terminé.')
            return
    elif os.path.exists(CONFIG['pretrain_ckpt']):
        print(f'\n  Chargement pretrain : {CONFIG["pretrain_ckpt"]}')
        pt = torch.load(CONFIG['pretrain_ckpt'], map_location='cpu', weights_only=False)
        missing, unexpected = raw_model.load_state_dict(
            pt['model_state_dict'], strict=False)
        if missing:
            print(f'  ⚠️  Clés manquantes : {missing[:5]}{"..." if len(missing) > 5 else ""}')
        if unexpected:
            print(f'  ⚠️  Clés inattendues : {unexpected[:5]}{"..." if len(unexpected) > 5 else ""}')
        del pt; gc.collect()
        print('  ✅ Poids pretrain chargés')
    else:
        print(f'\n  ⚠️  Aucun pretrain trouvé ({CONFIG["pretrain_ckpt"]}) — entraînement from scratch')

    p = raw_model.count_parameters()
    print(f'  Params total : {p["total_M"]}M')
    print(f'  Naylis       : {p["naylis_K"]}K = {p["naylis_pct"]}')

    # torch.compile
    if CONFIG['use_compile'] and DEVICE == 'cuda':
        print('\ntorch.compile...')
        try:
            import torch._dynamo as _dynamo
            _dynamo.config.cache_size_limit = 256
            _dynamo.config.suppress_errors  = True
            model = torch.compile(model, mode=CONFIG['compile_mode'])
            print('  OK')
        except Exception as e:
            print(f'  FAIL : {e}')
    else:
        print('\ntorch.compile : désactivé')

    raw_model  = model._orig_mod if hasattr(model, '_orig_mod') else model
    optimizers = configure_optimizers(
        raw_model, CONFIG['learning_rate'], CONFIG['weight_decay'],
        (CONFIG['adam_beta1'], CONFIG['adam_beta2']), CONFIG['adam_eps'],
    )
    muon_opt, adamw_opt = optimizers

    if cp is not None:
        muon_opt.load_state_dict(cp.get('muon_state_dict', {}))
        adamw_opt.load_state_dict(cp.get('adamw_state_dict', {}))

    scheduler = WSDScheduler(
        list(optimizers),
        max_lr       = CONFIG['learning_rate'],
        total_steps  = total_steps,
        warmup_ratio = CONFIG['warmup_ratio'],
        decay_ratio  = CONFIG['decay_ratio'],
        min_lr_ratio = CONFIG['min_lr_ratio'],
    )
    if cp is not None:
        scheduler.load_state_dict(cp.get('scheduler_state_dict', {}))

    history = {'config': CONFIG, 'epochs': [], 'validations': []}

    print('\n' + '=' * 70)
    print(f'  SFT START — {total_steps:,} steps  ({CONFIG["num_epochs"]} epochs)')
    print('=' * 70)

    for epoch in range(current_epoch, CONFIG['num_epochs'] + 1):
        print(f'\nEPOCH {epoch}/{CONFIG["num_epochs"]}')
        try:
            global_step, total_time = train_epoch(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizers=optimizers,
                scheduler=scheduler,
                ckpt_mgr=ckpt_mgr,
                history=history,
                global_step=global_step,
                total_time=total_time,
                epoch=epoch,
                total_steps=total_steps,
            )
            cp = None

        except KeyboardInterrupt:
            print('\n  CTRL+C')
            return

        except Exception:
            print(f'\n  ERREUR :\n{traceback.format_exc()}')
            ckpt_mgr.save(model, optimizers, scheduler, {
                'global_step': global_step, 'current_epoch': epoch,
                'total_training_time': total_time,
            })
            raise

    print(f'\n{"="*70}\n  SFT TERMINÉ\n{"="*70}')
    print(f'  Steps : {global_step:,}  |  Temps : {total_time / 3600:.2f}h')

    ckpt_mgr.save(model, optimizers, scheduler, {
        'global_step': global_step,
        'current_epoch': CONFIG['num_epochs'] + 1,
        'total_training_time': total_time,
    })
    hist_path = CONFIG['sft_ckpt'].replace('.pt', '_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    print(f'  History : {hist_path}')
    print('  DONE\nBye')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrompu')
    except Exception:
        print(traceback.format_exc())
    finally:
        print('\nBye')