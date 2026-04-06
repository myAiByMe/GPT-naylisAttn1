#!/usr/bin/env python3
"""
sft.py — Naylis v1 — Supervised Fine-Tuning
============================================
Fine-tune NaylisGPT sur des données instruction/conversation.
Le script télécharge automatiquement les données depuis HuggingFace.

DATASETS DISPONIBLES (--dataset) :
  smoltalk    → HuggingFaceTB/smol-smoltalk      (485K — recette officielle SmolLM2) ← DÉFAUT
  openhermes  → teknium/OpenHermes-2.5            (~1M exemples, qualité GPT-4)
  ultrachat   → HuggingFaceH4/ultrachat_200k      (200K multi-tour)
  alpaca      → vicgalle/alpaca-gpt4              (52K, simple et propre)
  magpie      → Magpie-Align/Magpie-Ultra-v0.1   (~200K instructions ultra-qualité)
  mix         → openhermes(100k) + magpie(200k) + ultrachat(50k) + smoltalk(50k)
                (~400K total — optimal pour < 200M params)

RECETTE RECOMMANDÉE pour < 200M params :
  - smoltalk seul  (485k, 1 epoch) : option la plus simple et la plus reproductible
  - mix (~400k, 1 epoch)           : plus diversifié mais plus long à télécharger

USAGE :
  python sft.py                                     ← smoltalk par défaut
  python sft.py --dataset mix
  python sft.py --dataset mix --max_examples 200000
  python sft.py --pretrain ./Model/naylis_pretrain.pt --output ./Model/naylis_sft.pt
  python sft.py --no-compile
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
import pickle
import hashlib
import traceback
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Optional, List, Dict, Any

torch.set_float32_matmul_precision('high')

# ── Paths Core ────────────────────────────────────────────────────────────────
_root = os.path.dirname(__file__)
sys.path.append(os.path.join(_root, 'Core', 'Model'))
sys.path.append(os.path.join(_root, 'Core', 'Attention'))
sys.path.append(os.path.join(_root, 'Core', 'FeedForward'))
sys.path.append(os.path.join(_root, 'Core', 'TransformerBlock'))

from HessGpt import NaylisGPT

# ── Args ──────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description='Naylis v1 — SFT')
    p.add_argument('--pretrain',     default='./Model/naylis_pretrain.pt',
                   help='Checkpoint pretrain à fine-tuner')
    p.add_argument('--output',       default='./Model/naylis_sft.pt',
                   help='Chemin de sauvegarde du modèle SFT')
    p.add_argument('--dataset',      default='smoltalk',
                   # smoltalk par défaut : recette officielle SmolLM2, 485k exemples,
                   # déjà filtré et optimisé pour les modèles < 200M params.
                   choices=['openhermes', 'ultrachat', 'alpaca',
                            'magpie', 'smoltalk', 'mix'],
                   help='Dataset à télécharger (défaut: smoltalk)')
    p.add_argument('--max_examples', type=int, default=None,
                   help='Nombre max d\'exemples global (None = recette par défaut)')
    p.add_argument('--neftune_alpha', type=float, default=0.1,
                   # NEFTune activé par défaut (α=0.1) — réduit le sur-apprentissage
                   # de style sur les petits modèles sans sacrifier le raisonnement.
                   # Ref : https://arxiv.org/abs/2310.05914
                   help='Bruit NEFTune sur les embeddings (0 = désactivé, défaut 0.1)')
    p.add_argument('--no-compile',   action='store_true')
    p.add_argument('--compile-mode', default='default',
                   choices=['default', 'reduce-overhead', 'max-autotune'])
    return p.parse_args()

ARGS   = get_args()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    # Modèle — doit correspondre au pretrain
    'vocab_size'            : 49_152,
    'embed_dim'             : 768,
    'num_heads'             : 12,
    'num_layers'            : 18,
    'max_seq_len'           : 1024,   # adapter si pretrain = 512
    'dropout'               : 0.0,
    'use_rope'              : True,
    'use_yarn'              : False,
    'yarn_scale'            : 1.0,
    'yarn_original_max_len' : 1024,
    'use_swiglu'            : True,
    'n_kv_heads'            : 4,
    'use_qk_norm'           : True,
    'soft_cap'              : None,
    'use_flash_attn'        : True,
    'rel_rank'              : 32,

    # ── Hyperparamètres SFT (recette SmolLM2-style pour < 200M params) ────────
    #
    # learning_rate : 5e-5 — compromis entre 2e-5 (trop prudent, déplace peu
    #   les poids pour apprendre la structure de dialogue) et 1e-4 (risque
    #   d'oubli catastrophique sur 160M params). SmolLM2-135M utilise ~1e-3
    #   mais avec un corpus 10× plus large. 5e-5 est le bon équilibre ici.
    'learning_rate'         : 1e-4,

    'weight_decay'          : 0.01,
    'adam_beta1'            : 0.9,
    'adam_beta2'            : 0.95,
    'adam_eps'              : 1e-8,
    'max_grad_norm'         : 1.0,

    # num_epochs : 1 seule epoch — le sur-apprentissage arrive très vite sur
    #   les modèles < 200M. Au-delà d'une epoch le modèle imite le format
    #   de réponse sans logique ("effondrement d'instruction").
    'num_epochs'            : 1,

    # global batch = batch_size × gradient_accumulation = 4 × 32 = 128 séquences
    # Un GBS ≥ 128 est crucial pour la stabilité du gradient en SFT.
    'batch_size'            : 48,
    'gradient_accumulation' : 4,

    # warmup_ratio : 10% — plus long que le pretrain (5%) car les poids sont
    #   déjà convergés ; un warmup trop court crée des pics de gradient.
    'warmup_ratio'          : 0.10,

    # decay_ratio : 20% — cosine decay sur les derniers 20% des steps.
    'decay_ratio'           : 0.20,

    'min_lr_ratio'          : 0.1,

    # neftune_alpha : bruit uniforme ∝ α/√(L·d) sur les embeddings d'entrée.
    # Réduit l'over-fitting de style sans dégrader le raisonnement.
    # Désactiver avec --neftune_alpha 0 si vous voulez comparer.
    'neftune_alpha'         : ARGS.neftune_alpha,

    # ── Data ──────────────────────────────────────────────────────────────────
    'data_dir'              : './data_sft',
    'dataset'               : ARGS.dataset,
    'val_ratio'             : 0.02,
    'max_examples'          : ARGS.max_examples,

    # Validation / Save
    'validate_every_steps'  : 200,
    'val_batches'           : 30,
    'save_every_steps'      : 500,

    # Checkpoint SFT
    'checkpoint_file'       : ARGS.output,
    'pretrain_file'         : ARGS.pretrain,

    # Compile
    'use_compile'           : not ARGS.no_compile,
    'compile_mode'          : ARGS.compile_mode,
    'num_workers'           : 1,
}

SYSTEM_PROMPT_DEFAULT = "You are Naylis, a helpful and knowledgeable AI assistant."

# ── Téléchargement automatique des données ────────────────────────────────────
DATASET_REGISTRY = {
    'openhermes': {
        'hf_id'  : 'teknium/OpenHermes-2.5',
        'split'  : 'train',
        'format' : 'openhermes',
        'desc'   : '~1M exemples, qualité GPT-4',
    },
    'ultrachat': {
        'hf_id'  : 'HuggingFaceH4/ultrachat_200k',
        'split'  : 'train_sft',
        'format' : 'messages',
        'desc'   : '200K conversations multi-tour',
    },
    'alpaca': {
        'hf_id'  : 'vicgalle/alpaca-gpt4',
        'split'  : 'train',
        'format' : 'alpaca',
        'desc'   : '52K instructions simples et propres',
    },
    'magpie': {
        'hf_id'  : 'Magpie-Align/Magpie-Ultra-v0.1',
        'split'  : 'train',
        'format' : 'openhermes',   # même format conversations from/value
        'desc'   : '~200K instructions synthétiques très haute qualité',
    },
    'smoltalk': {
        'hf_id'  : 'HuggingFaceTB/smol-smoltalk',
        'split'  : 'train',
        'format' : 'messages',     # messages role/content standard
        'desc'   : '485K exemples — recette officielle SmolLM2 (RECOMMANDÉ)',
    },
}

# ── Recette mix recommandée : {dataset_name: max_examples} ───────────────────
#
# Total ~400K exemples — optimal pour < 200M params selon la littérature
# (SmolTalk, Magpie-Ultra, OpenHermes-2.5 papers).
#
# Composition :
#   - openhermes (100k) : échantillons de raisonnement et MMLU, qualité GPT-4
#   - magpie     (200k) : instructions synthétiques ultra-qualité (cœur de SmolTalk)
#   - ultrachat  ( 50k) : structure conversationnelle multi-tours
#   - smoltalk   ( 50k) : complément de la recette officielle HuggingFace
#
# Note : 1M+ d'exemples pour 160M de params = risque d'"effondrement d'instruction"
#   (imitation du format sans logique) + oubli catastrophique des capacités
#   acquises au pretrain (Winogrande 51.38%, PIQA 64.15%, etc.).
MIX_RECIPE = {
    'openhermes' : 100_000,   # 100K meilleurs raisonnements GPT-4
    'magpie'     : 200_000,   # 200K instructions synthétiques ultra
    'ultrachat'  :  50_000,   # 50K conversations multi-tour
    'smoltalk'   :  50_000,   # 50K recette SmolLM2
}


def _convert_openhermes(ex: dict) -> Optional[List[Dict]]:
    """teknium/OpenHermes-2.5 — champ 'conversations'."""
    convs = ex.get('conversations', [])
    msgs  = []
    for c in convs:
        role = c.get('from', '')
        val  = c.get('value', '').strip()
        if not val:
            continue
        if role == 'system':
            msgs.append({'role': 'system',    'content': val})
        elif role in ('human', 'user'):
            msgs.append({'role': 'user',      'content': val})
        elif role in ('gpt', 'assistant'):
            msgs.append({'role': 'assistant', 'content': val})
    has_assistant = any(m['role'] == 'assistant' for m in msgs)
    has_user      = any(m['role'] == 'user'      for m in msgs)
    return msgs if (has_assistant and has_user) else None


def _convert_messages(ex: dict) -> Optional[List[Dict]]:
    """Format standard {'messages': [...]} — UltraChat, SmolTalk, etc."""
    msgs = ex.get('messages', [])
    out  = []
    for m in msgs:
        role    = m.get('role', '')
        content = m.get('content', '').strip()
        if role in ('system', 'user', 'assistant') and content:
            out.append({'role': role, 'content': content})
    has_assistant = any(m['role'] == 'assistant' for m in out)
    has_user      = any(m['role'] == 'user'      for m in out)
    return out if (has_assistant and has_user) else None


def _convert_alpaca(ex: dict) -> Optional[List[Dict]]:
    """Format Alpaca — instruction / input / output."""
    instruction = ex.get('instruction', '').strip()
    inp         = ex.get('input', '').strip()
    output      = ex.get('output', '').strip()
    if not instruction or not output:
        return None
    user_content = instruction
    if inp:
        user_content += '\n\n' + inp
    return [
        {'role': 'system',    'content': SYSTEM_PROMPT_DEFAULT},
        {'role': 'user',      'content': user_content},
        {'role': 'assistant', 'content': output},
    ]


CONVERTERS = {
    'openhermes': _convert_openhermes,
    'messages'  : _convert_messages,
    'alpaca'    : _convert_alpaca,
}


def download_and_save(dataset_name: str, data_dir: str,
                      limit: Optional[int] = None) -> str:
    """
    Télécharge un dataset HuggingFace, le convertit en JSONL et le sauvegarde.
    `limit` : si fourni, ne garde que les N premiers exemples valides.
    Skipe si le fichier (avec le bon suffix de limite) existe déjà.
    """
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        print('\n  ERREUR : `datasets` non installé.')
        print('  → pip install datasets')
        sys.exit(1)

    os.makedirs(data_dir, exist_ok=True)
    suffix   = f'_{limit//1000}k' if limit else ''
    out_path = os.path.join(data_dir, f'{dataset_name}{suffix}.jsonl')

    if os.path.exists(out_path):
        n_lines = sum(1 for _ in open(out_path, 'r', encoding='utf-8'))
        print(f'  ✅ {os.path.basename(out_path)} déjà présent ({n_lines:,} ex) — skip')
        return out_path

    info = DATASET_REGISTRY[dataset_name]
    lbl  = f'≤{limit//1000}K' if limit else 'complet'
    print(f'\n  ⬇️  Téléchargement : {info["hf_id"]} ({lbl} — {info["desc"]})')
    print(f'      Cela peut prendre quelques minutes...')

    ds        = hf_load(info['hf_id'], split=info['split'])
    converter = CONVERTERS[info['format']]

    n_ok = n_skip = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        for ex in tqdm(ds, desc=f'  Conversion {dataset_name}', leave=False):
            if limit and n_ok >= limit:
                break
            msgs = converter(ex)
            if msgs is None:
                n_skip += 1
                continue
            f.write(json.dumps({'messages': msgs}, ensure_ascii=False) + '\n')
            n_ok += 1

    print(f'  ✅ {os.path.basename(out_path)} sauvegardé : {n_ok:,} ex  '
          f'({n_skip} ignorés) → {out_path}')
    return out_path


def prepare_data(dataset_name: str, data_dir: str) -> None:
    """Télécharge les datasets nécessaires selon le choix de l'utilisateur."""
    if dataset_name == 'mix':
        names = ' + '.join(MIX_RECIPE.keys())
        total = sum(MIX_RECIPE.values())
        print(f'\n  Dataset : mix ({names})  —  ~{total//1000}K exemples total')
        print(f'  (recette optimisée pour < 200M params : diversité sans sur-apprentissage)')
        for ds_name, ds_limit in MIX_RECIPE.items():
            download_and_save(ds_name, data_dir, limit=ds_limit)
    else:
        info = DATASET_REGISTRY[dataset_name]
        print(f'\n  Dataset : {dataset_name} — {info["desc"]}')
        download_and_save(dataset_name, data_dir)

print('=' * 70)
print('  Naylis v1 — SFT (Supervised Fine-Tuning)')
print('=' * 70)
if DEVICE == 'cuda':
    print(f'  GPU  : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB')
print(f'  Pretrain : {CONFIG["pretrain_file"]}')
print(f'  Output   : {CONFIG["checkpoint_file"]}')
print(f'  Dataset  : {CONFIG["dataset"]}')
print(f'  LR={CONFIG["learning_rate"]:.1e}  epochs={CONFIG["num_epochs"]}  '
      f'batch={CONFIG["batch_size"]}×{CONFIG["gradient_accumulation"]}  '
      f'(GBS={CONFIG["batch_size"] * CONFIG["gradient_accumulation"]})')
print(f'  NEFTune α={CONFIG["neftune_alpha"]}  '
      f'warmup={CONFIG["warmup_ratio"]:.0%}  decay={CONFIG["decay_ratio"]:.0%}')


# ── Tokenizer ─────────────────────────────────────────────────────────────────
print('\nTokenizer...')
tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/cosmo2-tokenizer')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
EOS_ID = tokenizer.eos_token_id
print(f'  vocab={len(tokenizer)}  eos={EOS_ID}')

# Tokens ChatML (natifs au cosmo2-tokenizer)
IM_START = tokenizer.encode('<|im_start|>', add_special_tokens=False)
IM_END   = tokenizer.encode('<|im_end|>',   add_special_tokens=False)
print(f'  <|im_start|>={IM_START}  <|im_end|>={IM_END}')


# ── Dataset SFT ───────────────────────────────────────────────────────────────
def format_chatml(messages: List[Dict[str, str]]) -> str:
    """Formate une liste de messages en texte ChatML."""
    text = ''
    # Ajoute un system prompt par défaut si absent
    if not messages or messages[0]['role'] != 'system':
        text += f'<|im_start|>system\n{SYSTEM_PROMPT_DEFAULT}<|im_end|>\n'
    for msg in messages:
        role    = msg['role']
        content = msg['content'].strip()
        text   += f'<|im_start|>{role}\n{content}<|im_end|>\n'
    return text


def tokenize_with_mask(
    messages : List[Dict[str, str]],
    max_len  : int,
) -> Optional[tuple]:
    """
    Tokenise les messages et crée le masque de loss.
    Retourne (input_ids, labels) ou None si trop court.

    labels[i] = -100 si le token est du contexte (system/user)
    labels[i] = token_id si le token est de l'assistant (contribue à la loss)
    """
    full_text = format_chatml(messages)
    all_ids   = tokenizer.encode(full_text, add_special_tokens=False)

    # Tronque si nécessaire (garde la fin = tokens assistant)
    if len(all_ids) > max_len:
        all_ids = all_ids[-max_len:]

    labels = [-100] * len(all_ids)

    # Retrouve les spans assistant dans les tokens
    # Stratégie : re-tokeniser message par message et tracker les positions
    pos = 0
    if not messages or messages[0]['role'] != 'system':
        sys_text = f'<|im_start|>system\n{SYSTEM_PROMPT_DEFAULT}<|im_end|>\n'
        sys_ids  = tokenizer.encode(sys_text, add_special_tokens=False)
        pos     += len(sys_ids)

    for msg in messages:
        role    = msg['role']
        content = msg['content'].strip()
        prefix  = f'<|im_start|>{role}\n'
        suffix  = '<|im_end|>\n'

        prefix_ids  = tokenizer.encode(prefix,  add_special_tokens=False)
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        suffix_ids  = tokenizer.encode(suffix,  add_special_tokens=False)

        msg_len = len(prefix_ids) + len(content_ids) + len(suffix_ids)

        # Ajuste pour la troncature éventuelle
        if pos >= len(all_ids):
            break
        if pos < 0:
            pos = 0

        if role == 'assistant':
            # Début du contenu assistant = après le prefix
            content_start = pos + len(prefix_ids)
            # Fin = avant le suffix (on inclut le EOS token de fin de message)
            content_end   = content_start + len(content_ids) + len(suffix_ids)
            content_end   = min(content_end, len(all_ids))
            for i in range(content_start, content_end):
                if 0 <= i < len(labels):
                    labels[i] = all_ids[i]

        pos += msg_len

    # Filtre les exemples sans aucun token assistant (labels tous -100)
    if all(l == -100 for l in labels):
        return None

    input_ids = all_ids[:-1]
    targets   = labels[1:]

    if len(input_ids) < 4:
        return None

    return input_ids, targets


class SFTDataset(Dataset):
    """
    Dataset SFT — lit des fichiers JSONL depuis data_dir.
    Cache la tokenisation sur disque (data_dir/.cache/) pour éviter
    de re-tokeniser à chaque lancement.

    Format attendu : {"messages": [{"role": ..., "content": ...}, ...]}
    Supporte aussi : Alpaca et ShareGPT.
    """
    def __init__(
        self,
        data_dir    : str,
        max_len     : int,
        max_examples: Optional[int] = None,
        split       : str   = 'train',
        val_ratio   : float = 0.02,
        seed        : int   = 42,
    ):
        self.max_len  = max_len
        self.examples : List[tuple] = []

        all_files = sorted(Path(data_dir).glob('*.jsonl'))
        if not all_files:
            all_files = sorted(Path(data_dir).glob('*.json'))
        if not all_files:
            raise FileNotFoundError(
                f'Aucun fichier .jsonl trouvé dans {data_dir}'
            )

        # ── Clé de cache ─────────────────────────────────────────────────────
        # Basée sur : noms+tailles des fichiers, max_len, max_examples,
        #             val_ratio, seed, split → si l'un change, cache invalide.
        file_sig = '_'.join(
            f'{f.name}:{f.stat().st_size}' for f in all_files
        )
        cache_key = hashlib.md5(
            f'{file_sig}|{max_len}|{max_examples}|{val_ratio}|{seed}|{split}'.encode()
        ).hexdigest()[:12]

        cache_dir  = Path(data_dir) / '.cache'
        cache_dir.mkdir(exist_ok=True)
        cache_path = cache_dir / f'tokenized_{split}_{cache_key}.pkl'

        # ── Charge depuis le cache si disponible ──────────────────────────────
        if cache_path.exists():
            size_mb = cache_path.stat().st_size / 1e6
            print(f'  ✅ Cache {split} trouvé ({size_mb:.0f} MB) — chargement...')
            t0 = time.time()
            with open(cache_path, 'rb') as f:
                self.examples = pickle.load(f)
            print(f'  ✅ {len(self.examples):,} exemples chargés en {time.time()-t0:.1f}s')
            return

        # ── Lecture des JSONL ─────────────────────────────────────────────────
        raw = []
        for fpath in all_files:
            print(f'  Chargement : {fpath.name}')
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj  = json.loads(line)
                        msgs = self._parse_example(obj)
                        if msgs:
                            raw.append(msgs)
                    except json.JSONDecodeError:
                        continue
            print(f'    → {len(raw):,} exemples au total')

        # Shuffle reproductible puis split
        rng = random.Random(seed)
        rng.shuffle(raw)
        if max_examples is not None:
            raw = raw[:max_examples]
        n_val = max(1, int(len(raw) * val_ratio))
        raw   = raw[-n_val:] if split == 'val' else raw[:-n_val]

        print(f'  Split={split}  exemples bruts={len(raw):,}')

        # ── Tokenisation ──────────────────────────────────────────────────────
        n_skip = 0
        for msgs in tqdm(raw, desc=f'  Tokenisation {split}', leave=False):
            result = tokenize_with_mask(msgs, max_len)
            if result is None:
                n_skip += 1
                continue
            self.examples.append(result)

        print(f'  {split} : {len(self.examples):,} exemples valides  '
              f'({n_skip} ignorés)')

        # ── Sauvegarde du cache ───────────────────────────────────────────────
        print(f'  💾 Sauvegarde cache {split}...')
        t0 = time.time()
        with open(cache_path, 'wb') as f:
            pickle.dump(self.examples, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = cache_path.stat().st_size / 1e6
        print(f'  ✅ Cache sauvegardé : {cache_path.name}  '
              f'({size_mb:.0f} MB, {time.time()-t0:.1f}s)')

    def _parse_example(self, obj: Dict[str, Any]) -> Optional[List[Dict]]:
        """Parse différents formats de données."""
        # Format standard : {"messages": [...]}
        if 'messages' in obj:
            msgs = obj['messages']
            if isinstance(msgs, list) and len(msgs) >= 2:
                return msgs

        # Format Alpaca : {"instruction": ..., "input": ..., "output": ...}
        if 'instruction' in obj and 'output' in obj:
            user_content = obj['instruction']
            if obj.get('input', '').strip():
                user_content += '\n\n' + obj['input'].strip()
            return [
                {'role': 'system',    'content': SYSTEM_PROMPT_DEFAULT},
                {'role': 'user',      'content': user_content},
                {'role': 'assistant', 'content': obj['output']},
            ]

        # Format ShareGPT : {"conversations": [{"from": "human", "value": ...}]}
        if 'conversations' in obj:
            convs = obj['conversations']
            msgs  = []
            for c in convs:
                role = c.get('from', c.get('role', ''))
                val  = c.get('value', c.get('content', ''))
                if role in ('system', 'gpt', 'human', 'user', 'assistant'):
                    role = {'gpt': 'assistant', 'human': 'user'}.get(role, role)
                    msgs.append({'role': role, 'content': val})
            if len(msgs) >= 2:
                return msgs

        return None

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids, labels = self.examples[idx]
        return torch.tensor(input_ids, dtype=torch.long), \
               torch.tensor(labels,    dtype=torch.long)


def sft_collate_fn(batch):
    """Padding dynamique à la longueur max du batch."""
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    x_pad   = torch.zeros(len(xs), max_len, dtype=torch.long)
    y_pad   = torch.full((len(ys), max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        x_pad[i, :x.size(0)] = x
        y_pad[i, :y.size(0)] = y
    return x_pad, y_pad


# ── Muon + MARS-M (copié du pretrain) ────────────────────────────────────────
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
    """Même split que le pretrain : Muon pour blocs, AdamW pour le reste."""
    EXCLUDE = {'token_embeddings.weight', 'output_head.weight'}
    muon_params, adamw_decay, adamw_nodecay = [], [], []
    for pn, p in model.named_parameters():
        if not p.requires_grad: continue
        if pn in EXCLUDE:
            (adamw_decay if p.dim() >= 2 else adamw_nodecay).append(p)
        elif p.dim() >= 2 and pn.startswith('blocks.'):
            muon_params.append(p)
        elif p.dim() < 2 and pn.startswith('blocks.'):
            adamw_nodecay.append(p)
        elif p.dim() >= 2:
            adamw_decay.append(p)
        else:
            adamw_nodecay.append(p)

    muon_opt = Muon(
        [{'params': muon_params, 'is_muon': True}],
        lr=lr, momentum=0.95, nesterov=True,
        ns_steps=3, weight_decay=0.0, use_mars=True, mars_gamma=0.025,
    )
    muon_opt.param_groups[0]['is_muon'] = True
    adamw_opt = torch.optim.AdamW(
        [{'params': adamw_decay,   'weight_decay': weight_decay, 'is_muon': False},
         {'params': adamw_nodecay, 'weight_decay': 0.0,          'is_muon': False}],
        lr=lr, betas=betas, eps=eps, fused=(DEVICE == 'cuda'),
    )
    n_muon  = sum(p.numel() for p in muon_params)
    n_adamw = sum(p.numel() for p in adamw_decay + adamw_nodecay)
    print(f'\n  Muon+MARS  : {n_muon / 1e6:.2f}M params  lr={lr * 5:.2e}  (base×5 via scheduler)')
    print(f'  AdamW      : {n_adamw / 1e6:.2f}M params  lr={lr:.2e}')
    return muon_opt, adamw_opt


# ── WSD Scheduler (identique au pretrain) ────────────────────────────────────
class WSDScheduler:
    def __init__(self, optimizers, max_lr, total_steps,
                 warmup_ratio=0.05, decay_ratio=0.20, min_lr_ratio=0.1):
        self.optimizers   = optimizers if isinstance(optimizers, list) else [optimizers]
        self.max_lr       = max_lr
        self.min_lr       = max_lr * min_lr_ratio
        self.total_steps  = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.decay_steps  = int(total_steps * decay_ratio)
        self.stable_steps = total_steps - self.warmup_steps - self.decay_steps
        self.current_step = 0

    def get_lr(self) -> float:
        s = self.current_step
        if s < self.warmup_steps:
            return self.max_lr * (s / max(self.warmup_steps, 1))
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


# ── Checkpoint ────────────────────────────────────────────────────────────────
class CheckpointManager:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    def save(self, model, optimizers, scheduler, metadata: dict):
        m           = model._orig_mod if hasattr(model, '_orig_mod') else model
        muon, adamw = optimizers
        cp = {
            'model_state_dict'    : m.state_dict(),
            'muon_state_dict'     : muon.state_dict(),
            'adamw_state_dict'    : adamw.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metadata'            : metadata,
            'config'              : CONFIG,
        }
        tmp = self.path + '.tmp'
        torch.save(cp, tmp)
        os.replace(tmp, self.path)
        step = metadata.get('global_step', '?')
        loss = metadata.get('train_loss', float('nan'))
        print(f'  💾 SAVE  step={step}  train_loss={loss:.4f}  [{self.path}]')

    def load_pretrain(self, path: str, model) -> None:
        """Charge uniquement les poids du pretrain (pas les optimiseurs)."""
        print(f'\n  Chargement pretrain : {path}')
        ckpt  = torch.load(path, map_location='cpu', weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f'  ⚠️  {len(missing)} clés manquantes du pretrain')
        if unexpected:
            print(f'  ⚠️  {len(unexpected)} clés inattendues')
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f'  ✅ {params:.1f}M params chargés')

    def load_sft(self) -> Optional[dict]:
        """Charge un checkpoint SFT existant pour reprendre."""
        if not os.path.exists(self.path):
            return None
        print(f'\n  Checkpoint SFT trouvé : {self.path} — reprise')
        return torch.load(self.path, map_location='cpu', weights_only=False)


# ── Validation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, val_loader, max_batches: int = 30) -> tuple:
    model.eval()
    total_loss, n_tokens = 0.0, 0
    ae  = (DEVICE == 'cuda')
    adt = torch.bfloat16 if ae else torch.float32
    try:
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches: break
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast(DEVICE, dtype=adt, enabled=ae):
                logits, _, _ = model(x)
                # Loss uniquement sur tokens non masqués
                mask = (y != -100)
                if mask.sum() == 0:
                    continue
                loss = F.cross_entropy(
                    logits.view(-1, CONFIG['vocab_size'])[mask.view(-1)],
                    y.view(-1)[mask.view(-1)],
                )
            total_loss += loss.item() * mask.sum().item()
            n_tokens   += mask.sum().item()
    finally:
        model.train()
    if n_tokens == 0:
        return float('inf'), float('inf')
    avg = total_loss / n_tokens
    return math.exp(min(avg, 10)), avg


# ── Forward SFT avec loss masquée ────────────────────────────────────────────
def sft_loss(model, x, y):
    """
    Calcule la cross-entropy uniquement sur les tokens assistant (y != -100).
    """
    logits, _, _ = model(x)
    mask = (y != -100)
    if mask.sum() == 0:
        return None
    loss = F.cross_entropy(
        logits.view(-1, CONFIG['vocab_size'])[mask.view(-1)],
        y.view(-1)[mask.view(-1)],
    )
    return loss


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print('\n' + '=' * 70)
    print('  CHARGEMENT DONNÉES')
    print('=' * 70)

    data_dir = CONFIG['data_dir']

    # Téléchargement automatique si nécessaire
    prepare_data(CONFIG['dataset'], data_dir)

    train_ds = SFTDataset(
        data_dir, CONFIG['max_seq_len'],
        max_examples = CONFIG['max_examples'],
        split='train', val_ratio=CONFIG['val_ratio'],
    )
    val_ds = SFTDataset(
        data_dir, CONFIG['max_seq_len'],
        max_examples = CONFIG['max_examples'],
        split='val', val_ratio=CONFIG['val_ratio'],
    )

    if len(train_ds) == 0:
        print('\n  ERREUR : aucun exemple valide dans le dataset.')
        sys.exit(1)

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG['batch_size'],
        shuffle=True, num_workers=CONFIG['num_workers'],
        pin_memory=True, drop_last=False,
        collate_fn=sft_collate_fn,
        persistent_workers=(CONFIG['num_workers'] > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG['batch_size'],
        shuffle=False, num_workers=1, pin_memory=True,
        collate_fn=sft_collate_fn,
    )

    steps_per_epoch = math.ceil(
        len(train_ds) / (CONFIG['batch_size'] * CONFIG['gradient_accumulation'])
    )
    total_steps = steps_per_epoch * CONFIG['num_epochs']
    print(f'\n  {len(train_ds)} exemples train  |  {len(val_ds)} exemples val')
    print(f'  {steps_per_epoch} steps/epoch × {CONFIG["num_epochs"]} epochs = {total_steps} steps')

    # ── Modèle ────────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('  CRÉATION MODÈLE')
    print('=' * 70)

    ckpt_mgr = CheckpointManager(CONFIG['checkpoint_file'])

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
    )

    # Essaie de reprendre un SFT existant, sinon charge le pretrain
    sft_cp = ckpt_mgr.load_sft()
    global_step   = 0
    start_epoch   = 1

    if sft_cp is not None:
        state = sft_cp.get('model_state_dict', sft_cp)
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        meta         = sft_cp.get('metadata', {})
        global_step  = meta.get('global_step', 0)
        start_epoch  = meta.get('epoch', 1)
        print(f'  Reprise depuis step={global_step}  epoch={start_epoch}')
    else:
        ckpt_mgr.load_pretrain(CONFIG['pretrain_file'], model)

    dtype = torch.bfloat16 if DEVICE == 'cuda' else torch.float32
    model = model.to(dtype).to(DEVICE)

    p = model.count_parameters()
    print(f'  Params total : {p["total_M"]}M  |  Naylis : {p["naylis_pct"]}')

    # ── NEFTune — bruit uniforme sur les embeddings ────────────────────────────
    # Réduit le sur-apprentissage de style sans sacrifier le raisonnement.
    # α=0.1 activé par défaut — désactiver avec --neftune_alpha 0.
    # Pour les < 200M params, NEFTune est particulièrement bénéfique car ces
    # modèles sur-apprennent très vite le format ChatML exact.
    # Ref : https://arxiv.org/abs/2310.05914
    neftune_alpha = CONFIG['neftune_alpha']
    neftune_hook  = None
    if neftune_alpha > 0:
        raw_model_for_nef = model._orig_mod if hasattr(model, '_orig_mod') else model
        embed_layer = raw_model_for_nef.token_embeddings  # nn.Embedding

        def _neftune_hook(module, inp, output):
            if not module.training:
                return output
            L, d = output.shape[-2], output.shape[-1]
            noise = torch.zeros_like(output).uniform_(-1, 1)
            noise = noise * (neftune_alpha / math.sqrt(L * d))
            return output + noise

        neftune_hook = embed_layer.register_forward_hook(_neftune_hook)
        print(f'\n  NEFTune activé  α={neftune_alpha}  '
              f'(bruit ∝ {neftune_alpha}/√(L·{raw_model_for_nef.embed_dim}))')
    else:
        print('\n  NEFTune : désactivé')

    # torch.compile
    if CONFIG['use_compile'] and DEVICE == 'cuda':
        print('\ntorch.compile...')
        import torch._dynamo as _dynamo
        _dynamo.config.cache_size_limit = 256
        _dynamo.config.suppress_errors  = True
        try:
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

    if sft_cp is not None:
        muon_opt.load_state_dict(sft_cp.get('muon_state_dict', {}))
        adamw_opt.load_state_dict(sft_cp.get('adamw_state_dict', {}))

    scheduler = WSDScheduler(
        list(optimizers), max_lr=CONFIG['learning_rate'],
        total_steps=total_steps,
        warmup_ratio=CONFIG['warmup_ratio'],
        decay_ratio =CONFIG['decay_ratio'],
        min_lr_ratio=CONFIG['min_lr_ratio'],
    )
    if sft_cp is not None:
        scheduler.load_state_dict(sft_cp.get('scheduler_state_dict', {}))

    # ── Training ──────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print(f'  SFT START — {total_steps:,} steps')
    print('=' * 70)

    ae  = (DEVICE == 'cuda')
    adt = torch.bfloat16 if ae else torch.float32
    best_val_loss = float('inf')

    for epoch in range(start_epoch, CONFIG['num_epochs'] + 1):
        print(f'\nEPOCH {epoch}/{CONFIG["num_epochs"]}')
        model.train()

        running_loss  = 0.0
        running_tok   = 0
        acc_steps     = 0
        t0            = time.time()

        pbar = tqdm(train_loader, desc=f'  E{epoch}', leave=True, dynamic_ncols=True)

        for batch in pbar:
            try:
                x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)

                with torch.amp.autocast(DEVICE, dtype=adt, enabled=ae):
                    loss = sft_loss(model, x, y)

                if loss is None:
                    continue

                loss_scaled = loss / CONFIG['gradient_accumulation']

                if torch.isnan(loss_scaled) or torch.isinf(loss_scaled):
                    muon_opt.zero_grad(set_to_none=True)
                    adamw_opt.zero_grad(set_to_none=True)
                    acc_steps = 0
                    continue

                loss_scaled.backward()

                n_tok         = (y != -100).sum().item()
                running_loss += loss.item() * n_tok
                running_tok  += n_tok
                acc_steps    += 1

                if acc_steps >= CONFIG['gradient_accumulation']:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), CONFIG['max_grad_norm'])
                    muon_opt.step()
                    adamw_opt.step()
                    lr = scheduler.step()
                    muon_opt.zero_grad(set_to_none=True)
                    adamw_opt.zero_grad(set_to_none=True)
                    acc_steps   = 0
                    global_step += 1

                    avg_loss = running_loss / max(running_tok, 1)
                    pbar.set_postfix(
                        loss=f'{avg_loss:.4f}',
                        ppl =f'{math.exp(min(avg_loss, 10)):.2f}',
                        lr  =f'{lr:.2e}',
                    )

                    # Validation
                    if global_step % CONFIG['validate_every_steps'] == 0:
                        val_ppl, val_loss = validate(model, val_loader, CONFIG['val_batches'])
                        pbar.write(
                            f'  [val  step={global_step:,}] '
                            f'loss={val_loss:.4f}  ppl={val_ppl:.2f}'
                        )
                        # Sauvegarde si meilleur modèle val
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_path = CONFIG['checkpoint_file'].replace('.pt', '_best.pt')
                            os.makedirs(os.path.dirname(best_path) or '.', exist_ok=True)
                            m = model._orig_mod if hasattr(model, '_orig_mod') else model
                            torch.save({'model_state_dict': m.state_dict(),
                                        'val_loss': val_loss, 'config': CONFIG}, best_path)
                            pbar.write(f'  ⭐ Meilleur val_loss={val_loss:.4f} → {best_path}')

                    # Checkpoint périodique
                    if global_step % CONFIG['save_every_steps'] == 0:
                        avg = running_loss / max(running_tok, 1)
                        ckpt_mgr.save(model, optimizers, scheduler, {
                            'global_step': global_step, 'epoch': epoch,
                            'train_loss': avg,
                        })

                    # Affichage graph_scale Naylis
                    if global_step % 500 == 0:
                        raw = model._orig_mod if hasattr(model, '_orig_mod') else model
                        scales = [b.attention.graph_scale.detach().abs().mean().item()
                                  for b in raw.blocks]
                        avg_s = sum(scales) / len(scales)
                        pbar.write(
                            f'  [naylis step={global_step:,}] '
                            f'|graph_scale| avg={avg_s:.5f}  '
                            f'min={min(scales):.5f}  max={max(scales):.5f}'
                        )

            except torch.cuda.OutOfMemoryError:
                print(f'\n  OOM — skip batch')
                torch.cuda.empty_cache()
                muon_opt.zero_grad(set_to_none=True)
                adamw_opt.zero_grad(set_to_none=True)
                acc_steps = 0
                gc.collect()
                model.train()
                continue

        pbar.close()
        elapsed  = time.time() - t0
        avg_loss = running_loss / max(running_tok, 1)
        print(f'\n  Epoch {epoch} terminée | '
              f'train_loss={avg_loss:.4f} | ppl={math.exp(min(avg_loss, 10)):.2f} | '
              f'{elapsed / 60:.1f}min')

        # Sauvegarde fin d'epoch
        ckpt_mgr.save(model, optimizers, scheduler, {
            'global_step': global_step, 'epoch': epoch + 1,
            'train_loss': avg_loss,
        })

    # ── Sauvegarde finale ─────────────────────────────────────────────────────
    print(f'\n{"=" * 70}')
    print('  SFT TERMINÉ')
    print(f'{"=" * 70}')

    m   = model._orig_mod if hasattr(model, '_orig_mod') else model
    out = {
        'model_state_dict' : m.state_dict(),
        'config'           : CONFIG,
        'metadata'         : {
            'global_step' : global_step,
            'num_epochs'  : CONFIG['num_epochs'],
            'final_loss'  : avg_loss,
            'sft_data'    : CONFIG['data_dir'],
            'dataset'     : CONFIG['dataset'],
            'neftune_alpha': CONFIG['neftune_alpha'],
        },
    }
    torch.save(out, CONFIG['checkpoint_file'])
    print(f'  ✅ Modèle SFT sauvegardé : {CONFIG["checkpoint_file"]}')
    print(f'  → Lance maintenant : python3.10 inference.py --model {CONFIG["checkpoint_file"]}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n\n  CTRL+C — arrêt propre.')
