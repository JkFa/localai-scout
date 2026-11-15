"""Mapeamento das respostas da API do Hugging Face para entradas do catálogo (M3).

Funções puras: recebem os JSONs já baixados por scout.hf e não tocam a rede.
Nada é preenchido de memória: campo que não veio da API fica None, com
motivo, e o modelo recebe complete=False (DECISIONS.md, campo incompleto).
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from scout.hf import BASE, model_card_url

# Campos estruturais que tornam o modelo incompleto quando None (catalog.schema.json).
STRUCTURAL_FIELDS: tuple[str, ...] = (
    "license", "license_source", "tasks", "params_total", "params_active", "n_layers",
    "n_heads", "n_kv_heads", "head_dim", "ctx_max", "kv_model",
)
ARCH_FIELDS: tuple[str, ...] = ("n_layers", "n_heads", "n_kv_heads", "head_dim", "ctx_max", "kv_model")

# pipeline_tag da API -> tarefas do catálogo. Tag fora da tabela -> tasks None.
# "code" não tem pipeline_tag próprio no HF: não é derivável da API.
PIPELINE_TASKS: dict[str, tuple[str, ...]] = {
    "text-generation": ("chat",),
    "image-text-to-text": ("vlm",),
    "automatic-speech-recognition": ("stt",),
    "feature-extraction": ("embeddings",),
    "sentence-similarity": ("embeddings",),
    "text-to-image": ("diffusion",),
}

# Chaves que indicam MoE e contam os experts roteados.
MOE_EXPERT_KEYS: tuple[str, ...] = ("num_experts", "n_routed_experts", "num_local_experts")
# Ativações de FFN com porta (gate, up, down = 3 matrizes por expert).
GATED_ACTS: tuple[str, ...] = ("silu", "swish", "swiglu")

STALE_DAYS = 60
TASK_ENUM: tuple[str, ...] = ("chat", "code", "embeddings", "vlm", "stt", "diffusion")
BPW_MAX = 40  # catalog.schema.json; F32 com metadados passa de 32

_QUANT_RE = re.compile(r"(?i)(?<![A-Za-z0-9])((?:I?Q\d+(?:_[A-Z0-9]+)*)|BF16|FP16|F16|F32)(?![A-Za-z0-9])")
_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})$")


def _int(v: Any) -> int | None:
    """Inteiro positivo ou None (bool não conta como int)."""
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        return None
    return v


def _arch_source(config: dict) -> dict:
    """VLMs aninham o modelo de linguagem em text_config: os campos vêm só de lá,
    para não misturar com o encoder de visão do nível de cima."""
    tc = config.get("text_config")
    return tc if isinstance(tc, dict) else config


def map_config(config: dict | None) -> dict:
    """config.json -> campos de arquitetura, flags de MoE e motivos de None.

    Chaves: n_layers, n_heads, n_kv_heads, head_dim, ctx_max, kv_model,
    is_moe, moe_inactive_params, null_reasons {campo: motivo}.
    """
    out: dict[str, Any] = {f: None for f in ARCH_FIELDS}
    out.update(is_moe=False, moe_inactive_params=None, null_reasons={})
    reasons: dict[str, str] = out["null_reasons"]
    if config is None:
        for f in ARCH_FIELDS:
            reasons[f] = "config.json indisponível"
        return out

    c = _arch_source(config)
    src = "text_config." if c is not config else ""

    out["n_layers"] = _int(c.get("num_hidden_layers"))
    out["n_heads"] = _int(c.get("num_attention_heads"))
    out["ctx_max"] = _int(c.get("max_position_embeddings"))

    if "num_key_value_heads" in c and c["num_key_value_heads"] is not None:
        out["n_kv_heads"] = _int(c["num_key_value_heads"])
    else:
        out["n_kv_heads"] = out["n_heads"]  # ausente => MHA (PROMPT.md M3)

    hd = c.get("head_dim")
    if hd is not None:
        out["head_dim"] = _int(hd)
        if out["head_dim"] is None:
            reasons["head_dim"] = f"{src}head_dim inválido"
    else:
        hidden, heads = _int(c.get("hidden_size")), out["n_heads"]
        if hidden is not None and heads is not None:
            if hidden % heads == 0:
                out["head_dim"] = hidden // heads
            else:
                reasons["head_dim"] = f"{src}hidden_size não é divisível por num_attention_heads"
        else:
            reasons["head_dim"] = f"{src}head_dim ausente e sem hidden_size/num_attention_heads"

    out["kv_model"] = "mla" if c.get("kv_lora_rank") is not None else "standard"

    key_names = {"n_layers": "num_hidden_layers", "n_heads": "num_attention_heads",
                 "ctx_max": "max_position_embeddings", "n_kv_heads": "num_key_value_heads"}
    for f, k in key_names.items():
        if out[f] is None and f not in reasons:
            reasons[f] = f"{src}{k} ausente ou inválido"

    n_experts = next((c[k] for k in MOE_EXPERT_KEYS if _int(c.get(k)) is not None), None)
    if n_experts is not None and n_experts > 1:
        out["is_moe"] = True
        out["moe_inactive_params"], why = moe_inactive_params(c, n_experts, out["n_layers"])
        if why:
            reasons["params_active"] = why
    return out


def moe_inactive_params(c: dict, n_experts: int, n_layers: int | None) -> tuple[int | None, str | None]:
    """Parâmetros dos experts roteados que NÃO são lidos por token.

    inativos = (n_experts - experts_por_token) * camadas_moe * 3 * hidden * ffn_expert

    3 matrizes por expert (gate, up, down) só vale para FFN com porta; outra
    ativação, ou qualquer dado faltando, devolve None com motivo. Experts
    compartilhados (n_shared_experts) sempre rodam e não entram na conta.
    Ignora vieses dos experts (ordem de 1e-4 do total).
    """
    k = _int(c.get("num_experts_per_tok"))
    hidden = _int(c.get("hidden_size"))
    act = c.get("hidden_act")
    if "moe_intermediate_size" in c:
        ffn = _int(c.get("moe_intermediate_size"))
    elif "num_local_experts" in c:  # estilo Mixtral: intermediate_size é o do expert
        ffn = _int(c.get("intermediate_size"))
    else:
        ffn = None
    if k is None or hidden is None or ffn is None or n_layers is None:
        return None, "MoE sem num_experts_per_tok, hidden_size, tamanho do FFN do expert ou num_hidden_layers"
    if act not in GATED_ACTS:
        return None, f"MoE com hidden_act {act!r}: fórmula de 3 matrizes por expert não se aplica"
    if k >= n_experts:
        return None, "num_experts_per_tok >= número de experts"

    step = c.get("decoder_sparse_step", 1)
    freq = c.get("moe_layer_freq", 1)
    if step != 1 or freq != 1:
        return None, "MoE intercalado (decoder_sparse_step/moe_layer_freq != 1) não suportado"
    dense_layers = c.get("first_k_dense_replace", 0) or 0
    mlp_only = c.get("mlp_only_layers") or []
    if not isinstance(dense_layers, int) or not isinstance(mlp_only, list):
        return None, "first_k_dense_replace/mlp_only_layers com formato inesperado"
    moe_layers = n_layers - dense_layers - len(set(mlp_only))
    if moe_layers < 1:
        return None, "nenhuma camada MoE pela config"
    return (n_experts - k) * moe_layers * 3 * hidden * ffn, None


def _sibling_size(s: dict) -> int | None:
    lfs = s.get("lfs")
    if isinstance(lfs, dict) and _int(lfs.get("size")) is not None:
        return lfs["size"]
    return _int(s.get("size"))


def gguf_quants(blob_info: dict, params_total: int | None,
                warnings: list[str] | None = None) -> list[dict]:
    """Quants GGUF do repo (API com ?blobs=true). bpw = file_bytes*8/params_total.

    Por quant (mesmo nome de arquivo sem o sufixo de shard) pode haver um
    arquivo único e/ou conjuntos de shards (-00001-of-0000N). Gera UMA quant:
    o arquivo único quando existe; senão o único conjunto completo de shards,
    somado. Conjunto incompleto é descartado. Se arquivo único e shards
    divergem em mais de 1% no total, fica o arquivo único e entra um aviso em
    `warnings`. mmproj (projetor de visão) e arquivos sem nome de quant são
    ignorados.
    """
    if warnings is None:
        warnings = []
    repo = blob_info.get("id")
    # key -> {"single": (path, size) | None, "sets": {n_total: {idx: (path, size)}}}
    groups: dict[str, dict[str, Any]] = {}
    for s in blob_info.get("siblings") or []:
        path = s.get("rfilename")
        if not isinstance(path, str) or not path.lower().endswith(".gguf"):
            continue
        if "mmproj" in path.lower():
            continue
        stem = path[: -len(".gguf")]
        m = _SHARD_RE.search(stem)
        key = stem[: m.start()] if m else stem
        g = groups.setdefault(key, {"single": None, "sets": {}})
        if m:
            g["sets"].setdefault(int(m.group(2)), {})[int(m.group(1))] = (path, _sibling_size(s))
        else:
            g["single"] = (path, _sibling_size(s))

    named: list[tuple[str, str, str, int | None]] = []  # (nome, key, url_path, file_bytes)
    for key in sorted(groups):
        g = groups[key]
        matches = _QUANT_RE.findall(key.rsplit("/", 1)[-1]) or _QUANT_RE.findall(key)
        if not matches:
            continue
        complete_sets = []
        for n_total in sorted(g["sets"]):
            parts = g["sets"][n_total]
            if sorted(parts) == list(range(1, n_total + 1)):
                sizes = [parts[i][1] for i in sorted(parts)]
                total = sum(sizes) if all(x is not None for x in sizes) else None
                complete_sets.append((parts[1][0], total))
        if g["single"] is not None:
            path, size = g["single"]
            for _, total in complete_sets:
                if size is not None and total is not None and abs(total - size) * 100 > size:
                    warnings.append(f"{key}: arquivo único ({size} bytes) e shards ({total} bytes) "
                                    "divergem mais de 1%; usado o arquivo único")
            named.append((matches[-1].upper(), key, path, size))
        elif len(complete_sets) == 1:
            path, total = complete_sets[0]
            named.append((matches[-1].upper(), key, path, total))
        elif len(complete_sets) > 1:
            warnings.append(f"{key}: mais de um conjunto de shards e nenhum arquivo único; quant ignorada")

    counts: dict[str, int] = {}
    for name, _, _, _ in named:
        counts[name] = counts.get(name, 0) + 1

    quants: list[dict] = []
    for name, key, path, file_bytes in named:
        bpw = None
        if file_bytes is not None and params_total is not None:
            bpw = round(file_bytes * 8 / params_total, 4)
            if not 0 < bpw <= BPW_MAX:
                warnings.append(f"{key}: bpw calculado {bpw} fora de (0, {BPW_MAX}]; bpw fica null")
                bpw = None
        quants.append({
            "format": "GGUF",
            "name": name if counts[name] == 1 else key,
            "bpw": bpw,
            "bpw_source": "file_bytes" if bpw is not None else None,
            "file_bytes": file_bytes,
            "file_url": f"{BASE}/{repo}/resolve/main/{path}" if repo else None,
        })
    return sorted(quants, key=lambda q: (q["name"]))


def license_fields(info: dict, policy: dict) -> dict:
    """cardData.license literal + URL do cartão + commercial_ok pela tabela (Q4)."""
    card = info.get("cardData") if isinstance(info.get("cardData"), dict) else {}
    lic = card.get("license")
    if isinstance(lic, list):
        lic = lic[0] if len(lic) == 1 else None  # várias licenças: ambíguo
    if not isinstance(lic, str) or not lic.strip():
        lic = None
    repo = info.get("id")
    table = policy.get("licenses", {})
    ok = table.get(lic.strip().lower()) if lic is not None else None
    return {
        "license": lic,
        "license_source": model_card_url(repo) if lic is not None and repo else None,
        "commercial_ok": ok if isinstance(ok, bool) else None,
    }


def tasks_from_info(info: dict) -> list[str] | None:
    tag = info.get("pipeline_tag")
    tasks = PIPELINE_TASKS.get(tag) if isinstance(tag, str) else None
    return list(tasks) if tasks else None


def build_entry(repo: str, info: dict, config: dict | None, gguf_info: dict | None,
                policy: dict, today: str) -> dict:
    """Entrada do catálogo, válida contra catalog.schema.json."""
    entry, _, _ = build_entry_with_reasons(repo, info, config, gguf_info, policy, today)
    return entry


def build_entry_with_reasons(repo: str, info: dict, config: dict | None, gguf_info: dict | None,
                             policy: dict, today: str) -> tuple[dict, dict[str, str], list[str]]:
    """Como build_entry, mais motivos dos campos null e avisos (ex.: shards divergentes)."""
    warnings: list[str] = []
    arch = map_config(config)
    reasons = dict(arch["null_reasons"])

    st = info.get("safetensors")
    params_total = _int(st.get("total")) if isinstance(st, dict) else None
    if params_total is None:
        reasons["params_total"] = "API sem safetensors.total"

    if not arch["is_moe"]:
        params_active = params_total
        if params_total is None:
            reasons["params_active"] = "depende de params_total"
    elif params_total is None or arch["moe_inactive_params"] is None:
        params_active = None
        reasons.setdefault("params_active", "depende de params_total")
    else:
        params_active = params_total - arch["moe_inactive_params"]
        if not 1 <= params_active < params_total:
            reasons["params_active"] = "parâmetros inativos calculados incompatíveis com safetensors.total"
            params_active = None

    lic = license_fields({**info, "id": repo}, policy)
    if lic["license"] is None:
        reasons["license"] = "API sem cardData.license"
        reasons["license_source"] = "sem licença"
    tasks = tasks_from_info(info)
    if tasks is None:
        reasons["tasks"] = f"pipeline_tag {info.get('pipeline_tag')!r} fora da tabela"

    quants = gguf_quants(gguf_info, params_total, warnings) if gguf_info else []
    if not quants:
        reasons["quants"] = "nenhuma quant GGUF encontrada" if gguf_info else "sem --gguf-repo"

    entry: dict[str, Any] = {
        "id": repo,
        "source_url": model_card_url(repo),
        "gguf_repo": gguf_info.get("id") if gguf_info else None,
        **lic,
        "tasks": tasks,
        "params_total": params_total,
        "params_active": params_active,
        **{f: arch[f] for f in ARCH_FIELDS},
        "quants": quants,
        "verified_at": today,
    }
    incomplete = [f for f in STRUCTURAL_FIELDS if entry[f] is None]
    if not quants:
        incomplete.append("quants")
    entry["complete"] = not incomplete
    entry["incomplete_fields"] = incomplete
    return entry, {f: reasons.get(f, "ausente") for f in incomplete}, warnings


def apply_task_overrides(entry: dict, overrides: dict) -> tuple[dict, list[str]]:
    """Override revisado por humano (data/task_overrides.json): só ACRESCENTA tarefas.

    Não remove tarefas nem toca outro campo. Se tasks veio null da API, o
    override não se aplica (preencheria um campo incompleto) e sai um aviso.
    """
    o = (overrides.get("overrides") or {}).get(entry["id"])
    if not o:
        return entry, []
    add = [t for t in o.get("add_tasks", []) if t in TASK_ENUM]
    bad = [t for t in o.get("add_tasks", []) if t not in TASK_ENUM]
    msgs = [f"override com tarefa desconhecida ignorada: {t}" for t in bad]
    if entry["tasks"] is None:
        return entry, msgs + ["tasks null pela API: override de tarefas não aplicado"]
    new = list(entry["tasks"]) + [t for t in add if t not in entry["tasks"]]
    if new == entry["tasks"]:
        return entry, msgs
    out = dict(entry, tasks=new)
    return out, msgs + [f"tasks +{','.join(t for t in new if t not in entry['tasks'])} por override "
                        f"({o.get('reason', 'sem motivo')}; {o.get('source', 'sem fonte')})"]


def remove(catalog: dict, model_id: str) -> dict:
    """Catálogo sem o modelo; KeyError se o id não existe."""
    models = catalog.get("models", [])
    if not any(m.get("id") == model_id for m in models):
        raise KeyError(model_id)
    return dict(catalog, models=[m for m in models if m.get("id") != model_id])


def upsert(catalog: dict, entry: dict) -> dict:
    """Substitui ou acrescenta por id; modelos ordenados por id."""
    models = [m for m in catalog.get("models", []) if m.get("id") != entry["id"]] + [entry]
    out = {k: v for k, v in catalog.items() if k != "models"}
    out.update(schema_version=1, kind="catalog", models=sorted(models, key=lambda m: m["id"]))
    return out


def check_catalog(catalog: dict, today: str) -> tuple[list[str], list[str]]:
    """Coerência que o schema não cobre. Devolve (erros, avisos)."""
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    ref = date.fromisoformat(today)
    for i, m in enumerate(catalog.get("models", [])):
        mid = m.get("id", f"models[{i}]")
        if mid in seen:
            errors.append(f"{mid}: id duplicado")
        seen.add(mid)
        if not m.get("source_url"):
            errors.append(f"{mid}: source_url nulo")
        nulls = [f for f in STRUCTURAL_FIELDS if m.get(f) is None]
        if not m.get("quants"):
            nulls.append("quants")
        listed = list(m.get("incomplete_fields") or [])
        if m.get("complete"):
            if nulls:
                errors.append(f"{mid}: complete=true com campos nulos: {', '.join(nulls)}")
            if not m.get("license_source"):
                errors.append(f"{mid}: complete=true sem license_source")
        if sorted(nulls) != sorted(listed):
            errors.append(f"{mid}: incomplete_fields {sorted(listed)} não bate com campos nulos {sorted(nulls)}")
        if bool(m.get("complete")) == bool(nulls):
            errors.append(f"{mid}: complete={m.get('complete')} incoerente com campos nulos")
        try:
            age = (ref - date.fromisoformat(m.get("verified_at", ""))).days
        except (TypeError, ValueError):
            errors.append(f"{mid}: verified_at inválido")
            continue
        if age > STALE_DAYS:
            warnings.append(f"{mid}: verificado há {age} dias (> {STALE_DAYS}); rode scout catalog add de novo")
    return errors, warnings
