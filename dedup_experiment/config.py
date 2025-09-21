from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class DatasetConfig:
    name: str = "togethercomputer/RedPajama-Data-V2"
    subset: Optional[str] = "sample"
    split: str = "train"
    text_field: str = "text"
    max_documents: int = 200
    streaming: bool = True
    shuffle_seed: int = 13
    shuffle_buffer: int = 1000
    local_cache_dir: Optional[str] = None
    train_fraction: float = 0.98
    val_fraction: float = 0.01
    test_fraction: float = 0.01


@dataclass
class NearDedupConfig:
    enabled: bool = True
    shingle_size: int = 5
    num_permutations: int = 128
    band_size: int = 4
    threshold: float = 0.85
    skip_short_tokens: int = 200


@dataclass
class DedupConfig:
    chunk_tokens: int = 1024
    stride_tokens: int = 1024
    min_chunk_tokens: int = 128
    lowercase: bool = True
    strip_html: bool = True
    collapse_whitespace: bool = True
    keep_metadata: bool = False
    near: NearDedupConfig = field(default_factory=NearDedupConfig)
    cross_split: bool = True


@dataclass
class TrainingConfig:
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    grad_clip: float = 1.0
    batch_size: int = 32
    max_steps: int = 300
    warmup_steps: int = 30
    eval_interval: int = 50
    log_interval: int = 10
    device: str = "auto"
    precision: str = "bf16"
    save_checkpoints: bool = False
    max_train_tokens: Optional[int] = None


@dataclass
class EvaluationConfig:
    val_batch_size: int = 8
    max_val_tokens: int = 32768
    downstream_tasks: Dict[str, int] = field(default_factory=lambda: {
        "lambada_openai": 200,
        "piqa": 200,
        "hellaswag": 200,
    })
    canary_prompts: int = 100
    canary_decode_max_tokens: int = 32


@dataclass
class RunConfig:
    name: str
    apply_exact: bool = True
    apply_near: bool = False
    near_threshold: Optional[float] = None
    description: Optional[str] = None


@dataclass
class ExperimentConfig:
    seed: int = 17
    output_dir: str = "outputs"
    report_file: str = "outputs/report.json"
    table_file: str = "outputs/summary.tsv"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    runs: List[RunConfig] = field(default_factory=lambda: [
        RunConfig(name="no_dedup", apply_exact=False, apply_near=False, description="Raw chunks"),
        RunConfig(name="exact", apply_exact=True, apply_near=False, description="Exact hash dedup"),
        RunConfig(name="near", apply_exact=True, apply_near=True, near_threshold=0.85, description="Exact + MinHash (0.85)")
    ])
    num_canaries: int = 100
    canary_length: int = 24
    canary_prefix: str = "CANARY"


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _merge_dict_into_dataclass(dc, data: dict):
    for field_info in dataclasses.fields(dc):
        name = field_info.name
        if name not in data:
            continue
        value = getattr(dc, name)
        incoming = data[name]
        if dataclasses.is_dataclass(value):
            _merge_dict_into_dataclass(value, incoming)
        elif isinstance(value, list) and value and dataclasses.is_dataclass(value[0]):
            new_list = []
            for item in incoming:
                new_item = dataclasses.replace(value[0])
                _merge_dict_into_dataclass(new_item, item)
                new_list.append(new_item)
            setattr(dc, name, new_list)
        else:
            setattr(dc, name, incoming)


def load_config(path: str) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if path is None:
        return cfg
    data = _load_yaml(Path(path))
    if data is None:
        return cfg
    _merge_dict_into_dataclass(cfg, data)
    # Apply run-specific threshold overrides if given.
    for run in cfg.runs:
        if run.apply_near and run.near_threshold is None:
            run.near_threshold = cfg.dedup.near.threshold
    return cfg
