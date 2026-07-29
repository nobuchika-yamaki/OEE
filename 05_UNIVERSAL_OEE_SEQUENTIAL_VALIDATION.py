#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_UNIVERSAL_OEE_SEQUENTIAL_VALIDATION.py

Single-file, restartable test of the proposition that hereditary possibility-space
expansion and organism-driven environmental construction jointly sustain new ecological possibilities across prespecified abstract, non-Earth-specific worlds.

The script first verifies a prespecified global ecological turnover scale in a
demographic/numerical preflight independent of open-endedness outcomes. It then runs
the same worlds, seeds, horizon, instantaneous model equations, and outcome
definitions for all conditions. Primary trajectories are indexed by cumulative births
(evolutionary opportunities), not elapsed simulation time.

Default execution:
    python3 05_UNIVERSAL_OEE_SEQUENTIAL_VALIDATION.py

Output:
    ~/Desktop/universal_oee_sequential_validation_v6
"""

from __future__ import annotations

# Avoid nested BLAS/OpenMP parallelism inside worker processes.
import os
for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import copy
import csv
import dataclasses
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import pickle
import platform
import random
import shutil
import statistics
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from scipy.optimize import curve_fit
    from scipy.stats import t as student_t
    SCIPY_AVAILABLE = True
except Exception:
    curve_fit = None
    student_t = None
    SCIPY_AVAILABLE = False


try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    psutil = None
    PSUTIL_AVAILABLE = False


SCRIPT_VERSION = "6.0.0"
SCHEMA_VERSION = 7
ROOT_SEED_DEFAULT = 20260726


# -----------------------------------------------------------------------------
# Deterministic utilities and atomic I/O
# -----------------------------------------------------------------------------

def stable_seed(*parts: Any, modulus: int = 2**63 - 1) -> int:
    payload = "|".join(str(x) for x in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    return int.from_bytes(digest, "little") % modulus


def stable_id(*parts: Any, n: int = 16) -> str:
    payload = "|".join(str(x) for x in parts).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=n).hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def write_json_gz(path: Path, obj: Any) -> None:
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    atomic_write_bytes(path, gzip.compress(raw, compresslevel=1))


def read_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(json_safe(x) for x in value)
    if dataclasses.is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(x) for x in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_write_text(path, "")
        return
    keys: List[str] = []
    seen: Set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})
    os.replace(tmp, path)


def nested_float_dict() -> defaultdict:
    return defaultdict(float)


def mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not vals:
        return math.nan, math.nan
    if len(vals) == 1:
        return vals[0], 0.0
    return float(statistics.mean(vals)), float(statistics.stdev(vals))


def paired_mean_ci(values: Sequence[float], alpha: float = 0.05) -> Tuple[float, float, float]:
    vals = np.asarray([float(x) for x in values if math.isfinite(float(x))], dtype=float)
    if vals.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(vals.mean())
    if vals.size == 1:
        return mean, math.nan, math.nan
    se = float(vals.std(ddof=1) / math.sqrt(vals.size))
    if SCIPY_AVAILABLE and student_t is not None:
        crit = float(student_t.ppf(1.0 - alpha / 2.0, vals.size - 1))
    else:
        crit = 1.959963984540054
    return mean, mean - crit * se, mean + crit * se


def exact_sign_flip_p(
    values: Sequence[float],
    seed: int,
    max_draws: int = 20_000,
    alternative: str = "two-sided",
) -> float:
    vals = np.asarray([float(x) for x in values if math.isfinite(float(x))], dtype=float)
    if vals.size == 0:
        return math.nan
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(f"Unsupported alternative: {alternative}")
    observed_mean = float(vals.mean())

    def count_exceed(candidates: np.ndarray) -> int:
        if alternative == "greater":
            return int(np.count_nonzero(candidates >= observed_mean - 1e-15))
        if alternative == "less":
            return int(np.count_nonzero(candidates <= observed_mean + 1e-15))
        return int(np.count_nonzero(np.abs(candidates) >= abs(observed_mean) - 1e-15))

    n = vals.size
    if n <= 16:
        total = 1 << n
        masks = np.arange(total, dtype=np.uint32)[:, None]
        bits = (masks >> np.arange(n, dtype=np.uint32)[None, :]) & 1
        signs = 1.0 - 2.0 * bits.astype(np.float64)
        candidates = (signs @ vals) / n
        return count_exceed(candidates) / total

    rng = np.random.default_rng(seed)
    exceed = 1  # Monte Carlo plus-one correction.
    remaining = int(max_draws)
    batch_size = 4096
    while remaining > 0:
        batch = min(batch_size, remaining)
        bits = rng.integers(0, 2, size=(batch, n), dtype=np.int8)
        signs = 1.0 - 2.0 * bits.astype(np.float64)
        candidates = (signs @ vals) / n
        exceed += count_exceed(candidates)
        remaining -= batch
    return exceed / (max_draws + 1)


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    p = np.asarray([float(x) if x is not None else math.nan for x in p_values], dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p))
    if valid.size == 0:
        return q.tolist()
    order = valid[np.argsort(p[valid], kind="mergesort")]
    m = order.size
    running = 1.0
    for rank_from_end, idx in enumerate(order[::-1], start=1):
        rank = m - rank_from_end + 1
        running = min(running, float(p[idx]) * m / rank)
        q[idx] = min(max(running, 0.0), 1.0)
    return q.tolist()


def wilson_lower(successes: int, trials: int, z: float = 1.959963984540054) -> float:
    if trials <= 0:
        return 0.0
    phat = successes / trials
    denom = 1.0 + z * z / trials
    center = phat + z * z / (2.0 * trials)
    margin = z * math.sqrt(phat * (1.0 - phat) / trials + z * z / (4.0 * trials * trials))
    return max(0.0, (center - margin) / denom)


@dataclass
class CompensatedSum:
    """Kahan-compensated scalar ledger, safe to pickle in checkpoints."""
    total: float = 0.0
    correction: float = 0.0
    updates: int = 0

    def add(self, value: float) -> None:
        x = float(value)
        y = x - self.correction
        t = self.total + y
        self.correction = (t - self.total) - y
        self.total = t
        self.updates += 1

    def value(self) -> float:
        return float(self.total)


# -----------------------------------------------------------------------------
# Experimental design
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Condition:
    name: str
    constructive: bool
    extensible: bool
    mutation: bool
    # Additional half-life of non-primitive, organism-constructed substrates in
    # effective ecological time units. None means no additional removal; 0 means
    # removal after every ecological update.
    constructed_half_life: Optional[float] = None
    factorial: bool = True
    retention_sweep: bool = False


RETENTION_HALF_LIVES: Tuple[float, ...] = (1.0, 4.0, 16.0, 64.0)

CONDITIONS: Dict[str, Condition] = {
    # Prespecified 2 x 2 factorial: environmental construction (C) x hereditary
    # possibility-space extensibility (G). All four cells retain mutation.
    "full": Condition("full", True, True, True),
    "construction_only": Condition("construction_only", True, False, True),
    "extensibility_only": Condition("extensibility_only", False, True, True),
    "closed_control": Condition("closed_control", False, False, True),
    # Mechanistic negative controls outside the factorial.
    "products_erased": Condition(
        "products_erased", True, True, True,
        constructed_half_life=0.0, factorial=False, retention_sweep=True,
    ),
    "no_mutation": Condition("no_mutation", True, True, False, factorial=False),
    # Continuous intervention on the persistence of constructed environmental states.
    **{
        f"retention_hl_{int(h)}": Condition(
            f"retention_hl_{int(h)}", True, True, True,
            constructed_half_life=h, factorial=False, retention_sweep=True,
        )
        for h in RETENTION_HALF_LIVES
    },
}

FACTORIAL_CONDITIONS = tuple(k for k, v in CONDITIONS.items() if v.factorial)
RETENTION_CONDITIONS = (
    "products_erased",
    *(f"retention_hl_{int(h)}" for h in RETENTION_HALF_LIVES),
    "full",
)

CORE_CONDITIONS: Tuple[str, ...] = (
    "full", "construction_only", "extensibility_only",
    "closed_control", "products_erased", "no_mutation",
)
RETENTION_INTERMEDIATE_CONDITIONS: Tuple[str, ...] = tuple(
    f"retention_hl_{int(h)}" for h in RETENTION_HALF_LIVES
)
EXPECTED_CORE_RUNS_DEFAULT = 4 * 3 * 2 * 3 * len(CORE_CONDITIONS)
EXPECTED_RETENTION_RUNS_DEFAULT = 4 * 3 * 2 * 3 * len(RETENTION_INTERMEDIATE_CONDITIONS)


@dataclass
class CampaignConfig:
    n_sites: int = 64
    world_replicates: int = 2
    evolutionary_seeds: int = 3
    steps: int = 120_000
    establishment_steps: int = 8_000
    window_size: int = 500
    checkpoint_interval: int = 20_000
    causal_repeats: int = 8
    causal_horizon: int = 2_000
    conditions: Tuple[str, ...] = (
        "full", "construction_only", "extensibility_only",
        "closed_control", "products_erased", "no_mutation",
    )
    topologies: Tuple[str, ...] = ("lattice", "random_regular", "small_world", "modular")
    forcings: Tuple[str, ...] = ("constant", "periodic", "stochastic_switching")

    # Birth-indexed operational gate. Equal birth thirds are used rather than equal
    # time thirds, so conditions are compared per evolutionary opportunity.
    min_analysis_births_for_epochs: int = 60
    family_direction_required: float = 1.0
    alpha: float = 0.05

    # Integrated mechanism analyses.
    birth_grid_bins: int = 12
    mechanism_bootstrap_draws: int = 1_000
    boundary_bootstrap_draws: int = 1_000
    retention_bootstrap_draws: int = 2_000

    # Prespecified continuation. In auto mode, long-term continuation is triggered
    # when finite-horizon OEE classification remains mixed or trajectory fits remain
    # insufficient in the Full condition.
    longterm_multiplier: int = 3
    longterm_conditions: Tuple[str, ...] = ("full", "products_erased")
    longterm_mode: str = "auto"  # auto, always, never
    longterm_min_valid_trajectory_fraction: float = 0.80

    # Preflight examines demographic/numerical adequacy only, never OEE outcomes.
    preflight_steps: int = 10_000
    preflight_turnover_candidates: Tuple[float, ...] = (128.0,)
    preflight_min_births_each_world: int = 7
    preflight_median_births: int = 15
    preflight_max_population: int = 512
    preflight_max_extinction_fraction: float = 0.0
    preflight_min_expected_functional_mutants_main: float = 15.0


@dataclass
class ModelConfig:
    # Numerical dynamics
    dt: float = 0.01
    mean_degree: int = 4
    transport_range: Tuple[float, float] = (0.01, 0.05)
    dissipation_range: Tuple[float, float] = (0.0005, 0.003)
    source_strength_range: Tuple[float, float] = (0.15, 0.45)
    normalized_source_total: float = 1.0
    periodic_amplitude: float = 0.50
    periodic_period: float = 100.0
    switch_rate: float = 0.002
    initial_substrate: float = 2.0
    substrate_prune_eps: float = 1e-10
    max_fraction_consumed_per_step: float = 0.25
    transport_interval: int = 1
    prune_interval: int = 50
    quality_check_interval: int = 100

    # Organisms and energy. turnover_scale multiplies source, transport,
    # dissipation, reaction, maintenance, and forcing rates and is selected once
    # from preflight, then frozen for every condition.
    turnover_scale: float = 1.0
    founders: int = 32
    founder_energy: float = 1.0
    reproduction_threshold: float = 2.0
    basal_cost: float = 0.004
    module_maintenance_cost: float = 0.001
    activation_cost: float = 0.0005
    energy_capture: float = 0.8
    dispersal_probability: float = 0.25

    # Instantaneous module defaults and bounds. No delay state or inherited temporal
    # buffer is present in this version.
    founder_gamma: float = 1.0
    founder_beta: float = 2.0
    founder_sigma: float = 0.1
    founder_k: float = 1.0
    founder_kappa: float = 1.0
    theta_min: float = 1e-4
    theta_max: float = 20.0
    gamma_max: float = 4.0

    # Mutation per birth
    mu_scalar: float = 0.05
    scalar_sigma: float = 0.08
    mu_symbol_sub: float = 0.07
    mu_symbol_ins: float = 0.025
    mu_symbol_del: float = 0.025
    mu_operation: float = 0.025
    mu_module_dup: float = 0.03
    mu_module_del: float = 0.015
    functional_mutation_probability_extensible_lower: float = 0.0
    functional_mutation_probability_closed_lower: float = 0.0

    # Novelty and persistence
    flux_threshold: float = 0.01
    interaction_threshold: float = 0.001
    substrate_presence_threshold: float = 0.01
    persistence_windows: int = 10
    min_organisms: int = 5
    min_generations: int = 3
    min_causal_events_for_oee_fit: int = 5

    # Safety stops do not alter the dynamics.
    safety_max_population: int = 2_048
    safety_max_substrates: int = 20_000
    safety_max_genome_modules: int = 256
    safety_max_string_length: int = 256
    mass_balance_relative_tolerance: float = 1e-8
    mass_balance_warning_tolerance: float = 1e-9

    # Logging/runtime
    progress_interval: int = 20_000
    record_lineage_events: bool = True


# -----------------------------------------------------------------------------
# Abstract worlds
# -----------------------------------------------------------------------------

TOPOLOGIES = ("lattice", "random_regular", "small_world", "modular")
FORCINGS = ("constant", "periodic", "stochastic_switching")
ALPHABET_SIZES = (2, 3, 4)


@dataclass(frozen=True)
class WorldKey:
    topology: str
    forcing: str
    replicate: int
    alphabet_size: int

    @property
    def family(self) -> str:
        return f"{self.topology}__{self.forcing}"

    @property
    def world_id(self) -> str:
        return f"W_{self.topology}_{self.forcing}_r{self.replicate:02d}_a{self.alphabet_size}"


@dataclass(frozen=True)
class ModuleGene:
    op: int  # 0 ligation, 1 cleavage
    u: Tuple[int, ...]
    v: Tuple[int, ...]
    k: float
    kappa: float
    gamma: float
    beta: float
    sigma: float

    def function_key(self) -> str:
        if self.op == 0:
            return f"L:{encode_string(self.u)}+{encode_string(self.v)}->{encode_string(self.u + self.v)}"
        return f"C:{encode_string(self.u + self.v)}->{encode_string(self.u)}+{encode_string(self.v)}"

    def reactants(self) -> Tuple[Tuple[int, ...], ...]:
        return (self.u, self.v) if self.op == 0 else (self.u + self.v,)

    def products(self) -> Tuple[Tuple[int, ...], ...]:
        return (self.u + self.v,) if self.op == 0 else (self.u, self.v)


@dataclass
class WorldSpec:
    key: WorldKey
    seed: int
    n_sites: int
    edges_u: np.ndarray
    edges_v: np.ndarray
    neighbors: Tuple[Tuple[int, ...], ...]
    symbol_potential: np.ndarray
    pair_potential: np.ndarray
    primitive_strings: Tuple[Tuple[int, ...], ...]
    source_sites_a: Dict[Tuple[int, ...], np.ndarray]
    source_sites_b: Dict[Tuple[int, ...], np.ndarray]
    source_strength_a: Dict[Tuple[int, ...], np.ndarray]
    source_strength_b: Dict[Tuple[int, ...], np.ndarray]
    source_phase: Dict[Tuple[int, ...], np.ndarray]
    transport: float
    dissipation: float
    periodic_amplitude: float
    periodic_period: float
    switch_rate: float
    founder_module: ModuleGene

    def potential(self, s: Tuple[int, ...]) -> float:
        total = float(sum(self.symbol_potential[a] for a in s))
        if len(s) > 1:
            total += float(sum(self.pair_potential[s[j], s[j + 1]] for j in range(len(s) - 1)))
        return total


def encode_string(s: Tuple[int, ...]) -> str:
    return ".".join(str(x) for x in s)


def decode_string(s: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in s.split(".")) if s else tuple()


def reaction_delta_phi(world: WorldSpec, gene: ModuleGene) -> float:
    if gene.op == 0:
        return world.potential(gene.u) + world.potential(gene.v) - world.potential(gene.u + gene.v)
    return world.potential(gene.u + gene.v) - world.potential(gene.u) - world.potential(gene.v)


def lattice_edges(n: int) -> Tuple[np.ndarray, np.ndarray]:
    side = int(round(math.sqrt(n)))
    if side * side == n:
        rows = cols = side
    else:
        rows = int(math.floor(math.sqrt(n)))
        while rows > 1 and n % rows:
            rows -= 1
        cols = n // rows
    edges: Set[Tuple[int, int]] = set()
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            for rr, cc in (((r + 1) % rows, c), (r, (c + 1) % cols)):
                v = rr * cols + cc
                if u != v:
                    edges.add((min(u, v), max(u, v)))
    ordered = sorted(edges)
    return np.asarray([x[0] for x in ordered], dtype=np.int32), np.asarray([x[1] for x in ordered], dtype=np.int32)


def random_regular_edges(n: int, degree: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if degree >= n or (n * degree) % 2:
        raise ValueError("Invalid random-regular graph parameters")
    for _ in range(1000):
        stubs = np.repeat(np.arange(n, dtype=np.int32), degree)
        rng.shuffle(stubs)
        edges: Set[Tuple[int, int]] = set()
        valid = True
        for j in range(0, stubs.size, 2):
            a, b = int(stubs[j]), int(stubs[j + 1])
            edge = (min(a, b), max(a, b))
            if a == b or edge in edges:
                valid = False
                break
            edges.add(edge)
        if valid and len(edges) == n * degree // 2:
            ordered = sorted(edges)
            return np.asarray([x[0] for x in ordered], dtype=np.int32), np.asarray([x[1] for x in ordered], dtype=np.int32)
    # Deterministic degree-preserving fallback.
    edges = set()
    for a in range(n):
        for d in range(1, degree // 2 + 1):
            b = (a + d) % n
            edges.add((min(a, b), max(a, b)))
    ordered = sorted(edges)
    return np.asarray([x[0] for x in ordered], dtype=np.int32), np.asarray([x[1] for x in ordered], dtype=np.int32)


def small_world_edges(n: int, degree: int, rng: np.random.Generator, p_rewire: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    if degree % 2:
        degree += 1
    edges: Set[Tuple[int, int]] = set()
    for a in range(n):
        for d in range(1, degree // 2 + 1):
            b = (a + d) % n
            edges.add((min(a, b), max(a, b)))
    original = sorted(edges)
    for edge in original:
        if rng.random() >= p_rewire:
            continue
        a, _ = edge
        candidates = [b for b in range(n) if b != a and (min(a, b), max(a, b)) not in edges]
        if not candidates:
            continue
        edges.remove(edge)
        b = int(rng.choice(candidates))
        edges.add((min(a, b), max(a, b)))
    ordered = sorted(edges)
    return np.asarray([x[0] for x in ordered], dtype=np.int32), np.asarray([x[1] for x in ordered], dtype=np.int32)


def modular_edges(n: int, degree: int, rng: np.random.Generator, modules: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    groups = np.array_split(np.arange(n, dtype=np.int32), modules)
    edges: Set[Tuple[int, int]] = set()
    for group in groups:
        half = min(max(1, degree // 2), max(1, (len(group) - 1) // 2))
        for j, a in enumerate(group):
            for d in range(1, half + 1):
                b = int(group[(j + d) % len(group)])
                edges.add((min(int(a), b), max(int(a), b)))
    for j in range(modules):
        a = int(rng.choice(groups[j]))
        b = int(rng.choice(groups[(j + 1) % modules]))
        edges.add((min(a, b), max(a, b)))
    ordered = sorted(edges)
    return np.asarray([x[0] for x in ordered], dtype=np.int32), np.asarray([x[1] for x in ordered], dtype=np.int32)


def build_neighbors(n: int, u: np.ndarray, v: np.ndarray) -> Tuple[Tuple[int, ...], ...]:
    adj: List[List[int]] = [[] for _ in range(n)]
    for a, b in zip(u.tolist(), v.tolist()):
        adj[a].append(b)
        adj[b].append(a)
    return tuple(tuple(sorted(x)) for x in adj)


def _normalized_strengths(
    primitive: Sequence[Tuple[int, ...]],
    sites: Mapping[Tuple[int, ...], np.ndarray],
    rng: np.random.Generator,
    low: float,
    high: float,
    target_total: float,
) -> Dict[Tuple[int, ...], np.ndarray]:
    raw: Dict[Tuple[int, ...], np.ndarray] = {}
    total = 0.0
    for s in primitive:
        arr = rng.uniform(low, high, size=len(sites[s])).astype(np.float64)
        raw[s] = arr
        total += float(arr.sum())
    # Mean total external material input is invariant to alphabet size and source count.
    factor = target_total / max(total, 1e-12)
    return {s: arr * factor for s, arr in raw.items()}


def generate_world(key: WorldKey, n_sites: int, cfg: ModelConfig, root_seed: int) -> WorldSpec:
    seed = stable_seed("world", root_seed, key.world_id)
    rng = np.random.default_rng(seed)
    n = int(n_sites)
    if key.topology == "lattice":
        edges_u, edges_v = lattice_edges(n)
    elif key.topology == "random_regular":
        edges_u, edges_v = random_regular_edges(n, cfg.mean_degree, rng)
    elif key.topology == "small_world":
        edges_u, edges_v = small_world_edges(n, cfg.mean_degree, rng)
    elif key.topology == "modular":
        edges_u, edges_v = modular_edges(n, cfg.mean_degree, rng)
    else:
        raise ValueError(key.topology)
    neighbors = build_neighbors(n, edges_u, edges_v)

    alphabet = key.alphabet_size
    primitive: List[Tuple[int, ...]] = [(a,) for a in range(alphabet)]
    pairs = [(a, b) for a in range(alphabet) for b in range(alphabet)]
    rng.shuffle(pairs)
    primitive.extend(tuple(x) for x in pairs[:max(1, alphabet // 2)])
    primitive_t = tuple(sorted(set(primitive)))

    founder: Optional[ModuleGene] = None
    h = np.empty(alphabet)
    J = np.empty((alphabet, alphabet))
    for _ in range(20_000):
        h = rng.uniform(-1.0, 1.0, size=alphabet).astype(np.float64)
        J = rng.uniform(-1.0, 1.0, size=(alphabet, alphabet)).astype(np.float64)
        candidates: List[ModuleGene] = []
        for su in primitive_t:
            for sv in primitive_t:
                gene = ModuleGene(
                    0, su, sv, cfg.founder_k, cfg.founder_kappa,
                    cfg.founder_gamma, cfg.founder_beta, cfg.founder_sigma,
                )
                # For ligation, symbol terms cancel and only the new boundary contributes.
                dphi = -float(J[su[-1], sv[0]])
                if 0.75 <= dphi <= 1.00:
                    candidates.append(gene)
        if candidates:
            founder = candidates[int(rng.integers(len(candidates)))]
            break
    if founder is None:
        raise RuntimeError(f"Unable to generate viable world {key.world_id}")

    n_source = max(4, n // 16)
    niche_count = min(n_source, max(4, cfg.founders // 8))
    niche_sites = np.sort(rng.choice(n, size=niche_count, replace=False).astype(np.int32))
    founder_reactants = set(founder.reactants())

    source_sites_a: Dict[Tuple[int, ...], np.ndarray] = {}
    source_sites_b: Dict[Tuple[int, ...], np.ndarray] = {}
    source_phase: Dict[Tuple[int, ...], np.ndarray] = {}
    for s in primitive_t:
        if s in founder_reactants:
            remaining = n_source - niche_count
            pool = np.setdiff1d(np.arange(n, dtype=np.int32), niche_sites, assume_unique=False)
            extra_a = rng.choice(pool, size=remaining, replace=False).astype(np.int32) if remaining > 0 else np.empty(0, dtype=np.int32)
            extra_b = rng.choice(pool, size=remaining, replace=False).astype(np.int32) if remaining > 0 else np.empty(0, dtype=np.int32)
            sites_a = np.sort(np.unique(np.concatenate([niche_sites, extra_a])).astype(np.int32))
            sites_b = np.sort(np.unique(np.concatenate([niche_sites, extra_b])).astype(np.int32))
        else:
            sites_a = np.sort(rng.choice(n, size=n_source, replace=False).astype(np.int32))
            sites_b = np.sort(rng.choice(n, size=n_source, replace=False).astype(np.int32))
        source_sites_a[s] = sites_a
        source_sites_b[s] = sites_b
        source_phase[s] = rng.uniform(0.0, 2.0 * math.pi, size=len(sites_a)).astype(np.float64)

    source_strength_a = _normalized_strengths(
        primitive_t, source_sites_a, rng, cfg.source_strength_range[0], cfg.source_strength_range[1],
        cfg.normalized_source_total,
    )
    source_strength_b = _normalized_strengths(
        primitive_t, source_sites_b, rng, cfg.source_strength_range[0], cfg.source_strength_range[1],
        cfg.normalized_source_total,
    )

    return WorldSpec(
        key=key,
        seed=seed,
        n_sites=n,
        edges_u=edges_u,
        edges_v=edges_v,
        neighbors=neighbors,
        symbol_potential=h,
        pair_potential=J,
        primitive_strings=primitive_t,
        source_sites_a=source_sites_a,
        source_sites_b=source_sites_b,
        source_strength_a=source_strength_a,
        source_strength_b=source_strength_b,
        source_phase=source_phase,
        transport=float(rng.uniform(*cfg.transport_range)),
        dissipation=float(rng.uniform(*cfg.dissipation_range)),
        periodic_amplitude=cfg.periodic_amplitude,
        periodic_period=cfg.periodic_period,
        switch_rate=cfg.switch_rate,
        founder_module=founder,
    )


def build_world_keys(campaign: CampaignConfig, root_seed: int) -> List[WorldKey]:
    keys: List[WorldKey] = []
    for topology in campaign.topologies:
        for forcing in campaign.forcings:
            for rep in range(campaign.world_replicates):
                alphabet = ALPHABET_SIZES[
                    stable_seed(root_seed, "alphabet", topology, forcing, rep) % len(ALPHABET_SIZES)
                ]
                keys.append(WorldKey(topology, forcing, rep, alphabet))
    return keys



# -----------------------------------------------------------------------------
# Dynamic environmental substrate store
# -----------------------------------------------------------------------------

class SubstrateStore:
    def __init__(self, n_sites: int, primitive: Sequence[Tuple[int, ...]], initial_capacity: int = 32):
        self.n_sites = int(n_sites)
        self.capacity = max(initial_capacity, len(primitive) * 2, 8)
        self.q = np.zeros((n_sites, self.capacity), dtype=np.float64)
        self.string_to_id: Dict[Tuple[int, ...], int] = {}
        self.id_to_string: List[Tuple[int, ...]] = []
        self.primitive_ids: Set[int] = set()
        self.ever_biological: Set[int] = set()
        self._active_cache = np.empty(0, dtype=np.int32)
        self._active_cache_step = -1
        self._dirty = True
        for s in primitive:
            sid = self.ensure(s)
            self.primitive_ids.add(sid)

    def lookup(self, s: Tuple[int, ...]) -> Optional[int]:
        return self.string_to_id.get(s)

    def ensure(self, s: Tuple[int, ...]) -> int:
        sid = self.string_to_id.get(s)
        if sid is not None:
            return sid
        sid = len(self.id_to_string)
        if sid >= self.capacity:
            new_capacity = self.capacity * 2
            new_q = np.zeros((self.n_sites, new_capacity), dtype=np.float64)
            new_q[:, :self.capacity] = self.q
            self.q = new_q
            self.capacity = new_capacity
        self.string_to_id[s] = sid
        self.id_to_string.append(s)
        self._dirty = True
        return sid

    @property
    def size(self) -> int:
        return len(self.id_to_string)

    def active_ids(self, eps: float, step: int, refresh_interval: int = 20) -> np.ndarray:
        if self._dirty or self._active_cache_step < 0 or step - self._active_cache_step >= refresh_interval:
            if self.size == 0:
                active = np.empty(0, dtype=np.int32)
            else:
                totals = self.q[:, :self.size].sum(axis=0)
                active = np.flatnonzero(totals > eps).astype(np.int32)
                if self.primitive_ids:
                    active = np.unique(np.concatenate([
                        active,
                        np.fromiter(self.primitive_ids, dtype=np.int32),
                    ])).astype(np.int32)
            self._active_cache = active
            self._active_cache_step = step
            self._dirty = False
        return self._active_cache

    def prune(self, eps: float) -> float:
        """Prune negligible nonnegative quantities and return removed symbol mass.

        V4 zeroed these values without adding the removed mass to the material ledger,
        producing the six false-positive mass-balance stops near 1e-8. This version accounts
        for every pruned amount explicitly.
        """
        if self.size == 0:
            return 0.0
        arr = self.q[:, :self.size]
        mask = (arr >= 0.0) & (arr < eps)
        if not np.any(mask):
            return 0.0
        lengths = np.asarray([len(s) for s in self.id_to_string], dtype=np.float64)
        removed = float(np.sum(np.where(mask, arr, 0.0) * lengths[None, :]))
        arr[mask] = 0.0
        self._dirty = True
        return removed

    def total_symbol_mass(self) -> float:
        if self.size == 0:
            return 0.0
        lengths = np.asarray([len(s) for s in self.id_to_string], dtype=np.float64)
        return float(np.sum(self.q[:, :self.size] * lengths[None, :]))


# -----------------------------------------------------------------------------
# Individuals, inheritance, and mutation
# -----------------------------------------------------------------------------

@dataclass
class Individual:
    individual_id: int
    parent_id: int
    lineage_id: int
    site: int
    energy: float
    age: int
    generation: int
    genome: Tuple[ModuleGene, ...]
    x: np.ndarray
    tagged: bool = False


def clone_individual(ind: Individual) -> Individual:
    return Individual(
        individual_id=ind.individual_id,
        parent_id=ind.parent_id,
        lineage_id=ind.lineage_id,
        site=ind.site,
        energy=ind.energy,
        age=ind.age,
        generation=ind.generation,
        genome=ind.genome,
        x=ind.x.copy(),
        tagged=ind.tagged,
    )


def mutate_string(
    s: Tuple[int, ...],
    alphabet_size: int,
    rng: np.random.Generator,
    cfg: ModelConfig,
    *,
    extensible: bool,
) -> Tuple[Tuple[int, ...], bool]:
    out = list(s)
    changed = False
    for j in range(len(out)):
        if rng.random() < cfg.mu_symbol_sub:
            choices = [x for x in range(alphabet_size) if x != out[j]]
            if choices:
                out[j] = int(rng.choice(choices))
                changed = True

    # Draw structural variates in both G levels. Apply them only in the extensible
    # condition, preserving paired mutation streams until structures diverge.
    ins_trigger = rng.random() < cfg.mu_symbol_ins
    ins_pos = int(rng.integers(0, len(out) + 1))
    ins_symbol = int(rng.integers(alphabet_size))
    if extensible and ins_trigger:
        out.insert(ins_pos, ins_symbol)
        changed = True

    del_trigger = rng.random() < cfg.mu_symbol_del
    del_index = int(rng.integers(max(1, len(out))))
    if extensible and len(out) > 1 and del_trigger:
        del out[del_index % len(out)]
        changed = True
    return tuple(out), changed


def mutate_scalar(
    value: float,
    rng: np.random.Generator,
    cfg: ModelConfig,
    *,
    gamma: bool = False,
) -> Tuple[float, bool]:
    if rng.random() >= cfg.mu_scalar:
        return value, False
    new_value = float(value * math.exp(float(rng.normal(0.0, cfg.scalar_sigma))))
    if gamma:
        new_value = min(max(new_value, cfg.theta_min), cfg.gamma_max)
    else:
        new_value = min(max(new_value, cfg.theta_min), cfg.theta_max)
    return new_value, not math.isclose(new_value, value, rel_tol=0.0, abs_tol=0.0)


@dataclass
class MutationResult:
    genome: Tuple[ModuleGene, ...]
    parent_mapping: List[Optional[int]]
    state_compatible: List[bool]
    heritable_change: bool
    transformation_change: bool
    structural_change: bool


def mutate_genome(
    genome: Tuple[ModuleGene, ...],
    condition: Condition,
    alphabet_size: int,
    rng: np.random.Generator,
    cfg: ModelConfig,
) -> MutationResult:
    if not condition.mutation:
        return MutationResult(genome, list(range(len(genome))), [True] * len(genome), False, False, False)

    modules: List[Tuple[ModuleGene, Optional[int], bool]] = []
    any_change = False
    transformation_change = False
    for idx, g in enumerate(genome):
        op = g.op
        op_changed = False
        if rng.random() < cfg.mu_operation:
            op = 1 - op
            op_changed = True
        u, u_changed = mutate_string(g.u, alphabet_size, rng, cfg, extensible=condition.extensible)
        v, v_changed = mutate_string(g.v, alphabet_size, rng, cfg, extensible=condition.extensible)
        k, c1 = mutate_scalar(g.k, rng, cfg)
        kappa, c2 = mutate_scalar(g.kappa, rng, cfg)
        gamma, c3 = mutate_scalar(g.gamma, rng, cfg, gamma=True)
        beta, c4 = mutate_scalar(g.beta, rng, cfg)
        sigma, c5 = mutate_scalar(g.sigma, rng, cfg)
        structural = op_changed or u_changed or v_changed
        scalar_changed = c1 or c2 or c3 or c4 or c5
        changed = structural or scalar_changed
        any_change |= changed
        transformation_change |= structural
        ng = ModuleGene(op, u, v, k, kappa, gamma, beta, sigma)
        # Changed modules are initialized without inherited activation to avoid carrying a
        # state generated by a different transformation or parameterization.
        modules.append((ng, idx, not changed))

    genome_structural_change = False
    # Draw the structural-mutation variates in every factorial condition so that
    # fixed-architecture and extensible runs share the same mutation stream until
    # their genome lengths actually diverge.
    dup_trigger = rng.random() < cfg.mu_module_dup
    dup_src = int(rng.integers(len(modules)))
    dup_pos = int(rng.integers(len(modules) + 1))
    del_trigger = rng.random() < cfg.mu_module_del
    del_index_draw = int(rng.integers(max(1, len(modules) + 1)))
    if condition.extensible and dup_trigger:
        g, parent_idx, _ = modules[dup_src]
        modules.insert(dup_pos, (g, parent_idx, True))
        any_change = True
        genome_structural_change = True
    if condition.extensible and len(modules) > 1 and del_trigger:
        del modules[del_index_draw % len(modules)]
        any_change = True
        genome_structural_change = True

    new_genome = tuple(x[0] for x in modules)
    old_signature = tuple(g.function_key() for g in genome)
    new_signature = tuple(g.function_key() for g in new_genome)
    transformation_change |= set(old_signature) != set(new_signature)
    return MutationResult(
        genome=new_genome,
        parent_mapping=[x[1] for x in modules],
        state_compatible=[x[2] for x in modules],
        heritable_change=any_change,
        transformation_change=transformation_change,
        structural_change=genome_structural_change,
    )


def estimate_functional_mutation_probability(
    cfg: ModelConfig,
    *,
    extensible: bool,
    alphabet_size: int = 3,
    draws: int = 20_000,
) -> Dict[str, float]:
    rng = np.random.default_rng(stable_seed(
        "mutation_probability", SCRIPT_VERSION, alphabet_size, int(extensible)
    ))
    founder = ModuleGene(
        0, (0,), (1,), cfg.founder_k, cfg.founder_kappa,
        cfg.founder_gamma, cfg.founder_beta, cfg.founder_sigma,
    )
    cond = Condition("mutation_assay", True, extensible, True)
    changed = 0
    for _ in range(draws):
        res = mutate_genome((founder,), cond, alphabet_size, rng, cfg)
        changed += int(res.transformation_change)
    return {
        "draws": float(draws),
        "changed": float(changed),
        "probability": changed / draws,
        "wilson_lower_95": wilson_lower(changed, draws),
    }


def closed_function_space_upper_bound(world: WorldSpec) -> int:
    g = world.founder_module
    # In G=0, module count and each reactant-string length are invariant. Symbol
    # substitution and operation switching remain possible, but the transformation
    # repertoire is finite and exactly bounded by this expression for one module.
    return int(2 * (world.key.alphabet_size ** (len(g.u) + len(g.v))))


# -----------------------------------------------------------------------------
# Novelty trackers
# -----------------------------------------------------------------------------

@dataclass
class FunctionTracker:
    first_window: int
    baseline: bool
    first_carrier_id: int
    first_generation: int
    consecutive: int = 0
    persistent: bool = False
    persistent_window: Optional[int] = None
    enabling_depth: int = 0
    causal_tested: bool = False
    causal_retained: bool = False
    causal_mean: Optional[float] = None
    causal_ci_low: Optional[float] = None
    causal_ci_high: Optional[float] = None
    causal_p: Optional[float] = None


@dataclass
class InteractionTracker:
    first_window: int
    baseline: bool = False
    consecutive: int = 0
    persistent: bool = False
    persistent_window: Optional[int] = None


@dataclass
class SubstrateTracker:
    first_window: int
    baseline: bool = False
    consecutive: int = 0
    persistent: bool = False
    persistent_window: Optional[int] = None


class SafetyStop(RuntimeError):
    pass

# -----------------------------------------------------------------------------
# Simulation engine
# -----------------------------------------------------------------------------

class SimulationEngine:
    def __init__(
        self,
        world: WorldSpec,
        condition: Condition,
        cfg: ModelConfig,
        campaign: CampaignConfig,
        evolutionary_seed: int,
        run_id: str,
        *,
        causal_enabled: bool,
        preflight: bool = False,
    ):
        self.world = world
        self.condition = condition
        self.cfg = cfg
        self.campaign = campaign
        self.evolutionary_seed = int(evolutionary_seed)
        self.run_id = run_id
        self.causal_enabled = causal_enabled
        self.preflight = preflight
        self.in_causal_branch = False
        self.knockout_function: Optional[str] = None
        self.step_index = 0
        self.window_index = 0
        self.switch_state = 0
        self.next_individual_id = 1
        self.next_lineage_id = 1
        self.start_wall = time.time()

        base = stable_seed("run", world.key.world_id, evolutionary_seed, cfg.turnover_scale)
        self.rng_env = np.random.default_rng(stable_seed(base, "env"))
        self.rng_noise = np.random.default_rng(stable_seed(base, "noise"))
        self.rng_order = np.random.default_rng(stable_seed(base, "order"))
        self.rng_repro = np.random.default_rng(stable_seed(base, "repro"))
        self.rng_mut = np.random.default_rng(stable_seed(base, "mut"))

        self.store = SubstrateStore(world.n_sites, world.primitive_strings)
        for s in world.primitive_strings:
            sid = self.store.lookup(s)
            assert sid is not None
            self.store.q[:, sid] = cfg.initial_substrate / max(1, len(world.primitive_strings))

        self.initial_symbol_mass = self.store.total_symbol_mass()
        self.cumulative_source_symbol_mass = CompensatedSum()
        self.cumulative_dissipated_symbol_mass = CompensatedSum()
        self.cumulative_discarded_symbol_mass = CompensatedSum()
        self.cumulative_numerical_correction_symbol_mass = CompensatedSum()
        self.cumulative_retention_removed_symbol_mass = CompensatedSum()
        self.cumulative_constructed_production_symbol_mass = CompensatedSum()
        self.cumulative_constructed_consumption_symbol_mass = CompensatedSum()

        # Window-sampled residence-time integral for constructed environmental mass.
        self.constructed_mass_auc = 0.0
        self.constructed_mass_last = 0.0
        self.constructed_mass_last_step = 0

        self.individuals: List[Individual] = []
        self.lineage_parent: Dict[int, int] = {1: 0}
        self.lineage_birth_step: Dict[int, int] = {1: 0}
        self.lineage_signature: Dict[int, Tuple[str, ...]] = {
            1: (world.founder_module.function_key(),)
        }

        founder_sites = self._founder_sites()
        founder_genome = (world.founder_module,)
        for site in founder_sites:
            self.individuals.append(self._new_founder(site, founder_genome))

        self.function_key_cache: Dict[ModuleGene, str] = {}
        self.dphi_cache: Dict[ModuleGene, float] = {}

        # Current-window accumulators.
        self.window_function_flux: Dict[str, float] = defaultdict(float)
        self.window_function_carriers: Dict[str, Set[int]] = defaultdict(set)
        self.window_function_lineages: Dict[str, Set[int]] = defaultdict(set)
        self.window_function_generations: Dict[str, Tuple[int, int]] = {}
        self.window_production_function: Dict[int, Dict[str, float]] = defaultdict(nested_float_dict)
        self.window_consumption_function: Dict[int, Dict[str, float]] = defaultdict(nested_float_dict)
        self.window_function_lineage_presence: Dict[str, Set[int]] = defaultdict(set)
        self.window_births = 0
        self.window_deaths = 0
        self.window_reactions = 0
        self.window_energy_yield = 0.0
        self.window_endogenic_energy_spent = 0.0
        self.window_blocked_endogenic_flux = 0.0
        self.causal_tagged_births = 0

        self.function_trackers: Dict[str, FunctionTracker] = {}
        self.interaction_trackers: Dict[str, InteractionTracker] = {}
        self.substrate_trackers: Dict[int, SubstrateTracker] = {}
        self.function_depth: Dict[str, int] = {}
        self.substrate_persistent_producers: Dict[str, Set[str]] = defaultdict(set)
        self.baseline_functions: Set[str] = {world.founder_module.function_key()}
        self.establishment_functions: Set[str] = set(self.baseline_functions)
        self.establishment_interactions: Set[str] = set()
        self.analysis_baseline_substrate_ids: Set[int] = set()
        self.analysis_cumulative_births = 0
        self.analysis_birth_counter_exact = 0
        self.analysis_started = campaign.establishment_steps <= 0
        self.ever_expressed_functions: Set[str] = set(self.baseline_functions)
        self.ever_interactions: Set[str] = set()

        # Exact event ordering required to distinguish niche-first from genotype-first
        # innovation. Founder functions and establishment-period states are baselines.
        self.genomic_function_first_birth: Dict[str, int] = {
            world.founder_module.function_key(): 0
        }
        self.genomic_function_first_step: Dict[str, int] = {
            world.founder_module.function_key(): 0
        }
        self.substrate_first_production_birth: Dict[int, int] = {}
        self.substrate_first_production_step: Dict[int, int] = {}
        self.substrate_first_producer_function: Dict[int, str] = {}
        self.substrate_persistent_birth: Dict[int, int] = {}

        self.window_rows: List[Dict[str, Any]] = []
        self.function_events: List[Dict[str, Any]] = []
        self.interaction_events: List[Dict[str, Any]] = []
        self.substrate_events: List[Dict[str, Any]] = []
        self.genome_function_events: List[Dict[str, Any]] = []
        self.substrate_origin_events: List[Dict[str, Any]] = []
        self.niche_origin_rows: List[Dict[str, Any]] = []
        self.causal_rows: List[Dict[str, Any]] = []
        self.lineage_events: List[Dict[str, Any]] = []
        self.qc: Dict[str, Any] = {
            "max_population": len(self.individuals),
            "max_substrates": self.store.size,
            "max_genome_modules": 1,
            "max_string_length": max(len(s) for s in world.primitive_strings),
            "negative_substrate_corrections": 0,
            "nonfinite_energy_events": 0,
            "max_abs_mass_balance_residual": 0.0,
            "max_relative_mass_balance_residual": 0.0,
            "blocked_endogenic_flux": 0.0,
            "retention_removed_symbol_mass": 0.0,
            "pruned_symbol_mass": 0.0,
            "numerical_correction_symbol_mass": 0.0,
            "mass_balance_warnings": 0,
            "closed_space_violations": 0,
            "safety_stop": None,
        }

    def _founder_sites(self) -> List[int]:
        reactants = self.world.founder_module.reactants()
        common: Optional[Set[int]] = None
        for s in reactants:
            sites = set(int(x) for x in self.world.source_sites_a[s].tolist())
            common = sites if common is None else common & sites
        candidates = sorted(common or set())
        if not candidates:
            candidates = sorted({
                int(x)
                for arr in self.world.source_sites_a.values()
                for x in arr.tolist()
            })
        if not candidates:
            candidates = list(range(self.world.n_sites))
        return [int(x) for x in self.rng_repro.choice(candidates, size=self.cfg.founders, replace=True)]

    def _new_founder(self, site: int, genome: Tuple[ModuleGene, ...]) -> Individual:
        iid = self.next_individual_id
        self.next_individual_id += 1
        L = len(genome)
        return Individual(
            individual_id=iid,
            parent_id=0,
            lineage_id=1,
            site=site,
            energy=self.cfg.founder_energy,
            age=0,
            generation=0,
            genome=genome,
            x=np.zeros(L, dtype=np.float64),
        )

    def _fkey(self, gene: ModuleGene) -> str:
        key = self.function_key_cache.get(gene)
        if key is None:
            key = gene.function_key()
            self.function_key_cache[gene] = key
        return key

    def _dphi(self, gene: ModuleGene) -> float:
        value = self.dphi_cache.get(gene)
        if value is None:
            value = reaction_delta_phi(self.world, gene)
            self.dphi_cache[gene] = value
        return value

    def _source_update(self) -> None:
        w, cfg = self.world, self.cfg
        dt = cfg.dt
        scale = cfg.turnover_scale
        if w.key.forcing == "stochastic_switching":
            if self.rng_env.random() < w.switch_rate * scale * dt:
                self.switch_state = 1 - self.switch_state
        for s in w.primitive_strings:
            sid = self.store.lookup(s)
            assert sid is not None
            if w.key.forcing == "constant":
                sites = w.source_sites_a[s]
                strength = w.source_strength_a[s]
            elif w.key.forcing == "periodic":
                sites = w.source_sites_a[s]
                strength = w.source_strength_a[s] * (
                    1.0 + w.periodic_amplitude * np.sin(
                        2.0 * math.pi * self.step_index * dt * scale / w.periodic_period + w.source_phase[s]
                    )
                )
                strength = np.maximum(strength, 0.0)
            else:
                if self.switch_state == 0:
                    sites = w.source_sites_a[s]
                    strength = w.source_strength_a[s]
                else:
                    sites = w.source_sites_b[s]
                    strength = w.source_strength_b[s]
            additions = strength * scale * dt
            self.store.q[sites, sid] += additions
            self.cumulative_source_symbol_mass.add(float(additions.sum()) * len(s))
        self.store._dirty = True

    def _transport_and_dissipation(self) -> None:
        cfg, w = self.cfg, self.world
        ids = self.store.active_ids(cfg.substrate_prune_eps, self.step_index)
        if ids.size == 0:
            return
        q = self.store.q
        dt = cfg.dt
        decay = math.exp(-w.dissipation * cfg.turnover_scale * dt)
        before = q[:, ids].copy()
        q[:, ids] *= decay
        lengths = np.asarray([len(self.store.id_to_string[int(sid)]) for sid in ids], dtype=np.float64)
        self.cumulative_dissipated_symbol_mass.add(float(np.sum((before - q[:, ids]) * lengths[None, :])))

        if w.transport > 0.0 and w.edges_u.size:
            u = w.edges_u
            v = w.edges_v
            qu = q[u[:, None], ids[None, :]]
            qv = q[v[:, None], ids[None, :]]
            flux = w.transport * cfg.turnover_scale * dt * (qv - qu)
            # Conservative edge-local cap; additions and subtractions remain paired.
            flux = np.where(flux > 0.0, np.minimum(flux, 0.5 * qv), np.maximum(flux, -0.5 * qu))
            delta = np.zeros((w.n_sites, ids.size), dtype=np.float64)
            np.add.at(delta, u, flux)
            np.add.at(delta, v, -flux)
            q[:, ids] += delta

        selected = q[:, ids]
        negative_mask = selected < 0.0
        negatives = int(np.count_nonzero(selected < -1e-12))
        if np.any(negative_mask):
            correction = float(np.sum(
                np.where(negative_mask, -selected, 0.0) * lengths[None, :]
            ))
            if correction > 0.0:
                self.cumulative_numerical_correction_symbol_mass.add(correction)
                self.qc["numerical_correction_symbol_mass"] = (
                    self.cumulative_numerical_correction_symbol_mass.value()
                )
        if negatives:
            self.qc["negative_substrate_corrections"] += negatives
        np.maximum(selected, 0.0, out=selected)
        self.store._dirty = True

    def _availability(self, site: int, gene: ModuleGene) -> float:
        if gene.op == 0:
            uid = self.store.lookup(gene.u)
            vid = self.store.lookup(gene.v)
            if uid is None or vid is None:
                return 0.0
            if uid == vid:
                return float(self.store.q[site, uid] / 2.0)
            return float(min(self.store.q[site, uid], self.store.q[site, vid]))
        rid = self.store.lookup(gene.u + gene.v)
        if rid is None:
            return 0.0
        return float(self.store.q[site, rid])

    def _update_module_state(self, ind: Individual, r: int, gene: ModuleGene, availability: float) -> float:
        cfg = self.cfg
        x0 = float(ind.x[r])
        dx = (
            -gene.gamma * x0
            - gene.beta * math.tanh(x0)
            + gene.kappa * availability
        ) * cfg.dt + gene.sigma * float(self.rng_noise.normal()) * math.sqrt(cfg.dt)
        ind.x[r] += dx
        x = float(ind.x[r])
        if x >= 0.0:
            return 1.0 / (1.0 + math.exp(-min(x, 700.0)))
        ex = math.exp(max(x, -700.0))
        return ex / (1.0 + ex)

    def _execute_reaction(
        self,
        ind: Individual,
        gene: ModuleGene,
        activation: float,
        availability: float,
    ) -> Tuple[float, float]:
        cfg = self.cfg
        if availability <= 0.0 or activation <= 0.0:
            return 0.0, 0.0
        amount = cfg.turnover_scale * gene.k * activation * availability * cfg.dt
        amount = min(amount, cfg.max_fraction_consumed_per_step * availability)
        if amount <= 0.0:
            return 0.0, 0.0
        fkey = self._fkey(gene)
        if self.knockout_function is not None and fkey == self.knockout_function:
            return 0.0, 0.0

        dphi = self._dphi(gene)
        if dphi < 0.0:
            max_by_energy = max(ind.energy, 0.0) / max(cfg.energy_capture * abs(dphi), 1e-15)
            if amount > max_by_energy:
                self.window_blocked_endogenic_flux += amount - max_by_energy
                self.qc["blocked_endogenic_flux"] += amount - max_by_energy
                amount = max_by_energy
            if amount <= 0.0:
                return 0.0, 0.0

        site = ind.site
        q = self.store.q
        if gene.op == 0:
            uid = self.store.lookup(gene.u)
            vid = self.store.lookup(gene.v)
            if uid is None or vid is None:
                return 0.0, 0.0
            if uid == vid:
                amount = min(amount, q[site, uid] / 2.0)
                q[site, uid] -= 2.0 * amount
            else:
                amount = min(amount, q[site, uid], q[site, vid])
                q[site, uid] -= amount
                q[site, vid] -= amount
            if amount <= 0.0:
                return 0.0, 0.0
            self.window_consumption_function[uid][fkey] += amount * (2.0 if uid == vid else 1.0)
            if uid not in self.store.primitive_ids:
                self.cumulative_constructed_consumption_symbol_mass.add(
                    amount * (2.0 if uid == vid else 1.0) * len(self.store.id_to_string[uid])
                )
            if uid != vid:
                self.window_consumption_function[vid][fkey] += amount
                if vid not in self.store.primitive_ids:
                    self.cumulative_constructed_consumption_symbol_mass.add(
                        amount * len(self.store.id_to_string[vid])
                    )
            product = gene.u + gene.v
            if self.condition.constructive or product in self.world.primitive_strings:
                pid = self.store.ensure(product)
                q = self.store.q  # ensure() may grow and replace the backing array
                q[site, pid] += amount
                self.window_production_function[pid][fkey] += amount
                if pid not in self.store.primitive_ids:
                    self.store.ever_biological.add(pid)
                    self.cumulative_constructed_production_symbol_mass.add(amount * len(product))
                    self._record_substrate_first_production(pid, fkey)
            else:
                self.cumulative_discarded_symbol_mass.add(amount * len(product))
        else:
            reactant = gene.u + gene.v
            rid = self.store.lookup(reactant)
            if rid is None:
                return 0.0, 0.0
            amount = min(amount, q[site, rid])
            if amount <= 0.0:
                return 0.0, 0.0
            q[site, rid] -= amount
            self.window_consumption_function[rid][fkey] += amount
            if rid not in self.store.primitive_ids:
                self.cumulative_constructed_consumption_symbol_mass.add(
                    amount * len(self.store.id_to_string[rid])
                )
            for product in (gene.u, gene.v):
                if self.condition.constructive or product in self.world.primitive_strings:
                    pid = self.store.ensure(product)
                    q = self.store.q  # ensure() may grow and replace the backing array
                    q[site, pid] += amount
                    self.window_production_function[pid][fkey] += amount
                    if pid not in self.store.primitive_ids:
                        self.store.ever_biological.add(pid)
                        self.cumulative_constructed_production_symbol_mass.add(amount * len(product))
                        self._record_substrate_first_production(pid, fkey)
                else:
                    self.cumulative_discarded_symbol_mass.add(amount * len(product))

        energy = cfg.energy_capture * amount * dphi
        ind.energy += energy
        if energy < 0.0:
            self.window_endogenic_energy_spent += -energy
        self.window_function_flux[fkey] += amount
        self.window_function_carriers[fkey].add(ind.individual_id)
        self.window_function_lineages[fkey].add(ind.lineage_id)
        self.window_function_lineage_presence[fkey].add(ind.lineage_id)
        lo, hi = self.window_function_generations.get(fkey, (ind.generation, ind.generation))
        self.window_function_generations[fkey] = (min(lo, ind.generation), max(hi, ind.generation))
        self.window_reactions += 1
        self.window_energy_yield += energy

        self.store._dirty = True
        return amount, energy

    def _process_individual(self, ind: Individual) -> None:
        cfg = self.cfg
        maintenance = cfg.turnover_scale * (
            cfg.basal_cost + cfg.module_maintenance_cost * len(ind.genome)
        ) * cfg.dt
        ind.energy -= maintenance
        activation_sum = 0.0
        if ind.energy > 0.0:
            for r, gene in enumerate(ind.genome):
                availability = self._availability(ind.site, gene)
                z = self._update_module_state(ind, r, gene, availability)
                activation_sum += z
                activation_cost = cfg.turnover_scale * cfg.activation_cost * z * cfg.dt
                ind.energy -= activation_cost
                if ind.energy <= 0.0:
                    break
                self._execute_reaction(ind, gene, z, availability)
        ind.age += 1
        if not math.isfinite(ind.energy):
            self.qc["nonfinite_energy_events"] += 1

    def _reproduce(self, parent: Individual) -> Optional[Individual]:
        cfg = self.cfg
        if parent.energy < cfg.reproduction_threshold:
            return None
        parent.energy *= 0.5
        child_energy = parent.energy
        mutation = mutate_genome(
            parent.genome,
            self.condition,
            self.world.key.alphabet_size,
            self.rng_mut,
            cfg,
        )
        genome = mutation.genome
        if len(genome) > cfg.safety_max_genome_modules:
            raise SafetyStop("genome_module_limit")
        max_len = max(max(len(g.u), len(g.v)) for g in genome)
        if max_len > cfg.safety_max_string_length:
            raise SafetyStop("substrate_string_length_limit")

        if mutation.heritable_change:
            self.next_lineage_id += 1
            lineage = self.next_lineage_id
            self.lineage_parent[lineage] = parent.lineage_id
            self.lineage_birth_step[lineage] = self.step_index
            self.lineage_signature[lineage] = tuple(g.function_key() for g in genome)
        else:
            lineage = parent.lineage_id

        if self.rng_repro.random() < cfg.dispersal_probability and self.world.neighbors[parent.site]:
            site = int(self.rng_repro.choice(self.world.neighbors[parent.site]))
        else:
            site = parent.site

        L = len(genome)
        x = np.zeros(L, dtype=np.float64)
        for j, src in enumerate(mutation.parent_mapping):
            if src is not None and src < len(parent.genome) and mutation.state_compatible[j]:
                x[j] = parent.x[src]

        iid = self.next_individual_id
        self.next_individual_id += 1
        child = Individual(
            individual_id=iid,
            parent_id=parent.individual_id,
            lineage_id=lineage,
            site=site,
            energy=child_energy,
            age=0,
            generation=parent.generation + 1,
            genome=genome,
            x=x,
            tagged=parent.tagged,
        )
        if parent.tagged:
            self.causal_tagged_births += 1
        if self.analysis_started:
            self.analysis_birth_counter_exact += 1
            if not self.in_causal_branch:
                for gene in genome:
                    fkey = gene.function_key()
                    if fkey in self.genomic_function_first_birth:
                        continue
                    self.genomic_function_first_birth[fkey] = int(self.analysis_birth_counter_exact)
                    self.genomic_function_first_step[fkey] = int(self.step_index)
                    self.genome_function_events.append({
                        "event": "first_genomic_appearance",
                        "step": self.step_index,
                        "birth_index": self.analysis_birth_counter_exact,
                        "function": fkey,
                        "child_id": iid,
                        "child_lineage": lineage,
                        "parent_lineage": parent.lineage_id,
                        "generation": child.generation,
                        "genome_modules": len(genome),
                    })
        if cfg.record_lineage_events and mutation.heritable_change and not self.in_causal_branch:
            self.lineage_events.append({
                "step": self.step_index,
                "child_id": iid,
                "parent_id": parent.individual_id,
                "child_lineage": lineage,
                "parent_lineage": parent.lineage_id,
                "generation": child.generation,
                "transformation_change": mutation.transformation_change,
                "structural_change": mutation.structural_change,
                "genome_modules": len(genome),
            })
        return child

    def _record_substrate_first_production(self, sid: int, producer_function: str) -> None:
        if not self.analysis_started or self.in_causal_branch:
            return
        if sid in self.substrate_first_production_birth:
            return
        self.substrate_first_production_birth[sid] = int(self.analysis_birth_counter_exact)
        self.substrate_first_production_step[sid] = int(self.step_index)
        self.substrate_first_producer_function[sid] = producer_function
        self.substrate_origin_events.append({
            "event": "first_production",
            "step": self.step_index,
            "birth_index": self.analysis_birth_counter_exact,
            "substrate_id": sid,
            "substrate": encode_string(self.store.id_to_string[sid]),
            "producer_function": producer_function,
        })

    def _constructed_symbol_mass(self) -> float:
        ids = sorted(self.store.ever_biological)
        if not ids:
            return 0.0
        idx = np.asarray(ids, dtype=np.int32)
        lengths = np.asarray(
            [len(self.store.id_to_string[int(sid)]) for sid in idx], dtype=np.float64
        )
        return float(np.sum(self.store.q[:, idx] * lengths[None, :]))

    def _update_constructed_mass_auc(self) -> float:
        current = self._constructed_symbol_mass()
        elapsed = max(0, self.step_index - self.constructed_mass_last_step)
        effective_dt = elapsed * self.cfg.dt * self.cfg.turnover_scale
        self.constructed_mass_auc += 0.5 * (
            self.constructed_mass_last + current
        ) * effective_dt
        self.constructed_mass_last = current
        self.constructed_mass_last_step = self.step_index
        return current

    def _apply_constructed_retention(self) -> None:
        half_life = self.condition.constructed_half_life
        if half_life is None or not self.store.ever_biological:
            return
        ids = np.fromiter(sorted(self.store.ever_biological), dtype=np.int32)
        if ids.size == 0:
            return
        quantities = self.store.q[:, ids]
        lengths = np.asarray(
            [len(self.store.id_to_string[int(sid)]) for sid in ids], dtype=np.float64
        )
        before = float(np.sum(quantities * lengths[None, :]))
        if before <= 0.0:
            return
        if half_life <= 0.0:
            self.store.q[:, ids] = 0.0
            removed = before
        else:
            decay = math.exp(
                -math.log(2.0) * self.cfg.turnover_scale * self.cfg.dt / half_life
            )
            self.store.q[:, ids] *= decay
            after = float(np.sum(self.store.q[:, ids] * lengths[None, :]))
            removed = max(0.0, before - after)
        if removed > 0.0:
            self.cumulative_discarded_symbol_mass.add(removed)
            self.cumulative_retention_removed_symbol_mass.add(removed)
            self.qc["retention_removed_symbol_mass"] = (
                self.cumulative_retention_removed_symbol_mass.value()
            )
            self.store._dirty = True

    def _mass_balance_check(self) -> None:
        actual = self.store.total_symbol_mass()
        expected = (
            self.initial_symbol_mass
            + self.cumulative_source_symbol_mass.value()
            + self.cumulative_numerical_correction_symbol_mass.value()
            - self.cumulative_dissipated_symbol_mass.value()
            - self.cumulative_discarded_symbol_mass.value()
        )
        residual = actual - expected
        relative = abs(residual) / max(1.0, abs(expected))
        self.qc["max_abs_mass_balance_residual"] = max(
            float(self.qc["max_abs_mass_balance_residual"]), abs(residual)
        )
        self.qc["max_relative_mass_balance_residual"] = max(
            float(self.qc["max_relative_mass_balance_residual"]), relative
        )
        if relative > self.cfg.mass_balance_warning_tolerance:
            self.qc["mass_balance_warnings"] = int(self.qc["mass_balance_warnings"]) + 1
        if relative > self.cfg.mass_balance_relative_tolerance:
            raise SafetyStop(f"mass_balance_residual:{relative:.3e}")

    def _quality_checks(self) -> None:
        cfg = self.cfg
        pop = len(self.individuals)
        self.qc["max_population"] = max(int(self.qc["max_population"]), pop)
        self.qc["max_substrates"] = max(int(self.qc["max_substrates"]), self.store.size)
        if self.individuals:
            self.qc["max_genome_modules"] = max(
                int(self.qc["max_genome_modules"]), max(len(i.genome) for i in self.individuals)
            )
            self.qc["max_string_length"] = max(
                int(self.qc["max_string_length"]),
                max(max(max(len(g.u), len(g.v)) for g in i.genome) for i in self.individuals),
            )
        if not self.condition.extensible and self.individuals:
            founder = self.world.founder_module
            for ind in self.individuals:
                valid = len(ind.genome) == 1 and all(
                    len(g.u) == len(founder.u) and len(g.v) == len(founder.v)
                    for g in ind.genome
                )
                if not valid:
                    self.qc["closed_space_violations"] += 1
                    raise SafetyStop("closed_hereditary_space_violation")
        if pop > cfg.safety_max_population:
            raise SafetyStop("population_limit")
        if self.store.size > cfg.safety_max_substrates:
            raise SafetyStop("substrate_limit")
        self._mass_balance_check()

    def step(self, *, record: bool = True, allow_causal: bool = True) -> None:
        self._source_update()
        if self.step_index % self.cfg.transport_interval == 0:
            self._transport_and_dissipation()

        if self.individuals:
            order = self.rng_order.permutation(len(self.individuals))
            for idx in order.tolist():
                self._process_individual(self.individuals[idx])

            self._apply_constructed_retention()

            survivors: List[Individual] = []
            newborns: List[Individual] = []
            for ind in self.individuals:
                if ind.energy <= 0.0 or not math.isfinite(ind.energy):
                    self.window_deaths += 1
                    continue
                survivors.append(ind)
                child = self._reproduce(ind)
                if child is not None:
                    newborns.append(child)
                    self.window_births += 1
            self.individuals = survivors + newborns

        if not self.individuals:
            self._apply_constructed_retention()

        self.step_index += 1
        if self.step_index % self.cfg.prune_interval == 0:
            pruned = self.store.prune(self.cfg.substrate_prune_eps)
            if pruned > 0.0:
                self.cumulative_discarded_symbol_mass.add(pruned)
                self.qc["pruned_symbol_mass"] = float(self.qc["pruned_symbol_mass"]) + pruned
        if self.step_index % self.cfg.quality_check_interval == 0:
            self._quality_checks()
        if record and self.step_index % self.campaign.window_size == 0:
            self._close_window(allow_causal=allow_causal)

    def _function_reactants(self, key: str) -> Tuple[Tuple[int, ...], ...]:
        if key.startswith("L:"):
            left = key[2:].split("->", 1)[0]
            a, b = left.split("+", 1)
            return decode_string(a), decode_string(b)
        return (decode_string(key[2:].split("->", 1)[0]),)

    def _function_products(self, key: str) -> Tuple[Tuple[int, ...], ...]:
        right = key.split("->", 1)[1]
        if key.startswith("L:"):
            return (decode_string(right),)
        a, b = right.split("+", 1)
        return decode_string(a), decode_string(b)

    def _compute_enabling_depth(self, fkey: str) -> int:
        depth = 1
        for reactant in self._function_reactants(fkey):
            if reactant in self.world.primitive_strings:
                continue
            producers = self.substrate_persistent_producers.get(encode_string(reactant), set())
            if producers:
                depth = max(depth, 1 + max(self.function_depth.get(p, 1) for p in producers))
        return depth

    def _derive_current_interactions(self) -> Tuple[Dict[str, float], Dict[str, Set[int]]]:
        current_interactions: Dict[str, float] = defaultdict(float)
        current_interaction_lineages: Dict[str, Set[int]] = defaultdict(set)
        for sid, producers in self.window_production_function.items():
            if sid in self.store.primitive_ids:
                continue
            consumers = self.window_consumption_function.get(sid)
            if not consumers:
                continue
            total_consumption = sum(consumers.values())
            if total_consumption <= 0.0:
                continue
            substrate = encode_string(self.store.id_to_string[sid])
            for producer_function, production in producers.items():
                for consumer_function, consumption in consumers.items():
                    if producer_function == consumer_function:
                        continue
                    weight = production * consumption / total_consumption
                    if weight < self.cfg.interaction_threshold:
                        continue
                    ikey = f"{producer_function}|{substrate}|{consumer_function}"
                    current_interactions[ikey] += weight
                    current_interaction_lineages[ikey].update(
                        self.window_function_lineage_presence.get(producer_function, set())
                    )
                    current_interaction_lineages[ikey].update(
                        self.window_function_lineage_presence.get(consumer_function, set())
                    )
        return current_interactions, current_interaction_lineages

    def _append_window_snapshot(
        self,
        *,
        phase: str,
        expressed: Set[str],
        persistent_count: int,
        causal_count: int,
        new_persistent: Sequence[str],
        persistent_interactions: int,
        persistent_substrates: int,
        new_persistent_substrates: int,
        dmax: int,
    ) -> None:
        genome_lengths = [len(i.genome) for i in self.individuals]
        active_modules = sum(
            sum(1 for g in i.genome if self._fkey(g) in expressed)
            for i in self.individuals
        )
        energies = [i.energy for i in self.individuals]
        self.window_rows.append({
            "run_id": self.run_id,
            "world_id": self.world.key.world_id,
            "world_family": self.world.key.family,
            "topology": self.world.key.topology,
            "forcing": self.world.key.forcing,
            "alphabet_size": self.world.key.alphabet_size,
            "condition": self.condition.name,
            "constructive": int(self.condition.constructive),
            "extensible": int(self.condition.extensible),
            "mutation": int(self.condition.mutation),
            "constructed_half_life": (
                self.condition.constructed_half_life
                if self.condition.constructed_half_life is not None else "inf"
            ),
            "retention_sweep": int(self.condition.retention_sweep),
            "factorial": int(self.condition.factorial),
            "evolutionary_seed": self.evolutionary_seed,
            "pair_id": f"{self.world.key.world_id}__{self.evolutionary_seed}",
            "window": self.window_index,
            "step": self.step_index,
            "analysis_phase": phase,
            "population": len(self.individuals),
            "births": self.window_births,
            "deaths": self.window_deaths,
            "reactions": self.window_reactions,
            "energy_yield": self.window_energy_yield,
            "endogenic_energy_spent": self.window_endogenic_energy_spent,
            "blocked_endogenic_flux": self.window_blocked_endogenic_flux,
            "expressed_functions": len(expressed),
            "persistent_functions": persistent_count,
            "causal_persistent_functions": causal_count,
            "new_persistent_functions": len(new_persistent),
            "new_causal_functions": sum(
                1 for f in new_persistent
                if f in self.function_trackers and self.function_trackers[f].causal_retained
            ),
            "persistent_interactions": persistent_interactions,
            "persistent_derived_substrates": persistent_substrates,
            "new_persistent_derived_substrates": new_persistent_substrates,
            "analysis_cumulative_births": self.analysis_birth_counter_exact,
            "environmental_substrates": len(self.store.primitive_ids | self.store.ever_biological),
            "constructed_substrates_ever": len(self.store.ever_biological),
            "current_derived_substrates": sum(
                1 for sid in self.store.ever_biological
                if float(self.store.q[:, sid].sum()) >= self.cfg.substrate_presence_threshold
            ),
            "analysis_constructed_substrates_ever": (
                len(self.store.ever_biological - self.analysis_baseline_substrate_ids)
                if phase == "analysis" else 0
            ),
            "constructed_symbol_mass": self._update_constructed_mass_auc(),
            "constructed_mass_auc": self.constructed_mass_auc,
            "constructed_production_symbol_mass": self.cumulative_constructed_production_symbol_mass.value(),
            "constructed_consumption_symbol_mass": self.cumulative_constructed_consumption_symbol_mass.value(),
            "constructed_utilization_ratio": (
                self.cumulative_constructed_consumption_symbol_mass.value()
                / max(self.cumulative_constructed_production_symbol_mass.value(), 1e-15)
            ),
            "empirical_constructed_residence_time": (
                self.constructed_mass_auc
                / max(self.cumulative_constructed_production_symbol_mass.value(), 1e-15)
            ),
            "enabling_depth_max": dmax,
            "mean_genome_modules": float(np.mean(genome_lengths)) if genome_lengths else 0.0,
            "max_genome_modules": max(genome_lengths, default=0),
            "active_modules": active_modules,
            "mean_energy": float(np.mean(energies)) if energies else 0.0,
            "min_energy": min(energies, default=0.0),
            "max_energy": max(energies, default=0.0),
        })

    def _reset_window_accumulators(self) -> None:
        self.window_index += 1
        self.window_function_flux.clear()
        self.window_function_carriers.clear()
        self.window_function_lineages.clear()
        self.window_function_generations.clear()
        self.window_production_function.clear()
        self.window_consumption_function.clear()
        self.window_function_lineage_presence.clear()
        self.window_births = 0
        self.window_deaths = 0
        self.window_reactions = 0
        self.window_energy_yield = 0.0
        self.window_endogenic_energy_spent = 0.0
        self.window_blocked_endogenic_flux = 0.0

    def _close_window(self, *, allow_causal: bool) -> None:
        cfg = self.cfg
        widx = self.window_index
        alive_ids = {i.individual_id for i in self.individuals}
        expressed = {k for k, v in self.window_function_flux.items() if v >= cfg.flux_threshold}
        new_persistent: List[str] = []

        # Establishment is used only to initialize the ecological state. Functions and
        # interactions already present at the analysis boundary are baseline, not novelty.
        if self.step_index <= self.campaign.establishment_steps:
            self.establishment_functions.update(expressed)
            establishment_interactions, _ = self._derive_current_interactions()
            self.establishment_interactions.update(establishment_interactions)
            self._append_window_snapshot(
                phase="establishment",
                expressed=expressed,
                persistent_count=0,
                causal_count=0,
                new_persistent=(),
                persistent_interactions=0,
                persistent_substrates=0,
                new_persistent_substrates=0,
                dmax=0,
            )
            self._reset_window_accumulators()
            return

        if not self.analysis_started:
            self.analysis_started = True
            # All hereditary functions present at the analysis boundary are baseline.
            for ind in self.individuals:
                for gene in ind.genome:
                    fkey = self._fkey(gene)
                    self.genomic_function_first_birth.setdefault(fkey, 0)
                    self.genomic_function_first_step.setdefault(fkey, self.step_index)
            self.analysis_baseline_substrate_ids = {
                sid for sid in self.store.ever_biological
                if float(self.store.q[:, sid].sum()) >= cfg.substrate_presence_threshold
            }
            for sid in self.analysis_baseline_substrate_ids:
                self.substrate_trackers[sid] = SubstrateTracker(
                    first_window=widx,
                    baseline=True,
                    consecutive=cfg.persistence_windows,
                    persistent=True,
                    persistent_window=widx,
                )
            self.baseline_functions.update(self.establishment_functions)
            for fkey in sorted(self.baseline_functions):
                tracker = self.function_trackers.get(fkey)
                if tracker is None:
                    tracker = FunctionTracker(
                        first_window=widx,
                        baseline=True,
                        consecutive=cfg.persistence_windows,
                        first_carrier_id=-1,
                        first_generation=0,
                        persistent_window=widx,
                        persistent=True,
                        enabling_depth=1,
                    )
                    self.function_trackers[fkey] = tracker
                else:
                    tracker.baseline = True
                    tracker.persistent = True
                    tracker.persistent_window = widx
                    tracker.enabling_depth = 1
                self.function_depth[fkey] = 1
                self.ever_expressed_functions.add(fkey)
                for product in self._function_products(fkey):
                    self.substrate_persistent_producers[encode_string(product)].add(fkey)
            for ikey in sorted(self.establishment_interactions):
                self.interaction_trackers[ikey] = InteractionTracker(
                    first_window=widx,
                    baseline=True,
                    consecutive=cfg.persistence_windows,
                    persistent=True,
                    persistent_window=widx,
                )
                self.ever_interactions.add(ikey)

        self.analysis_cumulative_births = int(self.analysis_birth_counter_exact)

        for fkey in expressed:
            carriers = self.window_function_carriers.get(fkey, set())
            gen_lo, gen_hi = self.window_function_generations.get(fkey, (0, 0))
            tracker = self.function_trackers.get(fkey)
            if tracker is None:
                tracker = FunctionTracker(
                    first_window=widx,
                    baseline=fkey in self.baseline_functions,
                    first_carrier_id=min(carriers) if carriers else -1,
                    first_generation=gen_lo,
                )
                self.function_trackers[fkey] = tracker
                self.ever_expressed_functions.add(fkey)
                if not tracker.baseline:
                    self.function_events.append({
                        "event": "first_expression",
                        "window": widx,
                        "step": self.step_index,
                        "birth_index": self.analysis_cumulative_births,
                        "function": fkey,
                        "genomic_first_birth": self.genomic_function_first_birth.get(fkey),
                        "flux": self.window_function_flux[fkey],
                        "organisms": len(carriers),
                        "lineages": len(self.window_function_lineages.get(fkey, set())),
                    })
            tracker.consecutive += 1
            founder_dead = tracker.first_carrier_id not in alive_ids
            generation_span = gen_hi - tracker.first_generation
            if (
                not tracker.persistent
                and tracker.consecutive >= cfg.persistence_windows
                and len(carriers) >= cfg.min_organisms
                and generation_span >= cfg.min_generations
                and founder_dead
            ):
                tracker.persistent = True
                tracker.persistent_window = widx
                tracker.enabling_depth = self._compute_enabling_depth(fkey)
                self.function_depth[fkey] = tracker.enabling_depth
                for product in self._function_products(fkey):
                    self.substrate_persistent_producers[encode_string(product)].add(fkey)
                if not tracker.baseline:
                    new_persistent.append(fkey)
                    self.function_events.append({
                        "event": "persistent",
                        "window": widx,
                        "step": self.step_index,
                        "birth_index": self.analysis_cumulative_births,
                        "function": fkey,
                        "genomic_first_birth": self.genomic_function_first_birth.get(fkey),
                        "flux": self.window_function_flux[fkey],
                        "organisms": len(carriers),
                        "lineages": len(self.window_function_lineages.get(fkey, set())),
                        "generation_span": generation_span,
                        "enabling_depth": tracker.enabling_depth,
                    })

        for fkey, tracker in self.function_trackers.items():
            if not tracker.baseline and fkey not in expressed:
                tracker.consecutive = 0

        current_interactions, current_interaction_lineages = self._derive_current_interactions()
        for ikey, weight in current_interactions.items():
            tracker = self.interaction_trackers.get(ikey)
            if tracker is None:
                tracker = InteractionTracker(first_window=widx)
                self.interaction_trackers[ikey] = tracker
                self.ever_interactions.add(ikey)
                self.interaction_events.append({
                    "event": "first_expression",
                    "window": widx,
                    "step": self.step_index,
                    "birth_index": self.analysis_cumulative_births,
                    "interaction": ikey,
                    "weight": weight,
                    "lineages": len(current_interaction_lineages[ikey]),
                })
            tracker.consecutive += 1
            producer_function, _, consumer_function = ikey.split("|", 2)
            producer_persistent = bool(
                producer_function in self.function_trackers
                and self.function_trackers[producer_function].persistent
            )
            consumer_persistent = bool(
                consumer_function in self.function_trackers
                and self.function_trackers[consumer_function].persistent
            )
            if (
                not tracker.persistent
                and tracker.consecutive >= cfg.persistence_windows
                and len(current_interaction_lineages[ikey]) >= 2
                and producer_persistent
                and consumer_persistent
            ):
                tracker.persistent = True
                tracker.persistent_window = widx
                if not tracker.baseline:
                    self.interaction_events.append({
                        "event": "persistent",
                        "window": widx,
                        "step": self.step_index,
                        "birth_index": self.analysis_cumulative_births,
                        "interaction": ikey,
                        "weight": weight,
                        "lineages": len(current_interaction_lineages[ikey]),
                    })
        for ikey, tracker in self.interaction_trackers.items():
            if not tracker.baseline and ikey not in current_interactions:
                tracker.consecutive = 0

        active_derived = {
            sid for sid in self.store.ever_biological
            if float(self.store.q[:, sid].sum()) >= cfg.substrate_presence_threshold
        }
        new_persistent_substrates: List[int] = []
        for sid in active_derived:
            tracker = self.substrate_trackers.get(sid)
            if tracker is None:
                tracker = SubstrateTracker(
                    first_window=widx,
                    baseline=sid in self.analysis_baseline_substrate_ids,
                )
                self.substrate_trackers[sid] = tracker
                if not tracker.baseline:
                    self.substrate_events.append({
                        "event": "first_presence",
                        "window": widx,
                        "step": self.step_index,
                        "birth_index": self.analysis_cumulative_births,
                        "substrate": encode_string(self.store.id_to_string[sid]),
                        "substrate_id": sid,
                        "total_amount": float(self.store.q[:, sid].sum()),
                    })
            tracker.consecutive += 1
            if (
                not tracker.persistent
                and tracker.consecutive >= cfg.persistence_windows
                and not tracker.baseline
            ):
                tracker.persistent = True
                tracker.persistent_window = widx
                self.substrate_persistent_birth[sid] = int(self.analysis_birth_counter_exact)
                new_persistent_substrates.append(sid)
                self.substrate_events.append({
                    "event": "persistent",
                    "window": widx,
                    "step": self.step_index,
                    "birth_index": self.analysis_cumulative_births,
                    "substrate": encode_string(self.store.id_to_string[sid]),
                    "substrate_id": sid,
                    "total_amount": float(self.store.q[:, sid].sum()),
                })
        for sid, tracker in self.substrate_trackers.items():
            if not tracker.baseline and sid not in active_derived:
                tracker.consecutive = 0

        if allow_causal and self.causal_enabled and not self.in_causal_branch:
            for fkey in new_persistent:
                self._causal_assay(fkey)

        causal_count = sum(
            1 for tr in self.function_trackers.values()
            if not tr.baseline and tr.causal_retained
        )
        persistent_count = sum(
            1 for tr in self.function_trackers.values()
            if not tr.baseline and tr.persistent
        )
        persistent_interactions = sum(
            1 for tr in self.interaction_trackers.values()
            if not tr.baseline and tr.persistent
        )
        persistent_substrates = sum(
            1 for tr in self.substrate_trackers.values()
            if not tr.baseline and tr.persistent
        )
        dmax = max(
            (depth for fkey, depth in self.function_depth.items()
             if fkey in self.function_trackers and not self.function_trackers[fkey].baseline),
            default=0,
        )
        self._append_window_snapshot(
            phase="analysis",
            expressed=expressed,
            persistent_count=persistent_count,
            causal_count=causal_count,
            new_persistent=new_persistent,
            persistent_interactions=persistent_interactions,
            persistent_substrates=persistent_substrates,
            new_persistent_substrates=len(new_persistent_substrates),
            dmax=dmax,
        )
        self._reset_window_accumulators()

    def _tag_function_carriers(self, fkey: str) -> int:
        count = 0
        for ind in self.individuals:
            ind.tagged = any(self._fkey(g) == fkey for g in ind.genome)
            count += int(ind.tagged)
        return count

    def _branch_blob(self) -> bytes:
        branch = pickle.loads(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))
        branch.window_rows = []
        branch.function_events = []
        branch.interaction_events = []
        branch.substrate_events = []
        branch.genome_function_events = []
        branch.substrate_origin_events = []
        branch.niche_origin_rows = []
        branch.causal_rows = []
        branch.lineage_events = []
        branch.function_trackers = {}
        branch.interaction_trackers = {}
        branch.substrate_trackers = {}
        branch.window_function_flux.clear()
        branch.window_function_carriers.clear()
        branch.window_function_lineages.clear()
        branch.window_function_generations.clear()
        branch.window_production_function.clear()
        branch.window_consumption_function.clear()
        branch.window_function_lineage_presence.clear()
        branch.causal_tagged_births = 0
        return pickle.dumps(branch, protocol=pickle.HIGHEST_PROTOCOL)

    def _causal_assay(self, fkey: str) -> None:
        tracker = self.function_trackers[fkey]
        tracker.causal_tested = True
        carrier_count = self._tag_function_carriers(fkey)
        if carrier_count == 0:
            return
        base_blob = self._branch_blob()
        diffs: List[float] = []
        birth_diffs: List[float] = []
        intact_counts: List[int] = []
        knockout_counts: List[int] = []
        for rep in range(self.campaign.causal_repeats):
            seed = stable_seed(self.run_id, "causal", fkey, self.step_index, rep)
            intact: SimulationEngine = pickle.loads(base_blob)
            knockout: SimulationEngine = pickle.loads(base_blob)
            for branch in (intact, knockout):
                branch.in_causal_branch = True
                branch.causal_enabled = False
                branch.rng_env = np.random.default_rng(stable_seed(seed, "env"))
                branch.rng_noise = np.random.default_rng(stable_seed(seed, "noise"))
                branch.rng_order = np.random.default_rng(stable_seed(seed, "order"))
                branch.rng_repro = np.random.default_rng(stable_seed(seed, "repro"))
                branch.rng_mut = np.random.default_rng(stable_seed(seed, "mut"))
            knockout.knockout_function = fkey
            for _ in range(self.campaign.causal_horizon):
                intact.step(record=False, allow_causal=False)
                knockout.step(record=False, allow_causal=False)
                if not intact.individuals and not knockout.individuals:
                    break
            intact_count = sum(1 for ind in intact.individuals if ind.tagged)
            knockout_count = sum(1 for ind in knockout.individuals if ind.tagged)
            intact_counts.append(intact_count)
            knockout_counts.append(knockout_count)
            diffs.append(float(intact_count - knockout_count))
            birth_diffs.append(float(intact.causal_tagged_births - knockout.causal_tagged_births))

        mean, low, high = paired_mean_ci(diffs)
        p_value = exact_sign_flip_p(
            diffs, stable_seed(self.run_id, fkey, "causal_p"), alternative="greater"
        )
        # Descendant abundance is primary. Tagged births are retained as a diagnostic.
        tracker.causal_mean = mean
        tracker.causal_ci_low = low
        tracker.causal_ci_high = high
        tracker.causal_p = p_value
        causal_candidate = bool(
            math.isfinite(mean)
            and mean > 0.0
            and math.isfinite(low)
            and low > 0.0
            and math.isfinite(p_value)
            and p_value < 0.05
            and sum(x > 0.0 for x in diffs) >= self.campaign.causal_repeats - 1
        )
        tracker.causal_retained = False
        row = {
            "run_id": self.run_id,
            "world_id": self.world.key.world_id,
            "world_family": self.world.key.family,
            "condition": self.condition.name,
            "evolutionary_seed": self.evolutionary_seed,
            "step": self.step_index,
            "window": self.window_index,
            "birth_index": self.analysis_cumulative_births,
            "function": fkey,
            "carrier_count": carrier_count,
            "repeats": self.campaign.causal_repeats,
            "horizon": self.campaign.causal_horizon,
            "intact_mean": float(np.mean(intact_counts)),
            "knockout_mean": float(np.mean(knockout_counts)),
            "difference_mean": mean,
            "ci_low": low,
            "ci_high": high,
            "sign_flip_p": p_value,
            "positive_repeats": sum(x > 0.0 for x in diffs),
            "tagged_birth_difference_mean": float(np.mean(birth_diffs)),
            "causal_candidate": causal_candidate,
            "causal_q_bh": math.nan,
            "causal_retained": False,
        }
        self.causal_rows.append(row)
        self.function_events.append({
            "event": "causal_assay",
            "window": self.window_index,
            "step": self.step_index,
            "birth_index": self.analysis_cumulative_births,
            "function": fkey,
            "difference_mean": mean,
            "ci_low": low,
            "ci_high": high,
            "sign_flip_p": p_value,
            "causal_candidate": causal_candidate,
            "causal_q_bh": math.nan,
            "causal_retained": False,
        })

    def _finalize_causal_classification(self) -> None:
        if not self.causal_rows:
            for row in self.window_rows:
                if row.get("analysis_phase") == "analysis":
                    row["new_causal_functions"] = 0
                    row["causal_persistent_functions"] = 0
            return
        q_values = benjamini_hochberg([
            float(row.get("sign_flip_p", math.nan)) for row in self.causal_rows
        ])
        retained_by_window: Dict[int, int] = defaultdict(int)
        for row, q_value in zip(self.causal_rows, q_values):
            retained = bool(row.get("causal_candidate", False)) and math.isfinite(q_value) and q_value <= self.campaign.alpha
            row["causal_q_bh"] = q_value
            row["causal_retained"] = retained
            fkey = str(row["function"])
            tracker = self.function_trackers.get(fkey)
            if tracker is not None:
                tracker.causal_retained = retained
            if retained:
                retained_by_window[int(row["window"])] += 1

        cumulative = 0
        for row in self.window_rows:
            if row.get("analysis_phase") != "analysis":
                continue
            new_count = retained_by_window.get(int(row["window"]), 0)
            cumulative += new_count
            row["new_causal_functions"] = new_count
            row["causal_persistent_functions"] = cumulative

        q_by_function = {
            str(row["function"]): (row["causal_q_bh"], row["causal_retained"])
            for row in self.causal_rows
        }
        for event in self.function_events:
            if event.get("event") == "causal_assay":
                q_value, retained = q_by_function.get(
                    str(event.get("function")), (math.nan, False)
                )
                event["causal_q_bh"] = q_value
                event["causal_retained"] = retained

    def _classify_niche_origins(self) -> None:
        rows: List[Dict[str, Any]] = []
        for fkey, tracker in sorted(self.function_trackers.items()):
            if tracker.baseline or not tracker.persistent:
                continue
            genetic_birth = self.genomic_function_first_birth.get(fkey)
            genetic_step = self.genomic_function_first_step.get(fkey)
            reactants = self._function_reactants(fkey)
            nonprimitive = [r for r in reactants if r not in self.world.primitive_strings]
            details: List[Dict[str, Any]] = []
            production_births: List[int] = []
            persistent_births: List[int] = []
            missing = False
            for reactant in nonprimitive:
                sid = self.store.lookup(reactant)
                if sid is None:
                    missing = True
                    details.append({"reactant": encode_string(reactant), "missing": True})
                    continue
                prod_birth = self.substrate_first_production_birth.get(sid)
                pers_birth = self.substrate_persistent_birth.get(sid)
                if prod_birth is None:
                    missing = True
                else:
                    production_births.append(int(prod_birth))
                if pers_birth is not None:
                    persistent_births.append(int(pers_birth))
                details.append({
                    "reactant": encode_string(reactant),
                    "substrate_id": sid,
                    "first_production_birth": prod_birth,
                    "persistent_birth": pers_birth,
                    "producer_function": self.substrate_first_producer_function.get(sid),
                })

            if not nonprimitive:
                origin = "external_supported"
                strict_origin = "external_supported"
            elif genetic_birth is None or missing:
                origin = "unresolved"
                strict_origin = "unresolved"
            else:
                comparisons = [b - int(genetic_birth) for b in production_births]
                if comparisons and all(x < 0 for x in comparisons):
                    origin = "niche_first"
                elif comparisons and all(x > 0 for x in comparisons):
                    origin = "genotype_first"
                elif comparisons and all(x == 0 for x in comparisons):
                    origin = "coincident"
                else:
                    origin = "mixed_order"
                if len(persistent_births) != len(nonprimitive):
                    strict_origin = "no_persistent_niche_precondition"
                else:
                    strict_cmp = [b - int(genetic_birth) for b in persistent_births]
                    if all(x < 0 for x in strict_cmp):
                        strict_origin = "persistent_niche_first"
                    elif all(x > 0 for x in strict_cmp):
                        strict_origin = "genotype_first"
                    elif all(x == 0 for x in strict_cmp):
                        strict_origin = "coincident"
                    else:
                        strict_origin = "mixed_order"

            rows.append({
                "run_id": self.run_id,
                "world_id": self.world.key.world_id,
                "world_family": self.world.key.family,
                "condition": self.condition.name,
                "evolutionary_seed": self.evolutionary_seed,
                "pair_id": f"{self.world.key.world_id}__{self.evolutionary_seed}",
                "function": fkey,
                "genomic_first_birth": genetic_birth,
                "genomic_first_step": genetic_step,
                "persistent_window": tracker.persistent_window,
                "enabling_depth": tracker.enabling_depth,
                "causal_retained": tracker.causal_retained,
                "nonprimitive_reactant_count": len(nonprimitive),
                "origin_class": origin,
                "strict_origin_class": strict_origin,
                "reactant_timing_json": json.dumps(details, sort_keys=True),
            })
        self.niche_origin_rows = rows

    def save_checkpoint(self, path: Path) -> None:
        payload = {
            "script_version": SCRIPT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "engine": self,
        }
        raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        atomic_write_bytes(path, gzip.compress(raw, compresslevel=1))

    @staticmethod
    def load_checkpoint(path: Path) -> "SimulationEngine":
        with gzip.open(path, "rb") as f:
            payload = pickle.load(f)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("Checkpoint schema mismatch")
        if payload.get("script_version") != SCRIPT_VERSION:
            raise RuntimeError("Checkpoint script version mismatch")
        return payload["engine"]

    def run(self, total_steps: int, checkpoint_path: Optional[Path] = None) -> None:
        next_progress = ((self.step_index // self.cfg.progress_interval) + 1) * self.cfg.progress_interval
        while self.step_index < total_steps:
            self.step(record=True, allow_causal=True)
            if checkpoint_path and self.step_index % self.campaign.checkpoint_interval == 0:
                self.save_checkpoint(checkpoint_path)
            if self.step_index >= next_progress and not self.in_causal_branch:
                print(
                    f"[run] {self.run_id[:10]} step={self.step_index}/{total_steps} "
                    f"pop={len(self.individuals)} births={sum(int(r['births']) for r in self.window_rows)}",
                    flush=True,
                )
                next_progress += self.cfg.progress_interval
            if not self.individuals:
                # The replicate terminates at extinction. Later event rates are zero and the
                # requested horizon remains recorded in the run summary.
                if self.step_index % self.campaign.window_size != 0:
                    while self.step_index % self.campaign.window_size != 0 and self.step_index < total_steps:
                        self._source_update()
                        self._transport_and_dissipation()
                        self.step_index += 1
                    if self.step_index % self.campaign.window_size == 0:
                        self._close_window(allow_causal=False)
                break
        self._quality_checks()

    def finalize(self, requested_steps: int) -> Dict[str, Any]:
        if self.step_index % self.campaign.window_size != 0 and (
            self.window_function_flux or self.window_births or self.window_deaths
        ):
            self._close_window(allow_causal=False)
        self._finalize_causal_classification()
        self._classify_niche_origins()
        summary = summarize_run(self.window_rows, requested_steps, self.campaign, self.cfg, self.condition)
        summary.update({
            "run_id": self.run_id,
            "world_id": self.world.key.world_id,
            "world_family": self.world.key.family,
            "topology": self.world.key.topology,
            "forcing": self.world.key.forcing,
            "alphabet_size": self.world.key.alphabet_size,
            "world_replicate": self.world.key.replicate,
            "world_seed": self.world.seed,
            "condition": self.condition.name,
            "constructive": int(self.condition.constructive),
            "extensible": int(self.condition.extensible),
            "mutation": int(self.condition.mutation),
            "constructed_half_life": (
                self.condition.constructed_half_life
                if self.condition.constructed_half_life is not None else "inf"
            ),
            "retention_sweep": int(self.condition.retention_sweep),
            "factorial": int(self.condition.factorial),
            "evolutionary_seed": self.evolutionary_seed,
            "pair_id": f"{self.world.key.world_id}__{self.evolutionary_seed}",
            "requested_steps": requested_steps,
            "completed_steps": self.step_index,
            "extinct": int(len(self.individuals) == 0),
            "final_population": len(self.individuals),
            "turnover_scale": self.cfg.turnover_scale,
            "runtime_seconds": time.time() - self.start_wall,
            "script_version": SCRIPT_VERSION,
        })
        return {
            "summary": json_safe(summary),
            "window_rows": json_safe(self.window_rows),
            "function_events": json_safe(self.function_events),
            "interaction_events": json_safe(self.interaction_events),
            "substrate_events": json_safe(self.substrate_events),
            "genome_function_events": json_safe(self.genome_function_events),
            "substrate_origin_events": json_safe(self.substrate_origin_events),
            "niche_origin_rows": json_safe(self.niche_origin_rows),
            "causal_rows": json_safe(self.causal_rows),
            "lineage_events": json_safe(self.lineage_events),
            "qc": json_safe(self.qc),
        }

# -----------------------------------------------------------------------------
# Birth-indexed OEE trajectory models and finite-time operational classification
# -----------------------------------------------------------------------------

def fit_trajectory_models(t: np.ndarray, y: np.ndarray, minimum_events: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "model_comparison_valid": False,
        "linear_rmse": math.nan,
        "power_rmse": math.nan,
        "saturation_rmse": math.nan,
        "linear_loglik": math.nan,
        "power_loglik": math.nan,
        "saturation_loglik": math.nan,
        "best_model": None,
        "saturation_outperforms_both": False,
    }
    mask = np.isfinite(t) & np.isfinite(y)
    t = np.asarray(t[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    if t.size < 12 or int(y[-1]) < minimum_events or np.unique(y).size < minimum_events:
        return out
    # Remove duplicate birth coordinates caused by zero-birth windows, retaining the
    # latest cumulative outcome at each coordinate.
    unique_t: List[float] = []
    unique_y: List[float] = []
    for tx, yy in zip(t.tolist(), y.tolist()):
        if unique_t and math.isclose(tx, unique_t[-1], rel_tol=0.0, abs_tol=1e-12):
            unique_y[-1] = yy
        else:
            unique_t.append(tx)
            unique_y.append(yy)
    t = np.asarray(unique_t, dtype=float)
    y = np.asarray(unique_y, dtype=float)
    if t.size < 12:
        return out
    scale = max(float(t[-1]), 1.0)
    x = t / scale
    split = max(8, int(math.floor(0.70 * x.size)))
    if split >= x.size - 3:
        return out
    tx, ty = x[:split], y[:split]
    vx, vy = x[split:], y[split:]

    def score(pred: np.ndarray) -> Tuple[float, float]:
        pred = np.maximum(np.asarray(pred, dtype=float), 0.0)
        residual = vy - pred
        mse = max(float(np.mean(residual * residual)), 1e-12)
        return float(math.sqrt(mse)), float(-0.5 * len(vy) * (math.log(2.0 * math.pi * mse) + 1.0))

    # Linear model.
    X = np.column_stack([np.ones_like(tx), tx])
    coef, *_ = np.linalg.lstsq(X, ty, rcond=None)
    pred = np.column_stack([np.ones_like(vx), vx]) @ coef
    out["linear_rmse"], out["linear_loglik"] = score(pred)
    out["linear_params"] = [float(coef[0]), float(coef[1] / scale)]

    # Power model y = a + b x^c. Grid over c and solve a,b by least squares.
    best_train = math.inf
    best_power = None
    for exponent in np.linspace(0.05, 3.0, 240):
        basis = np.power(np.maximum(tx, 1e-12), exponent)
        XP = np.column_stack([np.ones_like(tx), basis])
        beta, *_ = np.linalg.lstsq(XP, ty, rcond=None)
        if beta[1] < 0.0:
            continue
        train_mse = float(np.mean((ty - XP @ beta) ** 2))
        if train_mse < best_train:
            best_train = train_mse
            best_power = (float(beta[0]), float(beta[1]), float(exponent))
    if best_power is not None:
        a, b, exponent = best_power
        pred = a + b * np.power(np.maximum(vx, 1e-12), exponent)
        out["power_rmse"], out["power_loglik"] = score(pred)
        out["power_params"] = [a, b / (scale ** exponent), exponent]

    # Saturation model y = a + K(1-exp(-lambda x)). Grid over lambda and solve a,K.
    best_train = math.inf
    best_sat = None
    for lam in np.logspace(-3, 3, 320):
        basis = 1.0 - np.exp(-lam * np.maximum(tx, 0.0))
        XS = np.column_stack([np.ones_like(tx), basis])
        beta, *_ = np.linalg.lstsq(XS, ty, rcond=None)
        if beta[1] < 0.0:
            continue
        train_mse = float(np.mean((ty - XS @ beta) ** 2))
        if train_mse < best_train:
            best_train = train_mse
            best_sat = (float(beta[0]), float(beta[1]), float(lam))
    if best_sat is not None:
        a, K, lam = best_sat
        pred = a + K * (1.0 - np.exp(-lam * np.maximum(vx, 0.0)))
        out["saturation_rmse"], out["saturation_loglik"] = score(pred)
        out["saturation_params"] = [K, lam / scale, a]

    rmses = {
        name: float(out[f"{name}_rmse"])
        for name in ("linear", "power", "saturation")
        if math.isfinite(float(out[f"{name}_rmse"]))
    }
    if len(rmses) >= 2:
        out["best_model"] = min(rmses, key=rmses.get)
    if len(rmses) == 3:
        out["model_comparison_valid"] = True
        out["saturation_outperforms_both"] = bool(
            out["saturation_rmse"] < out["linear_rmse"]
            and out["saturation_rmse"] < out["power_rmse"]
        )
    return out


def summarize_run(
    rows: Sequence[Mapping[str, Any]],
    requested_steps: int,
    campaign: CampaignConfig,
    cfg: ModelConfig,
    condition: Condition,
) -> Dict[str, Any]:
    all_rows = list(rows)
    analysis_rows = [r for r in all_rows if r.get("analysis_phase", "analysis") == "analysis"]
    total_births_all = int(sum(int(r.get("births", 0)) for r in all_rows))
    total_reactions_all = int(sum(int(r.get("reactions", 0)) for r in all_rows))
    last_any = all_rows[-1] if all_rows else {}
    empty = {
        "oee_operational": False,
        "oee_evaluable": False,
        "late_causal_novelty_rate": 0.0,
        "late_causal_novelty_per_1000_births": 0.0,
        "late_persistent_novelty_rate": 0.0,
        "late_persistent_novelty_per_1000_births": 0.0,
        "events_in_all_birth_thirds": False,
        "late_ecological_expansion": False,
        "saturation_not_preferred": False,
        "final_causal_functions": 0,
        "final_persistent_functions": 0,
        "final_persistent_interactions": 0,
        "final_persistent_derived_substrates": 0,
        "final_enabling_depth": 0,
        "total_births": total_births_all,
        "analysis_births": 0,
        "expected_functional_mutants": 0.0,
        "mutation_supply_adequate": False,
        "birth_epochs_evaluable": False,
        "trajectory_model_required": False,
        "trajectory_model_valid": False,
        "total_reactions": total_reactions_all,
        "analysis_reactions": 0,
        "causal_novelty_per_1000_births": 0.0,
        "persistent_novelty_per_1000_births": 0.0,
        "persistent_interactions_per_1000_births": 0.0,
        "persistent_derived_substrates_per_1000_births": 0.0,
        "final_mean_genome_modules": float(last_any.get("mean_genome_modules", 0.0)),
        "final_max_genome_modules": int(last_any.get("max_genome_modules", 0)),
        "constructed_symbol_mass": float(last_any.get("constructed_symbol_mass", 0.0)),
        "constructed_mass_auc": float(last_any.get("constructed_mass_auc", 0.0)),
        "constructed_production_symbol_mass": float(last_any.get("constructed_production_symbol_mass", 0.0)),
        "constructed_consumption_symbol_mass": float(last_any.get("constructed_consumption_symbol_mass", 0.0)),
        "constructed_utilization_ratio": float(last_any.get("constructed_utilization_ratio", 0.0)),
        "empirical_constructed_residence_time": float(last_any.get("empirical_constructed_residence_time", 0.0)),
    }
    if not analysis_rows:
        return empty

    cumulative_births = np.asarray(
        [float(r.get("analysis_cumulative_births", 0.0)) for r in analysis_rows], dtype=float
    )
    cumulative_causal = np.asarray(
        [float(r.get("causal_persistent_functions", 0.0)) for r in analysis_rows], dtype=float
    )
    cumulative_persistent = np.asarray(
        [float(r.get("persistent_functions", 0.0)) for r in analysis_rows], dtype=float
    )
    interactions = np.asarray(
        [float(r.get("persistent_interactions", 0.0)) for r in analysis_rows], dtype=float
    )
    substrates = np.asarray(
        [float(r.get("persistent_derived_substrates", 0.0)) for r in analysis_rows], dtype=float
    )
    dmax = np.asarray([float(r.get("enabling_depth_max", 0.0)) for r in analysis_rows], dtype=float)
    new_causal = np.asarray([float(r.get("new_causal_functions", 0.0)) for r in analysis_rows], dtype=float)
    new_persistent = np.asarray(
        [float(r.get("new_persistent_functions", 0.0)) for r in analysis_rows], dtype=float
    )
    analysis_births = int(cumulative_births[-1]) if cumulative_births.size else 0
    birth_epochs_evaluable = analysis_births >= campaign.min_analysis_births_for_epochs

    b1 = analysis_births / 3.0
    b2 = 2.0 * analysis_births / 3.0
    event_flags = (
        float(new_causal[cumulative_births <= b1].sum()) > 0.0,
        float(new_causal[(cumulative_births > b1) & (cumulative_births <= b2)].sum()) > 0.0,
        float(new_causal[cumulative_births > b2].sum()) > 0.0,
    ) if analysis_births > 0 else (False, False, False)
    late_mask = cumulative_births > b2
    late_events = float(new_causal[late_mask].sum()) if late_mask.any() else 0.0
    late_persistent_events = float(new_persistent[late_mask].sum()) if late_mask.any() else 0.0
    late_birth_exposure = analysis_births / 3.0 if analysis_births > 0 else 0.0
    late_rate = late_events / late_birth_exposure if late_birth_exposure > 0.0 else 0.0
    late_persistent_rate = (
        late_persistent_events / late_birth_exposure if late_birth_exposure > 0.0 else 0.0
    )

    late_ecological_expansion = False
    if np.count_nonzero(late_mask) >= 2:
        late_ecological_expansion = bool(
            interactions[late_mask][-1] > interactions[late_mask][0]
            or substrates[late_mask][-1] > substrates[late_mask][0]
            or dmax[late_mask][-1] > dmax[late_mask][0]
        )

    fits = fit_trajectory_models(
        cumulative_births, cumulative_causal, cfg.min_causal_events_for_oee_fit
    )
    mutation_probability = (
        cfg.functional_mutation_probability_extensible_lower
        if condition.extensible else cfg.functional_mutation_probability_closed_lower
    ) if condition.mutation else 0.0
    expected_functional_mutants = analysis_births * mutation_probability
    mutation_supply_adequate = bool(
        condition.mutation
        and expected_functional_mutants >= campaign.preflight_min_expected_functional_mutants_main
    )
    final_causal_event_count = int(cumulative_causal[-1])
    trajectory_model_required = final_causal_event_count >= cfg.min_causal_events_for_oee_fit
    trajectory_model_valid = bool(fits.get("model_comparison_valid", False))
    evaluable = bool(
        mutation_supply_adequate
        and birth_epochs_evaluable
        and (not trajectory_model_required or trajectory_model_valid)
    )
    saturation_not_preferred = bool(
        trajectory_model_required
        and trajectory_model_valid
        and not fits.get("saturation_outperforms_both", False)
    )
    oee = bool(
        evaluable
        and trajectory_model_required
        and late_rate > 0.0
        and all(event_flags)
        and late_ecological_expansion
        and saturation_not_preferred
    )
    last = analysis_rows[-1]
    per_1000 = 1000.0 / analysis_births if analysis_births > 0 else 0.0
    out = {
        "oee_operational": oee,
        "oee_evaluable": evaluable,
        "late_causal_novelty_rate": late_rate,
        "late_causal_novelty_per_1000_births": 1000.0 * late_rate,
        "late_persistent_novelty_rate": late_persistent_rate,
        "late_persistent_novelty_per_1000_births": 1000.0 * late_persistent_rate,
        "late_causal_events": late_events,
        "late_persistent_events": late_persistent_events,
        "late_births": late_birth_exposure,
        "events_first_birth_third": event_flags[0],
        "events_middle_birth_third": event_flags[1],
        "events_final_birth_third": event_flags[2],
        "events_in_all_birth_thirds": all(event_flags),
        "late_ecological_expansion": late_ecological_expansion,
        "saturation_not_preferred": saturation_not_preferred,
        "final_causal_functions": final_causal_event_count,
        "final_persistent_functions": int(cumulative_persistent[-1]),
        "final_persistent_interactions": int(interactions[-1]),
        "final_persistent_derived_substrates": int(substrates[-1]),
        "final_constructed_substrates_ever": int(last.get("analysis_constructed_substrates_ever", 0)),
        "final_current_derived_substrates": int(last.get("current_derived_substrates", 0)),
        "final_enabling_depth": int(dmax[-1]),
        "total_births": total_births_all,
        "analysis_births": analysis_births,
        "expected_functional_mutants": expected_functional_mutants,
        "mutation_supply_adequate": mutation_supply_adequate,
        "birth_epochs_evaluable": birth_epochs_evaluable,
        "trajectory_model_required": trajectory_model_required,
        "trajectory_model_valid": trajectory_model_valid,
        "total_reactions": total_reactions_all,
        "analysis_reactions": int(sum(int(r.get("reactions", 0)) for r in analysis_rows)),
        "causal_novelty_per_1000_births": final_causal_event_count * per_1000,
        "persistent_novelty_per_1000_births": int(cumulative_persistent[-1]) * per_1000,
        "persistent_interactions_per_1000_births": int(interactions[-1]) * per_1000,
        "persistent_derived_substrates_per_1000_births": int(substrates[-1]) * per_1000,
        "final_mean_genome_modules": float(last.get("mean_genome_modules", 0.0)),
        "final_max_genome_modules": int(last.get("max_genome_modules", 0)),
        "constructed_symbol_mass": float(last.get("constructed_symbol_mass", 0.0)),
        "constructed_mass_auc": float(last.get("constructed_mass_auc", 0.0)),
        "constructed_production_symbol_mass": float(last.get("constructed_production_symbol_mass", 0.0)),
        "constructed_consumption_symbol_mass": float(last.get("constructed_consumption_symbol_mass", 0.0)),
        "constructed_utilization_ratio": float(last.get("constructed_utilization_ratio", 0.0)),
        "empirical_constructed_residence_time": float(last.get("empirical_constructed_residence_time", 0.0)),
    }
    out.update(fits)
    return out


# -----------------------------------------------------------------------------
# Campaign tasks, preflight, and worker execution
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    world_key: WorldKey
    condition_name: str
    seed_index: int
    evolutionary_seed: int
    total_steps: int
    run_id: str


@dataclass(frozen=True)
class LongTermTask:
    base_task: Task
    source_run_id: str
    total_steps: int
    run_id: str


_WORKER_WORLD_CACHE: Dict[str, WorldSpec] = {}


def task_paths(out_dir: Path, task: Task) -> Tuple[Path, Path, Path]:
    return (
        out_dir / "runs" / f"{task.run_id}.json.gz",
        out_dir / "checkpoints" / f"{task.run_id}.pkl.gz",
        out_dir / "failures" / f"{task.run_id}.json",
    )


def terminal_state_path(out_dir: Path, run_id: str) -> Path:
    return out_dir / "states" / f"{run_id}.pkl.gz"


def group_task_blocks(tasks: Sequence[Task]) -> List[List[Task]]:
    grouped: Dict[Tuple[str, int], List[Task]] = defaultdict(list)
    for task in tasks:
        grouped[(task.world_key.world_id, task.seed_index)].append(task)
    condition_order = {name: i for i, name in enumerate(CONDITIONS)}
    blocks = []
    for key in sorted(grouped):
        block = sorted(grouped[key], key=lambda t: condition_order.get(t.condition_name, 10_000))
        blocks.append(block)
    return blocks


def build_tasks(campaign: CampaignConfig, root_seed: int, cfg: ModelConfig) -> List[Task]:
    tasks: List[Task] = []
    config_hash = stable_id(json.dumps(json_safe(asdict(cfg)), sort_keys=True), n=8)
    for key in build_world_keys(campaign, root_seed):
        for seed_index in range(campaign.evolutionary_seeds):
            evolutionary_seed = stable_seed(root_seed, "evolution", key.world_id, seed_index)
            # Interleave conditions within a paired world-seed block.
            for condition_name in campaign.conditions:
                run_id = stable_id(
                    SCRIPT_VERSION,
                    config_hash,
                    key.world_id,
                    seed_index,
                    condition_name,
                    campaign.steps,
                )
                tasks.append(Task(
                    world_key=key,
                    condition_name=condition_name,
                    seed_index=seed_index,
                    evolutionary_seed=evolutionary_seed,
                    total_steps=campaign.steps,
                    run_id=run_id,
                ))
    return tasks


def build_longterm_tasks(
    main_tasks: Sequence[Task], campaign: CampaignConfig, cfg: ModelConfig
) -> List[LongTermTask]:
    long_steps = int(campaign.steps * campaign.longterm_multiplier)
    out: List[LongTermTask] = []
    config_hash = stable_id(json.dumps(json_safe(asdict(cfg)), sort_keys=True), n=8)
    for task in main_tasks:
        if task.condition_name not in campaign.longterm_conditions:
            continue
        run_id = stable_id(
            SCRIPT_VERSION, "longterm", config_hash, task.world_key.world_id,
            task.seed_index, task.condition_name, long_steps,
        )
        out.append(LongTermTask(task, task.run_id, long_steps, run_id))
    return out


def execute_block(payload: Mapping[str, Any]) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    for task_payload in payload["tasks"]:
        single = dict(payload)
        single.pop("tasks", None)
        single["task"] = task_payload
        records.append(execute_task(single))
    return {"block_id": payload.get("block_id"), "records": records}


def _rewrite_engine_run_id(engine: SimulationEngine, new_run_id: str) -> None:
    engine.run_id = new_run_id
    for rows in (
        engine.window_rows, engine.function_events, engine.interaction_events,
        engine.substrate_events, engine.genome_function_events,
        engine.substrate_origin_events, engine.niche_origin_rows,
        engine.causal_rows, engine.lineage_events,
    ):
        for row in rows:
            if isinstance(row, dict) and "run_id" in row:
                row["run_id"] = new_run_id


def execute_longterm_task(payload: Mapping[str, Any]) -> Dict[str, Any]:
    base = Task(
        world_key=WorldKey(**payload["task"]["world_key"]),
        condition_name=str(payload["task"]["condition_name"]),
        seed_index=int(payload["task"]["seed_index"]),
        evolutionary_seed=int(payload["task"]["evolutionary_seed"]),
        total_steps=int(payload["task"]["base_total_steps"]),
        run_id=str(payload["task"]["source_run_id"]),
    )
    long_task = Task(
        world_key=base.world_key,
        condition_name=base.condition_name,
        seed_index=base.seed_index,
        evolutionary_seed=base.evolutionary_seed,
        total_steps=int(payload["task"]["total_steps"]),
        run_id=str(payload["task"]["run_id"]),
    )
    campaign = CampaignConfig(**payload["campaign"])
    cfg = ModelConfig(**payload["model_config"])
    out_dir = Path(payload["out_dir"])
    main_out_dir = Path(payload["main_out_dir"])
    result_path, checkpoint_path, failure_path = task_paths(out_dir, long_task)
    if result_path.exists() and not bool(payload.get("overwrite", False)):
        return {"run_id": long_task.run_id, "status": "skipped_complete", "result": str(result_path)}
    source_path = terminal_state_path(main_out_dir, base.run_id)
    if not source_path.exists():
        record = {
            "run_id": long_task.run_id,
            "status": "dependency_missing",
            "reason": f"terminal state not found: {source_path}",
            "task": json_safe(payload["task"]),
        }
        atomic_write_text(failure_path, json.dumps(record, indent=2, ensure_ascii=False))
        return record
    try:
        if checkpoint_path.exists() and bool(payload.get("resume", True)):
            engine = SimulationEngine.load_checkpoint(checkpoint_path)
        else:
            engine = SimulationEngine.load_checkpoint(source_path)
            _rewrite_engine_run_id(engine, long_task.run_id)
            engine.start_wall = time.time()
        engine.run(long_task.total_steps, checkpoint_path=checkpoint_path)
        result = engine.finalize(long_task.total_steps)
        result["summary"]["continuation_from_run_id"] = base.run_id
        result["summary"]["continuation_start_steps"] = base.total_steps
        write_json_gz(result_path, result)
        checkpoint_path.unlink(missing_ok=True)
        failure_path.unlink(missing_ok=True)
        return {
            "run_id": long_task.run_id,
            "status": "complete",
            "result": str(result_path),
            "runtime_seconds": result["summary"].get("runtime_seconds"),
        }
    except SafetyStop as exc:
        record = {
            "run_id": long_task.run_id, "status": "safety_stop", "reason": str(exc),
            "task": json_safe(payload["task"]), "traceback": traceback.format_exc(),
        }
        atomic_write_text(failure_path, json.dumps(record, indent=2, ensure_ascii=False))
        return record
    except Exception as exc:
        record = {
            "run_id": long_task.run_id, "status": "failed", "reason": repr(exc),
            "task": json_safe(payload["task"]), "traceback": traceback.format_exc(),
        }
        atomic_write_text(failure_path, json.dumps(record, indent=2, ensure_ascii=False))
        return record


def execute_longterm_block(payload: Mapping[str, Any]) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    for task_payload in payload["tasks"]:
        single = dict(payload)
        single.pop("tasks", None)
        single["task"] = task_payload
        records.append(execute_longterm_task(single))
    return {"block_id": payload.get("block_id"), "records": records}


def execute_task(payload: Mapping[str, Any]) -> Dict[str, Any]:
    task = Task(
        world_key=WorldKey(**payload["task"]["world_key"]),
        condition_name=str(payload["task"]["condition_name"]),
        seed_index=int(payload["task"]["seed_index"]),
        evolutionary_seed=int(payload["task"]["evolutionary_seed"]),
        total_steps=int(payload["task"]["total_steps"]),
        run_id=str(payload["task"]["run_id"]),
    )
    campaign = CampaignConfig(**payload["campaign"])
    cfg = ModelConfig(**payload["model_config"])
    root_seed = int(payload["root_seed"])
    out_dir = Path(payload["out_dir"])
    resume = bool(payload.get("resume", True))
    overwrite = bool(payload.get("overwrite", False))
    result_path, checkpoint_path, failure_path = task_paths(out_dir, task)

    if result_path.exists() and not overwrite:
        return {"run_id": task.run_id, "status": "skipped_complete", "result": str(result_path)}
    if overwrite:
        for p in (result_path, checkpoint_path, failure_path):
            p.unlink(missing_ok=True)

    try:
        cache_key = f"{task.world_key.world_id}|{cfg.turnover_scale}|{campaign.n_sites}"
        world = _WORKER_WORLD_CACHE.get(cache_key)
        if world is None:
            world = generate_world(task.world_key, campaign.n_sites, cfg, root_seed)
            _WORKER_WORLD_CACHE[cache_key] = world
        condition = CONDITIONS[task.condition_name]
        if resume and checkpoint_path.exists():
            engine = SimulationEngine.load_checkpoint(checkpoint_path)
            if engine.run_id != task.run_id:
                raise RuntimeError("Checkpoint run_id mismatch")
        else:
            engine = SimulationEngine(
                world,
                condition,
                cfg,
                campaign,
                task.evolutionary_seed,
                task.run_id,
                causal_enabled=True,
            )
        engine.run(task.total_steps, checkpoint_path=checkpoint_path)
        result = engine.finalize(task.total_steps)
        write_json_gz(result_path, result)
        checkpoint_path.unlink(missing_ok=True)
        failure_path.unlink(missing_ok=True)
        return {
            "run_id": task.run_id,
            "status": "complete",
            "result": str(result_path),
            "runtime_seconds": result["summary"].get("runtime_seconds"),
        }
    except SafetyStop as exc:
        record = {
            "run_id": task.run_id,
            "status": "safety_stop",
            "reason": str(exc),
            "task": json_safe(payload["task"]),
            "traceback": traceback.format_exc(),
        }
        atomic_write_text(failure_path, json.dumps(record, indent=2, ensure_ascii=False))
        return record
    except Exception as exc:
        record = {
            "run_id": task.run_id,
            "status": "failed",
            "reason": repr(exc),
            "task": json_safe(payload["task"]),
            "traceback": traceback.format_exc(),
        }
        atomic_write_text(failure_path, json.dumps(record, indent=2, ensure_ascii=False))
        return record


def _preflight_one_world(
    key: WorldKey,
    campaign: CampaignConfig,
    base_cfg: ModelConfig,
    root_seed: int,
    turnover_scale: float,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.turnover_scale = float(turnover_scale)
    cfg.safety_max_population = campaign.preflight_max_population
    world = generate_world(key, campaign.n_sites, cfg, root_seed)
    seed = stable_seed(root_seed, "preflight", key.world_id)
    preflight_campaign = copy.deepcopy(campaign)
    preflight_campaign.establishment_steps = 0
    engine = SimulationEngine(
        world,
        CONDITIONS["no_mutation"],
        cfg,
        preflight_campaign,
        seed,
        run_id=f"preflight_{key.world_id}_{turnover_scale:g}",
        causal_enabled=False,
        preflight=True,
    )
    start = time.time()
    status = "complete"
    reason = ""
    try:
        engine.run(campaign.preflight_steps, checkpoint_path=None)
        result = engine.finalize(campaign.preflight_steps)
        summary = result["summary"]
    except Exception as exc:
        status = "failed"
        reason = repr(exc)
        summary = {
            "total_births": 0,
            "final_population": 0,
            "extinct": 1,
        }
    return {
        "turnover_scale": turnover_scale,
        "world_id": key.world_id,
        "world_family": key.family,
        "topology": key.topology,
        "forcing": key.forcing,
        "alphabet_size": key.alphabet_size,
        "status": status,
        "reason": reason,
        "births": int(summary.get("total_births", 0)),
        "final_population": int(summary.get("final_population", len(engine.individuals))),
        "extinct": int(len(engine.individuals) == 0),
        "max_population": int(engine.qc.get("max_population", 0)),
        "max_substrates": int(engine.qc.get("max_substrates", 0)),
        "mass_balance_relative": float(engine.qc.get("max_relative_mass_balance_residual", math.inf)),
        "runtime_seconds": time.time() - start,
    }


def preflight_signature(campaign: CampaignConfig, cfg: ModelConfig, root_seed: int) -> str:
    payload = {
        "script_version": SCRIPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "root_seed": int(root_seed),
        "campaign": json_safe(asdict(campaign)),
        "model_config": json_safe(asdict(cfg)),
    }
    return stable_id(json.dumps(json_safe(payload), sort_keys=True), n=16)


def run_preflight(
    out_dir: Path,
    campaign: CampaignConfig,
    base_cfg: ModelConfig,
    root_seed: int,
) -> Tuple[float, List[Dict[str, Any]], Dict[str, Any]]:
    world_keys = build_world_keys(campaign, root_seed)
    mutation_extensible = estimate_functional_mutation_probability(
        base_cfg, extensible=True
    )
    mutation_closed = estimate_functional_mutation_probability(
        base_cfg, extensible=False
    )
    all_rows: List[Dict[str, Any]] = []
    selected: Optional[float] = None
    selection_summary: Dict[str, Any] = {}

    print("[preflight] demographic and numerical adequacy check", flush=True)
    for scale in campaign.preflight_turnover_candidates:
        scale_rows: List[Dict[str, Any]] = []
        print(f"[preflight] turnover_scale={scale:g}", flush=True)
        for index, key in enumerate(world_keys, start=1):
            row = _preflight_one_world(key, campaign, base_cfg, root_seed, scale)
            scale_rows.append(row)
            all_rows.append(row)
            print(
                f"[preflight {index}/{len(world_keys)}] {key.world_id} "
                f"births={row['births']} pop={row['final_population']} status={row['status']}",
                flush=True,
            )
        births = np.asarray([r["births"] for r in scale_rows], dtype=float)
        extinct_fraction = float(np.mean([r["extinct"] for r in scale_rows]))
        max_population = max(r["max_population"] for r in scale_rows)
        failures = sum(r["status"] != "complete" for r in scale_rows)
        projected_births_min = float(births.min() * campaign.steps / campaign.preflight_steps)
        projected_functional_mutants_min = projected_births_min * min(
            mutation_extensible["wilson_lower_95"], mutation_closed["wilson_lower_95"]
        )
        criteria = {
            "all_completed": failures == 0,
            "min_births": float(births.min()) >= campaign.preflight_min_births_each_world,
            "median_births": float(np.median(births)) >= campaign.preflight_median_births,
            "population_bound": max_population <= campaign.preflight_max_population,
            "extinction_bound": extinct_fraction <= campaign.preflight_max_extinction_fraction,
            "mutation_supply": projected_functional_mutants_min >= campaign.preflight_min_expected_functional_mutants_main,
            "mass_balance": max(r["mass_balance_relative"] for r in scale_rows) <= base_cfg.mass_balance_relative_tolerance,
        }
        selection_summary = {
            "preflight_signature": preflight_signature(campaign, base_cfg, root_seed),
            "turnover_scale": scale,
            "functional_mutation_probability_extensible": mutation_extensible,
            "functional_mutation_probability_closed": mutation_closed,
            "births_min": float(births.min()),
            "births_median": float(np.median(births)),
            "births_max": float(births.max()),
            "projected_main_births_min": projected_births_min,
            "projected_main_functional_mutants_min": projected_functional_mutants_min,
            "extinction_fraction": extinct_fraction,
            "max_population": max_population,
            "failures": failures,
            "runtime_seconds_total": float(sum(r["runtime_seconds"] for r in scale_rows)),
            "criteria": criteria,
            "passed": all(criteria.values()),
        }
        atomic_write_text(
            out_dir / "preflight_latest.json",
            json.dumps(json_safe(selection_summary), indent=2, ensure_ascii=False),
        )
        if all(criteria.values()):
            selected = scale
            break

    write_csv(out_dir / "01_preflight_worlds.csv", all_rows)
    if selected is None:
        atomic_write_text(
            out_dir / "01_preflight_failure.json",
            json.dumps(json_safe(selection_summary), indent=2, ensure_ascii=False),
        )
        raise RuntimeError(
            "Preflight failed for every prespecified turnover scale. Main outcomes were not computed. "
            "Inspect 01_preflight_worlds.csv and 01_preflight_failure.json."
        )
    selection_summary["selected_turnover_scale"] = selected
    atomic_write_text(
        out_dir / "01_preflight_selection.json",
        json.dumps(json_safe(selection_summary), indent=2, ensure_ascii=False),
    )
    return selected, all_rows, selection_summary


def auto_workers(requested: str, task_count: int) -> int:
    if requested != "auto":
        return max(1, min(int(requested), max(task_count, 1)))
    cpus = max(1, (os.cpu_count() or 2) - 1)
    by_memory = cpus
    if PSUTIL_AVAILABLE and psutil is not None:
        available = int(psutil.virtual_memory().available)
        # Causal state-clone assays are the peak-memory operation. A conservative
        # 1.5 GiB/worker budget prevents swap while allowing more than two workers on
        # machines with adequate memory.
        by_memory = max(1, int(available // (1.5 * 1024**3)))
    platform_cap = 6 if sys.platform == "darwin" else 8
    return max(1, min(cpus, by_memory, platform_cap, max(task_count, 1)))

# -----------------------------------------------------------------------------
# Aggregation and statistical analysis
# -----------------------------------------------------------------------------

def flatten_event(base: Mapping[str, Any], row: Mapping[str, Any]) -> Dict[str, Any]:
    out = {
        "run_id": base.get("run_id"),
        "world_id": base.get("world_id"),
        "world_family": base.get("world_family"),
        "topology": base.get("topology"),
        "forcing": base.get("forcing"),
        "alphabet_size": base.get("alphabet_size"),
        "condition": base.get("condition"),
        "evolutionary_seed": base.get("evolutionary_seed"),
        "pair_id": base.get("pair_id"),
    }
    out.update(row)
    return out


PRIMARY_METRIC = "late_causal_novelty_per_1000_births"
ANALYSIS_METRICS = (
    PRIMARY_METRIC,
    "late_persistent_novelty_per_1000_births",
    "causal_novelty_per_1000_births",
    "persistent_novelty_per_1000_births",
    "persistent_interactions_per_1000_births",
    "persistent_derived_substrates_per_1000_births",
    "final_enabling_depth",
    "final_mean_genome_modules",
)


def condition_summary(run_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row["condition"])].append(row)
    metrics = (
        "oee_operational", "oee_evaluable", "birth_epochs_evaluable",
        *ANALYSIS_METRICS,
        "final_causal_functions", "final_persistent_functions",
        "final_persistent_interactions", "final_persistent_derived_substrates",
        "final_constructed_substrates_ever", "final_current_derived_substrates",
        "total_births", "analysis_births", "expected_functional_mutants",
        "mutation_supply_adequate", "trajectory_model_valid",
        "final_max_genome_modules", "extinct", "runtime_seconds",
    )
    output: List[Dict[str, Any]] = []
    for condition, rows in sorted(grouped.items()):
        record: Dict[str, Any] = {"condition": condition, "n_runs": len(rows)}
        for metric in metrics:
            values: List[float] = []
            for row in rows:
                value = row.get(metric)
                if isinstance(value, bool):
                    values.append(float(value))
                elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values.append(float(value))
            mean, sd = mean_sd(values)
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sd"] = sd
        output.append(record)
    return output


def _add_bh(rows: List[Dict[str, Any]], p_key: str = "sign_flip_p") -> None:
    q_values = benjamini_hochberg([float(r.get(p_key, math.nan)) for r in rows])
    for row, q_value in zip(rows, q_values):
        row["q_bh"] = q_value
        row["significant_bh"] = bool(
            math.isfinite(q_value) and q_value <= 0.05
            and isinstance(row.get("ci_low"), (int, float))
            and float(row["ci_low"]) > 0.0
        )


def paired_contrasts(run_rows: Sequence[Mapping[str, Any]], root_seed: int) -> List[Dict[str, Any]]:
    index = {(str(r["pair_id"]), str(r["condition"])): r for r in run_rows}
    pairs = sorted({str(r["pair_id"]) for r in run_rows})
    comparators = [condition for condition in CONDITIONS if condition != "full"]
    output: List[Dict[str, Any]] = []
    for comparator in comparators:
        for metric in ANALYSIS_METRICS:
            differences: List[float] = []
            for pair in pairs:
                full = index.get((pair, "full"))
                other = index.get((pair, comparator))
                if full is None or other is None:
                    continue
                a, b = full.get(metric), other.get(metric)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    if math.isfinite(float(a)) and math.isfinite(float(b)):
                        differences.append(float(a) - float(b))
            mean, low, high = paired_mean_ci(differences)
            output.append({
                "comparison": f"full_minus_{comparator}",
                "metric": metric,
                "n_pairs": len(differences),
                "mean_difference": mean,
                "ci_low": low,
                "ci_high": high,
                "positive_pair_fraction": (
                    sum(x > 0.0 for x in differences) / len(differences) if differences else math.nan
                ),
                "sign_flip_p": exact_sign_flip_p(
                    differences, stable_seed(root_seed, "paired", comparator, metric),
                    alternative="greater",
                ),
            })
    _add_bh(output)
    return output


def factorial_effects(run_rows: Sequence[Mapping[str, Any]], root_seed: int) -> List[Dict[str, Any]]:
    factorial = [r for r in run_rows if int(r.get("factorial", 0)) == 1]
    pairs = sorted({str(r["pair_id"]) for r in factorial})
    by_pair_condition = {(str(r["pair_id"]), str(r["condition"])): r for r in factorial}
    terms = {
        "C": lambda c, g: c,
        "G": lambda c, g: g,
        "C:G": lambda c, g: c * g,
    }
    output: List[Dict[str, Any]] = []
    for metric in ANALYSIS_METRICS:
        pair_effects: Dict[str, List[float]] = {term: [] for term in terms}
        for pair in pairs:
            cells: List[Tuple[int, int, float]] = []
            complete = True
            for condition_name in FACTORIAL_CONDITIONS:
                row = by_pair_condition.get((pair, condition_name))
                if row is None or not isinstance(row.get(metric), (int, float)):
                    complete = False
                    break
                value = float(row[metric])
                if not math.isfinite(value):
                    complete = False
                    break
                c = 1 if int(row["constructive"]) == 1 else -1
                g = 1 if int(row["extensible"]) == 1 else -1
                cells.append((c, g, value))
            if not complete or len(cells) != 4:
                continue
            for term, contrast in terms.items():
                # For a 2 x 2 design coded {-1,+1}, division by 2 gives the
                # difference-scale main effect or interaction contrast.
                effect = sum(contrast(c, g) * value for c, g, value in cells) / 2.0
                pair_effects[term].append(float(effect))
        for term, values in pair_effects.items():
            mean, low, high = paired_mean_ci(values)
            output.append({
                "metric": metric,
                "term": term,
                "n_complete_pairs": len(values),
                "effect": mean,
                "ci_low": low,
                "ci_high": high,
                "positive_pair_fraction": (
                    sum(x > 0.0 for x in values) / len(values) if values else math.nan
                ),
                "sign_flip_p": exact_sign_flip_p(
                    values, stable_seed(root_seed, "factorial", metric, term),
                    alternative="greater",
                ),
            })
    _add_bh(output)
    return output


def run_gee(run_rows: Sequence[Mapping[str, Any]], out_dir: Path) -> List[Dict[str, Any]]:
    """Poisson log-linear model with pair-clustered sandwich covariance.

    Event counts use cumulative births as the exposure offset. Paired factorial
    contrasts remain primary; this regression is a world-family-adjusted robustness
    analysis.
    """
    rows = [r for r in run_rows if int(r.get("factorial", 0)) == 1]
    status_path = out_dir / "14_gee_status.txt"
    if not rows:
        atomic_write_text(status_path, "Clustered Poisson skipped: no factorial rows\n")
        return []
    y = np.asarray([float(r.get("late_causal_events", 0.0) or 0.0) for r in rows], dtype=float)
    births = np.asarray([float(r.get("late_births", 0.0) or 0.0) for r in rows], dtype=float)
    valid = np.isfinite(y) & np.isfinite(births) & (y >= 0.0) & (births > 0.0)
    rows = [row for row, keep in zip(rows, valid.tolist()) if keep]
    y = y[valid]
    births = births[valid]
    if not rows or float(y.sum()) <= 0.0:
        atomic_write_text(status_path, "Clustered Poisson not estimable: no positive valid event counts\n")
        return []

    families = sorted({str(r.get("world_family", "unknown")) for r in rows})
    reference_family = families[0]
    family_terms = [f"world_family[{f}]" for f in families[1:]]
    terms = ["Intercept", "constructive", "extensible", "constructive:extensible"] + family_terms
    X = np.zeros((len(rows), len(terms)), dtype=float)
    for i, row in enumerate(rows):
        C = float(int(row.get("constructive", 0)))
        G = float(int(row.get("extensible", 0)))
        X[i, :4] = (1.0, C, G, C * G)
        family = str(row.get("world_family", "unknown"))
        for j, candidate in enumerate(families[1:], start=4):
            X[i, j] = float(family == candidate)
    offset = np.log(births)
    beta = np.zeros(X.shape[1], dtype=float)
    converged = False
    iterations = 0
    bread = None
    for iterations in range(1, 101):
        eta = np.clip(offset + X @ beta, -30.0, 30.0)
        mu = np.exp(eta)
        W = np.maximum(mu, 1e-12)
        z = eta + (y - mu) / W - offset
        XtW = X.T * W
        bread = XtW @ X + np.eye(X.shape[1]) * 1e-9
        try:
            updated = np.linalg.solve(bread, XtW @ z)
        except np.linalg.LinAlgError:
            updated = np.linalg.pinv(bread) @ (XtW @ z)
        if float(np.max(np.abs(updated - beta))) < 1e-9:
            beta = updated
            converged = True
            break
        beta = updated
    eta = np.clip(offset + X @ beta, -30.0, 30.0)
    mu = np.exp(eta)
    bread_inv = np.linalg.pinv((X.T * np.maximum(mu, 1e-12)) @ X + np.eye(X.shape[1]) * 1e-9)
    cluster_ids = [str(r["pair_id"]) for r in rows]
    meat = np.zeros((X.shape[1], X.shape[1]), dtype=float)
    unique_clusters = sorted(set(cluster_ids))
    for cluster in unique_clusters:
        idx = np.asarray([i for i, c in enumerate(cluster_ids) if c == cluster], dtype=int)
        score = X[idx].T @ (y[idx] - mu[idx])
        meat += np.outer(score, score)
    covariance = bread_inv @ meat @ bread_inv
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    output: List[Dict[str, Any]] = []
    for term, estimate, std_error in zip(terms, beta.tolist(), se.tolist()):
        z_value = estimate / std_error if std_error > 0.0 else math.nan
        p_value = math.erfc(abs(z_value) / math.sqrt(2.0)) if math.isfinite(z_value) else math.nan
        output.append({
            "term": term, "estimate": estimate, "std_error": std_error,
            "z": z_value, "p": p_value,
            "ci_low": estimate - 1.959963984540054 * std_error,
            "ci_high": estimate + 1.959963984540054 * std_error,
            "family": "poisson_pair_cluster_robust",
            "reference_world_family": reference_family,
            "n_rows": len(rows), "n_clusters": len(unique_clusters),
            "iterations": iterations, "converged": converged,
        })
    atomic_write_text(
        status_path,
        f"Clustered Poisson complete: converged={converged}; iterations={iterations}; "
        f"rows={len(rows)}; clusters={len(unique_clusters)}\n",
    )
    return output


def world_family_generality(run_rows: Sequence[Mapping[str, Any]], root_seed: int) -> List[Dict[str, Any]]:
    index = {(str(r["pair_id"]), str(r["condition"])): r for r in run_rows}
    pairs_by_family: Dict[str, Set[str]] = defaultdict(set)
    for row in run_rows:
        pairs_by_family[str(row["world_family"])].add(str(row["pair_id"]))
    output: List[Dict[str, Any]] = []
    for family, pairs in sorted(pairs_by_family.items()):
        for comparator in (
            "construction_only", "extensibility_only",
            "closed_control", "products_erased", "no_mutation",
        ):
            differences: List[float] = []
            for pair in sorted(pairs):
                full = index.get((pair, "full"))
                other = index.get((pair, comparator))
                if full is None or other is None:
                    continue
                a = full.get(PRIMARY_METRIC)
                b = other.get(PRIMARY_METRIC)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    if math.isfinite(float(a)) and math.isfinite(float(b)):
                        differences.append(float(a) - float(b))
            mean, low, high = paired_mean_ci(differences)
            output.append({
                "world_family": family,
                "comparison": f"full_minus_{comparator}",
                "metric": PRIMARY_METRIC,
                "n_pairs": len(differences),
                "mean_difference": mean,
                "ci_low": low,
                "ci_high": high,
                "positive_pair_fraction": (
                    sum(x > 0.0 for x in differences) / len(differences) if differences else math.nan
                ),
                "direction_positive": bool(math.isfinite(mean) and mean > 0.0),
                "sign_flip_p": exact_sign_flip_p(
                    differences, stable_seed(root_seed, "family", family, comparator),
                    alternative="greater",
                ),
            })
    return output


def assign_confirmatory_fdr(
    factorial: List[Dict[str, Any]],
    paired: List[Dict[str, Any]],
) -> None:
    """Correct only the prespecified proposition-defining hypothesis family.

    Exploratory q_bh values remain correction across every reported metric. The
    confirmatory family contains exactly six one-sided tests declared by the design:
    C x G synergy and Full against each single-mechanism condition, the closed
    control, the products-erased intervention, and the no-mutation control.
    """
    for row in [*factorial, *paired]:
        row["confirmatory_hypothesis"] = False
        row["confirmatory_q_bh"] = math.nan

    targets: List[Dict[str, Any]] = []
    for row in factorial:
        if row.get("metric") == PRIMARY_METRIC and row.get("term") == "C:G":
            targets.append(row)
    comparisons = {
        "full_minus_construction_only",
        "full_minus_extensibility_only",
        "full_minus_closed_control",
        "full_minus_products_erased",
        "full_minus_no_mutation",
    }
    for row in paired:
        if row.get("metric") == PRIMARY_METRIC and row.get("comparison") in comparisons:
            targets.append(row)

    q_values = benjamini_hochberg([
        float(row.get("sign_flip_p", math.nan)) for row in targets
    ])
    for row, q_value in zip(targets, q_values):
        row["confirmatory_hypothesis"] = True
        row["confirmatory_q_bh"] = q_value


def proposition_verdict(
    run_rows: Sequence[Mapping[str, Any]],
    factorial: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    qc_rows: Sequence[Mapping[str, Any]],
    expected_tasks: int,
    failures: int,
    campaign: CampaignConfig,
) -> Dict[str, Any]:
    def find_factor(term: str) -> Optional[Mapping[str, Any]]:
        return next((
            r for r in factorial
            if r.get("metric") == PRIMARY_METRIC and r.get("term") == term
        ), None)

    def find_pair(comparator: str) -> Optional[Mapping[str, Any]]:
        target = f"full_minus_{comparator}"
        return next((
            r for r in paired
            if r.get("metric") == PRIMARY_METRIC and r.get("comparison") == target
        ), None)

    def supported(row: Optional[Mapping[str, Any]], effect_key: str) -> bool:
        return bool(
            row is not None
            and isinstance(row.get(effect_key), (int, float))
            and math.isfinite(float(row[effect_key]))
            and float(row[effect_key]) > 0.0
            and isinstance(row.get("ci_low"), (int, float))
            and math.isfinite(float(row["ci_low"]))
            and float(row["ci_low"]) > 0.0
            and isinstance(row.get("confirmatory_q_bh"), (int, float))
            and math.isfinite(float(row["confirmatory_q_bh"]))
            and float(row["confirmatory_q_bh"]) <= campaign.alpha
        )

    cg = find_factor("C:G")
    full_construction = find_pair("construction_only")
    full_extensibility = find_pair("extensibility_only")
    full_closed = find_pair("closed_control")
    full_erased = find_pair("products_erased")
    full_nomut = find_pair("no_mutation")

    family_comparisons = {
        "full_minus_construction_only",
        "full_minus_extensibility_only",
        "full_minus_closed_control",
        "full_minus_products_erased",
    }
    family_targets = [
        r for r in family_rows if r.get("comparison") in family_comparisons
    ]
    family_positive_fraction = (
        sum(bool(r.get("direction_positive")) for r in family_targets) / len(family_targets)
        if family_targets else 0.0
    )
    expected_family_rows = len(TOPOLOGIES) * len(FORCINGS) * len(family_comparisons)
    family_rows_complete = len(family_targets) == expected_family_rows

    complete = len(run_rows) == expected_tasks and failures == 0
    no_safety_stops = all(not r.get("safety_stop") for r in qc_rows)
    closed_space_valid = all(
        int(r.get("closed_space_violations", 0) or 0) == 0 for r in qc_rows
    )
    analysis_conditions = {
        "full", "construction_only", "extensibility_only",
        "closed_control", "products_erased",
    }
    analysis_rows = [r for r in run_rows if r.get("condition") in analysis_conditions]
    exposure_adequate = bool(analysis_rows) and all(
        bool(r.get("birth_epochs_evaluable", False))
        and bool(r.get("mutation_supply_adequate", False))
        for r in analysis_rows
    )
    primary_rows_present = all(
        row is not None
        for row in (
            cg, full_construction, full_extensibility, full_closed, full_erased, full_nomut
        )
    )

    gates = {
        "complete_campaign": complete,
        "no_unresolved_safety_stops": no_safety_stops,
        "closed_structural_hereditary_space_verified": closed_space_valid,
        "birth_and_mutation_exposure_adequate": exposure_adequate,
        "all_confirmatory_tests_estimable": primary_rows_present,
        "constructive_extensible_synergy": supported(cg, "effect"),
        "full_exceeds_construction_only": supported(full_construction, "mean_difference"),
        "full_exceeds_extensibility_only": supported(full_extensibility, "mean_difference"),
        "full_exceeds_closed_control": supported(full_closed, "mean_difference"),
        "persistent_products_are_causally_required": supported(full_erased, "mean_difference"),
        "mutation_is_required": supported(full_nomut, "mean_difference"),
        "world_family_rows_complete": family_rows_complete,
        "direction_generalizes_across_all_prespecified_world_families": (
            family_rows_complete
            and family_positive_fraction >= campaign.family_direction_required
        ),
    }
    evaluable = bool(
        complete
        and no_safety_stops
        and closed_space_valid
        and exposure_adequate
        and primary_rows_present
        and family_rows_complete
    )
    inferential_gates = [
        gates["constructive_extensible_synergy"],
        gates["full_exceeds_construction_only"],
        gates["full_exceeds_extensibility_only"],
        gates["full_exceeds_closed_control"],
        gates["persistent_products_are_causally_required"],
        gates["mutation_is_required"],
        gates["direction_generalizes_across_all_prespecified_world_families"],
    ]
    verdict = (
        "SUPPORTED" if evaluable and all(inferential_gates)
        else ("NOT_SUPPORTED" if evaluable else "NOT_EVALUABLE")
    )
    return {
        "proposition": (
            "Across the prespecified abstract world families, expansion of the "
            "structural hereditary possibility space and organism-driven persistent "
            "environmental construction jointly increase birth-normalized, causally "
            "retained late novelty."
        ),
        "verdict": verdict,
        "primary_metric": PRIMARY_METRIC,
        "confirmatory_family_size": 6,
        "family_positive_fraction": family_positive_fraction,
        "analysis_rows_evaluable_fraction": (
            sum(
                bool(r.get("birth_epochs_evaluable", False))
                and bool(r.get("mutation_supply_adequate", False))
                for r in analysis_rows
            ) / len(analysis_rows) if analysis_rows else 0.0
        ),
        **{f"gate_{k}": v for k, v in gates.items()},
    }



def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def load_result_bundles(out_dir: Path) -> List[Dict[str, Any]]:
    bundles: List[Dict[str, Any]] = []
    for path in sorted((out_dir / "runs").glob("*.json.gz")):
        try:
            data = read_json_gz(path)
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("summary"), dict):
            bundles.append(data)
    return bundles


def bootstrap_mean_ci(
    values: Sequence[float], seed: int, draws: int = 2_000, alpha: float = 0.05
) -> Tuple[float, float, float]:
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if vals.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(vals.mean())
    if vals.size == 1 or draws <= 0:
        return mean, math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = np.empty(int(draws), dtype=float)
    batch = 256
    cursor = 0
    while cursor < draws:
        n = min(batch, draws - cursor)
        idx = rng.integers(0, vals.size, size=(n, vals.size))
        means[cursor:cursor + n] = vals[idx].mean(axis=1)
        cursor += n
    return mean, float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0))


def _birth_grid_from_windows(
    rows: Sequence[Mapping[str, Any]], bins: int
) -> Optional[Dict[str, np.ndarray]]:
    analysis = [
        r for r in rows
        if str(r.get("analysis_phase")) == "analysis"
        and math.isfinite(_finite_float(r.get("analysis_cumulative_births")))
    ]
    if not analysis:
        return None
    analysis.sort(key=lambda r: (
        _finite_float(r.get("analysis_cumulative_births"), 0.0),
        _finite_float(r.get("step"), 0.0),
    ))
    # Retain the latest state when several windows have the same cumulative birth count.
    collapsed: Dict[int, Mapping[str, Any]] = {}
    for row in analysis:
        collapsed[int(_finite_float(row.get("analysis_cumulative_births"), 0.0))] = row
    births = np.asarray(sorted(collapsed), dtype=float)
    if births.size < 3 or births[-1] <= 0.0:
        return None
    ordered = [collapsed[int(b)] for b in births.tolist()]
    edges = np.linspace(0.0, float(births[-1]), int(bins) + 1)
    cumulative_keys = {
        "substrates": "persistent_derived_substrates",
        "interactions": "persistent_interactions",
        "depth": "enabling_depth_max",
        "causal": "causal_persistent_functions",
        "genome": "mean_genome_modules",
        "constructed_ever": "analysis_constructed_substrates_ever",
    }
    sampled: Dict[str, List[float]] = {k: [] for k in cumulative_keys}
    j = 0
    for edge in edges[1:]:
        while j + 1 < births.size and births[j + 1] <= edge + 1e-12:
            j += 1
        row = ordered[j]
        for name, key in cumulative_keys.items():
            sampled[name].append(max(0.0, _finite_float(row.get(key), 0.0)))
    out: Dict[str, np.ndarray] = {
        name: np.asarray(values, dtype=float) for name, values in sampled.items()
    }
    for name in ("substrates", "interactions", "depth", "causal", "constructed_ever"):
        values = out[name]
        out[f"delta_{name}"] = np.maximum(np.diff(np.concatenate(([0.0], values))), 0.0)
    out["birth_edges"] = edges
    out["births_per_bin"] = np.asarray([float(births[-1]) / bins], dtype=float)
    return out


def _partial_standardized_slope(
    x: np.ndarray, y_future: np.ndarray, y_current: np.ndarray, trend: np.ndarray
) -> float:
    valid = np.isfinite(x) & np.isfinite(y_future) & np.isfinite(y_current) & np.isfinite(trend)
    x = np.asarray(x[valid], dtype=float)
    y_future = np.asarray(y_future[valid], dtype=float)
    y_current = np.asarray(y_current[valid], dtype=float)
    trend = np.asarray(trend[valid], dtype=float)
    if x.size < 5 or float(np.std(x)) <= 1e-12 or float(np.std(y_future)) <= 1e-12:
        return math.nan
    x = (x - x.mean()) / max(float(x.std(ddof=0)), 1e-12)
    y_future = (y_future - y_future.mean()) / max(float(y_future.std(ddof=0)), 1e-12)
    y_current = (y_current - y_current.mean()) / max(float(y_current.std(ddof=0)), 1e-12)
    trend = (trend - trend.mean()) / max(float(trend.std(ddof=0)), 1e-12)
    controls = np.column_stack([np.ones(x.size), y_current, trend])
    bx = np.linalg.lstsq(controls, x, rcond=None)[0]
    by = np.linalg.lstsq(controls, y_future, rcond=None)[0]
    rx = x - controls @ bx
    ry = y_future - controls @ by
    denom = float(rx @ rx)
    if denom <= 1e-12:
        return math.nan
    return float((rx @ ry) / denom)


def run_time_lagged_chain_analysis(
    core_dir: Path,
    campaign: CampaignConfig,
    root_seed: int,
    out_dir: Path,
) -> Dict[str, Any]:
    """Birth-indexed cross-lag analysis without altering the simulation.

    Each run is regridded into equal cumulative-birth bins. For each prespecified
    mechanism link, a standardized within-run partial slope predicts the future
    increment from the current increment while controlling the current outcome and
    secular birth-bin trend. Inference is across independent runs, not across windows.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bundles = load_result_bundles(core_dir)
    links = (
        ("constructed_substrates_to_interactions", "delta_substrates", "delta_interactions"),
        ("interactions_to_enabling_depth", "delta_interactions", "delta_depth"),
        ("enabling_depth_to_causal_novelty", "delta_depth", "delta_causal"),
        ("constructed_substrates_to_causal_novelty", "delta_substrates", "delta_causal"),
    )
    conditions = (
        "full", "products_erased", "construction_only",
        "extensibility_only", "closed_control", "no_mutation",
    )
    grids: List[Dict[str, Any]] = []
    for data in bundles:
        summary = data["summary"]
        grid = _birth_grid_from_windows(data.get("window_rows", []), campaign.birth_grid_bins)
        if grid is None:
            continue
        grids.append({
            "summary": summary,
            "grid": grid,
        })

    rows: List[Dict[str, Any]] = []
    per_run_rows: List[Dict[str, Any]] = []
    for condition in conditions:
        subset = [g for g in grids if str(g["summary"].get("condition")) == condition]
        for link_name, x_key, y_key in links:
            for lag in range(1, 5):
                slopes: List[float] = []
                birth_widths: List[float] = []
                for item in subset:
                    grid = item["grid"]
                    x = np.asarray(grid[x_key], dtype=float)
                    y = np.asarray(grid[y_key], dtype=float)
                    if x.size <= lag + 3:
                        continue
                    slope = _partial_standardized_slope(
                        x[:-lag], y[lag:], y[:-lag],
                        np.arange(x.size - lag, dtype=float),
                    )
                    if not math.isfinite(slope):
                        continue
                    slopes.append(slope)
                    birth_widths.append(float(grid["births_per_bin"][0]))
                    per_run_rows.append({
                        "run_id": item["summary"].get("run_id"),
                        "pair_id": item["summary"].get("pair_id"),
                        "world_family": item["summary"].get("world_family"),
                        "condition": condition,
                        "link": link_name,
                        "lag_bins": lag,
                        "lag_births": float(grid["births_per_bin"][0]) * lag,
                        "partial_standardized_slope": slope,
                    })
                mean, low, high = bootstrap_mean_ci(
                    slopes,
                    stable_seed(root_seed, "time_lag", condition, link_name, lag),
                    campaign.mechanism_bootstrap_draws,
                )
                rows.append({
                    "condition": condition,
                    "link": link_name,
                    "lag_bins": lag,
                    "median_lag_births": (
                        float(np.median(birth_widths)) * lag if birth_widths else math.nan
                    ),
                    "n_runs_estimable": len(slopes),
                    "mean_partial_standardized_slope": mean,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "ci_low": low,
                    "ci_high": high,
                    "positive_run_fraction": (
                        sum(v > 0.0 for v in slopes) / len(slopes) if slopes else math.nan
                    ),
                    "sign_flip_p": exact_sign_flip_p(
                        slopes,
                        stable_seed(root_seed, "time_lag_sign", condition, link_name, lag),
                        alternative="greater",
                    ),
                })
    _add_bh(rows)
    write_csv(out_dir / "01_time_lagged_chain_summary.csv", rows)
    write_csv(out_dir / "02_time_lagged_chain_run_slopes.csv", per_run_rows)

    full_rows = [r for r in rows if r["condition"] == "full"]
    dominant: List[Dict[str, Any]] = []
    for link_name, _, _ in links:
        candidates = [r for r in full_rows if r["link"] == link_name and r["n_runs_estimable"] > 0]
        best = max(candidates, key=lambda r: _finite_float(r.get("mean_partial_standardized_slope"), -math.inf), default=None)
        if best is not None:
            dominant.append(dict(best))
    write_csv(out_dir / "03_full_dominant_lags.csv", dominant)
    report = [
        "# Phase 2: Time-Lagged Mechanism Chain", "",
        f"- Completed core bundles: {len(bundles)}",
        f"- Birth-indexed grids evaluable: {len(grids)}",
        f"- Equal-birth bins per run: {campaign.birth_grid_bins}", "",
        "## Full-condition dominant lag by prespecified link", "",
    ]
    for row in dominant:
        report.append(
            f"- **{row['link']}**: lag={row['lag_bins']} bins "
            f"(median {row['median_lag_births']:.1f} births), "
            f"slope={row['mean_partial_standardized_slope']:.4f}, "
            f"bootstrap 95% CI [{row['bootstrap_ci_low']:.4f}, {row['bootstrap_ci_high']:.4f}], "
            f"q={row.get('q_bh', math.nan):.4g}, n={row['n_runs_estimable']}."
        )
    report.extend([
        "", "## Interpretation rule", "",
        "A positive lagged slope indicates that a current increment predicts a later increment after controlling the current outcome and birth-bin trend. The unit of inference is the run; windows are not treated as independent replicates. Lag selection is descriptive, while all link-by-lag q-values are reported as an exploratory family.",
    ])
    atomic_write_text(out_dir / "04_time_lagged_chain_report.md", "\n".join(report) + "\n")
    return {
        "bundles": len(bundles),
        "grids": len(grids),
        "summary_rows": len(rows),
        "dominant_lags": dominant,
    }


def _niche_lag_from_row(row: Mapping[str, Any], persistent: bool) -> float:
    genetic = _finite_float(row.get("genomic_first_birth"))
    if not math.isfinite(genetic):
        return math.nan
    try:
        details = json.loads(str(row.get("reactant_timing_json", "[]")))
    except Exception:
        return math.nan
    key = "persistent_birth" if persistent else "first_production_birth"
    births = [
        _finite_float(item.get(key)) for item in details
        if isinstance(item, dict) and math.isfinite(_finite_float(item.get(key)))
    ]
    if not births:
        return math.nan
    # All required reactants must be available; the latest required reactant is the
    # operative niche-availability time.
    return genetic - max(births)


def run_niche_origin_analysis(
    core_dir: Path,
    root_seed: int,
    out_dir: Path,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundles = load_result_bundles(core_dir)
    raw_rows: List[Dict[str, Any]] = []
    for data in bundles:
        summary = data["summary"]
        for row in data.get("niche_origin_rows", []):
            flat = flatten_event(summary, row)
            flat["production_to_genome_lag_births"] = _niche_lag_from_row(flat, persistent=False)
            flat["persistent_niche_to_genome_lag_births"] = _niche_lag_from_row(flat, persistent=True)
            # User-facing terminology: mutation-first is equivalent here to genomic
            # appearance preceding availability of all required constructed reactants.
            origin = str(flat.get("origin_class", "unresolved"))
            flat["innovation_order"] = (
                "mutation_first" if origin == "genotype_first" else origin
            )
            strict = str(flat.get("strict_origin_class", "unresolved"))
            flat["strict_innovation_order"] = (
                "mutation_first" if strict == "genotype_first" else strict
            )
            raw_rows.append(flat)
    write_csv(out_dir / "01_niche_origin_events.csv", raw_rows)

    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        subset = "causal_retained" if bool(row.get("causal_retained", False)) else "all_persistent"
        grouped[(str(row.get("condition")), "all_persistent")].append(row)
        if subset == "causal_retained":
            grouped[(str(row.get("condition")), "causal_retained")].append(row)

    summary_rows: List[Dict[str, Any]] = []
    classes = (
        "niche_first", "mutation_first", "coincident", "mixed_order",
        "external_supported", "unresolved",
    )
    for (condition, subset), rows in sorted(grouped.items()):
        total = len(rows)
        record: Dict[str, Any] = {
            "condition": condition,
            "subset": subset,
            "n_innovations": total,
        }
        for cls in classes:
            n = sum(str(r.get("innovation_order")) == cls for r in rows)
            record[f"n_{cls}"] = n
            record[f"fraction_{cls}"] = n / total if total else math.nan
        strict_n = sum(str(r.get("strict_innovation_order")) == "persistent_niche_first" for r in rows)
        record["n_persistent_niche_first"] = strict_n
        record["fraction_persistent_niche_first"] = strict_n / total if total else math.nan
        production_lags = [
            _finite_float(r.get("production_to_genome_lag_births")) for r in rows
            if str(r.get("innovation_order")) == "niche_first"
            and math.isfinite(_finite_float(r.get("production_to_genome_lag_births")))
        ]
        persistent_lags = [
            _finite_float(r.get("persistent_niche_to_genome_lag_births")) for r in rows
            if str(r.get("strict_innovation_order")) == "persistent_niche_first"
            and math.isfinite(_finite_float(r.get("persistent_niche_to_genome_lag_births")))
        ]
        record["median_niche_first_lag_births"] = float(np.median(production_lags)) if production_lags else math.nan
        record["median_persistent_niche_first_lag_births"] = float(np.median(persistent_lags)) if persistent_lags else math.nan
        summary_rows.append(record)
    write_csv(out_dir / "02_niche_origin_summary.csv", summary_rows)

    # Pair-level birth-normalized rates include zero-event runs and therefore avoid
    # conditioning the contrast on the presence of a causal innovation.
    pair_counts: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    analysis_births: Dict[Tuple[str, str], float] = {}
    for data in bundles:
        summary = data["summary"]
        key = (str(summary.get("pair_id")), str(summary.get("condition")))
        analysis_births[key] = max(_finite_float(summary.get("analysis_births"), 0.0), 0.0)
    for row in raw_rows:
        if not bool(row.get("causal_retained", False)):
            continue
        key = (str(row.get("pair_id")), str(row.get("condition")))
        pair_counts[key]["total"] += 1
        if str(row.get("innovation_order")) == "niche_first":
            pair_counts[key]["niche_first"] += 1
        if str(row.get("strict_innovation_order")) == "persistent_niche_first":
            pair_counts[key]["persistent_niche_first"] += 1
    contrast_rows: List[Dict[str, Any]] = []
    full_pairs = sorted({pair for pair, condition in analysis_births if condition == "full"})
    for comparator in (
        "products_erased", "construction_only", "extensibility_only",
        "closed_control", "no_mutation",
    ):
        for metric in ("niche_first", "persistent_niche_first"):
            diffs: List[float] = []
            for pair in full_pairs:
                a_births = analysis_births.get((pair, "full"), 0.0)
                b_births = analysis_births.get((pair, comparator), 0.0)
                if a_births <= 0.0 or b_births <= 0.0:
                    continue
                a_rate = 1000.0 * pair_counts[(pair, "full")].get(metric, 0) / a_births
                b_rate = 1000.0 * pair_counts[(pair, comparator)].get(metric, 0) / b_births
                diffs.append(a_rate - b_rate)
            mean, low, high = paired_mean_ci(diffs)
            contrast_rows.append({
                "metric": f"causal_{metric}_per_1000_births",
                "comparison": f"full_minus_{comparator}",
                "n_pairs": len(diffs),
                "mean_difference": mean,
                "ci_low": low,
                "ci_high": high,
                "sign_flip_p": exact_sign_flip_p(
                    diffs,
                    stable_seed(root_seed, "niche_contrast", comparator, metric),
                    alternative="greater",
                ),
            })
    _add_bh(contrast_rows)
    write_csv(out_dir / "03_niche_origin_condition_contrasts.csv", contrast_rows)
    report = [
        "# Phase 3: Innovation-Order Classification", "",
        f"- Completed core bundles: {len(bundles)}",
        f"- Persistent innovation records classified: {len(raw_rows)}", "",
        "## Condition summaries for causally retained innovations", "",
    ]
    for row in summary_rows:
        if row.get("subset") != "causal_retained":
            continue
        report.append(
            f"- **{row['condition']}**: n={row['n_innovations']}, "
            f"niche-first={row.get('fraction_niche_first', math.nan):.3f}, "
            f"persistent-niche-first={row.get('fraction_persistent_niche_first', math.nan):.3f}, "
            f"mutation-first={row.get('fraction_mutation_first', math.nan):.3f}, "
            f"median niche lead={row.get('median_niche_first_lag_births', math.nan):.1f} births."
        )
    report.extend(["", "## Paired condition contrasts", ""])
    for row in contrast_rows:
        report.append(
            f"- **{row['comparison']} — {row['metric']}**: "
            f"difference={row['mean_difference']:.4f}, "
            f"95% CI [{row['ci_low']:.4f}, {row['ci_high']:.4f}], "
            f"q={row.get('q_bh', math.nan):.4g}, n={row['n_pairs']}."
        )
    report.extend([
        "", "## Classification rule", "",
        "Niche-first means that all required non-primitive reactants were produced before the hereditary function first appeared. Persistent-niche-first applies the stricter requirement that every required constructed reactant had already crossed the persistence criterion. Mutation-first means hereditary appearance preceded full constructed-reactant availability.",
    ])
    atomic_write_text(out_dir / "04_niche_origin_report.md", "\n".join(report) + "\n")
    return {
        "bundles": len(bundles),
        "events": len(raw_rows),
        "summary_rows": len(summary_rows),
        "contrast_rows": len(contrast_rows),
        "causal_condition_summaries": [r for r in summary_rows if r.get("subset") == "causal_retained"],
    }


def _cluster_robust_univariate(
    x: Sequence[float], y: Sequence[float], clusters: Sequence[str], seed: int
) -> Dict[str, Any]:
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    valid = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[valid], yv[valid]
    cv = np.asarray(clusters, dtype=object)[valid]
    if xv.size < 8 or np.unique(xv).size < 3:
        return {"n": int(xv.size), "n_clusters": len(set(str(c) for c in cv.tolist())), "slope_per_predictor_sd": math.nan, "std_error_cluster": math.nan, "ci_low": math.nan, "ci_high": math.nan, "p_wild_cluster": math.nan, "predictor_mean": math.nan, "predictor_sd": math.nan}
    x_mean, x_sd = float(xv.mean()), float(xv.std(ddof=0))
    if x_sd <= 1e-12:
        return {"n": int(xv.size), "n_clusters": len(set(str(c) for c in cv.tolist())), "slope_per_predictor_sd": math.nan, "std_error_cluster": math.nan, "ci_low": math.nan, "ci_high": math.nan, "p_wild_cluster": math.nan, "predictor_mean": math.nan, "predictor_sd": math.nan}
    z = (xv - x_mean) / x_sd
    X = np.column_stack([np.ones(z.size), z])
    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    residual = yv - X @ beta
    bread = np.linalg.pinv(X.T @ X)
    meat = np.zeros((2, 2), dtype=float)
    unique_clusters = sorted(set(str(c) for c in cv.tolist()))
    for cluster in unique_clusters:
        idx = np.asarray([i for i, c in enumerate(cv.tolist()) if str(c) == cluster], dtype=int)
        score = X[idx].T @ residual[idx]
        meat += np.outer(score, score)
    g = len(unique_clusters)
    n = len(yv)
    correction = (g / max(g - 1, 1)) * ((n - 1) / max(n - X.shape[1], 1))
    covariance = bread @ meat @ bread * correction
    se = math.sqrt(max(float(covariance[1, 1]), 0.0))

    # Exact wild-cluster sign enumeration for <=16 families; otherwise Monte Carlo.
    restricted_beta = np.linalg.lstsq(np.ones((n, 1)), yv, rcond=None)[0]
    fitted0 = np.full(n, restricted_beta[0], dtype=float)
    resid0 = yv - fitted0
    observed = abs(float(beta[1]))
    exceed = 0
    total = 0
    if g <= 16:
        for mask in range(1 << g):
            signs = {cluster: (-1.0 if (mask >> j) & 1 else 1.0) for j, cluster in enumerate(unique_clusters)}
            y_star = fitted0 + np.asarray([signs[str(c)] for c in cv.tolist()]) * resid0
            b_star = np.linalg.lstsq(X, y_star, rcond=None)[0][1]
            exceed += int(abs(float(b_star)) >= observed - 1e-15)
            total += 1
    else:
        rng = np.random.default_rng(seed)
        total = 20_000
        exceed = 1
        for _ in range(total):
            draws = rng.choice((-1.0, 1.0), size=g)
            sign_map = dict(zip(unique_clusters, draws.tolist()))
            y_star = fitted0 + np.asarray([sign_map[str(c)] for c in cv.tolist()]) * resid0
            b_star = np.linalg.lstsq(X, y_star, rcond=None)[0][1]
            exceed += int(abs(float(b_star)) >= observed - 1e-15)
        total += 1
    return {
        "n": int(n),
        "n_clusters": g,
        "slope_per_predictor_sd": float(beta[1]),
        "std_error_cluster": se,
        "ci_low": float(beta[1] - 1.959963984540054 * se),
        "ci_high": float(beta[1] + 1.959963984540054 * se),
        "p_wild_cluster": exceed / max(total, 1),
        "predictor_mean": x_mean,
        "predictor_sd": x_sd,
    }


def _ridge_lofo(
    rows: Sequence[Mapping[str, Any]], predictors: Sequence[str], outcome: str, cluster_key: str
) -> Dict[str, Any]:
    clean = [r for r in rows if math.isfinite(_finite_float(r.get(outcome))) and all(math.isfinite(_finite_float(r.get(p))) for p in predictors)]
    clusters = sorted({str(r.get(cluster_key)) for r in clean})
    if len(clean) < len(predictors) + 5 or len(clusters) < 5:
        return {"n": len(clean), "n_clusters": len(clusters), "lofo_r2": math.nan, "lofo_correlation": math.nan, "sign_accuracy": math.nan, "coefficients": {}}
    X = np.asarray([[_finite_float(r.get(p)) for p in predictors] for r in clean], dtype=float)
    y = np.asarray([_finite_float(r.get(outcome)) for r in clean], dtype=float)
    cluster_values = np.asarray([str(r.get(cluster_key)) for r in clean], dtype=object)
    mean = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd <= 1e-12] = 1.0
    Xz = (X - mean) / sd
    predictions = np.full(y.size, math.nan)
    lam = 1.0
    for cluster in clusters:
        test = cluster_values == cluster
        train = ~test
        Xa = np.column_stack([np.ones(int(train.sum())), Xz[train]])
        penalty = np.eye(Xa.shape[1]) * lam
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(Xa.T @ Xa + penalty, Xa.T @ y[train])
        predictions[test] = np.column_stack([np.ones(int(test.sum())), Xz[test]]) @ beta
    sse = float(np.sum((y - predictions) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0.0 else math.nan
    corr = float(np.corrcoef(y, predictions)[0, 1]) if y.size > 2 and np.std(predictions) > 0 else math.nan
    sign_accuracy = float(np.mean((predictions > 0.0) == (y > 0.0)))
    Xa = np.column_stack([np.ones(y.size), Xz])
    penalty = np.eye(Xa.shape[1]) * lam
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(Xa.T @ Xa + penalty, Xa.T @ y)
    return {
        "n": len(clean), "n_clusters": len(clusters), "ridge_lambda": lam,
        "lofo_r2": r2, "lofo_correlation": corr, "sign_accuracy": sign_accuracy,
        "coefficients": {p: float(b) for p, b in zip(predictors, beta[1:].tolist())},
    }


def run_world_boundary_analysis(
    core_dir: Path,
    root_seed: int,
    out_dir: Path,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundles = load_result_bundles(core_dir)
    run_rows = [dict(data["summary"]) for data in bundles]
    manifest_rows: List[Dict[str, Any]] = []
    manifest_path = core_dir / "00_environment_manifest.csv"
    if manifest_path.exists():
        with open(manifest_path, newline="", encoding="utf-8") as f:
            manifest_rows = [dict(r) for r in csv.DictReader(f)]
    manifest = {str(r.get("world_id")): r for r in manifest_rows}
    index = {(str(r.get("pair_id")), str(r.get("condition"))): r for r in run_rows}
    pair_ids = sorted({str(r.get("pair_id")) for r in run_rows if str(r.get("condition")) == "full"})
    effect_rows: List[Dict[str, Any]] = []
    for comparator in (
        "construction_only", "extensibility_only", "closed_control",
        "products_erased", "no_mutation",
    ):
        for pair_id in pair_ids:
            full = index.get((pair_id, "full"))
            other = index.get((pair_id, comparator))
            if full is None or other is None:
                continue
            a = _finite_float(full.get(PRIMARY_METRIC))
            b = _finite_float(other.get(PRIMARY_METRIC))
            if not (math.isfinite(a) and math.isfinite(b)):
                continue
            world_id = str(full.get("world_id"))
            env = manifest.get(world_id, {})
            row = {
                "pair_id": pair_id,
                "world_id": world_id,
                "world_family": full.get("world_family"),
                "topology": full.get("topology"),
                "forcing": full.get("forcing"),
                "comparison": f"full_minus_{comparator}",
                "effect": a - b,
                "transport": _finite_float(env.get("transport")),
                "dissipation": _finite_float(env.get("dissipation")),
                "mean_degree": _finite_float(env.get("mean_degree")),
                "alphabet_size": _finite_float(full.get("alphabet_size")),
                "full_residence_time": _finite_float(full.get("empirical_constructed_residence_time")),
                "full_utilization_ratio": _finite_float(full.get("constructed_utilization_ratio")),
                "full_constructed_mass_auc_per_birth": (
                    _finite_float(full.get("constructed_mass_auc"), 0.0)
                    / max(_finite_float(full.get("analysis_births"), 0.0), 1.0)
                ),
            }
            if math.isfinite(row["transport"]) and row["transport"] > 0:
                row["log_transport"] = math.log(row["transport"])
            else:
                row["log_transport"] = math.nan
            if math.isfinite(row["dissipation"]) and row["dissipation"] > 0:
                row["log_dissipation"] = math.log(row["dissipation"])
            else:
                row["log_dissipation"] = math.nan
            row["log1p_full_residence_time"] = math.log1p(max(row["full_residence_time"], 0.0)) if math.isfinite(row["full_residence_time"]) else math.nan
            effect_rows.append(row)
    write_csv(out_dir / "01_pair_level_world_effects.csv", effect_rows)

    predictors = (
        "log_transport", "log_dissipation", "mean_degree", "alphabet_size",
        "log1p_full_residence_time", "full_utilization_ratio",
        "full_constructed_mass_auc_per_birth",
    )
    regression_rows: List[Dict[str, Any]] = []
    for comparison in sorted({str(r["comparison"]) for r in effect_rows}):
        subset = [r for r in effect_rows if str(r["comparison"]) == comparison]
        for predictor in predictors:
            result = _cluster_robust_univariate(
                [_finite_float(r.get(predictor)) for r in subset],
                [_finite_float(r.get("effect")) for r in subset],
                [str(r.get("world_family")) for r in subset],
                stable_seed(root_seed, "world_boundary", comparison, predictor),
            )
            regression_rows.append({
                "comparison": comparison,
                "predictor": predictor,
                **result,
                "sign_flip_p": result.get("p_wild_cluster", math.nan),
            })
    _add_bh(regression_rows)
    for row in regression_rows:
        q = _finite_float(row.get("q_bh"))
        low = _finite_float(row.get("ci_low"))
        high = _finite_float(row.get("ci_high"))
        row["significant_bh"] = bool(
            math.isfinite(q) and q <= 0.05
            and math.isfinite(low) and math.isfinite(high)
            and (low > 0.0 or high < 0.0)
        )
    write_csv(out_dir / "02_boundary_univariate_cluster_tests.csv", regression_rows)

    ridge_predictors = (
        "log_transport", "log_dissipation", "mean_degree", "alphabet_size",
        "log1p_full_residence_time", "full_utilization_ratio",
    )
    ridge_rows: List[Dict[str, Any]] = []
    for comparison in sorted({str(r["comparison"]) for r in effect_rows}):
        subset = [r for r in effect_rows if str(r["comparison"]) == comparison]
        result = _ridge_lofo(subset, ridge_predictors, "effect", "world_family")
        base = {k: v for k, v in result.items() if k != "coefficients"}
        for predictor, coefficient in result.get("coefficients", {}).items():
            ridge_rows.append({
                "comparison": comparison,
                **base,
                "predictor": predictor,
                "standardized_ridge_coefficient": coefficient,
            })
    write_csv(out_dir / "03_boundary_multivariable_lofo_ridge.csv", ridge_rows)

    family_rows = world_family_generality(run_rows, root_seed)
    write_csv(out_dir / "04_world_family_effects.csv", family_rows)
    significant_predictors = [
        r for r in regression_rows
        if bool(r.get("significant_bh", False))
    ]
    report = [
        "# Phase 4: World Dependence and Physical Boundary Conditions", "",
        f"- Completed core bundles: {len(bundles)}",
        f"- Pair-level effects: {len(effect_rows)}",
        f"- World-family contrasts: {len(family_rows)}", "",
        "## Directional generality", "",
    ]
    for comparison in sorted({str(r.get('comparison')) for r in family_rows}):
        subset = [r for r in family_rows if str(r.get('comparison')) == comparison]
        positive = sum(bool(r.get('direction_positive', False)) for r in subset)
        report.append(f"- **{comparison}**: positive in {positive}/{len(subset)} world families.")
    report.extend(["", "## Boundary predictors surviving exploratory BH correction", ""])
    if significant_predictors:
        for row in significant_predictors:
            report.append(
                f"- **{row['comparison']} — {row['predictor']}**: "
                f"slope per predictor SD={row.get('slope_per_predictor_sd', math.nan):.4f}, "
                f"cluster 95% CI [{row.get('ci_low', math.nan):.4f}, {row.get('ci_high', math.nan):.4f}], "
                f"wild-cluster q={row.get('q_bh', math.nan):.4g}."
            )
    else:
        report.append("- None.")
    report.extend([
        "", "## Model boundary", "",
        "Univariate tests use world-family-clustered covariance and wild-cluster sign randomization. The multivariable ridge analysis is evaluated by leave-one-world-family-out prediction and is exploratory; it is not used as a confirmatory significance test.",
    ])
    atomic_write_text(out_dir / "05_world_boundary_report.md", "\n".join(report) + "\n")
    return {
        "bundles": len(bundles),
        "effect_rows": len(effect_rows),
        "univariate_rows": len(regression_rows),
        "ridge_rows": len(ridge_rows),
        "family_rows": len(family_rows),
        "significant_predictors": significant_predictors,
    }


def analyze_retention_sweep(
    core_dir: Path,
    retention_dir: Path,
    root_seed: int,
    out_dir: Path,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    core = load_result_bundles(core_dir)
    intermediate = load_result_bundles(retention_dir)
    selected: List[Dict[str, Any]] = []
    for data in core:
        if str(data["summary"].get("condition")) in {"products_erased", "full"}:
            selected.append(data)
    selected.extend(intermediate)
    run_rows = [dict(data["summary"]) for data in selected]
    nominal_map = {"products_erased": 0.0, "full": math.inf}
    nominal_map.update({f"retention_hl_{int(h)}": float(h) for h in RETENTION_HALF_LIVES})
    for row in run_rows:
        row["nominal_constructed_half_life"] = nominal_map.get(str(row.get("condition")), math.nan)
    write_csv(out_dir / "01_retention_run_summary.csv", run_rows)
    summary = condition_summary(run_rows)
    write_csv(out_dir / "02_retention_condition_summary.csv", summary)

    index = {(str(r.get("pair_id")), str(r.get("condition"))): r for r in run_rows}
    pairs = sorted({str(r.get("pair_id")) for r in run_rows if str(r.get("condition")) == "products_erased"})
    contrast_rows: List[Dict[str, Any]] = []
    for condition in (*RETENTION_INTERMEDIATE_CONDITIONS, "full"):
        differences: List[float] = []
        for pair in pairs:
            treated = index.get((pair, condition))
            zero = index.get((pair, "products_erased"))
            if treated is None or zero is None:
                continue
            a, b = _finite_float(treated.get(PRIMARY_METRIC)), _finite_float(zero.get(PRIMARY_METRIC))
            if math.isfinite(a) and math.isfinite(b):
                differences.append(a - b)
        mean, low, high = paired_mean_ci(differences)
        contrast_rows.append({
            "condition": condition,
            "nominal_half_life": nominal_map[condition],
            "comparison": f"{condition}_minus_products_erased",
            "n_pairs": len(differences),
            "mean_difference": mean,
            "ci_low": low,
            "ci_high": high,
            "positive_pair_fraction": sum(v > 0 for v in differences) / len(differences) if differences else math.nan,
            "sign_flip_p": exact_sign_flip_p(
                differences,
                stable_seed(root_seed, "retention_contrast", condition),
                alternative="greater",
            ),
        })
    _add_bh(contrast_rows)
    write_csv(out_dir / "03_retention_vs_erased_contrasts.csv", contrast_rows)

    threshold = next((
        row for row in sorted(
            [r for r in contrast_rows if r["condition"] != "full"],
            key=lambda r: float(r["nominal_half_life"]),
        )
        if bool(row.get("significant_bh", False))
        and _finite_float(row.get("ci_low"), -math.inf) > 0.0
    ), None)

    # Empirical residence-time response is preferred to assigning an arbitrary finite
    # numeric value to the no-extra-decay Full endpoint.
    x = [_finite_float(r.get("empirical_constructed_residence_time")) for r in run_rows]
    y = [_finite_float(r.get(PRIMARY_METRIC)) for r in run_rows]
    clusters = [str(r.get("world_family")) for r in run_rows]
    residence_model = _cluster_robust_univariate(
        [math.log1p(max(v, 0.0)) if math.isfinite(v) else math.nan for v in x],
        y,
        clusters,
        stable_seed(root_seed, "retention_residence_response"),
    )
    model_row = {
        "model": "primary_outcome_vs_log1p_empirical_residence_time",
        **residence_model,
        "lowest_significant_nominal_half_life": (
            threshold["nominal_half_life"] if threshold is not None else math.nan
        ),
    }
    write_csv(out_dir / "04_retention_threshold_and_residence_model.csv", [model_row])
    report = [
        "# Phase 5: Continuous Constructed-State Persistence Intervention", "",
        f"- Combined endpoint and intermediate runs: {len(run_rows)}",
        f"- Lowest intermediate nominal half-life significantly above immediate erasure: {model_row['lowest_significant_nominal_half_life']}",
        f"- Residence-response slope per log-residence SD: {residence_model.get('slope_per_predictor_sd', math.nan):.4f}",
        f"- Cluster 95% CI: [{residence_model.get('ci_low', math.nan):.4f}, {residence_model.get('ci_high', math.nan):.4f}]",
        f"- Wild-cluster p: {residence_model.get('p_wild_cluster', math.nan):.4g}", "",
        "## Half-life contrasts against immediate erasure", "",
    ]
    for row in contrast_rows:
        report.append(
            f"- **{row['condition']}**: difference={row['mean_difference']:.4f}, "
            f"95% CI [{row['ci_low']:.4f}, {row['ci_high']:.4f}], "
            f"q={row.get('q_bh', math.nan):.4g}, n={row['n_pairs']}."
        )
    report.extend([
        "", "## Interpretation rule", "",
        "The threshold is the lowest prespecified intermediate half-life whose paired improvement over immediate erasure has a positive confidence interval and survives BH correction. Empirical residence time is used for the continuous response model so that the Full endpoint does not require an arbitrary finite numerical substitute for infinity.",
    ])
    atomic_write_text(out_dir / "05_retention_intervention_report.md", "\n".join(report) + "\n")
    return {
        "runs": len(run_rows),
        "contrasts": len(contrast_rows),
        "lowest_significant_nominal_half_life": model_row["lowest_significant_nominal_half_life"],
        "residence_model": residence_model,
    }


def write_phase_status(root: Path, phase: int, status: Mapping[str, Any]) -> None:
    path = root / f"phase{phase}_status.json"
    atomic_write_text(path, json.dumps(json_safe(dict(status)), indent=2, ensure_ascii=False))


def write_integrated_mechanism_report(
    root: Path,
    phase1: Mapping[str, Any],
    phase2: Mapping[str, Any],
    phase3: Mapping[str, Any],
    phase4: Mapping[str, Any],
    phase5: Optional[Mapping[str, Any]] = None,
) -> None:
    verdict = phase1.get("verdict") or {}
    lines = [
        "# Sequential Open-Ended Evolution Mechanism Validation", "",
        "## Phase 1: Complete prespecified core campaign", "",
        f"- Complete campaign: **{phase1.get('complete_campaign')}**",
        f"- Completed runs: {phase1.get('completed')}/{phase1.get('expected')}",
        f"- Failure records: {phase1.get('failures')}",
        f"- Prespecified proposition verdict: **{verdict.get('verdict', 'UNKNOWN')}**", "",
        "## Phase 2: Time-lagged mechanism chain", "",
    ]
    dominant = phase2.get("dominant_lags", []) or []
    if dominant:
        for row in dominant:
            lines.append(
                f"- **{row.get('link')}**: dominant lag {row.get('lag_bins')} bins "
                f"(~{_finite_float(row.get('median_lag_births')):.1f} births), "
                f"slope={_finite_float(row.get('mean_partial_standardized_slope')):.4f}, "
                f"95% CI [{_finite_float(row.get('bootstrap_ci_low')):.4f}, "
                f"{_finite_float(row.get('bootstrap_ci_high')):.4f}], "
                f"q={_finite_float(row.get('q_bh')):.4g}."
            )
    else:
        lines.append("- No lag profile was estimable.")
    lines.extend(["", "## Phase 3: Innovation order", ""])
    causal_summaries = phase3.get("causal_condition_summaries", []) or []
    if causal_summaries:
        for row in causal_summaries:
            lines.append(
                f"- **{row.get('condition')}**: causal innovations={row.get('n_innovations')}, "
                f"niche-first={_finite_float(row.get('fraction_niche_first')):.3f}, "
                f"persistent-niche-first={_finite_float(row.get('fraction_persistent_niche_first')):.3f}, "
                f"mutation-first={_finite_float(row.get('fraction_mutation_first')):.3f}."
            )
    else:
        lines.append("- No causally retained innovation-order records were available.")
    lines.extend(["", "## Phase 4: World dependence and physical boundaries", ""])
    significant = phase4.get("significant_predictors", []) or []
    if significant:
        for row in significant:
            lines.append(
                f"- **{row.get('comparison')} — {row.get('predictor')}**: "
                f"slope={_finite_float(row.get('slope_per_predictor_sd')):.4f}, "
                f"q={_finite_float(row.get('q_bh')):.4g}."
            )
    else:
        lines.append("- No physical boundary predictor survived exploratory BH correction.")
    if phase5 is not None:
        analysis = phase5.get("analysis", {}) if isinstance(phase5, Mapping) else {}
        residence = analysis.get("residence_model", {}) if isinstance(analysis, Mapping) else {}
        lines.extend([
            "", "## Phase 5: Constructed-state persistence intervention", "",
            f"- Complete intervention campaign: **{phase5.get('campaign', {}).get('complete_campaign')}**",
            f"- Lowest significant intermediate half-life: {analysis.get('lowest_significant_nominal_half_life')}",
            f"- Empirical residence-response slope: {_finite_float(residence.get('slope_per_predictor_sd')):.4f}",
            f"- Wild-cluster p: {_finite_float(residence.get('p_wild_cluster')):.4g}",
        ])
    lines.extend([
        "", "## Interpretation guardrail", "",
        "Phase 1 retains the original six-test confirmatory FDR family. Phases 2–4 are prespecified mechanistic analyses performed only after core completion. Phase 5 is a separate intervention family and does not retroactively alter the Phase 1 confirmatory verdict.",
    ])
    atomic_write_text(root / "07_integrated_mechanism_report.md", "\n".join(lines) + "\n")


def aggregate_results(
    out_dir: Path,
    campaign: CampaignConfig,
    cfg: ModelConfig,
    root_seed: int,
    expected_tasks: int,
) -> Dict[str, Any]:
    result_files = sorted((out_dir / "runs").glob("*.json.gz"))
    run_rows: List[Dict[str, Any]] = []
    window_rows: List[Dict[str, Any]] = []
    function_rows: List[Dict[str, Any]] = []
    substrate_rows: List[Dict[str, Any]] = []
    interaction_rows: List[Dict[str, Any]] = []
    causal_rows: List[Dict[str, Any]] = []
    genome_function_rows: List[Dict[str, Any]] = []
    substrate_origin_rows: List[Dict[str, Any]] = []
    niche_origin_rows: List[Dict[str, Any]] = []
    lineage_rows: List[Dict[str, Any]] = []
    qc_rows: List[Dict[str, Any]] = []
    for path in result_files:
        try:
            data = read_json_gz(path)
        except Exception:
            continue
        summary = dict(data.get("summary", {}))
        run_rows.append(summary)
        window_rows.extend(dict(r) for r in data.get("window_rows", []))
        function_rows.extend(flatten_event(summary, r) for r in data.get("function_events", []))
        substrate_rows.extend(flatten_event(summary, r) for r in data.get("substrate_events", []))
        interaction_rows.extend(flatten_event(summary, r) for r in data.get("interaction_events", []))
        causal_rows.extend(flatten_event(summary, r) for r in data.get("causal_rows", []))
        genome_function_rows.extend(flatten_event(summary, r) for r in data.get("genome_function_events", []))
        substrate_origin_rows.extend(flatten_event(summary, r) for r in data.get("substrate_origin_events", []))
        niche_origin_rows.extend(flatten_event(summary, r) for r in data.get("niche_origin_rows", []))
        lineage_rows.extend(flatten_event(summary, r) for r in data.get("lineage_events", []))
        qc = dict(data.get("qc", {}))
        qc.update({
            "run_id": summary.get("run_id"), "world_id": summary.get("world_id"),
            "world_family": summary.get("world_family"), "condition": summary.get("condition"),
            "evolutionary_seed": summary.get("evolutionary_seed"),
        })
        qc_rows.append(qc)

    model_rows = [{
        "run_id": r.get("run_id"), "world_id": r.get("world_id"),
        "condition": r.get("condition"), "model_comparison_valid": r.get("model_comparison_valid"),
        "linear_rmse": r.get("linear_rmse"), "power_rmse": r.get("power_rmse"),
        "saturation_rmse": r.get("saturation_rmse"), "linear_loglik": r.get("linear_loglik"),
        "power_loglik": r.get("power_loglik"), "saturation_loglik": r.get("saturation_loglik"),
        "best_model": r.get("best_model"),
        "saturation_outperforms_both": r.get("saturation_outperforms_both"),
    } for r in run_rows]

    conditions = condition_summary(run_rows)
    paired = paired_contrasts(run_rows, root_seed)
    factorial = factorial_effects(run_rows, root_seed)
    assign_confirmatory_fdr(factorial, paired)
    family_rows = world_family_generality(run_rows, root_seed)
    gee = run_gee(run_rows, out_dir)
    failures = len(list((out_dir / "failures").glob("*.json")))
    verdict = proposition_verdict(
        run_rows, factorial, paired, family_rows, qc_rows,
        expected_tasks, failures, campaign,
    )

    write_csv(out_dir / "02_run_summary.csv", run_rows)
    write_csv(out_dir / "03_window_metrics.csv", window_rows)
    write_csv(out_dir / "04_function_events.csv", function_rows)
    write_csv(out_dir / "05_substrate_events.csv", substrate_rows)
    write_csv(out_dir / "06_interaction_events.csv", interaction_rows)
    write_csv(out_dir / "07_causal_assays.csv", causal_rows)
    write_csv(out_dir / "08_oee_model_fits.csv", model_rows)
    write_csv(out_dir / "09_condition_summary.csv", conditions)
    write_csv(out_dir / "10_factorial_effects.csv", factorial)
    write_csv(out_dir / "11_paired_contrasts.csv", paired)
    write_csv(out_dir / "12_world_family_generality.csv", family_rows)
    write_csv(out_dir / "13_gee_effects.csv", gee)
    write_csv(out_dir / "14_quality_control.csv", qc_rows)
    write_csv(out_dir / "15_lineage_events.csv", lineage_rows)
    write_csv(out_dir / "16_proposition_verdict.csv", [verdict])
    write_csv(out_dir / "18_genome_function_origin_events.csv", genome_function_rows)
    write_csv(out_dir / "19_substrate_origin_events.csv", substrate_origin_rows)
    write_csv(out_dir / "20_niche_origin_classification.csv", niche_origin_rows)

    completed = len(run_rows)
    report = [
        "# Hereditary–Environmental Open-Endedness Validation Report", "",
        "## Prespecified proposition", "", verdict["proposition"], "",
        f"**Verdict: {verdict['verdict']}**", "",
        "## Design", "",
        "- Primary factorial: persistent environmental construction (C) × hereditary possibility-space extensibility (G).",
        "- Negative controls: products erased after every ecological update; no mutation.",
        "- Primary outcome: late causally retained novelty per 1,000 births.",
        "- Sustained novelty is evaluated across equal cumulative-birth thirds, not equal time thirds.",
        "- Causal assays use one-sided paired sign-flip tests and within-run Benjamini–Hochberg correction.",
        "- Six proposition-defining tests form one prespecified confirmatory FDR family; all remaining q-values are exploratory.",
        "- G=0 fixes module number and reactant-string lengths; the counted structural transformation repertoire is finite and checked during every run.", "",
        "## Integrity", "",
        f"- Completed run files: {completed}/{expected_tasks}",
        f"- Failure records: {failures}",
        f"- Maximum relative material-balance residual: {max((float(r.get('max_relative_mass_balance_residual', 0.0)) for r in qc_rows), default=math.nan):.3e}",
        f"- Closed-space violations: {sum(int(r.get('closed_space_violations', 0) or 0) for r in qc_rows)}",
        f"- Analysis rows with adequate birth and mutation exposure: {verdict['analysis_rows_evaluable_fraction']:.3f}", "",
        "## Proposition gates", "",
    ]
    for key, value in verdict.items():
        if key.startswith("gate_"):
            report.append(f"- {key[5:].replace('_', ' ')}: **{value}**")
    report.extend(["", "## Condition summaries", ""])
    for row in conditions:
        report.append(
            f"- **{row['condition']}**: n={row['n_runs']}, "
            f"primary={row.get(PRIMARY_METRIC + '_mean', math.nan):.6g} per 1,000 births, "
            f"strict OEE prevalence={row.get('oee_operational_mean', math.nan):.3f}, "
            f"analysis births={row.get('analysis_births_mean', math.nan):.1f}, "
            f"extinction={row.get('extinct_mean', math.nan):.3f}."
        )
    report.extend([
        "", "## Interpretation boundary", "",
        "SUPPORTED refers only to the prespecified sampled world families and the operational definitions above. "
        "It is not a proof over all mathematically possible worlds. NOT_SUPPORTED is interpretable only when the "
        "integrity, mutation-supply, birth-epoch, causal-event, and trajectory-evaluability checks are satisfied.",
    ])
    atomic_write_text(out_dir / "17_report.md", "\n".join(report) + "\n")
    return {
        "completed": completed, "expected": expected_tasks, "failures": failures,
        "verdict": verdict, "out_dir": str(out_dir),
    }


# -----------------------------------------------------------------------------
# Manifest and command-line entry point
# -----------------------------------------------------------------------------

def write_environment_manifest(
    out_dir: Path,
    campaign: CampaignConfig,
    cfg: ModelConfig,
    root_seed: int,
) -> None:
    rows: List[Dict[str, Any]] = []
    for key in build_world_keys(campaign, root_seed):
        world = generate_world(key, campaign.n_sites, cfg, root_seed)
        total_a = sum(float(x.sum()) for x in world.source_strength_a.values())
        total_b = sum(float(x.sum()) for x in world.source_strength_b.values())
        rows.append({
            "world_id": key.world_id,
            "world_family": key.family,
            "topology": key.topology,
            "forcing": key.forcing,
            "replicate": key.replicate,
            "alphabet_size": key.alphabet_size,
            "world_seed": world.seed,
            "n_sites": world.n_sites,
            "n_edges": int(world.edges_u.size),
            "mean_degree": 2.0 * world.edges_u.size / world.n_sites,
            "primitive_substrates": len(world.primitive_strings),
            "normalized_source_rate_a": total_a,
            "normalized_source_rate_b": total_b,
            "effective_source_rate_a": total_a * cfg.turnover_scale,
            "effective_source_rate_b": total_b * cfg.turnover_scale,
            "transport": world.transport,
            "dissipation": world.dissipation,
            "founder_function": world.founder_module.function_key(),
            "founder_delta_phi": reaction_delta_phi(world, world.founder_module),
        })
    write_csv(out_dir / "00_environment_manifest.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--workers", default="auto", help="integer or auto")
    parser.add_argument("--out", default=None, help="root output directory")
    parser.add_argument("--root-seed", type=int, default=ROOT_SEED_DEFAULT)
    parser.add_argument(
        "--through", type=int, choices=(1, 2, 3, 4, 5), default=4,
        help="run sequentially through this phase; phase 5 adds the retention intervention",
    )
    parser.add_argument("--overwrite", action="store_true", help="delete the complete v6 output root before running")
    parser.add_argument(
        "--analysis-only", action="store_true",
        help="do not launch simulations; rebuild requested analyses from existing complete run files",
    )
    parser.add_argument("--self-test", action="store_true", help="run a small end-to-end test")
    return parser.parse_args()


def prepare_output_directory(path: Path, overwrite: bool) -> None:
    if overwrite and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def prepare_campaign_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for subdir in ("runs", "checkpoints", "failures", "logs"):
        (path / subdir).mkdir(exist_ok=True)


def load_or_run_preflight(
    out_dir: Path,
    campaign: CampaignConfig,
    cfg: ModelConfig,
    root_seed: int,
) -> Tuple[float, Dict[str, Any]]:
    selection_path = out_dir / "01_preflight_selection.json"
    expected_signature = preflight_signature(campaign, cfg, root_seed)
    if selection_path.exists():
        with open(selection_path, "r", encoding="utf-8") as f:
            selection = json.load(f)
        if selection.get("preflight_signature") != expected_signature:
            raise SystemExit(
                "Existing preflight does not match this script/configuration. "
                "Use --overwrite to start a clean v6 campaign."
            )
        selected = float(selection["selected_turnover_scale"])
        print(f"[preflight] reusing frozen turnover_scale={selected:g}", flush=True)
        return selected, selection
    selected, _, selection = run_preflight(out_dir, campaign, cfg, root_seed)
    return selected, selection


def write_campaign_manifest(
    out_dir: Path,
    campaign: CampaignConfig,
    cfg: ModelConfig,
    root_seed: int,
    preflight_summary: Mapping[str, Any],
    phase_name: str,
) -> None:
    manifest = {
        "script_version": SCRIPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "phase_name": phase_name,
        "campaign": json_safe(asdict(campaign)),
        "model_config": json_safe(asdict(cfg)),
        "root_seed": root_seed,
        "preflight": json_safe(dict(preflight_summary)),
        "equal_horizon_all_conditions": True,
        "command": sys.argv,
    }
    path = out_dir / "campaign_manifest.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            previous = json.load(f)
        previous_core = {
            k: previous.get(k) for k in (
                "script_version", "schema_version", "phase_name",
                "campaign", "model_config", "root_seed",
            )
        }
        current_core = {
            k: manifest.get(k) for k in (
                "script_version", "schema_version", "phase_name",
                "campaign", "model_config", "root_seed",
            )
        }
        if previous_core != current_core:
            raise SystemExit(
                f"Existing manifest in {out_dir} does not match this configuration. "
                "Use --overwrite to start clean."
            )
    atomic_write_text(path, json.dumps(json_safe(manifest), indent=2, ensure_ascii=False))
    write_environment_manifest(out_dir, campaign, cfg, root_seed)


def run_task_blocks(
    tasks: Sequence[Task],
    campaign: CampaignConfig,
    cfg: ModelConfig,
    out_dir: Path,
    root_seed: int,
    workers_requested: str,
    phase_label: str,
) -> Dict[str, int]:
    blocks = group_task_blocks(tasks)
    workers = auto_workers(workers_requested, len(blocks))
    payloads: List[Dict[str, Any]] = []
    for block_index, block in enumerate(blocks):
        payloads.append({
            "block_id": f"{phase_label}_{block_index:04d}",
            "tasks": [{
                "world_key": asdict(task.world_key),
                "condition_name": task.condition_name,
                "seed_index": task.seed_index,
                "evolutionary_seed": task.evolutionary_seed,
                "total_steps": task.total_steps,
                "run_id": task.run_id,
            } for task in block],
            "campaign": asdict(campaign),
            "model_config": asdict(cfg),
            "out_dir": str(out_dir),
            "root_seed": root_seed,
            "resume": True,
            "overwrite": False,
        })
    print(
        f"[{phase_label}] runs={len(tasks)} blocks={len(blocks)} workers={workers} "
        f"steps/run={campaign.steps}",
        flush=True,
    )
    status_counts: Dict[str, int] = defaultdict(int)
    log_path = out_dir / "logs" / f"{phase_label}.jsonl"
    start = time.time()
    methods = mp.get_all_start_methods()
    start_method = "fork" if "fork" in methods else "spawn"
    context = mp.get_context(start_method)
    with context.Pool(processes=workers, maxtasksperchild=2) as pool:
        for done, block_record in enumerate(
            pool.imap_unordered(execute_block, payloads, chunksize=1), start=1
        ):
            for record in block_record.get("records", []):
                status = str(record.get("status", "unknown"))
                status_counts[status] += 1
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(json_safe(record), ensure_ascii=False) + "\n")
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0.0 else 0.0
            eta = (len(blocks) - done) / rate if rate > 0.0 else math.nan
            print(
                f"[{phase_label} {done}/{len(blocks)} blocks] "
                f"elapsed={elapsed / 60.0:.1f} min eta={eta / 60.0:.1f} min "
                f"statuses={dict(status_counts)}",
                flush=True,
            )
    return dict(status_counts)


def aggregate_retention_campaign(
    out_dir: Path, expected_tasks: int
) -> Dict[str, Any]:
    bundles = load_result_bundles(out_dir)
    run_rows = [dict(data["summary"]) for data in bundles]
    qc_rows: List[Dict[str, Any]] = []
    for data in bundles:
        summary = data["summary"]
        qc = dict(data.get("qc", {}))
        qc.update({
            "run_id": summary.get("run_id"),
            "world_id": summary.get("world_id"),
            "world_family": summary.get("world_family"),
            "condition": summary.get("condition"),
            "evolutionary_seed": summary.get("evolutionary_seed"),
        })
        qc_rows.append(qc)
    failures = len(list((out_dir / "failures").glob("*.json")))
    write_csv(out_dir / "02_run_summary.csv", run_rows)
    write_csv(out_dir / "03_condition_summary.csv", condition_summary(run_rows))
    write_csv(out_dir / "04_quality_control.csv", qc_rows)
    status = {
        "completed": len(run_rows),
        "expected": expected_tasks,
        "failures": failures,
        "complete_campaign": len(run_rows) == expected_tasks and failures == 0,
        "max_relative_mass_balance_residual": max(
            (_finite_float(r.get("max_relative_mass_balance_residual"), 0.0) for r in qc_rows),
            default=math.nan,
        ),
    }
    atomic_write_text(out_dir / "05_campaign_status.json", json.dumps(json_safe(status), indent=2))
    return status


def configure_self_test(campaign: CampaignConfig, cfg: ModelConfig) -> None:
    campaign.n_sites = 32
    campaign.world_replicates = 1
    campaign.evolutionary_seeds = 1
    campaign.topologies = ("lattice",)
    campaign.forcings = ("constant",)
    campaign.steps = 1_500
    campaign.establishment_steps = 150
    campaign.window_size = 75
    campaign.checkpoint_interval = 750
    campaign.causal_repeats = 3
    campaign.causal_horizon = 75
    campaign.preflight_steps = 300
    campaign.preflight_turnover_candidates = (64.0, 128.0)
    campaign.preflight_min_births_each_world = 0
    campaign.preflight_median_births = 0
    campaign.preflight_min_expected_functional_mutants_main = 0.0
    campaign.min_analysis_births_for_epochs = 0
    campaign.birth_grid_bins = 6
    campaign.mechanism_bootstrap_draws = 100
    campaign.boundary_bootstrap_draws = 100
    campaign.retention_bootstrap_draws = 100
    cfg.persistence_windows = 2
    cfg.min_organisms = 2
    cfg.min_generations = 1
    cfg.progress_interval = 750
    cfg.quality_check_interval = 25
    cfg.safety_max_population = 512


def main() -> int:
    args = parse_args()
    root = (
        Path(args.out).expanduser()
        if args.out else Path.home() / "Desktop" / (
            "universal_oee_sequential_validation_v6_selftest"
            if args.self_test else "universal_oee_sequential_validation_v6"
        )
    )
    prepare_output_directory(root, args.overwrite)

    core_dir = root / "01_core_432"
    lag_dir = root / "02_time_lagged_chain"
    niche_dir = root / "03_niche_first"
    boundary_dir = root / "04_world_boundaries"
    retention_run_dir = root / "05_retention_intervention_runs"
    retention_analysis_dir = root / "06_retention_intervention_analysis"
    prepare_campaign_directory(core_dir)

    campaign = CampaignConfig()
    campaign.conditions = CORE_CONDITIONS
    cfg = ModelConfig()
    if args.self_test:
        configure_self_test(campaign, cfg)

    selected_scale, preflight_summary = load_or_run_preflight(
        core_dir, campaign, cfg, args.root_seed
    )
    cfg.turnover_scale = selected_scale
    cfg.functional_mutation_probability_extensible_lower = float(
        preflight_summary["functional_mutation_probability_extensible"]["wilson_lower_95"]
    )
    cfg.functional_mutation_probability_closed_lower = float(
        preflight_summary["functional_mutation_probability_closed"]["wilson_lower_95"]
    )
    write_campaign_manifest(
        core_dir, campaign, cfg, args.root_seed, preflight_summary, "phase1_core"
    )
    core_tasks = build_tasks(campaign, args.root_seed, cfg)
    expected_core = len(build_world_keys(campaign, args.root_seed)) * campaign.evolutionary_seeds * len(CORE_CONDITIONS)
    if not args.self_test and expected_core != EXPECTED_CORE_RUNS_DEFAULT:
        raise RuntimeError(f"Core design must contain exactly 432 runs, got {expected_core}")

    if not args.analysis_only:
        phase1_statuses = run_task_blocks(
            core_tasks, campaign, cfg, core_dir, args.root_seed, args.workers, "phase1_core"
        )
    else:
        phase1_statuses = {"analysis_only": 1}
    core_aggregate = aggregate_results(
        core_dir, campaign, cfg, args.root_seed, expected_core
    )
    phase1_status = {
        "statuses": phase1_statuses,
        "completed": core_aggregate["completed"],
        "expected": expected_core,
        "failures": core_aggregate["failures"],
        "complete_campaign": (
            core_aggregate["completed"] == expected_core
            and core_aggregate["failures"] == 0
        ),
        "verdict": core_aggregate.get("verdict"),
    }
    write_phase_status(root, 1, phase1_status)
    if not phase1_status["complete_campaign"]:
        print(json.dumps(json_safe(phase1_status), indent=2, ensure_ascii=False))
        raise SystemExit(
            "Phase 1 is incomplete. Phases 2-5 are intentionally blocked until the "
            "prespecified core campaign is complete. Re-run the same command to resume."
        )
    if args.through == 1:
        print(json.dumps(json_safe(phase1_status), indent=2, ensure_ascii=False))
        return 0

    phase2 = run_time_lagged_chain_analysis(
        core_dir, campaign, args.root_seed, lag_dir
    )
    write_phase_status(root, 2, phase2)
    if args.through == 2:
        print(json.dumps(json_safe(phase2), indent=2, ensure_ascii=False))
        return 0

    phase3 = run_niche_origin_analysis(core_dir, args.root_seed, niche_dir)
    write_phase_status(root, 3, phase3)
    if args.through == 3:
        print(json.dumps(json_safe(phase3), indent=2, ensure_ascii=False))
        return 0

    phase4 = run_world_boundary_analysis(core_dir, args.root_seed, boundary_dir)
    write_phase_status(root, 4, phase4)
    if args.through == 4:
        final = {"phase1": phase1_status, "phase2": phase2, "phase3": phase3, "phase4": phase4}
        write_integrated_mechanism_report(root, phase1_status, phase2, phase3, phase4)
        print(json.dumps(json_safe(final), indent=2, ensure_ascii=False))
        return 0

    # Phase 5 is intentionally separated from the six-condition confirmatory family.
    retention_campaign = copy.deepcopy(campaign)
    retention_campaign.conditions = RETENTION_INTERMEDIATE_CONDITIONS
    prepare_campaign_directory(retention_run_dir)
    write_campaign_manifest(
        retention_run_dir, retention_campaign, cfg, args.root_seed,
        preflight_summary, "phase5_retention_intervention",
    )
    retention_tasks = build_tasks(retention_campaign, args.root_seed, cfg)
    expected_retention = (
        len(build_world_keys(retention_campaign, args.root_seed))
        * retention_campaign.evolutionary_seeds
        * len(RETENTION_INTERMEDIATE_CONDITIONS)
    )
    if not args.self_test and expected_retention != EXPECTED_RETENTION_RUNS_DEFAULT:
        raise RuntimeError(f"Retention design must contain exactly 288 new runs, got {expected_retention}")
    if not args.analysis_only:
        phase5_statuses = run_task_blocks(
            retention_tasks, retention_campaign, cfg, retention_run_dir,
            args.root_seed, args.workers, "phase5_retention",
        )
    else:
        phase5_statuses = {"analysis_only": 1}
    retention_status = aggregate_retention_campaign(retention_run_dir, expected_retention)
    retention_status["statuses"] = phase5_statuses
    if not retention_status["complete_campaign"]:
        write_phase_status(root, 5, retention_status)
        print(json.dumps(json_safe(retention_status), indent=2, ensure_ascii=False))
        raise SystemExit(
            "Phase 5 retention campaign is incomplete. Re-run the same command to resume."
        )
    phase5_analysis = analyze_retention_sweep(
        core_dir, retention_run_dir, args.root_seed, retention_analysis_dir
    )
    phase5 = {"campaign": retention_status, "analysis": phase5_analysis}
    write_phase_status(root, 5, phase5)
    final = {
        "phase1": phase1_status, "phase2": phase2, "phase3": phase3,
        "phase4": phase4, "phase5": phase5,
    }
    write_integrated_mechanism_report(root, phase1_status, phase2, phase3, phase4, phase5)
    print(json.dumps(json_safe(final), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
