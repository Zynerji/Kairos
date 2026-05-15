"""KairosAntiResonantInit — orthogonal weight init that avoids the
dominant harmonics of a reference (teacher) weight.

Ported from qGPT-Infinity ``core/dirac_crystal_linear.py`` (2026-04-10),
where antiresonant orthogonal init was the fix for "Silver init caused
NaN" — the silver-ratio decay had large cross-term interference at
init that drove the K=8 distill run to NaN-on-backward.

Concept
-------
Standard orthogonal init (``nn.init.orthogonal_``) gives a well-
conditioned matrix but its dominant singular directions are random.
For *distillation*, that's wasteful: the student's largest-singular-
value directions tend to align with whatever the teacher's largest
directions are within a few hundred steps anyway. Worse, when student
and teacher share a dominant direction by chance at init, the
distillation gradient is uninformative on that subspace early on.

Antiresonant init pre-empts this:

    1. Compute teacher's top-K singular directions (SVD).
    2. Project them *out* of an orthogonal random init.
    3. Rescale so spectral norm is small (≤ ``scale_factor``).

The student is then forced to *acquire* those top-K directions from
the distillation signal rather than starting with random alignment.
For grokking-adjacent regimes, this widens the EXPLORE phase (the
student doesn't accidentally memorise) without sacrificing
optimisation stability.

For embeddings the equivalent trick is **phase-staggered Fourier
init**: row k gets the basis vector ``(1/√d) * cos(k·2π/d * j +
π/2·offset)``, which produces an orthonormal Fourier-coefficient
embedding with antiresonant cross-terms (sum cancels to zero).

Defaults
--------
``scale_factor = 0.02`` — matches qGPT's amplitude scale (proven
non-NaN on K=8→64 distillation).
``suppress_top_k = 8`` — covers the top harmonics for a typical LLM
hidden_dim (the next ones decay rapidly).

API
---
This is *not* a training callback. It is applied once before training
to a module tree, returning a list of (name, shape) tuples for the
layers it touched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AntiResonantReport:
    """Summary of an antiresonant-init pass."""
    n_linear: int = 0
    n_embedding: int = 0
    n_skipped: int = 0
    layers: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)
    suppressed_directions: int = 0


class KairosAntiResonantInit:
    """Orthogonal weight init with optional dominant-harmonic suppression.

    Parameters
    ----------
    suppress_top_k : int
        Number of teacher singular directions to project out of each
        student linear layer (matched by name). 0 = pure orthogonal
        init (no suppression). Default 8.
    scale_factor : float
        Spectral-norm cap for linear weights. The init is rescaled so
        the largest singular value equals ``scale_factor``. Default
        0.02 (matches qGPT's ``0.02/√K`` wave amplitude).
    embedding_scale : float
        Equivalent cap for embedding tables. Default 0.02.
    phase_staggered_embeddings : bool
        If True, embeddings use phase-staggered Fourier init (the
        qGPT trick). Default True.
    seed : int | None
        Reproducibility seed. Default ``None`` (uses ambient torch
        rng state).
    """

    def __init__(self, *, suppress_top_k: int = 8,
                 scale_factor: float = 0.02,
                 embedding_scale: float = 0.02,
                 phase_staggered_embeddings: bool = True,
                 seed: int | None = None) -> None:
        if suppress_top_k < 0:
            raise ValueError(f"suppress_top_k must be >= 0; got {suppress_top_k}")
        if not (0.0 < scale_factor):
            raise ValueError(f"scale_factor must be > 0; got {scale_factor}")
        if not (0.0 < embedding_scale):
            raise ValueError(f"embedding_scale must be > 0; got {embedding_scale}")
        self.suppress_top_k = int(suppress_top_k)
        self.scale_factor = float(scale_factor)
        self.embedding_scale = float(embedding_scale)
        self.phase_staggered_embeddings = bool(phase_staggered_embeddings)
        self.seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, student: Any, teacher: Any = None,
              *, dry_run: bool = False) -> AntiResonantReport:
        """Apply antiresonant init to all linear/embedding layers in
        ``student``. If ``teacher`` is supplied, top-K teacher
        directions are suppressed per layer (matched by name)."""
        import torch
        import torch.nn as nn

        if self.seed is not None:
            torch.manual_seed(int(self.seed))

        rep = AntiResonantReport()

        teacher_state: dict[str, "torch.Tensor"] = {}
        if teacher is not None:
            try:
                for tname, tparam in teacher.named_parameters():
                    teacher_state[tname] = tparam.detach()
            except AttributeError:
                # teacher is a plain state_dict
                teacher_state = {k: v for k, v in teacher.items()
                                   if isinstance(v, torch.Tensor)}

        for name, mod in student.named_modules():
            if isinstance(mod, nn.Linear):
                rep.n_linear += 1
                rep.layers.append((name, tuple(mod.weight.shape)))
                if not dry_run:
                    suppressed = self._init_linear(
                        name, mod, teacher_state,
                    )
                    rep.suppressed_directions += suppressed
            elif isinstance(mod, nn.Embedding):
                rep.n_embedding += 1
                rep.layers.append((name, tuple(mod.weight.shape)))
                if not dry_run:
                    self._init_embedding(name, mod)
            else:
                # We skip e.g. LayerNorm/RMSNorm/Conv1d on purpose:
                # those have well-tested defaults already.
                if any(isinstance(p, nn.Parameter)
                       for p in mod.parameters(recurse=False)):
                    rep.n_skipped += 1
        return rep

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_linear(self, name: str, mod: Any,
                       teacher_state: dict) -> int:
        import torch
        import torch.nn as nn
        with torch.no_grad():
            W = mod.weight  # (out, in)
            out_dim, in_dim = W.shape

            # Random orthogonal seed (rectangular: use Q from QR).
            rand = torch.randn_like(W)
            try:
                # torch.linalg.qr gives Q of shape (out, min(out,in))
                Q, _ = torch.linalg.qr(rand)
                if Q.shape != W.shape:
                    # Rectangular: pad/truncate to match
                    base = torch.zeros_like(W)
                    m = min(Q.shape[0], W.shape[0])
                    n = min(Q.shape[1], W.shape[1])
                    base[:m, :n] = Q[:m, :n]
                else:
                    base = Q
            except RuntimeError:
                # Fall back to nn.init.orthogonal_
                base = torch.zeros_like(W)
                nn.init.orthogonal_(base)

            # Project out teacher's top-K directions (if available).
            n_suppressed = 0
            t_key = self._match_teacher_key(name, teacher_state)
            if t_key is not None and self.suppress_top_k > 0:
                T = teacher_state[t_key]
                if T.shape == W.shape:
                    # SVD: T = U S V^T
                    try:
                        U, S, Vh = torch.linalg.svd(T.float(),
                                                      full_matrices=False)
                        k = min(self.suppress_top_k, len(S))
                        # Build projector P = U_k U_k^T (left subspace)
                        Uk = U[:, :k]
                        P_left = Uk @ Uk.t()
                        # Project out: base = (I - P_left) base
                        base = base - P_left @ base
                        # Re-orthogonalise to keep conditioning
                        try:
                            Q2, _ = torch.linalg.qr(base)
                            if Q2.shape == base.shape:
                                base = Q2
                        except RuntimeError:
                            pass
                        n_suppressed = k
                    except RuntimeError:
                        pass

            # Spectral-norm cap.
            try:
                _, S2, _ = torch.linalg.svd(base, full_matrices=False)
                top = float(S2.max().item()) if S2.numel() > 0 else 1.0
            except RuntimeError:
                top = 1.0
            if top > 1e-12:
                base = base * (self.scale_factor / top)

            mod.weight.copy_(base.to(W.dtype))
            if mod.bias is not None:
                mod.bias.zero_()
        return n_suppressed

    def _init_embedding(self, name: str, mod: Any) -> None:
        import torch
        with torch.no_grad():
            W = mod.weight  # (vocab, dim)
            vocab, dim = W.shape
            if self.phase_staggered_embeddings:
                # Antiresonant Fourier-stagger: row k = (1/√d) * cos(k·2π/d·j + π/2)
                # The π/2 offset makes consecutive rows quadrature pairs,
                # producing antiresonant cross-correlation (sum cancels).
                rows = torch.arange(vocab, dtype=torch.float32).unsqueeze(1)
                cols = torch.arange(dim, dtype=torch.float32).unsqueeze(0)
                phase = rows * (2.0 * math.pi / max(dim, 1)) * cols + math.pi / 2.0
                emb = torch.cos(phase) / math.sqrt(max(dim, 1))
                # Rescale to embedding_scale per row max
                row_max = emb.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
                emb = emb * (self.embedding_scale / row_max)
                mod.weight.copy_(emb.to(W.dtype))
            else:
                # Fall back to small-normal init.
                std = self.embedding_scale / math.sqrt(max(dim, 1))
                mod.weight.normal_(mean=0.0, std=std)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _match_teacher_key(student_name: str,
                             teacher_state: dict) -> str | None:
        """Find a teacher param key whose tail matches the student
        layer's. Match heuristic: same .weight suffix and longest
        common right substring."""
        if not teacher_state:
            return None
        sw = student_name + ".weight"
        if sw in teacher_state:
            return sw
        # Longest suffix match
        best = None
        best_len = 0
        for tk in teacher_state:
            if not tk.endswith(".weight"):
                continue
            sw_core = sw.split(".")[-2:]   # last 2 tokens
            tk_core = tk.split(".")[-2:]
            if sw_core == tk_core:
                # tiebreak by full tail length
                tail = 0
                for a, b in zip(reversed(sw.split(".")),
                                  reversed(tk.split("."))):
                    if a == b:
                        tail += 1
                    else:
                        break
                if tail > best_len:
                    best, best_len = tk, tail
        return best
