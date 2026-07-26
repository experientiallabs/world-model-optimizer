"""Cross-tokenizer on-policy distillation: scoring a student with a foreign teacher.

The same-tokenizer distill mode has the teacher score the student's exact
sampled token ids (`wmh.distill.teacher.TinkerTeacher`). That is only meaningful
when both models share a vocabulary. A GLM-5.2 teacher and a Qwen3.6 student do
not, so this package scores the teacher on its OWN tokenization of the same
conversation and compares the two side by side, span for span.

The pieces, in the order the loop uses them:

- `teacher_render` renders the canonical message list with the TEACHER's chat
  template and reports which teacher token ranges cover byte-identical message
  content (assistant reasoning, visible text, tool-call argument values). Turn
  framing and tool-call syntax differ between templates and are deliberately
  left uncovered.
- `byte_offsets` gives exact per-token byte offsets on both sides with no
  re-encoding, which is what lets the two token streams be compared in one byte
  space. Re-encoding is not an option on the student side: sampling emits
  non-canonical BPE, so `encode(decode(ids)) != ids` for a large minority of
  real spans.
- `chunks` holds the alignment result (`ChunkPlan`) and turns it into the
  per-token advantages the existing `importance_sampling` wire format already
  carries, via `attach_chunk_advantages`.
- `prompt_logprobs` is the teacher's only network surface: one
  `/v1/completions` call per trajectory against a self-hosted vLLM server,
  sending the prompt as token ids and reading `prompt_logprobs` back.

Invariants this package exists to protect, each of which was a real defect
found while building it:

- TITO is untouched. The student always trains on its exact sampled ids; every
  round trip here is scoring-side only.
- A chunk's influence is its reverse-KL gap, not its token count.
- Centering is over chunk TOTALS. Token-level centering inverts long chunks.
- A token no chunk covers keeps advantage exactly 0.0, which IS the mask on the
  wire, so centering must never touch it.
"""

from wmh.distill.xtoken.byte_offsets import span_byte_ends
from wmh.distill.xtoken.chunks import (
    ChunkAdvantageStats,
    ChunkPlan,
    ChunkSpan,
    attach_chunk_advantages,
)
from wmh.distill.xtoken.teacher_render import ContentIsland, TeacherRender, render_for_teacher

__all__ = [
    "ChunkAdvantageStats",
    "ChunkPlan",
    "ChunkSpan",
    "ContentIsland",
    "TeacherRender",
    "attach_chunk_advantages",
    "render_for_teacher",
    "span_byte_ends",
]
