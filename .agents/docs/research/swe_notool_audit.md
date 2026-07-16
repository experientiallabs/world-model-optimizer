# D90.2 audit: the swe "no-tool" episodes are format-confounded, not disengagement

**Question.** The swe substrate run's row of record (D86/D87) read "no trainable signal at
9B under gold test rubrics", with 477/600 episodes ending at steps=2 with zero parsed tool
calls as the headline evidence, interpreted as base-policy disengagement on long
PR-description tasks. Is the model actually disengaging, or is it emitting tool syntax the
scaffold's parsers reject?

**Method (rule 12: read the outputs).** The training rollouts were not retained, so the
probe (`.agents/scripts/swe_notool_probe.py`) reproduces the exact turn where those
episodes died: same system prompt template ({task}/{done_tool}), same "Begin." opener,
same tools payload rendered through the same chat template, same temperature 1.0 and
max_tokens 6000, raw text completions (matching the training rl_server, which returns
unparsed text), then the scaffold's own three-stage parse chain (structured -> JSON ->
Qwen-XML) applied verbatim. Two rounds: 12 scenarios x 2 and 30 scenarios x 2 from the
pinned training set, 84 first-turn completions total; 24 of the no-tool texts were read
in full by a human-equivalent pass, the rest classified mechanically.

**Result.** 50/84 first turns (60%) produce no parseable tool call, closely matching the
training run's shape. **All 50 are the model trying to act**: a competent analysis of the
issue (frequently identifying the correct file and fix) followed by a sensible first
command (`ls -la`, `find /testbed ...`) emitted inside a fenced
` ```mswea_bash_command ` block: the mini-swe-agent response format (THOUGHT + fenced
bash), i.e. the format of the harness that captured this corpus and, evidently, of the
model's own SWE training data. 35/50 additionally carry a corrupted mixed-syntax tail
(`</parameter> </function> </tool_call>`), the same schema-echo corruption D82 attributed
to RFT2: it is a base-model prior on swe-style tasks, which RFT amplified rather than
created. Zero completions surrender in prose. Raw texts:
`sonnet_era_wm_rows/swe_notool_probe.jsonl` and `swe_notool_probe2.jsonl`.

**Row rewording (correction appended to the PR #73 board, dated 2026-07-16).** The row of
record becomes: "no reward signal observed at 9B under gold test rubrics, and the dominant
failure mode (477/600 no-tool episodes) is FORMAT-CONFOUNDED: the policy emits its native
mini-swe-agent tool syntax, which the scaffold's JSON/Qwen-XML parsers reject, ending the
episode at the first turn. Not policy disengagement." The prompt says "call the bash tool"
and the template renders a bash(command) tool; the model's SWE-format prior overrides the
prompt's tool contract at 9B.

**What a fixed scaffold might change.** Adding a fenced-block parser for the mswea format
would re-engage ~60% of first turns that currently die at steps=2, converting them into
multi-step episodes with WM feedback. Whether a reward signal then appears under gold
FAIL_TO_PASS rubrics is open, and the prior evidence cuts both ways: episodes that DID
parse tools still scored ~0 (5 successes among the 123 tool-using substrate episodes;
0/29 on the pre-substrate row), but those tool-using episodes were sampled from the same
format-unstable policy mid-drift, so they are not a clean capability read either. The
honest state: the capability question at 9B on swe is OPEN, not answered in the negative.
A rerun would need the parser fix pre-declared (D90's protocol discipline) and is out of
the D78 sprint scope.
