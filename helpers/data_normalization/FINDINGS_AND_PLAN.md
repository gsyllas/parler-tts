# Greek TTS prompt-data: findings + normalization pipeline spec

**Scope:** female dataset (human-transcribed, gold).
The same pipeline will apply to the male dataset once it is re-transcribed
by humans (current male is Whisper-v3-large, will be replaced shortly).

**Where this is implemented:** *not here.* This is a spec only. The
implementation lives in a separate normalization repo.

**Why no ASR / forced-alignment step:** the transcripts are human-made and
already faithful to the audio. The defects are formatting issues, not
mistranscriptions. The pipeline is therefore pure-text.

Training pipeline reminder (`training/run_parler_tts_training.py:390-394`):
the only thing done to a prompt before tokenization is `prompt.strip()`.
The multilingual prompt tokenizer is non-destructive on Greek (every
per-speaker sample we audited had `raw == roundtrip`). So whatever sits
in the `text` column reaches the model verbatim, and normalization at the
source is the right intervention point.

---

## Findings (female, spk_0001, 9 911 rows)

### 1. ~8% of rows have no terminal punctuation — primary suspect for the "drag-out" tails

Top last-char of stripped `text`:

| count | last char |
|------:|-----------|
| 8 703 | `.`       |
|   203 | `ς` (Greek final sigma, mid-word last letter) |
|   177 | `;` (Greek question mark) |
|   125 | `!`       |
|   117 | `α` … plus 100+ more counts on `ν Σ ο ι ε η ω υ` … |
|    48 | `,`       |
|    38 | `…`       |

`. ; !` cover ~91.6%. Remaining ~830 rows end on a Greek letter, comma,
or ellipsis. Hypothesis: at training the model sees audio frames past
the last text token in those rows → at inference it learned a small
"emit a tail anyway" distribution → audible drag at sentence end.

Per-speaker random sample of 25 caught 7 such rows (~28% of the sample,
explained by sampling variance and the rough ~8% base rate combined with
the small sample size).

Typical no-period rows are **complete utterances** that simply look like
headlines or list/date strings, e.g.:

- `'Με βάση τα παραπάνω ο ευρωβουλευτής του Κόμματος υπέβαλε τα εξής ερωτήματα'`
- `'Με επιτυχία ολοκληρώθηκαν οι αγώνες αθλοπαιδείας ενόργανης γυμναστικής στο Ηράκλειο'`
- `'Σχέδιο τριών αξόνων με έναρξη την πρώτη Ιανουαρίου δύο χιλιάδες είκοσι πέντε και ολοκλήρωση το δύο χιλιάδες είκοσι επτά'`
- `'ΑΝΔΡΕΑΣ ΠΑΠΑΔΟΝΤΩΝΑΚΗΣ'` (a standalone name)

They are safe to terminate with a `.`.

### 2. Ellipsis / dot-cluster pollution

- `U+2026 HORIZONTAL ELLIPSIS` (`…`): **148** occurrences. 38 of them are
  the final char of the row.
- `'….'` (ellipsis + period combo) on rows like
  `'για τα περαιτέρω….'`.
- `'..'` (plain double-dot, not real ellipsis):
  `'Έκανε το σωστό..'`, `'Η υγεία των μαθητών γίνεται..πολυτέλεια'`.

`…` is a learned "trailing off" cue → very plausibly contributes to the
silence-tail behavior.

### 3. Middle-dot used as bullet separator

`U+00B7 MIDDLE DOT` (`·`): 8 occurrences. Used like a list bullet,
e.g.:

- `'Οργάνωση · Σύλλογος Γονέων και Κηδεμόνων δέκατου τέταρτου Δημοτικού Σχολείου Αθήνας Δημήτρης Πικιώνης · Σύλλογος Γονέων ...'`

Also a few uses as a real Greek "ano teleia" punctuation (rough
equivalent of `;` in English), e.g.:

- `'Η φέτα δεν είναι απλώς ένα αγροδιατροφικό προϊόν· είναι κομμάτι ...'`
- `'Το σημερινό σχολείο δεν μορφώνει· εκπαιδεύει για να περάσεις εξετάσεις.'`

So we cannot blanket-replace `·`; we need a context rule.

### 4. Number-placeholder leak (1 row)

`row 2128`:
`'Σχολικό Έτος Σύνολο Αναπληρωτών μείον ----- δύο χιλιάδες είκοσι-δύο χιλιάδες είκοσι ένα πενήντα πέντε χιλιάδες συν'`

The `-` sign in front of a number got verbalized to `"μείον"` (= "minus")
but the number itself was replaced with `-----` by some upstream step.
Unrecoverable from text alone → drop.

### 5. Eval split mirrors the same issues

- Row 2: `'Έτσι καταντήσατε τους Εκπαιδευτικούς της χώρας κύριε Μητσοτάκη…!!!'`
  (ellipsis + multi-bang stack).
- Row 4: `'... ένα… διαβάστε τη συνέχεια εδώ'` (`…` mid-text + missing
  terminal punctuation).
- Row 70: starts with `'μείον Σταδιακός περιορισμός ...'` (the leaked
  word from a `-` bullet).
- Many rows ending on a Greek letter with no period.

So the W&B eval audio that exhibits the drag is being driven by these
exact rows.

### 6. Things confirmed clean

- The tokenizer round-trip is byte-identical on every sampled row, on
  both speakers. Tokenizer is not the source of any defect.
- Single-speaker pools are correctly single-speaker.
- Audio is human-transcribed for female → no alignment work needed.

---

## Pipeline specification

Run as a script (in the separate normalization repo) over a
HuggingFace `save_to_disk` dataset, producing a new `save_to_disk`
dataset with the same schema and the same audio bytes, only the prompt
column is modified, and rows that fail the drop policy are removed.

### Inputs / outputs

- **Input:** `save_to_disk` directory with at least these columns:
  `text` (the prompt — the only column we touch), plus the rest passed
  through unchanged (`filename`, `speaker_id`, `audio`,
  `transcription_original`, all the dataspeech feature columns, etc.).
  The dataset may be a `DatasetDict` with `train`/`eval` splits; process
  each independently with the same logic.
- **Output:** `save_to_disk` directory written next to the input,
  e.g. input ending in `05_merged_det/` → output ending in
  `06_normalized/`.
- Emit two side-files in the output dir:
  - `normalize_text.report.txt` — counters + before/after histograms.
  - `normalize_text.dropped.jsonl` — one record per dropped row with
    `{index, reason, original, cleaned}`.

### Configuration

| flag | default | meaning |
|------|---------|---------|
| `--input` | (required) | source `save_to_disk` dir |
| `--output` | (required) | destination `save_to_disk` dir |
| `--prompt-column` | `text` | column to normalize |
| `--dry-run` | off | run, print report, but don't write output |
| `--samples` | 20 | random before/after pairs to print |
| `--seed` | 0 | RNG seed for sampling and shuffling reports |

### Per-row transforms

Apply in this order. Each is idempotent. Count rows touched per step.

1. **NFC normalize.** `unicodedata.normalize("NFC", t)`.
2. **Strip zero-width chars.** Remove any of `U+200B U+200C U+200D U+FEFF`.
3. **Ellipsis family.** Single regex `r"…+\.*|\.…+|\.{2,}"` → `"."`.
   Covers `…`, `….`, `.…`, `..`, `...`, `....`, `…………`, etc.
4. **Middle-dot context-aware:**
   - **Bullet usage** (`·` with at least one space on each side, e.g.
     `' · '`) → `, ` (comma + space). This is the list-separator pattern.
   - **Ano-teleia usage** (`·` directly after a word with no leading
     space, like `προϊόν·`) → leave unchanged. It's legitimate Greek
     sentence-internal punctuation.
   - Implementation hint: regex `r"\s+·\s+"` for bullets, anything else
     left alone.
5. **Typographic-to-ASCII fold.** Single `str.translate` map:
   ```
   « » “ ” „ ‟ ‹ ›   →  "
   ‘ ’ ‚ ‛           →  '
   – —               →  -
   ```
6. **Junk ASCII stripping.** Replace any of
   `* | ~ ^ _ = + [ ] { } < > \ / @ # $ % &` with a space.
   These never appear in clean Greek prose in this corpus; they would
   each be a separate tokenizer subword for the model and a learned
   cue we don't want.
7. **Whitespace collapse.** `re.sub(r"\s+", " ", t).strip()`.
8. **Repeated terminal-punctuation collapse.** Collapse runs of `! ? .`
   to a single marker:
   - `!{2,}` → `!`
   - `\?{2,}` → `?`
   - Mixed stacks of `[!?\.]` of length ≥ 2 → keep the **strongest**
     marker in priority `? > ! > .`. So `…!!!` (now `.!!!`) → `!`;
     `!!?` → `?`; `..!` → `!`.
9. **Terminal-punctuation enforcement.** Look at the last char of the
   cleaned, stripped text:
   - If it is `. ; ! ? :`: leave alone (`;` is Greek question mark, `:` is acceptable).
   - If it is `,`: drop the comma, append `.`.
   - Otherwise (Greek/Latin letter, digit, or any other char): append `.`.

   Do **not** promote a missing terminator to `;` or `!` — we have no
   evidence of question/exclamation intent for those rows, and the
   training distribution is overwhelmingly `.`.

### Drop policy (applied after the transforms)

Drop a row if any of these hold. Record the reason for the report.

- `empty` — cleaned text is empty after stripping.
- `placeholder_dash_run` — cleaned text still contains `--` (a `-{2,}`
  run). This catches the `μείον -----` family — they are not recoverable
  from text alone.
- `too_short` — fewer than 3 whitespace-separated tokens. Catches stray
  one-word fragments. (NB: `'ΑΝΔΡΕΑΣ ΠΑΠΑΔΟΝΤΩΝΑΚΗΣ'` is 2 tokens and
  will be dropped under this rule — that's the correct call; we don't
  want the model trained on standalone names.)

### Report contents

- Row counts per split before/after, and drops per reason.
- Per-transform "rows touched" counts.
- Top-12 sentence-ending characters histogram **before** and **after**,
  per split — the after-histogram should be `. ; ! ? :` only.
- 20 random `(BEFORE, AFTER)` pairs from rows where the text actually
  changed, for human spot-check.

### Acceptance gates (run after writing the output)

Re-run `helpers/data_inspection/inspect_prompts.py overview` against the
output. Should hold:

- "Flagged characters" list is empty (or down to a handful of legitimate
  ano-teleia `·`).
- "Suspicious substrings" list is empty.
- Top sentence-ending chars top-12 contains only `. ; ! ? :`.
- Total row count = input count − reported drops.
- Random spot-check: play 5 audio samples and confirm the cleaned `text`
  matches what's spoken.

### Wiring into training (after the normalized dataset is built)

Per the
[parler-tts-greek-tokenizer memory](../../.claude/projects/-home-gsyllas-parler-tts/memory/parler-tts-greek-tokenizer.md):

- Update `leonardo/configs/female_*.json` so
  `train_dataset_name` / `eval_dataset_name` point at `06_normalized`.
- Delete `leonardo/runs/<exp>/dataset_audio` (precomputed prompt cache —
  it would otherwise load the old `prompt_input_ids` blindly).
- Delete `leonardo/runs/<exp>/audio_code_tmp` only if rows were dropped
  (it is row-aligned; tokenizer-independent so it doesn't strictly
  require invalidation otherwise). Safer to clear it after a drop pass.
- Commit + push from local, then `git pull` on Leonardo
  (per [leonardo-repo-sync memory](../../.claude/projects/-home-gsyllas-parler-tts/memory/leonardo-repo-sync.md)).
- Short 1–2 epoch warm-resume from the existing checkpoint, listen to
  W&B eval audio, confirm the tail-drag is gone before committing to a
  full retrain.

### Expected magnitude on the female dataset

Rough estimates from the audit (will be confirmed by the report):

- `period_appended`: ~600–900 rows (the ~8% with no terminator).
- `comma_terminal_replaced`: ~48 rows.
- `ellipsis_replaced`: ~150 rows.
- `middot_replaced` (bullet form): single digits.
- `punct_stack_collapsed`: a handful (the `…!!!` stack and similar).
- `drop_placeholder`: 1 row (the `μείον -----` row).
- `drop_too_short`: a handful (standalone names like the
  all-caps `'ΑΝΔΡΕΑΣ ΠΑΠΑΔΟΝΤΩΝΑΚΗΣ'`).

Total rows changed: ~10% of the dataset. Total rows dropped: <0.1%.

---

## Notes for when the male human transcriptions land

The Whisper-v3-large male dataset has additional defects that the new
human transcriptions will not have — primarily mid-word text truncation
(rows ending `'... Ζελένσκι, χ'`, `'... οφη'`, `'... έγινε με'`) and
multiple unfilled number placeholders per sentence (`'... ήταν -- στην ΕΕ
και -- στη ζώνη του Ευρώ ...'`).

When the human male arrives:

1. Run `inspect_prompts.py overview` and `per-speaker` on it first, like
   we did for female.
2. If the histogram looks similar to female (i.e. only formatting
   issues, no mid-word amputations, no `--` placeholders), run the
   pipeline above unchanged.
3. If new defect classes show up, extend this spec — do **not** quietly
   patch the implementation without updating this doc.
