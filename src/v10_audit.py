"""WS-A — the manuscript number audit. A number the script cannot trace does not ship.

Two modes, both driven by audit/manifest_v10.yaml:

  REPLACE  for `*`-marked values: read the artifact, write the value into the
           output draft, log `old -> new`.
  VERIFY   for unmarked values: compare against the artifact within rounding;
           any mismatch is a hard FAIL naming the manifest id.

Plus the folded checks the freeze gate needs:

  * UNMATCHED POLICY — a number in the draft with no manifest entry is a
    SHIP-BLOCKER. The whitelist is narrow (years, section/figure refs, version
    strings, the alpha-grid literals, and the budget constants, which are
    themselves asserted once against src/v9_gate.py).
  * TERMINOLOGY BANS -> zero hits.
  * `[TODO` -> zero outside an explicitly labelled limitations block.
  * `*` residue -> zero in number contexts after replacement.
  * `[CITE:` -> an INVENTORY report, not a failure; citations are the one
    manual fill. The companion reference gets its own named line.
  * EXPECTATION CHECK — every registered CI status in v9_diff_report.json is
    re-asserted after replacement, so a manifest typo cannot silently flip a
    claim.
  * FIGURE CSVs — figures/fig*_data.csv are audited alongside the text, so a
    figure and the sentence about it cannot disagree.

Usage:
  python3 src/v10_audit.py --draft binary_debiaser_draft_v3.md \
      --manifest audit/manifest_v10.yaml --figures figures
  python3 src/v10_audit.py ... --replace --out binary_debiaser_draft_v3.1.md
"""
import os, sys, re, json, csv, glob, argparse, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10_common as C

try:
    import yaml
except ImportError:
    yaml = None

BANNED_TERMS = ["entire signal", "zero inference cost", "base-only",
                "no cost at", "beats full precision"]
# A banned term is allowed only when the surrounding text is explicitly
# retracting or quarantining it. "correction"/"claimed" are included because
# the manuscript legitimately narrates its own earlier error.
RETRACTION_MARKERS = ("struck", "retract", "quarantin", "banned", "must not",
                      "is false", "no longer", "withdrawn", "never", "correction",
                      "which claimed", "earlier drafts")

# Numbers that need no artifact. Kept deliberately short.
WHITELIST_PATTERNS = [
    (r"^(19|20)\d{2}$", "year"),
    (r"^\d+(\.\d+)*$", "section-ref"),       # only when context says section/fig/app
    (r"^v\d+(\.\d+)*$", "version-string"),
]
# The alpha-grid literals and the budget constants. Whitelisted as a set, and
# asserted once against src/v9_gate.py rather than traced per occurrence.
GATE_LITERALS = {"4", "16", "200", "1.10", "1e-9", "0.02", "9", "1"}

# The frozen sparsity/density grid. These are experiment PARAMETERS (which
# configurations were run), not measured outcomes, so they are asserted once
# against the panels rather than traced per mention.
GRID_PCTS = {"1%", "0.5%", "50%", "90%", "95%", "97%", "99%", "99.5%", "99.9%",
             "0.9", "0.95", "0.97", "0.99", "0.995", "0.999", "0.0", "0.5"}
GRID_CONTEXT = re.compile(
    r"sparsit|densit|coordinat|drop|zeroing|top-k|retain|sparse|cliff|s\s*=|"
    r"dense|outperform|free\b", re.I)

# Model-tier descriptors: "<=3.8B", "7-9B", "at 8B", "2.6-8B". The digits name a
# parameter count, and the model list in sec 5.1 is the thing that would be
# audited, not each mention of a tier.
TIER_NUMS = {"1.5", "1.7", "2", "2.6", "3", "3.2", "3.5", "3.8", "7", "8", "9",
             "10", "14", "27", "32"}

# Numbers that belong to CITED PRIOR WORK, not to this paper's measurements.
# They are checked against the citation, not against an artifact of ours.
CITED_CONTEXT = re.compile(
    r"dare|bitdelta|ties|ilharco|dige|schick|inlp\b.*cite|\[cite:", re.I)

# Frozen protocol constants: grid points, minimum-sample thresholds, replicate
# counts. Parameters of the design, not outcomes of it.
PROTOCOL_NUMS = {"20", "6", "3", "50", "40", "541", "200"}
PROTOCOL_CONTEXT = re.compile(
    r"replicat|informative preference|fewer than|per cell|seeds? per|"
    r"items per|prompts|configurations|selection space|grid", re.I)

NUMBER_RE = re.compile(r"[+−-]?\d[\d,]*(?:\.\d+)?\s*%?")


# ------------------------------------------------------------ artifact I/O ---

def resolve_key(obj, key):
    """Dotted key path with support for quoted segments holding dots/spaces:
       diff."DEC :: Delta_selection".v9_point"""
    cur = obj
    for seg in re.findall(r'"([^"]*)"|([^.]+)', key):
        k = seg[0] if seg[0] else seg[1]
        if cur is None:
            return None
        if isinstance(cur, list):
            try:
                cur = cur[int(k)]
                continue
            except (ValueError, IndexError):
                return None
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


CI_RE = re.compile(r"\s*([+-][\d.]+)\s*\[\s*([+-][\d.]+),\s*([+-][\d.]+)\]\s*\(n=(\d+)")


def ci_extract(s, part):
    """Pull point/lo/hi/n out of a formatted CI string such as
    '+0.2835 [+0.1692, +0.4072] (n=10, excludes 0)'."""
    m = CI_RE.match(s or "")
    if not m:
        return None
    return dict(zip(("point", "lo", "hi", "n"), m.groups())).get(part)


class Artifacts:
    def __init__(self):
        self._cache = {}

    def get(self, path):
        if path not in self._cache:
            self._cache[path] = C.jload(path)
        return self._cache[path]

    def value(self, entry):
        """Resolve one manifest entry to a raw python value."""
        if "artifact" in entry and "key" in entry:
            obj = self.get(entry["artifact"])
            if obj is None:
                return None, f"artifact missing: {entry['artifact']}"
            v = resolve_key(obj, entry["key"])
            if v is None:
                return None, f"key not found: {entry['key']} in {entry['artifact']}"
            if entry.get("extract"):
                x = ci_extract(v, entry["extract"])
                if x is None:
                    return None, (f"could not extract {entry['extract']} from "
                                  f"{entry['key']}")
                try:
                    return float(x), None
                except (TypeError, ValueError):
                    return x, None
            return v, None
        if "derived" in entry:
            env = {}
            for name, spec in (entry.get("inputs") or {}).items():
                obj = self.get(spec["artifact"])
                if obj is None:
                    return None, f"artifact missing: {spec['artifact']}"
                val = resolve_key(obj, spec["key"])
                if val is None:
                    return None, f"key not found: {spec['key']}"
                env[name] = val
            try:
                return eval(entry["derived"], {"__builtins__": {}}, env), None
            except Exception as e:                       # noqa: BLE001
                return None, f"derivation failed: {e}"
        if "literal" in entry:
            return entry["literal"], None
        return None, "entry has neither artifact+key, derived, nor literal"


# -------------------------------------------------------------- formatting ---

def fmt_value(v, spec):
    if v is None:
        return None
    if spec is None:
        return str(v)
    try:
        # Integer formats ('d', ',d') reject a float, and the bare except below
        # then silently returned str(v) — so every count entry has been
        # formatting itself by accident rather than by its declared spec.
        if spec.endswith("d"):
            return format(int(round(float(v))), spec)
        if float(v) == 0:
            v = 0.0          # normalise -0.0; a signed zero reads as a measurement
        if spec.endswith("%"):
            # Two conventions. ".0%" is Python's native percent type: the value
            # is a FRACTION and gets scaled. ".1f%" means the value is ALREADY
            # in percent units and just needs a literal sign appended.
            if spec[-2:-1].isalpha():
                return format(float(v), spec[:-1]) + "%"
            return format(float(v), spec)
        return format(float(v), spec)
    except (TypeError, ValueError):
        return str(v)


def normalize_num(s):
    """Draft text uses U+2212 MINUS and thin spaces; normalize for comparison."""
    if s is None:
        return None
    s = str(s).replace("−", "-").replace(" ", "").replace(" ", "")
    s = s.replace(",", "").replace(" ", "")
    return s.strip()


def numbers_equal(a, b, tol=None):
    na, nb = normalize_num(a), normalize_num(b)
    if na == nb:
        return True
    try:
        fa = float(na.rstrip("%")); fb = float(nb.rstrip("%"))
    except (TypeError, ValueError):
        return False
    if tol is None:
        # equal if they agree at the printed precision of the draft value
        dec = len(na.split(".")[1].rstrip("%")) if "." in na else 0
        tol = 0.5 * (10 ** -dec) + 1e-12
    return abs(fa - fb) <= tol


# ------------------------------------------------------------------ audit ----

def near_misses(shown, text, hint=None, ulps=2):
    """Numbers in the draft that are CLOSE to a resolved value but not equal:
    the signature of a rounding or transcription error rather than a gap.

    A near miss only counts inside a CONTEXTUALLY RELEVANT sentence. Without
    that constraint a small estimand like +0.004 "nearly matches" every +0.002
    and +0.005 in every unrelated table in the paper, and the check becomes
    noise. The hint is what makes it specific."""
    if not hint:
        return []
    hint = hint.lower()
    tv = None
    try:
        tv = float(normalize_num(shown).rstrip("%"))
    except (TypeError, ValueError, AttributeError):
        return []
    out = []
    for n in scan_numbers(text):
        norm = normalize_num(n["raw"]).rstrip("%")
        if not re.match(r"^[+-]?\d+(\.\d+)?$", norm):
            continue
        try:
            nv = float(norm)
        except ValueError:
            continue
        dec = len(norm.split(".")[1]) if "." in norm else 0
        if dec == 0:
            continue
        step = 10 ** -dec
        d = abs(nv - tv)
        if not (0.5 * step + 1e-12 < d <= ulps * step + 1e-12):
            continue
        if hint not in n["context"].lower():
            continue
        out.append({"line": n["line"], "raw": n["raw"],
                    "context": n["context"].strip()})
    return out


PROSE_EDITS = "audit/prose_edits.yaml"


def apply_prose_edits(text):
    """Apply the registered prose edits from audit/prose_edits.yaml.

    The author's draft is never modified: draft_v3.1 is generated as
    v3 + these edits + number replacement. Each edit exists because a number
    alone could not make its sentence true -- a population-dependent estimate
    printed with no population named, or an assertion the measurement
    contradicts. Every edit preserves the original numeric tokens so the
    manifest can still bind and fill them afterwards.
    """
    fp = os.path.join(C.HERE, PROSE_EDITS)
    if not os.path.exists(fp):
        return text, []
    try:
        import yaml as _y
        spec = _y.safe_load(open(fp)) or {}
    except Exception as e:                                 # noqa: BLE001
        return text, [{"error": f"could not read {PROSE_EDITS}: {e}"}]
    applied = []
    for e in spec.get("edits", []):
        old, new = e.get("old"), e.get("new")
        if not old:
            continue
        n = text.count(old)
        if n == 1:
            text = text.replace(old, new, 1)
            applied.append({"id": e.get("id"), "line": e.get("line"),
                            "status": "applied", "why": e.get("why")})
        else:
            applied.append({"id": e.get("id"), "line": e.get("line"),
                            "status": f"NOT APPLIED: matched {n} times, need exactly 1"})
    return text, applied


def load_manifest(path):
    if yaml is None:
        raise SystemExit("PyYAML required: pip install pyyaml")
    with open(path) as fh:
        m = yaml.safe_load(fh) or {}
    return m.get("entries", m if isinstance(m, list) else [])


# Structural numerals that are not claims: markdown ordered-list markers,
# table rules, and numerals inside model identifiers.
LIST_MARKER_RE = re.compile(r"^\s*\d+\.\s")
HEADER_RE = re.compile(r"^\s*#{1,6}\s*[\d.]+")
TABLE_RULE_RE = re.compile(r"^\s*\|[\s:|-]*\|\s*$")
MODEL_TOKEN_RE = re.compile(
    r"(gemma|qwen|llama|phi|smol|olmo|falcon|granite)[\w.-]*", re.I)


def _is_starred(line, start, end):
    """Is this token marked with the draft's `*` work-order marker?

    The marker collides with markdown emphasis. `**+0.353**` is a BOLD value
    with no marker; `**+0.119***` is a bold value that DOES carry one; `+0.198*`
    is a plain value that carries one. Counting a bold-close `*` as a marker
    makes replace-mode eat the delimiter and corrupt the table.
    """
    trailing = len(line[end:]) - len(line[end:].lstrip("*"))
    if trailing == 0:
        return False
    bold_open = line[max(0, start - 2):start].endswith("**")
    if bold_open:
        return trailing >= 3          # 2 close the bold, a 3rd is the marker
    return True


def scan_numbers(text):
    """Every numeric token in the draft that could be a CLAIM, with its line
    and a context window. Structural numerals (list markers, table rules,
    digits inside model names) are not claims and are excluded here rather
    than whitelisted one by one downstream."""
    out = []
    for i, line in enumerate(text.split("\n"), 1):
        if TABLE_RULE_RE.match(line):
            continue
        hm = HEADER_RE.match(line)
        skip = []
        m = LIST_MARKER_RE.match(line)
        if m:
            skip.append((0, m.end()))
        if hm:
            skip.append((0, hm.end()))
        for mm in MODEL_TOKEN_RE.finditer(line):
            skip.append((mm.start(), mm.end()))
        for mt in NUMBER_RE.finditer(line):
            if not re.search(r"\d", mt.group(0)):
                continue
            if any(a <= mt.start() < b for a, b in skip):
                continue
            # A leading "-" that follows a word character is a hyphen, not a
            # sign: "rank-16", "top-k", "1e-9", "bbq_Age-2". Treating it as a
            # negative number invents claims that were never made.
            prev = line[mt.start() - 1] if mt.start() else ""
            prev2 = line[mt.start() - 2] if mt.start() > 1 else ""
            # A dash after a LETTER is a hyphen ("rank-16", "top-k", "1e-9").
            # A dash after a DIGIT is a numeric range ("46-92%", "8-54%") and
            # its right-hand side is a real number that must stay auditable.
            if prev in "-–—" and (prev2.isalpha() or prev2 == "_"):
                continue
            if prev.isalpha() or prev == "_":
                continue
            # A dot after a digit means we are inside a dotted version string
            # ("5.14.1"): NUMBER_RE stops at the second dot and the trailing
            # ".1" would otherwise surface as its own claim.
            if prev == "." and prev2.isdigit():
                continue
            raw = mt.group(0).strip().rstrip(",")
            starred = _is_starred(line, mt.start(), mt.end())
            out.append({"line": i, "raw": raw, "starred": starred,
                        "col": mt.start(),
                        "context": line[max(0, mt.start() - 55):mt.end() + 25]})
    return out


def audit(draft_path, manifest_path, figdir, do_replace=False, out_path=None):
    text = open(draft_path).read()
    text, prose = apply_prose_edits(text)
    entries = load_manifest(manifest_path)
    art = Artifacts()

    report = {"draft": draft_path, "manifest": manifest_path,
              "n_entries": len(entries), "replace_mode": bool(do_replace),
              "entries": [], "fails": [], "warns": [], "replacements": [],
              "prose_edits": prose,
              "unmatched": [], "citations": {}, "greps": {}, "expectation": {},
              "figure_csvs": {}}

    # ---- 1. resolve every manifest entry -----------------------------------
    covered = set()
    deferred = []
    newtext = text
    # Which minus sign does this document actually use for numbers?
    doc_minus = ("\u2212" if text.count("\u2212") >= text.count(" -0.") else "-")
    for e in entries:
        eid = e.get("id", "<no id>")
        if e.get("blocked_as_run"):
            report["entries"].append({"id": eid, "status": "BLOCKED_AS_RUN",
                                      "as_run_artifact": e.get("as_run_artifact"),
                                      "why": e.get("why")})
            report["fails"].append(
                f"{eid}: BLOCKED — no v9 source. {e.get('why')}")
            for loc in e.get("occurrences", []):
                covered.add((loc["line"], normalize_num(loc["raw"])))
            continue
        if e.get("whitelist"):
            for loc in e.get("occurrences", []):
                covered.add((loc["line"], normalize_num(loc["raw"])))
            report["entries"].append({"id": eid, "status": "WHITELIST",
                                      "reason": e.get("whitelist")})
            continue

        val, err = art.value(e)
        shown = fmt_value(val, e.get("format"))
        rec = {"id": eid, "artifact": e.get("artifact") or e.get("inputs"),
               "key": e.get("key") or e.get("derived"),
               "resolved": shown, "status": None}

        if err:
            rec["status"] = "UNRESOLVED"; rec["error"] = err
            report["fails"].append(f"{eid}: {err}")
            report["entries"].append(rec)
            continue

        for loc in e.get("occurrences", []):
            ln, raw = loc["line"], loc["raw"]
            covered.add((ln, normalize_num(raw)))
            starred = loc.get("starred", False)
            if loc.get("stale") and not loc.get("expect_matched"):
                # A line anchor alone is not enough to identify WHICH number on
                # the line belongs to this estimand: a Table A row carries four
                # starred values. Without a confirmed expected-old-value match,
                # replacing would overwrite a neighbouring column.
                report["warns"].append(
                    f"{eid} (L{ln}): stale occurrence not confirmed by an "
                    f"expected old value; left untouched")
                continue
            if (starred or loc.get("expect_matched")) and do_replace and not (
                    loc.get("expect_matched") or loc.get("hint_matched")):
                # Never WRITE into the manuscript on an ambiguous binding.
                # Values that round to zero (+0.000 / -0.000) match dozens of
                # unrelated cells numerically, so a bare value match is not
                # provenance. Verification can stay permissive; replacement
                # cannot.
                report["warns"].append(
                    f"{eid} (L{ln}): ambiguous binding for {raw!r} — not replaced")
                continue
            if (starred or loc.get("expect_matched")) and do_replace:
                # replace this occurrence on its line
                lines = newtext.split("\n")
                # keep the draft's own minus glyph; swapping U+2212 for ASCII
                # is a silent typographic regression across the whole paper
                out = shown
                # Adopt the DOCUMENT's minus glyph, not just the token's. A
                # replacement can flip sign (+0.004 -> -0.013), so keying off
                # the old token alone leaves an ASCII hyphen sitting beside
                # U+2212 in the same interval.
                if out.startswith("-") and doc_minus == "\u2212":
                    out = "\u2212" + out[1:]
                col = loc.get("col")
                cur = lines[ln - 1] if ln - 1 < len(lines) else ""
                # Replace AT THE BOUND COLUMN. A first-match string replace
                # rewrites the wrong token whenever the same digits occur
                # earlier on the line -- an "n = 3" entry once overwrote the
                # trailing 3 of "+0.243", producing "+0.2415".
                done = False
                if col is not None and cur[col:col + len(raw)] == raw:
                    tail = col + len(raw)
                    # Consume the work-order marker, and ONLY it. `_is_starred`
                    # already knows the three shapes: `v*` (marker),
                    # `**v**` (bold, no marker), `**v***` (bold + marker).
                    if _is_starred(cur, col, tail):
                        tail += 1
                    lines[ln - 1] = cur[:col] + out + cur[tail:]
                    done = True
                elif raw in cur:                       # legacy entries with no col
                    lines[ln - 1] = cur.replace(raw + "*", out, 1) \
                        if raw + "*" in cur else cur.replace(raw, out, 1)
                    done = True
                if done:
                    newtext = "\n".join(lines)
                    report["replacements"].append(
                        {"id": eid, "line": ln, "old": raw, "new": out,
                         "by_column": col is not None})
            elif not starred and not loc.get("expect_matched"):
                if not numbers_equal(raw, shown, e.get("tol")):
                    report["fails"].append(
                        f"{eid} (L{ln}): draft prints {raw!r} but "
                        f"{e.get('artifact') or 'derivation'} gives {shown!r}")
                    rec["status"] = "MISMATCH"
        if not e.get("occurrences"):
            rec["status"] = "UNCITED"
            deferred.append((e, shown, rec))
            report["entries"].append(rec)
            continue
        rec["status"] = rec["status"] or ("REPLACED" if do_replace else "VERIFIED")
        report["entries"].append(rec)

    # ---- 1b. near-miss pass, AFTER every entry has bound -------------------
    # An entry that resolves but appears nowhere is either simply uncited (fine)
    # or the manuscript prints a DIFFERENT value for that estimand — a
    # transcription/rounding error, which a naive "unbound" report would hide.
    # Running this only once binding is complete stops a number already claimed
    # by another estimand from masquerading as a near miss.
    for e, shown, rec in deferred:
        near = [nm for nm in near_misses(shown, text, e.get("hint"))
                if (nm["line"], normalize_num(nm["raw"])) not in covered]
        if not near:
            continue
        rec["status"] = "SUSPECTED_MISMATCH"
        rec["near_misses"] = near
        for nm in near:
            report["fails"].append(
                f"{e.get('id')} (L{nm['line']}): artifact gives {shown!r} but the "
                f"draft prints {nm['raw']!r} in a matching context "
                f"({nm['context'][:60]!r})")

    # ---- 2. unmatched policy ------------------------------------------------
    for n in scan_numbers(text):
        if (n["line"], normalize_num(n["raw"])) in covered:
            continue
        ctx = n["context"].lower()
        wl = None
        nn = normalize_num(n["raw"])
        if nn in GATE_LITERALS and re.search(
                r"item|budget|mmlu|perplexity|ppl|α|alpha|rank|grid|toler|seed|"
                r"configurat|replicat", ctx):
            wl = "gate/alpha-grid literal (asserted against src/v9_gate.py)"
        elif n["raw"].strip() in GRID_PCTS and GRID_CONTEXT.search(ctx):
            wl = "sparsity/density grid parameter"
        elif nn in TIER_NUMS and re.search(
                r"\d\s*[-–—]\s*\d*\s*b\b|\db\b|≤\s*\d*\.?\d*b|"
                r"at\s+\d+\.?\d*b|tier", ctx):
            wl = "model-tier / parameter-count descriptor"
        elif re.search(r"transformers|torch|peft|python|version|\bv\d", ctx) \
                and re.match(r"^\d+\.\d+(\.\d+)?$", nn):
            wl = "software version string"
        elif re.search(r"forward fix|the forward|≥|>=|at least", ctx) \
                and nn in {"1000", "1,000"}:
            wl = "proposed forward fix, not a measurement"
        elif nn == "0" and re.search(
                r"covers 0|excludes 0|≈ ?0|approximately 0|nan→0|nan-&gt;0|nan\s*→\s*0|"
                r"0 differing|score 0|→ ?0\b|to 0\b|sit at zero", ctx):
            wl = "reference zero (a CI verdict, a limit, or the nan→0 rule), not a measurement"
        elif nn in {"1", "2"} and re.search(r"bit each|bit per|one bit|of 200 items|"
                                            r"within one item|per tensor|-bit|bit sign|"
                                            r"ci flip|ci status|axis, seed", ctx):
            wl = "structural constant of the method"
        elif nn == "1%" or (n["raw"].strip() == "1%" and re.search(
                r"sparsif|densit|coordinat|sensitivity|batch-composition", ctx)):
            wl = "density / tolerance grid parameter"
        elif nn == "0" and re.search(r"cover 0|seed 0|axis, seed", ctx):
            wl = "reference zero or a seed index"
        elif re.match(r"^\d+\.\d+$", nn) and re.search(r"transformers|rebuild|→ ?4\.|5\.14", ctx):
            wl = "software version string"
        elif re.match(r"^\d+$", nn) and re.search(
                r"\b" + re.escape(nn) + r"/" + re.escape(nn) + r"\b", ctx):
            wl = "self-ratio (n of n), a determinism/unanimity count stated as a fraction"
        elif CITED_CONTEXT.search(ctx):
            wl = "number from cited prior work, not an estimand of this paper"
        elif nn in PROTOCOL_NUMS and PROTOCOL_CONTEXT.search(ctx):
            wl = "frozen protocol constant"
        for pat, why in WHITELIST_PATTERNS:
            if wl:
                break
            if re.match(pat, normalize_num(n["raw"])):
                if why == "section-ref" and not re.search(r"§|section|fig|app\.", ctx):
                    continue
                wl = why
                break
        if wl:
            report["warns"].append(f"L{n['line']}: {n['raw']} whitelisted as {wl}")
            continue
        report["unmatched"].append(
            {"line": n["line"], "raw": n["raw"], "starred": n["starred"],
             "context": n["context"].strip()})

    # ---- 3. folded greps ----------------------------------------------------
    low = text.lower()
    banned = []
    for t in BANNED_TERMS:
        for mt in re.finditer(re.escape(t), low):
            ln = low[:mt.start()].count("\n") + 1
            seg = low[max(0, mt.start() - 300):mt.start() + 300]
            if any(w in seg for w in RETRACTION_MARKERS):
                report["warns"].append(
                    f"L{ln}: banned term '{t}' present but in a retraction context")
                continue
            banned.append({"line": ln, "term": t})
    report["greps"]["banned_terms"] = banned
    if banned:
        report["fails"] += [f"L{b['line']}: banned term '{b['term']}'" for b in banned]

    todo = []
    for mt in re.finditer(r"\[TODO", text):
        ln = text[:mt.start()].count("\n") + 1
        if "limitation" not in text[max(0, mt.start() - 300):mt.start()].lower():
            todo.append(ln)
    report["greps"]["todo_outside_limitations"] = todo
    report["fails"] += [f"L{l}: [TODO outside a limitations block" for l in todo]

    # `*` residue in number contexts (only meaningful after replacement)
    residue = [{"line": n["line"], "raw": n["raw"]}
               for n in scan_numbers(newtext if do_replace else text) if n["starred"]]
    report["greps"]["star_residue"] = residue
    if do_replace and residue:
        report["fails"] += [f"L{r['line']}: '*' residue on {r['raw']} after replacement"
                            for r in residue]

    # ---- 3b. CI coherence: a point must lie inside its own interval --------
    # Replacing a point estimate but not its bounds (or vice versa) produces an
    # interval that does not contain its own point. That is worse than leaving
    # the stale value alone, and it is exactly what a per-token manifest makes
    # easy to do by accident.
    incoherent = []
    for m in re.finditer(
            r"([+\u2212-]\d+\.\d+)\s*\[\s*([+\u2212-]\d+\.\d+),\s*"
            r"([+\u2212-]\d+\.\d+)\s*\]", newtext if do_replace else text):
        try:
            pt, lo, hi = (float(normalize_num(g)) for g in m.groups())
        except (TypeError, ValueError):
            continue
        if not (lo <= pt <= hi):
            ln = (newtext if do_replace else text)[:m.start()].count("\n") + 1
            incoherent.append({"line": ln, "text": m.group(0),
                               "point": pt, "lo": lo, "hi": hi})
    report["greps"]["ci_incoherent"] = incoherent
    report["fails"] += [
        f"L{c['line']}: point outside its own CI: {c['text']}" for c in incoherent]

    # ---- 4. citation inventory (report, never a failure) --------------------
    cites = collections.Counter(re.findall(r"\[CITE:([^\]]+)\]", text))
    # a single bracket may hold several keys: "[CITE:sentencedebias; CITE:inlp]"
    keys = sorted({k.strip().removeprefix("CITE:").strip()
                   for grp in cites for k in grp.split(";")
                   if k.strip().removeprefix("CITE:").strip() not in ("", "*")})
    report["citations"] = {
        "n_occurrences": sum(cites.values()), "n_distinct": len(keys), "keys": keys,
        "companion_present": any("companion" in k for k in keys),
        "note": "citations are the one manual fill; this is an inventory, not a gate",
    }

    # ---- 5. expectation check: re-assert registered CI statuses -------------
    diff = (C.jload("results/v9/v9_diff_report.json") or {}).get("diff", {})
    exp = {}
    body = newtext if do_replace else text
    for name, d in diff.items():
        v9 = d.get("v9", "")
        st = "excludes 0" if "excludes 0" in v9 else ("covers 0" if "covers 0" in v9 else None)
        pt = d.get("v9_point")
        if pt is None:
            continue
        printed = [f"{pt:+.3f}", f"{pt:+.4f}"]
        found = any(normalize_num(p) in normalize_num(body) for p in printed)
        exp[name] = {"v9_point": pt, "registered_status": st, "point_appears_in_draft": found}
    report["expectation"] = {
        "n_registered": len(exp), "detail": exp,
        "note": ("re-asserts the registered CI status of every estimand in "
                 "v9_diff_report.json so a manifest typo cannot flip a claim"),
    }

    # ---- 6. figure CSVs -----------------------------------------------------
    for fp in sorted(glob.glob(os.path.join(figdir, "fig*_data.csv"))):
        try:
            with open(fp) as fh:
                rows = list(csv.DictReader(fh))
            report["figure_csvs"][os.path.basename(fp)] = {
                "rows": len(rows), "columns": list(rows[0].keys()) if rows else []}
        except Exception as e:                            # noqa: BLE001
            report["fails"].append(f"figure CSV unreadable: {fp}: {e}")

    # ---- 7. gate-constant assertion ----------------------------------------
    try:
        import v9_gate
        gate_ok = (v9_gate.N_ITEMS_DEFAULT == 200 and abs(v9_gate.PPL_BUDGET - 1.10) < 1e-12
                   and round(v9_gate.DMMLU_BUDGET * v9_gate.N_ITEMS_DEFAULT) == 4)
        report["greps"]["gate_constants"] = {
            "n_items": v9_gate.N_ITEMS_DEFAULT, "ppl_budget": v9_gate.PPL_BUDGET,
            "allowed_drop_items": round(v9_gate.DMMLU_BUDGET * v9_gate.N_ITEMS_DEFAULT),
            "ok": gate_ok}
        if not gate_ok:
            report["fails"].append("gate constants do not match the whitelist assertion")
    except Exception as e:                                # noqa: BLE001
        report["fails"].append(f"could not assert gate constants: {e}")

    # unmatched numbers are ship-blockers
    for u in report["unmatched"]:
        report["fails"].append(
            f"L{u['line']}: UNREGISTERED number {u['raw']!r} ({u['context'][:70]})")

    report["n_fails"] = len(report["fails"])
    report["n_unmatched"] = len(report["unmatched"])
    report["pass"] = report["n_fails"] == 0

    if do_replace and out_path:
        with open(out_path, "w") as fh:
            fh.write(newtext)
        report["output_draft"] = out_path

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", default="binary_debiaser_draft_v3.md")
    ap.add_argument("--manifest", default="audit/manifest_v10.yaml")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--out", default="binary_debiaser_draft_v3.1.md")
    ap.add_argument("--report", default="results/v10/audit_report.json")
    ap.add_argument("--max-print", type=int, default=40)
    a = ap.parse_args()

    for p in (a.draft, a.manifest):
        if not os.path.exists(p):
            print(f"MISSING: {p}")
            return 2

    rep = audit(a.draft, a.manifest, a.figures, a.replace, a.out)
    C.jdump(rep, a.report)

    print("=" * 72)
    print(f"v10 NUMBER-SOURCE AUDIT  ({'REPLACE' if a.replace else 'VERIFY'} mode)")
    print("=" * 72)
    print(f"manifest entries : {rep['n_entries']}")
    pe = rep.get("prose_edits") or []
    if pe:
        bad = [p for p in pe if p.get("status") != "applied"]
        print(f"prose edits      : {len(pe) - len(bad)}/{len(pe)} applied"
              + (f"  ** {len(bad)} FAILED **" if bad else ""))
        for p in bad:
            print(f"   X {p.get('id')}: {p.get('status')}")
    print(f"draft            : {a.draft}")
    print(f"figure CSVs      : {len(rep['figure_csvs'])} "
          f"({', '.join(rep['figure_csvs']) or 'none'})")
    print(f"citations        : {rep['citations']['n_occurrences']} occurrences, "
          f"{rep['citations']['n_distinct']} distinct keys; "
          f"companion={'present' if rep['citations']['companion_present'] else 'MISSING'}")
    print(f"                   {', '.join(rep['citations']['keys'])}")
    if rep["replacements"]:
        print(f"\nREPLACEMENTS ({len(rep['replacements'])}):")
        for r in rep["replacements"][:a.max_print]:
            print(f"  L{r['line']:>4} {r['id']:38s} {r['old']:>12} -> {r['new']}")
        if len(rep["replacements"]) > a.max_print:
            print(f"  ... {len(rep['replacements']) - a.max_print} more (see {a.report})")
    if rep["unmatched"]:
        print(f"\nUNREGISTERED NUMBERS ({len(rep['unmatched'])}) — ship-blockers:")
        for u in rep["unmatched"][:a.max_print]:
            print(f"  L{u['line']:>4} {u['raw']:>12}{'*' if u['starred'] else ' '} "
                  f"| {u['context'][:62]}")
        if len(rep["unmatched"]) > a.max_print:
            print(f"  ... {len(rep['unmatched']) - a.max_print} more (see {a.report})")
    if rep["fails"]:
        n_other = [f for f in rep["fails"] if "UNREGISTERED" not in f]
        if n_other:
            print(f"\nFAILURES ({len(n_other)} excluding unregistered):")
            for f in n_other[:a.max_print]:
                print(f"  X {f}")
    print()
    print(f"unregistered: {rep['n_unmatched']}   total failures: {rep['n_fails']}")
    print("AUDIT PASS" if rep["pass"] else "AUDIT FAIL — a number the script cannot trace does not ship.")
    print(f"report: {a.report}")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
