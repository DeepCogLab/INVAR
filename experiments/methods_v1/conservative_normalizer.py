"""Conservative current-message normalization; no model or private case lookup.

Known whole modules may be reordered. Unknown/repeated headings fall back to
the original messages. Optional strict persona-table conversion removes only
identified Markdown decoration/row numbers and retains the column-name words.
No GPU dependencies. Run --self-test or --audit-corpus /path/to/baseline.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import re
from pathlib import Path

from run_prompt_methods import sanitized_messages, split_material_question


VERSION = "INVAR_conservative_current_messages_v1"
ORDER = ("个体设定", "实验规则", "当前情况", "本次明确目标")
CORE = frozenset(ORDER[:3])
HEADING = re.compile(r"(?m)^[ \t]*【([^】\r\n]+)】")
TABLE_LINE = re.compile(r"^(?P<leading>[ \t]*)\|(?P<left>[^|\r\n]*)\|(?P<right>[^|\r\n]*)\|(?P<trailing>[ \t]*)(?P<eol>\r?\n|$)")


def digest(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class UnsafeNormalization(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise UnsafeNormalization(reason)


def eol(line):
    return "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""


def removed_segments(raw, retained_spans):
    """Identify exact deleted original spans; newly inserted list markers are separate."""
    cursor, segments = 0, []
    for start, end in sorted(retained_spans):
        require(cursor <= start <= end <= len(raw), "invalid_retention_span")
        if start > cursor:
            segments.append({"start": cursor, "end": start, "characters": start - cursor,
                             "sha256": text_hash(raw[cursor:start])})
        cursor = end
    if cursor < len(raw):
        segments.append({"start": cursor, "end": len(raw), "characters": len(raw) - cursor,
                         "sha256": text_hash(raw[cursor:])})
    return segments


def parse_blocks(material):
    matches = list(HEADING.finditer(material))
    require(matches, "no_explicit_module_headings")
    prefix = material[:matches[0].start()]
    require(not prefix.strip(), "unclassified_material_before_first_heading")
    names = [match.group(1) for match in matches]
    require(all(name in ORDER for name in names), "unknown_heading")
    require(len(set(names)) == len(names), "repeated_heading")
    require(CORE <= set(names), "missing_core_module")
    blocks = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(material)
        block = material[match.start():end]
        lines = block.splitlines(keepends=True)
        require(lines, "empty_module")
        header = lines[0]
        after_header = header[header.index("】") + 1:]
        require(not after_header.strip(), "heading_has_inline_content")
        require(header.endswith("\n"), "heading_without_line_boundary")
        require(block.endswith("\n"), "module_without_safe_trailing_line_boundary")
        blocks[names[i]] = {"raw": block, "header": header, "body": block[len(header):], "source_index": i}
    return prefix, names, blocks


def persona_table_to_list(block):
    """Return converted block + auditable structural deletions, or unchanged.

    All retained cell text is byte-for-byte identical after removing only outer
    Markdown cell padding. The words 条目 and 内容 are retained, not discarded.
    """
    lines = block["body"].splitlines(keepends=True)
    if not any(line.lstrip().startswith("|") for line in lines):
        return block["raw"], None
    active = [i for i, line in enumerate(lines) if line.strip()]
    require(active and active == list(range(active[0], active[-1] + 1)), "persona_table_has_internal_gaps")
    require(len(active) >= 3, "persona_table_too_short")
    parsed = []
    for index in active:
        match = TABLE_LINE.fullmatch(lines[index])
        require(match is not None, "unsupported_persona_table_shape")
        parsed.append((index, match))
    first = parsed[0][1]
    require(first.group("left").strip(" \t") == "条目" and first.group("right").strip(" \t") == "内容",
            "unsupported_persona_table_header")
    separator = parsed[1][1]
    require(all(re.fullmatch(r":?-{3,}:?", separator.group(name).strip()) for name in ("left", "right")),
            "unsupported_persona_table_separator")
    values, removals = [], []
    # Removed separator is purely Markdown punctuation, padding, and newline.
    separator_line = lines[parsed[1][0]]
    require(all(char in "|:- \t\r\n" for char in separator_line), "unexpected_separator_characters")
    removals.append({"body_line": parsed[1][0], "kind": "markdown_separator",
                     "removed_sha256": text_hash(separator_line), "removed_characters": len(separator_line)})
    header_labels = [first.group("left").strip(" \t"), first.group("right").strip(" \t")]
    header_raw = lines[parsed[0][0]]
    header_spans = []
    for cell_name, label in zip(("left", "right"), header_labels):
        start = first.start(cell_name) + len(first.group(cell_name)) - len(first.group(cell_name).lstrip(" \t"))
        header_spans.append((start, start + len(label)))
    if first.group("eol"):
        header_spans.append(first.span("eol"))
    header_removed = removed_segments(header_raw, header_spans)
    require(all(char in "| \t" for segment in header_removed
                for char in header_raw[segment["start"]:segment["end"]]), "unclassified_header_decoration")
    removals.append({"body_line": parsed[0][0], "kind": "markdown_header_pipes_and_padding",
                     "removed_segments": header_removed,
                     "removed_characters": sum(segment["characters"] for segment in header_removed)})
    for sequence, (index, match) in enumerate(parsed[2:], 1):
        row_number = match.group("left").strip()
        require(re.fullmatch(r"[0-9]+", row_number) is not None, "persona_row_number_not_plain_digits")
        require(int(row_number) == sequence, "persona_row_numbers_not_consecutive_from_one")
        value = match.group("right").strip(" \t")
        require(value, "empty_persona_cell")
        values.append(value)
        # Verify only pipes, the explicitly identified numeric index, and outer
        # Markdown padding are removed from the row.
        raw = lines[index]
        left_part = raw[:match.start("right")]
        right_part = raw[match.end("right"):]
        require(all(char in "|0123456789 \t" for char in left_part), "unclassified_row_prefix")
        require(all(char in "| \t\r\n" for char in right_part), "unclassified_row_suffix")
        cell_start = match.start("right") + len(match.group("right")) - len(match.group("right").lstrip(" \t"))
        spans = [(cell_start, cell_start + len(value))]
        if match.group("eol"):
            spans.append(match.span("eol"))
        deleted = removed_segments(raw, spans)
        require(all(char in "|0123456789 \t" for segment in deleted
                    for char in raw[segment["start"]:segment["end"]]), "unclassified_persona_character_deletion")
        removals.append({"body_line": index, "kind": "markdown_pipes_padding_and_row_number",
                         "row_number": row_number,
                         "retained_cell_sha256": text_hash(value),
                         "removed_segments": deleted,
                         "removed_characters": sum(segment["characters"] for segment in deleted)})
    require(values, "no_persona_rows")
    header_out = " ".join(header_labels) + first.group("eol")
    rows_out = ["- " + value + match.group("eol") for value, (_, match) in zip(values, parsed[2:])]
    transformed_body = "".join(lines[:active[0]]) + header_out + "".join(rows_out) + "".join(lines[active[-1] + 1:])
    transformed = block["header"] + transformed_body
    # This exact literal ledger includes all non-decoration header/data content.
    retained = header_labels + values
    require(all(literal in transformed for literal in retained), "persona_content_retention_failed")
    record = {
        "kind": "strict_persona_two_column_table_to_list", "column_name_words_retained": header_labels,
        "row_count": len(values), "retained_cell_texts_sha256": digest(retained),
        "header_original_sha256": text_hash(lines[parsed[0][0]]),
        "before_sha256": text_hash(block["raw"]), "after_sha256": text_hash(transformed),
        "removed_fragments": removals,
        "allowed_removals": "recognized persona Markdown pipes/separator/padding/row numbers only; header words retained",
        "added_characters": "single space joining original header labels; list markers '- '",
    }
    return transformed, record


def normalize_messages(messages, convert_persona_table=False):
    """Only currently visible messages are accepted; no case/provenance lookup."""
    original = sanitized_messages({"messages": messages})
    audit = {
        "version": VERSION, "convert_persona_table": bool(convert_persona_table),
        "input_messages_sha256": digest(original), "private_metadata_read": False,
        "status": "fallback", "fallback_reason": None, "reordered": False,
        "persona_table_converted": False, "blocks": [], "question_preserved_exactly": True,
        "retention_verified": True,
    }
    try:
        material, question, user_index = split_material_question(original)
        require(material.endswith("\n"), "question_not_at_safe_line_boundary")
        prefix, names, blocks = parse_blocks(material)
        target_order = [name for name in ORDER if name in blocks]
        transformed_blocks = {name: block["raw"] for name, block in blocks.items()}
        conversion = None
        if convert_persona_table:
            transformed_blocks["个体设定"], conversion = persona_table_to_list(blocks["个体设定"])
        transformed_material = prefix + "".join(transformed_blocks[name] for name in target_order)
        output = copy.deepcopy(original)
        output[user_index]["content"] = transformed_material + question
        require(output[user_index]["content"].endswith(question), "question_changed")
        for name in target_order:
            before, after = blocks[name]["raw"], transformed_blocks[name]
            unchanged = before == after
            require(unchanged or name == "个体设定" and conversion is not None,
                    "unexpected_module_text_change")
            audit["blocks"].append({
                "heading": name, "source_index": blocks[name]["source_index"],
                "target_index": target_order.index(name), "original_exact_sha256": text_hash(before),
                "retained_exact_sha256": text_hash(after), "verbatim_unchanged": unchanged,
                "original_characters": len(before), "retained_characters": len(after),
            })
        audit.update(
            status="applied" if output != original else "no_change",
            reordered=names != target_order, persona_table_converted=conversion is not None,
            original_order=names, target_order=target_order,
            prefix_exact_sha256=text_hash(prefix), question_exact_sha256=text_hash(question),
            exact_retained_blocks_sha256=digest([{name: transformed_blocks[name]} for name in target_order]),
            persona_conversion=conversion,
        )
    except (UnsafeNormalization, ValueError, KeyError, TypeError) as error:
        output = copy.deepcopy(original)
        audit.update(status="fallback", fallback_reason=str(error), reordered=False,
                     persona_table_converted=False, blocks=[],
                     exact_retained_blocks_sha256=None,
                     fallback_original_messages_sha256=digest(original))
    audit["output_messages_sha256"] = digest(output)
    return {"messages": output, "transformation_audit": audit}


def normalize_row(row, convert_persona_table=False):
    """Convenience boundary: select ONLY messages, never serialize the full row."""
    return normalize_messages(row["messages"], convert_persona_table=convert_persona_table)


def audit_corpus(path, out):
    path = Path(path)
    if path.is_dir():
        path = path / "inputs.jsonl"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    counters = collections.defaultdict(collections.Counter)
    fallback = collections.defaultdict(collections.Counter)
    count = 0
    with path.open(encoding="utf-8-sig") as source, (out / "transformation_audit.jsonl").open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            task_type = row.get("task_type", "unknown")
            for convert in (False, True):
                mode = "reorder_and_strict_persona_table" if convert else "reorder_only"
                normalized = normalize_row(row, convert)
                audit = normalized["transformation_audit"]
                poisoned = copy.deepcopy(row)
                poisoned.update(prompt_id="NORM_SECRET_PID_941", case_id="NORM_SECRET_CASE_942",
                                pair_id="NORM_SECRET_PAIR_943", variant_id="NORM_SECRET_VARIANT_944",
                                expected_answer="NORM_SECRET_TRUTH_945", expected_actions=["NORM_SECRET_ACTION_946"])
                poisoned_result = normalize_row(poisoned, convert)
                require(poisoned_result == normalized, "Private metadata changed the normalization")
                require("NORM_SECRET_" not in json.dumps(normalized["messages"], ensure_ascii=False),
                        "ID/truth metadata leaked into messages")
                for group in ("all", task_type):
                    key = (mode, group)
                    counters[key]["inputs"] += 1
                    counters[key][audit["status"]] += 1
                    counters[key]["reordered"] += audit["reordered"]
                    counters[key]["persona_table_converted"] += audit["persona_table_converted"]
                    counters[key]["retention_verified"] += audit["retention_verified"]
                    if audit["fallback_reason"]:
                        fallback[key][audit["fallback_reason"]] += 1
                record = {"prompt_id": row["prompt_id"], "task_type": task_type, "mode": mode,
                          "metadata_sentinel_check": "passed", "transformation_audit": audit}
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "version": VERSION, "input_file": str(path.resolve()),
        "inputs_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "unique_input_rows": count, "metadata_sentinel_check": "all_rows_passed",
        "coverage_definition": "applied is actual text change; no_change is safely parsed already normalized; fallback retains original",
        "groups": [{"mode": mode, "task_type": group, **dict(counts),
                    "fallback_reasons": dict(fallback[(mode, group)])}
                   for (mode, group), counts in sorted(counters.items())],
        "claim": "conservative known-template preprocessing baseline; no innovation or external-data generalization claim",
    }
    (out / "corpus_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def self_test():
    role = "【个体设定】\n- 你是P2，资源10点。\n\n"
    rules = "【实验规则】\n- P1自主选择；若RIGHT则P2决定。\n| 行动 | P1 | P2 |\n|---|---|---|\n| LEFT | 7 | 3 |\n| RIGHT | 1 | 9 |\n\n"
    situation = "【当前情况】\nP1尚未采取行动。\n\n"
    goal = "【本次明确目标】\n最大化本人点数，不改变他人决定权。\n\n"
    question = "【本次决策】\n只输出JSON。"
    messages = [{"role": "system", "content": "按原题作答。"},
                {"role": "user", "content": goal + situation + rules + role + question}]
    result = normalize_messages(messages)
    require(result["messages"][1]["content"] == role + rules + situation + goal + question,
            "Whole-module reorder or explicit-goal preservation failed")
    require(result["messages"][0] == messages[0], "System message changed")
    require(result["transformation_audit"]["blocks"][1]["original_exact_sha256"] == text_hash(rules),
            "Rules table exact hash changed")
    require(messages[1]["content"] == goal + situation + rules + role + question, "Input mutated")
    for inserted in ("【未知模块】\n未知事实必须保留。\n", "【实验规则】\n重复规则不得丢弃。\n"):
        strange = copy.deepcopy(messages)
        strange[1]["content"] = inserted + strange[1]["content"]
        fallback = normalize_messages(strange)
        require(fallback["transformation_audit"]["status"] == "fallback" and fallback["messages"] == strange,
                "Unknown/repeated heading was not retained exactly")
    table_role = "【个体设定】\n| 条目 | 内容 |\n|---|---|\n| 1 | 参与者编号：Q7。 |\n| 2 | 你有10点，不能决定Q4的行动。 |\n\n"
    table_input = [{"role": "user", "content": rules + table_role + situation + goal + question}]
    converted = normalize_messages(table_input, True)
    content = converted["messages"][0]["content"]
    require("条目 内容\n- 参与者编号：Q7。\n- 你有10点，不能决定Q4的行动。\n" in content,
            "Strict persona table content/header labels not retained")
    require(rules in content and goal in content and content.endswith(question),
            "Rule table, goal, or question altered by persona conversion")
    require(converted["transformation_audit"]["persona_table_converted"], "Persona table was not converted")
    require(normalize_messages(converted["messages"], True)["messages"] == converted["messages"], "Not idempotent")
    for replacement in ("| 序号 | 内容 |", "| 条目 | 内容 | 额外 |", "| 1a | 参与者编号：Q7。 |"):
        bad = copy.deepcopy(table_input)
        target = "| 1 | 参与者编号：Q7。 |" if "1a" in replacement else "| 条目 | 内容 |"
        bad[0]["content"] = bad[0]["content"].replace(target, replacement)
        rejected = normalize_messages(bad, True)
        require(rejected["transformation_audit"]["status"] == "fallback" and rejected["messages"] == bad,
                "Unsupported persona table was partially normalized")
    private = {"messages": table_input, "prompt_id": "NORM_SECRET_PID", "pair_id": "NORM_SECRET_PAIR",
               "variant_id": "NORM_SECRET_VARIANT", "expected_answer": "NORM_SECRET_TRUTH"}
    require("NORM_SECRET_" not in json.dumps(normalize_row(private, True)["messages"]), "Metadata leaked")
    role_control = [{"role": "user", "content": "你是P2。\n【事实理解检查】\n编号是什么？"}]
    require(normalize_messages(role_control)["messages"] == role_control, "Non-modular role control changed")
    print(json.dumps({"self_test": "passed", "checks": ["explicit_goal_retained", "full_rules_and_payoff_table",
                      "unknown_heading_fallback", "repeated_heading_fallback", "strict_persona_table",
                      "header_words_retained", "unknown_table_fallback", "no_metadata_leak",
                      "idempotence", "input_not_mutated", "exact_retained_block_hashes"]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-corpus", type=Path, help="Baseline directory or inputs.jsonl")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "normalizer_audit")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.audit_corpus:
        audit_corpus(args.audit_corpus, args.out)
    else:
        parser.error("Use --self-test or --audit-corpus")


if __name__ == "__main__":
    main()
