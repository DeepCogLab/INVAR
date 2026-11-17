"""Audited PIQ572 single-node adaptations. No participant answers are used.

Only the 18 unique protocol payoff tuples are read from allresults.tab.
Figure 1 of van Leeuwen & Alger (2024) and the replication README define
the branch semantics. See audit/piq_audit.md for deviations from the source.
"""
from __future__ import annotations

import csv
import copy
import hashlib
import json
from pathlib import Path


SOURCE_URL = "https://doi.org/10.7910/DVN/PIQ572"
PAPER_URL = "https://www.journals.uchicago.edu/doi/full/10.1086/732125"
EXPECTED = {
    1: ("SPD", 90, 45, 15, 10), 2: ("SPD", 90, 55, 20, 10),
    3: ("SPD", 80, 65, 25, 20), 4: ("SPD", 90, 65, 25, 10),
    5: ("SPD", 80, 75, 30, 20), 6: ("SPD", 90, 75, 30, 10),
    7: ("TG", 80, 50, 30, 20), 8: ("TG", 90, 50, 30, 10),
    9: ("TG", 80, 60, 30, 20), 10: ("TG", 90, 60, 30, 10),
    11: ("TG", 80, 70, 30, 20), 12: ("TG", 90, 70, 30, 10),
    13: ("UG", 60, 50, 40, 10), 14: ("UG", 65, 50, 35, 10),
    15: ("UG", 70, 50, 30, 10), 16: ("UG", 75, 50, 25, 10),
    17: ("UG", 80, 50, 20, 10), 18: ("UG", 85, 50, 15, 10),
}


def _source_dir(root: Path) -> Path:
    for base in (root, root.parent):
        p = base / "research" / "evidence" / "PIQ572"
        if p.is_dir():
            return p
    raise FileNotFoundError("research/evidence/PIQ572 is required")


def _protocols(root: Path) -> tuple[dict, Path]:
    source = _source_dir(root)
    meta = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    license_name = meta["data"]["latestVersion"]["license"]["name"]
    if license_name != "CC0 1.0":
        raise ValueError(f"Source license changed: {license_name}")
    protocols = {}
    with (source / "allresults.tab").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            # Deliberately do not access id, keuzea, keuzebl, keuzebr or beliefs.
            no = int(float(row["game_no"]))
            values = (row["game_type"], *(int(float(row[k])) for k in ("t", "r", "p", "s")))
            if no in protocols and protocols[no] != values:
                raise ValueError(f"Nonconstant protocol values for game {no}")
            protocols[no] = values
    if protocols != EXPECTED:
        raise ValueError("Source game tuples do not match audited Table 1")
    return protocols, source


def _leaf(p1: int, p2: int, focal: str) -> dict:
    return {"P1": p1, "P2": p2, "self": p1 if focal == "P1" else p2,
            "other": p2 if focal == "P1" else p1}


def _common(focal: str) -> dict:
    return {
        "focal_id": focal,
        "entities": {"P1": "第一行动者", "P2": "第二行动者"},
        "role_facts": [f"你是{focal}。", f"你的角色是{'第一' if focal == 'P1' else '第二'}行动者。"],
        "rules": ["P1与P2匿名配对，彼此不知道对方的姓名。",
                  "两人都知道本题的规则和各分支收益；收益以点数计，原实验1点对应0.17欧元。"],
        "situation": [],
        "question": "请作出你在当前情况下的一次选择。",
        "action_space": {"type": "categorical", "values": []},
        "action_labels": {},
        "semantic_action_map": {},
        "outcome": {"kind": "probability", "target_action": None, "denominator": 1},
        "payoffs": {},
    }


def _actions(case: dict, values: list[str], target: str, meanings: dict) -> None:
    case["action_space"]["values"] = values
    case["action_labels"] = {v: f"选择{v}" for v in values}
    case["outcome"]["target_action"] = target
    case["semantic_action_map"] = meanings


def _tree(kind: str, t: int, r: int, p: int, s: int) -> dict:
    if kind == "SPD":
        return {"LEFT": {"WEST": {"P1": t, "P2": s}, "SOUTH": {"P1": p, "P2": p}},
                "RIGHT": {"NORTH": {"P1": r, "P2": r}, "EAST": {"P1": s, "P2": t}}}
    if kind == "TG":
        return {"LEFT": {"P1": p, "P2": p},
                "RIGHT": {"NORTH": {"P1": r, "P2": r}, "EAST": {"P1": s, "P2": t}}}
    return {"LEFT": {"P1": r, "P2": r},
            "RIGHT": {"NORTH": {"P1": s, "P2": s}, "EAST": {"P1": t, "P2": p}}}


def _spd(t: int, r: int, p: int, s: int, focal: str, condition: int) -> dict:
    c = _common(focal)
    c["source_game_tree"] = _tree("SPD", t, r, p, s)
    if focal == "P1":
        _actions(c, ["LEFT", "RIGHT"], "RIGHT", {"LEFT": "defect", "RIGHT": "cooperate"})
        if condition == 0:
            c["rules"] += ["P1先选择LEFT或RIGHT。随后P2知道P1的选择，并自主选择自己的行动。",
                f"P1选LEFT后：P2选WEST时P1得{t}点、P2得{s}点；P2选SOUTH时双方各得{p}点。",
                f"P1选RIGHT后：P2选NORTH时双方各得{r}点；P2选EAST时P1得{s}点、P2得{t}点。"]
            c["payoffs"] = {"LEFT": {"WEST": _leaf(t, s, focal), "SOUTH": _leaf(p, p, focal)},
                            "RIGHT": {"NORTH": _leaf(r, r, focal), "EAST": _leaf(s, t, focal)}}
        else:
            c["rules"] += ["P1先选择LEFT或RIGHT。P2的后续行动按事前公布的固定规则执行，P2不能更改。",
                f"P1选LEFT后，固定执行SOUTH，双方各得{p}点。",
                f"P1选RIGHT后，固定执行NORTH，双方各得{r}点。"]
            c["payoffs"] = {"LEFT": _leaf(p, p, focal), "RIGHT": _leaf(r, r, focal)}
        c["situation"] = ["现在轮到P1选择；P1尚未采取行动。"]
        c["relation_alt"] = {"section": "situation", "index": 0,
                             "text": "P1还没有采取行动，当前的选择由P1作出。"}
    else:
        # Source node labels differ; align neutral output labels before any T/I manipulation.
        _actions(c, ["X", "Y"], "X", {"X": "cooperate", "Y": "defect"})
        c["rules"] += ["P1先自主选择LEFT或RIGHT；P2看到P1的选择后，选择X或Y。",
            f"若P1选LEFT：P2选X时P1得{t}点、P2得{s}点；P2选Y时双方各得{p}点。",
            f"若P1选RIGHT：P2选X时双方各得{r}点；P2选Y时P1得{s}点、P2得{t}点。"]
        previous = "LEFT" if condition == 0 else "RIGHT"
        c["situation"] = [f"P1已经自主选择{previous}，这一选择已经确定。", "现在轮到P2选择。"]
        c["payoffs"] = {"X": _leaf(t, s, focal), "Y": _leaf(p, p, focal)} if condition == 0 else {
            "X": _leaf(r, r, focal), "Y": _leaf(s, t, focal)}
        c["source_action_map"] = {"X": "WEST", "Y": "SOUTH"} if condition == 0 else {"X": "NORTH", "Y": "EAST"}
        c["relation_alt"] = {"section": "situation", "index": 0,
                             "text": f"当前已确定的{previous}是P1自主作出的选择。"}
    return c


def _tg(t: int, r: int, p: int, s: int, focal: str, condition: int) -> dict:
    c = _common(focal)
    c["source_game_tree"] = _tree("TG", t, r, p, s)
    if focal == "P1":
        _actions(c, ["LEFT", "RIGHT"], "RIGHT", {"LEFT": "not_invest", "RIGHT": "invest"})
        c["rules"] += [f"P1先选择LEFT或RIGHT。选LEFT则本题结束，双方各得{p}点。"]
        if condition == 0:
            c["rules"] += ["P1选RIGHT后，P2知道P1的选择，并自主选择NORTH或EAST。",
                f"P2选NORTH时双方各得{r}点；选EAST时P1得{s}点、P2得{t}点。"]
            c["payoffs"] = {"LEFT": _leaf(p, p, focal),
                            "RIGHT": {"NORTH": _leaf(r, r, focal), "EAST": _leaf(s, t, focal)}}
        else:
            c["rules"] += [f"P1选RIGHT后，按事前公布的固定规则直接执行NORTH，双方各得{r}点；P2不能更改。"]
            c["payoffs"] = {"LEFT": _leaf(p, p, focal), "RIGHT": _leaf(r, r, focal)}
        c["situation"] = ["现在轮到P1选择；P1尚未采取行动。"]
        c["relation_alt"] = {"section": "situation", "index": 0,
                             "text": "P1还没有采取行动，当前的选择由P1作出。"}
    else:
        _actions(c, ["NORTH", "EAST"], "NORTH", {"NORTH": "give_back", "EAST": "keep"})
        if condition == 0:
            c["rules"] += [f"P1可以自主选择LEFT或RIGHT：选LEFT则本题结束，双方各得{p}点；选RIGHT则由P2选择NORTH或EAST。"]
            c["situation"] = ["P1已经自主选择RIGHT，P1原本也可以选择LEFT。", "现在轮到P2选择。"]
            c["relation_alt"] = {"section": "situation", "index": 0,
                                 "text": "当前的RIGHT由P1自主选择；当时LEFT也可供P1选择。"}
        else:
            c["rules"] += ["实验程序直接将本题设为RIGHT后的决策位置；P1没有选择LEFT或RIGHT的权利。"]
            c["situation"] = ["当前的RIGHT由程序指定，并非P1主动选择。", "现在轮到P2选择。"]
            c["relation_alt"] = {"section": "situation", "index": 0,
                                 "text": "P1未主动选择RIGHT；RIGHT是程序指定的当前情况。"}
        c["rules"] += [f"P2可自主选择NORTH或EAST：选NORTH时双方各得{r}点；选EAST时P1得{s}点、P2得{t}点。"]
        c["payoffs"] = {"NORTH": _leaf(r, r, focal), "EAST": _leaf(s, t, focal)}
    return c


def _ug(t: int, r: int, p: int, s: int, focal: str, condition: int) -> dict:
    c = _common(focal)
    c["source_game_tree"] = _tree("UG", t, r, p, s)
    if focal == "P1":
        _actions(c, ["LEFT", "RIGHT"], "LEFT", {"LEFT": "equal", "RIGHT": "unequal"})
        c["rules"] += [f"P1先选择LEFT或RIGHT。选LEFT则本题结束，双方各得{r}点。"]
        if condition == 0:
            c["rules"] += ["P1选RIGHT后，P2知道P1的选择，并自主选择NORTH或EAST。",
                f"P2选NORTH时双方各得{s}点；选EAST时P1得{t}点、P2得{p}点。"]
            c["payoffs"] = {"LEFT": _leaf(r, r, focal),
                            "RIGHT": {"NORTH": _leaf(s, s, focal), "EAST": _leaf(t, p, focal)}}
        else:
            c["rules"] += [f"P1选RIGHT后，直接执行EAST，P1得{t}点、P2得{p}点；P2没有选择NORTH的权利。"]
            c["payoffs"] = {"LEFT": _leaf(r, r, focal), "RIGHT": _leaf(t, p, focal)}
        c["situation"] = ["现在轮到P1选择；P1尚未采取行动。"]
        c["relation_alt"] = {"section": "situation", "index": 0,
                             "text": "P1还没有采取行动，当前的选择由P1作出。"}
    else:
        _actions(c, ["NORTH", "EAST"], "EAST", {"NORTH": "reject", "EAST": "accept"})
        if condition == 0:
            c["rules"] += [f"P1可以自主选择LEFT或RIGHT：选LEFT则本题结束，双方各得{r}点；选RIGHT则由P2选择NORTH或EAST。"]
            c["situation"] = ["P1已经自主选择RIGHT，P1原本也可以选择LEFT。", "现在轮到P2选择。"]
            c["relation_alt"] = {"section": "situation", "index": 0,
                                 "text": "当前的RIGHT由P1自主选择；当时LEFT也可供P1选择。"}
        else:
            c["rules"] += ["实验程序直接将本题设为RIGHT后的决策位置；P1没有选择LEFT或RIGHT的权利。"]
            c["situation"] = ["当前的RIGHT由程序指定，并非P1主动选择。", "现在轮到P2选择。"]
            c["relation_alt"] = {"section": "situation", "index": 0,
                                 "text": "P1未主动选择RIGHT；RIGHT是程序指定的当前情况。"}
        c["rules"] += [f"P2可自主选择NORTH或EAST：选NORTH时双方各得{s}点；选EAST时P1得{t}点、P2得{p}点。"]
        c["payoffs"] = {"NORTH": _leaf(s, s, focal), "EAST": _leaf(t, p, focal)}
    return c


def build_pairs(root: Path) -> list[dict]:
    protocols, source = _protocols(Path(root))
    source_hash = hashlib.sha256((source / "allresults.tab").read_bytes()).hexdigest()
    pairs = []
    for no, (kind, t, r, p, s) in sorted(protocols.items()):
        for focal in ("P1", "P2"):
            builder = {"SPD": _spd, "TG": _tg, "UG": _ug}[kind]
            original = kind == "SPD" and focal == "P2"
            descriptions = {
                ("SPD", "P1"): ("institution", "second_mover_rule", "P2自主回应改为事前公布的固定回应：LEFT后SOUTH、RIGHT后NORTH。"),
                ("SPD", "P2"): ("other_behavior", "first_mover_action", "P1已选LEFT改为已选RIGHT；按原树同步改变可得收益。"),
                ("TG", "P1"): ("institution", "return_decision_right", "RIGHT后P2自主选择改为固定执行NORTH。"),
                ("TG", "P2"): ("institution", "first_mover_agency", "相同RIGHT节点由P1自主选择改为程序指定，P1没有替代选择。"),
                ("UG", "P1"): ("institution", "rejection_right", "RIGHT后P2可选NORTH/EAST改为不能选NORTH、直接执行EAST。"),
                ("UG", "P2"): ("institution", "first_mover_agency", "相同RIGHT节点由P1自主选择改为程序指定，P1没有替代选择。"),
            }
            intervention_kind, factor, description = descriptions[kind, focal]
            adaptations = [
                "以原实验博弈树和实际收益参数编写中文文本，不使用任何参与者选择作为题目内容或正确答案。",
                "原实验要求每人填写所有角色的策略并预测他人行为；本题固定一个角色和决策位置，只作一次选择。",
                "原实验18题中随机选1题支付行为收益，另1题支付信念准确性；行为题随机配对并等概率分配角色。本题仅呈现所给角色与节点的条件收益，不保留跨题付款抽签、信念任务及角色不确定性。",
                "原研究没有公开给出参与者的完整Persona；仅使用匿名角色设定。",
            ]
            if kind == "SPD" and focal == "P2":
                adaptations.append("两个节点的源标签分别为WEST/SOUTH和NORTH/EAST；统一改为中性X/Y，消除干预与输出标签变更混杂，原映射另存。")
            if not original:
                adaptations.append("case1为研究者新增规则；原数据没有对应的真人干预组。")
            pair = {
                "pair_id": f"PIQ572_{kind}_{no:02d}_{focal}", "source_id": "PIQ572",
                "task": {"SPD": "sequential_prisoners_dilemma", "TG": "trust_game", "UG": "ultimatum_game"}[kind],
                "role": {("SPD", "P1"): "first_mover", ("SPD", "P2"): "second_mover", ("TG", "P1"): "trustor", ("TG", "P2"): "trustee", ("UG", "P1"): "proposer", ("UG", "P2"): "responder"}[kind, focal],
                "analysis_tier": "source_conditions" if original else "researcher_extension",
                "intervention": {"kind": intervention_kind, "factor": factor,
                    "origin": "original_conditions" if original else "researcher_constructed",
                    "description": description, "expected_direction": None},
                "provenance": {"file": "research/evidence/PIQ572/allresults.tab",
                    "locator": f"game_no={no}; unique protocol columns game_type,t,r,p,s; paper Figure 1 and Table 1; replication README section 0",
                    "url": SOURCE_URL, "paper_url": PAPER_URL, "sha256": source_hash,
                    "source_license": "CC0-1.0", "article_license": "CC-BY-NC-4.0",
                    "adaptations": adaptations,
                    "source_parameters": {"game_no": no, "game_type": kind, "t": t, "r": r, "p": p, "s": s}},
                "case0": builder(t, r, p, s, focal, 0),
                "case1": builder(t, r, p, s, focal, 1),
                "human_reference_status": "not_a_scoring_label_strategy_method_differs",
                "source_cluster": f"PIQ572_{kind}", "game_cluster": f"PIQ572_game_{no:02d}",
            }
            for idx in (0, 1):
                case = pair[f"case{idx}"]
                case["case_id"] = f"{pair['pair_id']}_C{idx}"
                case["source_action_map"] = case.get("source_action_map", {v: v for v in case["action_space"]["values"]})
                assert case["outcome"]["target_action"] in case["action_space"]["values"]
                assert set(case["payoffs"]) == set(case["action_space"]["values"])
            if pair["case0"]["action_space"] != pair["case1"]["action_space"]:
                raise AssertionError("Output actions must be invariant across intervention")
            pairs.append(pair)
    assert len(pairs) == 36
    assert sum(p["analysis_tier"] == "source_conditions" for p in pairs) == 6
    # Original payoff comparisons: adjacent pairs only, selected before model runs.
    base_pairs = {p["pair_id"]: p for p in pairs}
    comparisons = [("TG", a, b) for a, b in ((7, 9), (9, 11), (8, 10), (10, 12))]
    comparisons += [("UG", a, a + 1) for a in range(13, 18)]
    for kind, before, after in comparisons:
        _, t0, r0, p0, s0 = protocols[before]
        _, t1, r1, p1, s1 = protocols[after]
        if kind == "TG":
            assert (t0, p0, s0) == (t1, p1, s1) and r1 == r0 + 10
            factor = "mutual_return_payoff_R"
            description = (f"RIGHT→NORTH分支中P1与P2的收益分别由{r0}点增至{r1}点；"
                           f"T={t0}、P={p0}、S={s0}及所有行动权保持不变。")
        else:
            assert (r0, s0) == (r1, s1) and t0 + p0 == t1 + p1 == 100
            assert t1 == t0 + 5 and p1 == p0 - 5
            factor = "unequal_allocation_T_and_P"
            description = (f"RIGHT→EAST分支的分配由P1得{t0}点/P2得{p0}点改为P1得{t1}点/P2得{p1}点；"
                           f"双方总额100点、LEFT分支双方各{r0}点、RIGHT→NORTH分支双方各{s0}点及行动权不变。")
        for focal in ("P1", "P2"):
            old = base_pairs[f"PIQ572_{kind}_{before:02d}_{focal}"]
            new = base_pairs[f"PIQ572_{kind}_{after:02d}_{focal}"]
            pair = copy.deepcopy(old)
            pair["pair_id"] = f"PIQ572_{kind}_{before:02d}_TO_{after:02d}_{focal}"
            pair["analysis_tier"] = "source_conditions"
            pair["intervention"] = {"kind": "payoff", "factor": factor,
                "origin": "original_conditions", "description": description,
                "expected_direction": None}
            pair["case0"] = copy.deepcopy(old["case0"])
            pair["case1"] = copy.deepcopy(new["case0"])
            pair["provenance"]["locator"] = (f"game_no={before} and game_no={after}; "
                "paper Figure 1 and Table 1; replication README section 0")
            pair["provenance"]["source_parameters"] = {
                "before": old["provenance"]["source_parameters"],
                "after": new["provenance"]["source_parameters"]}
            pair["provenance"]["adaptations"] = [a for a in old["provenance"]["adaptations"]
                if not a.startswith("case1为")]
            pair["provenance"]["adaptations"].append(
                "两侧收益均为原研究真实设置，当前相邻条件配对由INVAR事前指定；不作单一心理机制识别或个体人类选择评分。")
            pair["game_cluster"] = f"PIQ572_{kind}_payoff_series"
            pair["source_game_ids"] = [before, after]
            pair["split_group"] = f"PIQ572_{kind}"
            for idx in (0, 1):
                pair[f"case{idx}"]["case_id"] = f"{pair['pair_id']}_C{idx}"
            assert pair["case0"]["action_space"] == pair["case1"]["action_space"]
            assert pair["case0"]["outcome"] == pair["case1"]["outcome"]
            assert pair["case0"]["role_facts"] == pair["case1"]["role_facts"]
            assert pair["case0"]["situation"] == pair["case1"]["situation"]
            pairs.append(pair)
    for pair in pairs:
        pair.setdefault("split_group", f"PIQ572_{pair['provenance']['source_parameters'].get('game_type', '')}")
        if "source_game_ids" not in pair:
            pair["source_game_ids"] = [pair["provenance"]["source_parameters"]["game_no"]]
    assert len(pairs) == 54
    assert sum(p["analysis_tier"] == "source_conditions" for p in pairs) == 24
    return pairs


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    result = build_pairs(root)
    print(json.dumps({"pairs": len(result), "source_conditions": 24, "researcher_extension": 30}, ensure_ascii=False))
