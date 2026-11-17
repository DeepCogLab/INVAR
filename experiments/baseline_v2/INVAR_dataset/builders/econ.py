"""Source-grounded Econographics adaptations; no human answer labels.

The eight comparisons are researcher extensions.  Text and payment rules are
checked against the locally archived design document, not inferred from answers.
Only Python's standard library is used.  ``root`` is the project directory.
"""

from copy import deepcopy
from pathlib import Path


SOURCE_FILE = "research/evidence/IGVOFO/WEP Robust Econographics Survey Document FINAL.pdf"
TEXT_FILE = SOURCE_FILE[:-4] + ".txt"
SOURCE_URL = "https://dataverse.harvard.edu/api/access/datafile/6573547"
MENU = list(range(0, 6001, 1000))


def _source(root: Path) -> None:
    path = Path(root) / TEXT_FILE
    text = path.read_text(encoding="utf-8")
    for phrase in (
        "III.I Altruism and Trust",
        "Whatever amount the",
        "other taker sends back to you will be doubled again",
        "text box that allows between 0 and 12,000",
        "then with 50% probability",
    ):
        if phrase not in text:
            raise ValueError(f"Econographics source check failed: {phrase}")


def _provenance(locator: str, adaptations: list[str]) -> dict:
    return {
        "file": SOURCE_FILE,
        "locator": locator,
        "url": SOURCE_URL,
        "adaptations": [
            "将英文调查说明适配为中文单次角色决策；P1、P2为本研究的无身份含义编号。",
            "条件化为本题已被选作结算，仅计算本题资源；不运行整份问卷的随机抽题及跨受访者真实付款。",
            *adaptations,
        ],
        "human_label_status": "disabled_adapted_task_and_researcher_intervention",
        "original_reference": "https://doi.org/10.1086/723044",
        "dataset_version": "1.0",
        "dataset_license": "CC0-1.0",
    }


def _pair(pair_id: str, task: str, role: str, factor: str,
          description: str, provenance: dict, case0: dict, case1: dict,
          kind: str = "institution") -> dict:
    return {
        "pair_id": pair_id,
        "source_id": "IGVOFO",
        "task": task,
        "role": role,
        "analysis_tier": "researcher_extension",
        "intervention": {
            "kind": kind,
            "factor": factor,
            "origin": "researcher_constructed",
            "description": description,
            "expected_direction": None,
        },
        "provenance": provenance,
        "case0": case0,
        "case1": case1,
    }


def _menu_space() -> dict:
    return {"type": "integer", "min": 0, "max": 6000,
            "step": 1000, "values": MENU.copy()}


def _dg() -> dict:
    role_facts = [
        "你的编号是P1，本次担任分配者。",
        "你可决定如何分配本题的6000点。",
        "P2是这次分配的接收者。",
    ]
    rules = [
        "本题已确定用于结算，只计算本题列出的点数。",
        "付款角色已确定为分配者，你本次实际获得的点数按你作出的分配结算。",
        "你可以给P2的点数为0、1000、2000、3000、4000、5000或6000；只能从这七项中选择。",
        "记你选择给P2的点数为g。P2不能拒绝或修改你的选择。",
        "这次分配直接执行：你得到6000−g点，P2得到g点；点数不会翻倍。",
    ]
    case0 = {
        "role_facts": role_facts,
        "rules": rules,
        "situation": ["现在由P1决定给P2多少点。", "P1尚未作出本次分配选择。"],
        "question": "你选择给P2多少点？请输出合法菜单中的点数。",
        "action_space": _menu_space(),
        "outcome": {"kind": "normalized_amount", "denominator": 6000},
        "payoffs": {
            "formula": {"P1": "6000-action", "P2": "action"},
            "by_action": {str(g): {"P1": 6000-g, "P2": g} for g in MENU},
            "scope": "conditional_on_own_allocation_payment_branch",
        },
        "focal_id": "P1",
        "entities": {"P1": "分配者", "P2": "接收者"},
        "relation_alt": {
            "section": "situation", "index": 0,
            "text": "现在P2将得到多少点，由P1决定。",
        },
    }
    case1 = deepcopy(case0)
    case1["rules"][4] = (
        "已公布的执行规则是：若g少于1200点，本次分配取消，你和P2都得到0点；"
        "若g至少为1200点，则你得到6000−g点，P2得到g点。P2不参与这项自动规则的执行。"
    )
    case1["payoffs"] = {
        "formula": {"P1": "0 if action<1200 else 6000-action",
                    "P2": "0 if action<1200 else action"},
        "by_action": {str(g): {"P1": 0 if g < 1200 else 6000-g,
                                "P2": 0 if g < 1200 else g} for g in MENU},
        "parameters": {"threshold": 1200},
        "scope": "conditional_on_own_allocation_payment_branch",
    }
    return _pair(
        "IGVOFO_DG_THRESHOLD", "dictator", "allocator",
        "automatic_minimum_allocation_threshold",
        "无最低分配要求→给P2少于1200点时双方本题收益归零；这是自动制度，不是接收者拒绝权。",
        _provenance(
            "III.I Q1, printed p.21; payment instructions printed p.23",
            [
                "原DG付款为50%按本人保留额、50%按另一个受访者给自己的金额；本题显式固定为本人分配支付分支。",
                "增加1200点自动最低分配阈值；原问卷没有这个处理组，七项选择菜单保留。",
                "1200点落在1000和2000两项之间；只有0和1000会触发归零，不把阈值当成既有人类规律。",
            ],
        ), case0, case1,
    )


def _tg_sender() -> dict:
    case0 = {
        "role_facts": [
            "你的编号是P1，本次担任先送出点数的人。",
            "你有6000点可在本次互动中保留或送出。",
            "P2是收到点数并返还的人。",
        ],
        "rules": [
            "本题已确定用于结算，只计算本次互动列出的点数。",
            "你送出s点后，自己先保留6000−s点；P2收到2s点。",
            "你只能选择送出0、1000、2000、3000、4000、5000或6000点。",
            "P2已自主提交一份条件返还计划，分别确定收到不同金额时返还多少。你不知道这份计划，执行时按你的送出额选取相应一项。",
            "记P2实际返还的点数为r，r可从0到2s；返还的点数再次翻倍，你实际收到2r点。若s为0，P2没有可返还点数，r为0。",
            "最后你得到6000−s＋2r点，P2得到2s−r点。",
        ],
        "situation": ["P1现在需要决定向P2送出多少点。", "P1尚不知道P2会返还多少点。"],
        "question": "你选择送出多少点给P2？请输出合法菜单中的点数。",
        "action_space": _menu_space(),
        "outcome": {"kind": "normalized_amount", "denominator": 6000},
        "payoffs": {
            "formula": {"P1": "6000-action+2*r", "P2": "2*action-r"},
            "unknowns": {"r": "P2预先自主选择的返还额，0<=r<=2*action；action=0时r=0"},
        },
        "focal_id": "P1", "entities": {"P1": "送出者", "P2": "返还者"},
        "relation_alt": {
            "section": "situation", "index": 0,
            "text": "P2将收到的转移由P1现在决定；P1需要选择送出的点数。",
        },
    }
    case1 = deepcopy(case0)
    case1["rules"][3] = (
        "已公开的制度固定返还额为P2收到点数的50%，由程序自动执行；"
        "P2不能自主选择其他返还额。你作选择前已经知道此规则。"
    )
    case1["rules"][4] = "P2收到2s点后，程序返还其中的s点；返还再次翻倍，你实际收到2s点。s为0时返还也为0。"
    case1["rules"][5] = "最后你得到6000＋s点，P2得到s点。"
    case1["situation"][1] = "P1已经知道P2将按公开规则返还收到点数的50%。"
    case1["payoffs"] = {
        "formula": {"P1": "6000+action", "P2": "action"},
        "by_action": {str(s): {"P1": 6000+s, "P2": s} for s in MENU},
        "parameters": {"return_fraction_of_received": 0.5,
                       "outgoing_multiplier": 2, "return_multiplier": 2},
    }
    return _pair(
        "IGVOFO_TG_SENDER_FIXED_RETURN", "trust", "sender",
        "return_determination_rule",
        "自主且对P1未知的条件返还计划→公开并自动执行的50%返还制度；自主性和不确定性共同改变，不能单独识别信任心理。",
        _provenance(
            "III.I Q2-Q3 and payment implementation, printed pp.22-23; IV.B printed p.44",
            [
                "保留原有七项送出菜单、送出翻倍与返还再次翻倍；基线使用匹配对象预先自主提交的计划。",
                "公开固定返还收到额50%为本研究新增制度；不是原数据处理组，也不是对真人平均返还的估计。",
                "只描述本次匹配，不为角色补充年龄、性别、职业或偏好。",
            ],
        ), case0, case1,
    )


def _tg_receiver(s: int) -> dict:
    received = 2*s
    kept_sender = 6000-s
    voluntary = f"P1原本可从0、1000、2000、3000、4000、5000、6000点中自主选择送出额；P1选择送出{s}点给P2。"
    forced = f"P1没有选择送出额的权利；程序规定从P1的6000点中转出{s}点给P2，P1不能改动或拒绝这次转移。"
    case0 = {
        "role_facts": [
            "你的编号是P2，本次担任收到点数后决定返还的人。",
            f"你在本次互动中已收到{received}点，可保留或返还。",
            "P1是原先拥有6000点的另一位参与者。",
        ],
        "rules": [
            "本题已确定用于结算，只计算本次互动列出的点数；本次收到额已给定，不再抽取其他收到额。",
            "从P1转出的点数会翻倍后交给P2。",
            f"你可以从0到{received}点之间选择返还额r；本题以整点作答，0和{received}都可以选择。",
            "你的返还再次翻倍后给P1；P1不能拒绝或改动你选择的返还额。",
            f"最后你得到{received}−r点，P1得到{kept_sender}＋2r点。",
        ],
        "situation": [
            voluntary,
            f"这{s}点已翻倍为{received}点给P2；P1暂时保留{kept_sender}点。",
            "现在由P2自主决定返还多少点。",
        ],
        "question": f"你选择返还多少点给P1？请输出0至{received}之间的一个整数。",
        "action_space": {"type": "integer", "min": 0, "max": received, "step": 1},
        "outcome": {"kind": "normalized_amount", "denominator": received},
        "payoffs": {
            "formula": {"P1": f"{kept_sender}+2*action", "P2": f"{received}-action"},
            "parameters": {"original_sent": s, "received": received,
                           "sender_kept": kept_sender, "return_multiplier": 2},
        },
        "focal_id": "P2", "entities": {"P1": "转出资源者", "P2": "返还者"},
        "relation_alt": {
            "section": "situation", "index": 1,
            "text": f"P2收到由转出的{s}点翻倍而成的{received}点；P1暂时留下{kept_sender}点。",
        },
    }
    case1 = deepcopy(case0)
    case1["situation"][0] = forced
    return _pair(
        f"IGVOFO_TG_RECEIVER_SENT{s}_AGENCY", "trust", "receiver",
        "source_of_initial_transfer",
        f"P1自主送出{s}点→程序强制转出相同{s}点；收到额、P1保留额、返还权限与收益公式相同。",
        _provenance(
            f"III.I Q3{chr(ord('a') + s//1000-1)}, printed pp.22-23; payment implementation printed p.23",
            [
                "原Q3要求事前填六格条件返还计划，本题固定其中一格并改写为已经发生的一次互动；不等同原策略表测量。",
                "原文本框只明确0至收到额的范围，未明确数值精度；本研究为一致接口显式规定整点，不能声称是原题限制。",
                "原Q3付款时从a-f六格等概率抽一格；本题显式固定当前收到额，不运行该抽格步骤。",
                "由程序强制转出是本研究新增条件；改变行动来源但不声称完全孤立了心理意图或互惠机制。",
            ],
        ), case0, case1, kind="institution",
    )


def build_pairs(root: Path) -> list[dict]:
    """Return 1 DG + 1 trust-sender + 6 trust-receiver researcher pairs."""
    _source(root)
    pairs = [_dg(), _tg_sender()]
    pairs.extend(_tg_receiver(s) for s in range(1000, 6001, 1000))
    return pairs
