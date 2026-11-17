"""Build source-grounded PGG and El Zein cases; no human answers enter prompts.

Only Python's standard library is used. ``root`` is the INVAR project root.
TPP offer=5 is deliberately excluded because the original half-integer rounding
rule cannot currently be verified. No implementation default fills that gap.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path


PGG_URL = "https://media.springernature.com/original/springer-static/esm/art%3A10.1007%2Fs10683-022-09756-9/MediaObjects/10683_2022_9756_MOESM1_ESM.pdf"
HPVBG_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC7062205/"
PGG_FILE = "research/evidence/PGG/instructions_appendix.pdf"
HPVBG_FILE = "research/evidence/HPVBG/article_fulltext.xml"
HPVBG_DATA = "research/evidence/HPVBG/Data_punishment.csv"
EXCLUSIONS = [
    {
        "candidate_id": "HPVBG_TPP_OFFER5_INDIV_GROUP",
        "source_id": "HPVBG",
        "reason": "惩罚后保留额为5×0.7=3.5；论文只说明取最近整数，现有材料不能确认半整数取整约定。",
        "status": "excluded_pending_source_rounding_verification",
        "reentry_requirement": "取得可验证的原实验实现或作者确认后，在模型测试前重新审定；不得猜测JavaScript/Python的默认舍入。",
    }
]


def _verify_sources(root: Path) -> None:
    """Fail closed if the verified stimuli have disappeared or changed."""
    digests = {
        PGG_FILE: "5d5f8b6579897ca3c010d173db4697b1c8cb276d35b68a88389edea1f963a1a2",
        HPVBG_DATA: "a45540e251775c374a9378095f6e9b041b2e94933eee7617821b1bcc5da5b934",
        HPVBG_FILE: "16b6183a6bb95eaefc9d896f111d6fdd2ae90432b69d23769f0e6a021671bc6a",
    }
    for rel, expected in digests.items():
        path = root / rel
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source hash mismatch: {rel}")
    # Verify stimulus availability without importing participants' responses.
    with (root / HPVBG_DATA).open(encoding="utf-8-sig", newline="") as handle:
        actual = {
            (int(row["role"]), int(row["groupcond"]), int(row["offercond"]))
            for row in csv.DictReader(handle)
            if row["role"] in {"2", "5"} and row["offercond"] in {str(i) for i in range(6)}
        }
    needed = {(2, 1, o) for o in range(6)} | {(5, g, o) for g in (1, 2) for o in range(5)}
    if not needed.issubset(actual):
        raise ValueError(f"Missing original El Zein conditions: {sorted(needed - actual)}")


def _pgg_case(treatment: int, c: int) -> dict:
    if treatment not in (2, 3, 4, 5) or c not in range(11):
        raise ValueError("Unsupported PGG condition")
    contribution_rule = {
        2: "P1、P2、P3各自向共同项目投入的点数，等于其本人在此前一次独立贡献任务中自行选择的贡献点数；本次不由他们重新选择。",
        3: "P1、P2、P3各自向共同项目投入的点数，分别采用从其他组随机选到的参与者在此前一次独立贡献任务中自行选择的贡献点数；本次不采用P1、P2、P3本人的此前选择。",
        4: "计算机分别从0至10的整数中为P1、P2、P3抽取贡献点数。每个数被抽中的概率相同，三次抽取相互独立；三人的实际贡献由这些抽取值决定。",
        5: "计算机分别从0至10的整数中为P1、P2、P3抽取贡献点数。每个数被抽中的概率相同，三次抽取相互独立；三人的实际贡献由这些抽取值决定。",
    }[treatment]
    display_rule = (
        "条件表上方的数值，是P1、P2、P3实际贡献点数的算术平均数取最近整数所得。"
        if treatment != 5 else
        "条件表上方的数值，是计算机另抽的三个0至10整数的算术平均数取最近整数所得。这三个数等概率抽取，相互独立，且与决定P1、P2、P3实际贡献的三个数独立；它们不决定任何人的实际贡献。"
    )
    relation_alt = (
        "P1、P2、P3实际贡献点数的算术平均数，取最近整数后，作为条件表上方的数值。"
        if treatment != 5 else
        "计算机另抽的三个0至10整数，其算术平均数取最近整数后，作为条件表上方的数值。这三个数等概率抽取，相互独立，且与决定P1、P2、P3实际贡献的三个数独立；它们不决定任何人的实际贡献。"
    )
    rules = [
        "四名参与者P1、P2、P3、P4匿名组成一组，每人有10点，其中没有投入共同项目的点数归本人保留。",
        "本情境中有三名X类参与者和一名Y类参与者。计算机随机确定谁是Y类，每个人被选中的机会相同。你填写的是自己若被选为Y类时使用的条件贡献表。",
        contribution_rule,
        display_rule,
        "条件表列出0至10的整数。填写时尚不知道实际会对应哪一列；若你被选为Y类，就以实际对应那一列中填写的数作为你的贡献。此处只要求填写其中一格，其余格不在本次作答范围内。",
        "每人的收益＝10－本人实际贡献＋0.75×全组四人的实际贡献总和。每人保留的点数只归本人，共同项目的收益由四人按相同规则分别获得。",
        "你可填写0至10中的任意整数作为贡献点数。你不知道其他三人的逐人贡献，不能由取整后的表头数值断定逐人贡献或准确总和。",
    ]
    latent = {
        "actual_contributions": {"P1": None, "P2": None, "P3": None},
        "actual_contribution_range": [0, 10],
        "displayed_column": c,
        "display_is_hypothetical_table_entry": True,
        "signal_related_to_actual_contributions": treatment != 5,
    }
    if treatment != 5:
        latent["constraint_if_this_column_applies"] = f"nearest_integer((x1+x2+x3)/3)={c}"
        if treatment == 4:
            latent["admissible_peer_sums_if_this_column_applies"] = [s for s in range(31) if abs(s - 3 * c) <= 1]
    else:
        latent["actual_contribution_distribution"] = "three independent discrete uniform draws on {0,...,10}"
        latent["signal_distribution"] = "nearest_integer(mean(three further independent discrete uniform draws on {0,...,10}))"
        latent["signal_independent_of_actual_contributions"] = True
    return {
        "focal_id": "P4",
        "entities": {"P1": "若P4为Y时的X类参与者", "P2": "若P4为Y时的X类参与者", "P3": "若P4为Y时的X类参与者", "P4": "填写条件贡献表的参与者"},
        "role_facts": ["参与者编号：P4。", "本次角色：填写自己若被抽为Y类时使用的条件贡献表。", "本情境资源：10点。"],
        "rules": rules,
        "situation": [f"现在只填写条件表表头为{c}的那一格。", "这个表头值是假设条件，不表示你已看到实际发生的贡献或实际抽取结果。"],
        "question": f"假如你被选为Y类，且条件表实际对应表头为{c}的这一列，你愿意向共同项目贡献多少点？请填写0至10中的一个整数。",
        "action_space": {"type": "integer", "min": 0, "max": 10},
        "outcome": {"kind": "normalized_amount", "denominator": 10},
        "payoffs": {
            "formula": {"self": "10-a+0.75*(a+x1+x2+x3)", "P1": "10-x1+0.75*(a+x1+x2+x3)", "P2": "10-x2+0.75*(a+x1+x2+x3)", "P3": "10-x3+0.75*(a+x1+x2+x3)"},
            "variables": {"a": "本格填写的P4贡献；仅该格被实际使用时执行", "x1": "P1实际贡献", "x2": "P2实际贡献", "x3": "P3实际贡献"},
            "latent": latent,
        },
        "relation_alt": {"section": "rules", "index": 3, "text": relation_alt},
        "source_condition": {"treatment": treatment, "table_column": c},
        "review_status": "program_validated_ai_source_reviewed_human_review_pending",
    }


def _ug_case(offer: int) -> dict:
    return {
        "focal_id": "P2",
        "entities": {"P1": "个人提议者", "P2": "接收者"},
        "role_facts": ["参与者编号：P2。", "本次角色：接收者，可以接受或拒绝P1提出的分配。"],
        "rules": [
            "本题是匿名的一次分配决策。P1是个人提议者，原有10点，由P1决定给P2多少点，其余留给P1。",
            f"你可以接受或拒绝。接受时，P2得到{offer}点，P1得到{10-offer}点；拒绝时，P1、P2都得到0点。",
        ],
        "situation": [f"P1提出给P2 {offer}点，给自己留下{10-offer}点。"],
        "question": "你作为P2，选择接受还是拒绝这一分配？",
        "action_space": {"type": "categorical", "values": ["accept", "reject"]},
        "action_labels": {"accept": "接受", "reject": "拒绝"},
        "outcome": {"kind": "probability", "target_action": "accept"},
        "payoffs": {"accept": {"self": offer, "P1": 10-offer}, "reject": {"self": 0, "P1": 0}},
        "relation_alt": {"section": "situation", "index": 0, "text": f"按照P1提出的分配，P2将得到{offer}点，P1给自己留下{10-offer}点。"},
        "source_condition": {"game": 1, "role": 2, "groupcond": 1, "offercond": offer},
        "review_status": "program_validated_ai_source_reviewed_human_review_pending",
    }


def _punished_remaining(offer: int) -> int:
    """Exact integer arithmetic; reject unverifiable half-integer outcomes."""
    numerator = 7 * (10 - offer)
    if numerator % 10 == 5:
        raise ValueError("Source half-integer rounding convention is unverified")
    return (numerator + 5) // 10


def _tpp_case(offer: int, group: bool) -> dict:
    remaining = 10 - offer
    punished = _punished_remaining(offer)
    proposers = ["P1", "P4", "P5"] if group else ["P1"]
    entities = {p: "提议者群体成员" if group else "个人提议者" for p in proposers}
    entities.update({"P2": "接收者", "P3": "第三方决策者"})
    if group:
        allocation_rule = "P1、P4、P5是三名提议者，各自基于10点提出分配。三人各自提议给接收者的数值取算术平均，成为共同执行的提议。每名提议者的保留收益都按这一共同提议计算，不是把一份剩余收益再分给三人。"
        situation = [f"P1、P4、P5共同执行的平均提议是给P2 {offer}点。三名提议者因此各保留{remaining}点，P2得到{offer}点。", "题目仅给出共同执行的平均提议，没有逐人列出三名成员原先的提议。"]
        relation_alt = f"P2将按P1、P4、P5共同执行的平均提议得到{offer}点；该方案使P1、P4、P5各保留{remaining}点。"
        consequence = f"不惩罚时，P3保留5点，P1、P4、P5各得{remaining}点，P2得{offer}点；惩罚时，P3保留4点，P1、P4、P5各自保留的{remaining}点均减少30%并取最近整数，分别变为{punished}点，P2仍得{offer}点。"
    else:
        allocation_rule = "P1是个人提议者，原有10点，由P1决定给P2多少点，其余留给P1。"
        situation = [f"P1提出给P2 {offer}点，给自己留下{remaining}点。"]
        relation_alt = f"按照P1提出的分配，P2将得到{offer}点，P1给自己留下{remaining}点。"
        consequence = f"不惩罚时，P3保留5点，P1得{remaining}点，P2得{offer}点；惩罚时，P3保留4点，P1保留的{remaining}点减少30%并取最近整数，变为{punished}点，P2仍得{offer}点。"
    no_punish = {"self": 5, "P2": offer, **{p: remaining for p in proposers}}
    punish = {"self": 4, "P2": offer, **{p: punished for p in proposers}}
    return {
        "focal_id": "P3", "entities": entities,
        "role_facts": ["参与者编号：P3。", "本次角色：第三方决策者，既不是提议者，也不是接收者。", "你拥有5点，可以自行决定是否花费1点进行惩罚。"],
        "rules": [
            "本题是匿名的一次分配决策。",
            allocation_rule,
            "P2只能接收分配，没有拒绝权。你可以不惩罚，也可以花费自己的1点进行惩罚；惩罚会使每位提议者保留的收益减少30%，再将其剩余收益取最近整数。扣除的点数不转给你或P2。",
            consequence,
        ],
        "situation": situation,
        "question": "你作为P3，选择惩罚还是不惩罚？",
        "action_space": {"type": "categorical", "values": ["punish", "do_not_punish"]},
        "action_labels": {"punish": "惩罚", "do_not_punish": "不惩罚"},
        "outcome": {"kind": "probability", "target_action": "punish"},
        "payoffs": {"punish": punish, "do_not_punish": no_punish},
        "relation_alt": {"section": "situation", "index": 0, "text": relation_alt},
        "source_condition": {"game": 2, "role": 5, "groupcond": 2 if group else 1, "offercond": offer},
        "review_status": "program_validated_ai_source_reviewed_human_review_pending",
    }


def build_pairs(root: Path) -> list[dict]:
    root = Path(root)
    _verify_sources(root)
    pairs = []
    pgg_adaptations = [
        "原英文印刷及图形指令整理为中文文字；匿名角色统一编号。",
        "从完整条件贡献表中抽取一格进行独立调用，仍明确为假设条件；保留角色随机选择和该格生效条件。",
        "不要求模型执行前置任务；原有前置选择只作为规则所依赖的过去行为，未知具体金额保持未知。",
        "本基准将回答统一为0至10的整数点数；原文明确整数条件表和随机数支持，但未取得原zTree程序确认贡献输入框的小数限制，故整数回答记录为本基准响应规范，不冒称已核验原软件精度。",
        "本任务保留点数收益计算，省去完整五情境流程、实验室物流和现金结算；不得直接作为原完整流程的人类行为复刻。",
        "未取得可复算的本研究微观回答；不提供或推断人类正确答案。",
    ]
    for before, after, kind, factor, description in (
        (2, 3, "institution", "origin_of_type_x_contributions", "X类实际贡献从其本人此前选择改为其他组随机抽取参与者的此前选择；当前条件表列不变。"),
        (4, 5, "information", "connection_between_display_and_actual_contributions", "表头从真实贡献的取整平均数改为另一组独立随机数的取整平均数；X类实际贡献仍由独立均匀随机数决定。"),
    ):
        for c in range(11):
            pairs.append({
                "pair_id": f"PGG_T{before}_T{after}_C{c:02d}", "source_id": "PGG",
                "task": "public_goods", "role": "conditional_contributor", "analysis_tier": "source_conditions",
                "intervention": {"kind": kind, "factor": factor, "origin": "original_conditions", "description": description, "expected_direction": None},
                "provenance": {"file": PGG_FILE, "locator": f"Appendix A, printed pp.3–4 and pp.{before+4}/{after+4}; within-subject treatments {before}/{after}, column {c}", "url": PGG_URL, "adaptations": pgg_adaptations.copy()},
                "case0": _pgg_case(before, c), "case1": _pgg_case(after, c),
            })
    hp_adaptations = [
        "原实验图示和说明适配为中文单次决策；不展示前后试次、不输入真实参与者特征或回答。",
        "保留参与者在原实验中被告知的社会情境；原研究实际使用计算机预设提议。数据来源据实披露，但不把这一幕后信息加入参与者视角刺激。",
        "省去完整60试次流程、图像、计时与现金结算；不把原样本选择当作当前模型的正确答案。",
    ]
    for low, high in ((0, 5), (1, 4), (2, 3)):
        pairs.append({
            "pair_id": f"HPVBG_UG_OFFER{low}_TO{high}", "source_id": "HPVBG", "task": "ultimatum", "role": "responder", "analysis_tier": "source_conditions",
            "intervention": {"kind": "other_behavior", "factor": "offer_to_receiver", "origin": "original_conditions", "description": f"个人提议给接收者的金额从{low}点增加到{high}点，提议者保留额同步改变。自己的收益与分配差距同时改变，不能单独识别公平偏好。", "expected_direction": None},
            "provenance": {"file": HPVBG_FILE, "locator": f"Methods: Ultimatum game; Design, conditions of interest; Data_punishment.csv: role=2, groupcond=1, offercond={low}/{high}", "url": HPVBG_URL, "adaptations": hp_adaptations.copy()},
            "case0": _ug_case(low), "case1": _ug_case(high),
        })
    for offer in range(5):
        pairs.append({
            "pair_id": f"HPVBG_TPP_OFFER{offer}_INDIV_GROUP", "source_id": "HPVBG", "task": "third_party_punishment", "role": "third_party", "analysis_tier": "source_conditions",
            "intervention": {"kind": "institution", "factor": "individual_or_group_offer", "origin": "original_conditions", "description": "相同提议金额，个人提议改为三人各自提议的平均方案；各成员分别受罚，受影响人数和总扣减同时变化，不能单独识别责任分散机制。", "expected_direction": None},
            "provenance": {"file": HPVBG_FILE, "locator": f"Methods: Dictator game with third-party punishment; Key experimental conditions; Design; Data_punishment.csv: role=5, groupcond=1/2, offercond={offer}", "url": HPVBG_URL, "adaptations": hp_adaptations.copy()},
            "case0": _tpp_case(offer, False), "case1": _tpp_case(offer, True),
        })
    assert len(pairs) == 30
    assert len({p["pair_id"] for p in pairs}) == len(pairs)
    return pairs
