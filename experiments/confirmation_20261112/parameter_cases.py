"""Fresh parameter diagnostics, not new empirical sources or benchmark replacements."""
from collections import Counter
from common import digest

POLICIES = {'own': '使你本人的最终收益最大', 'total': '使题目列出的所有参与者最终收益之和最大',
            'equality': '使题目列出的参与者中最高最终收益与最低最终收益之差最小'}


def build(family, i, condition):
    s = (i + 1) * 100
    spec = dict(family=family, index=i, condition=condition, scale=s)
    if family == 'dictator':
        step = (i + 5) * 100; endowment = 6 * step
        threshold = step + step // 3 + (step if condition and i < 4 else step // 6 if condition else 0)
        menu = list(range(0, endowment + 1, step))
        payoffs = {a: ([endowment - a, a] if a >= threshold else [0, 0]) for a in menu}
        spec.update(step=step, endowment=endowment, threshold=threshold)
        role = '你是分配者P1，另一名参与者是P2。'
        rules = [f'P1有{endowment}点可分配。P1选择给P2的点数a。',
                 f'只有a至少为{threshold}时，分配有效：P1最终得到{endowment}减a，P2最终得到a。',
                 '若a低于门槛，两人最终收益均为0；没有其他收益。']
        situation = '现在由你选择给P2的点数。门槛不是额外费用。'
    elif family == 'ultimatum':
        offer = 3 * s; endowment = 10 * s
        outside = offer - s + 2 * s * condition if i < 4 else s + s * condition
        menu = ['accept', 'reject']; payoffs = {'accept': [offer, endowment - offer], 'reject': [outside, 0]}
        spec.update(offer=offer, endowment=endowment, outside=outside)
        role = '你是最后通牒的接收者P2，提议者是P1。'
        rules = [f'P1原有{endowment}点，提出给P2 {offer}点。',
                 f'若你接受，你最终得到{offer}点，P1最终得到{endowment - offer}点。',
                 f'本参数化规则另设拒绝补偿：若你拒绝，你最终得到{outside}点，P1最终得到0点。没有其他收益。']
        situation = '现在你选择接受或拒绝；补偿只在拒绝时支付。'
    elif family == 'trust':
        scale = 2 * s; received = 6 * scale; base = 2 * scale
        # The rule changes how much reaches the investor per point paid back.
        numerator, denominator = ((1, 2) if condition == 0 else (2, 1)) if i < 4 else ((2, 1) if condition == 0 else (3, 1))
        menu = list(range(0, received + 1, scale))
        payoffs = {a: [received - a, base + a * numerator // denominator] for a in menu}
        assert all(a * numerator % denominator == 0 for a in menu)
        spec.update(received=received, investor_base=base, step=scale, numerator=numerator, denominator=denominator)
        role = '你是信任决策中的返还者P2，投资者是P1。'
        rules = [f'你当前可支配{received}点，P1另外保留{base}点。你选择从自己点数中支付返还额a。',
                 f'支付后你的最终收益为{received}减a。',
                 f'结算规则将返还额乘以{numerator}/{denominator}后计入P1：P1最终收益为{base}加a乘{numerator}/{denominator}。',
                 '该系数由外部结算规则提供，差额不再向你收取或返还。题中没有其他参与者或收益。']
        situation = '投资步骤已结束，你只决定本次返还支付额。'
    elif family == 'third_party_punishment':
        target = 'P1' if not condition or i >= 4 else 'P2'
        penalty = s if i < 4 or not condition else 2 * s
        menu = ['punish', 'do_not_punish']
        before = [5 * s, 10 * s, 2 * s]; after = [4 * s, 10 * s, 2 * s]
        after[1 if target == 'P1' else 2] -= penalty
        payoffs = {'do_not_punish': before, 'punish': after}
        spec.update(target=target, penalty=penalty)
        role = '你是第三方P3，另两名参与者是P1和P2。'
        rules = [f'不惩罚时，你最终有{5*s}点，P1有{10*s}点，P2有{2*s}点。',
                 f'若惩罚，你支付{s}点，最终保留{4*s}点；{target}再被扣除{penalty}点，另一名参与者的点数不变。',
                 '被扣除的点数不转给任何人；没有其他收益。']
        situation = '你现在决定是否实施一次惩罚。'
    elif family == 'sequential_prisoners_dilemma':
        fine = (s if condition == 0 else 3*s) if i < 4 else s * condition
        menu = ['cooperate', 'defect']; payoffs = {'cooperate': [3*s, 3*s], 'defect': [5*s-fine, 0]}
        spec.update(fine=fine)
        role = '你是序贯合作决策的第二行动者P2，第一行动者是P1。'
        rules = [f'你已观察到P1选择合作。若你也合作，你与P1最终各得{3*s}点。',
                 f'若你选择不合作，你先得{5*s}点，P1得0点；规则再从你的收益中扣除{fine}点罚金。',
                 '罚金不转给P1；没有其他收益。']
        situation = '现在由你选择合作或不合作。'
    else: raise ValueError(family)
    assert all(all(type(x) is int and x >= 0 for x in row) for row in payoffs.values())
    return dict(spec=spec, role=role, rules=rules, situation=situation, menu=menu, payoffs=payoffs)


def independent_payoffs(spec, action):
    """Separate formula implementation, checked against the materialized ledger."""
    f, i, c, s = (spec[k] for k in ['family', 'index', 'condition', 'scale'])
    if f == 'dictator':
        u = (i+5)*100; t = u + u//3
        if c: t += u if i < 4 else u//6
        return [6*u-action, action] if action >= t else [0, 0]
    if f == 'ultimatum':
        rejected = (2+2*c)*s if i < 4 else (1+c)*s
        return [3*s, 7*s] if action == 'accept' else [rejected, 0]
    if f == 'trust':
        ratio = (0.5 if not c else 2) if i < 4 else (2 if not c else 3)
        # Every product is an exact integer; independently reject rounding.
        z = action * ratio; assert z == int(z)
        return [12*s-action, 4*s+int(z)]
    if f == 'third_party_punishment':
        if action == 'do_not_punish': return [5*s, 10*s, 2*s]
        p1, p2 = 10*s, 2*s
        if i < 4 and c: p2 -= s
        else: p1 -= (2*s if i >= 4 and c else s)
        return [4*s, p1, p2]
    if f == 'sequential_prisoners_dilemma':
        if action == 'cooperate': return [3*s, 3*s]
        sanction = (1+2*c)*s if i < 4 else c*s
        return [5*s-sanction, 0]
    raise ValueError(f)


def optima(payoffs, policy):
    def value(row):
        return row[0] if policy == 'own' else sum(row) if policy == 'total' else -(max(row)-min(row))
    best = max(map(value, payoffs.values()))
    return [a for a, row in payoffs.items() if value(row) == best]


def generate():
    payloads = []; labels = []; quartets = []; cases = []; categories = Counter()
    families = ['dictator', 'ultimatum', 'trust', 'third_party_punishment', 'sequential_prisoners_dilemma']
    for family in families:
        for i in range(8):
            pair = f'PARAM_{family}_{i:02d}'; lookup = {}; sets = {}
            for c in [0, 1]:
                case = build(family, i, c); case_id = f'{pair}_c{c}'
                alternative = {a: independent_payoffs(case['spec'], a) for a in case['menu']}
                assert alternative == case['payoffs']
                cases.append(dict(case_id=case_id, pair_id=pair, **case))
                menu = '合法行动值为：' + '、'.join(map(str, case['menu'])) + '。只能从这些值中选择，不能填写其他数值或词。'
                for policy, objective in POLICIES.items():
                    expected = optima(case['payoffs'], policy)
                    # Independent objective enumeration, preserving all tied optima.
                    scores = {a: (row[0], sum(row), min(row)-max(row))[list(POLICIES).index(policy)] for a, row in alternative.items()}
                    assert set(expected) == {a for a, v in scores.items() if v == max(scores.values())}
                    sets[c, policy] = set(expected)
                    for variant in ['list', 'table']:
                        rules = '\n'.join('- '+x for x in case['rules']) if variant == 'list' else '| 规则编号 | 内容 |\n|---|---|\n'+'\n'.join(f'| {j+1} | {x} |' for j, x in enumerate(case['rules']))
                        text = f'【参与者】\n{case["role"]}\n\n【规则】\n{rules}\n\n【当前情况】\n{case["situation"]}\n\n【合法菜单】\n{menu}\n\n【明确目标】\n{objective}。只输出JSON对象，且仅包含action字段。'
                        assert all(text.count(x) == 1 for x in case['rules'])
                        pid = 'confirm_goal_' + digest([case_id, policy, variant, text])[:20]
                        schema = dict(type='object', properties=dict(action=dict(type='integer' if type(case['menu'][0]) is int else 'string', enum=case['menu'])), required=['action'], additionalProperties=False)
                        payloads.append(dict(prompt_id=pid, task_type='goal', messages=[dict(role='system', content='请根据材料中的完整规则与明确目标选择合法行动，只输出要求的JSON对象。'), dict(role='user', content=text)], output_schema=schema))
                        labels.append(dict(prompt_id=pid, task_type='goal', pair_id=pair, case_id=case_id, condition=c, policy=policy, task=family, source_id='PARAMETRIC_DIAGNOSTIC', variant_id=variant, expected_actions=expected, action_space=dict(type='categorical', values=case['menu'])))
                        lookup[c, policy, variant] = pid
            for policy in POLICIES:
                a, b = sets[0, policy], sets[1, policy]
                category = 'same_optimum_set' if a == b else 'disjoint_optimum_sets' if not a & b else 'changed_but_overlapping'
                categories[category] += 1
                quartets.append(dict(quartet_id=f'{pair}__{policy}', pair_id=pair, policy=policy, task=family, source_id='PARAMETRIC_DIAGNOSTIC', category=category,
                    A=lookup[0, policy, 'list'], B=lookup[0, policy, 'table'], C=lookup[1, policy, 'list'], D=lookup[1, policy, 'table']))
    assert len(payloads) == len({digest(p['messages']) for p in payloads}) == 480
    assert len(cases) == 80 and len(quartets) == 120
    assert categories['same_optimum_set'] and categories['disjoint_optimum_sets']
    for family in families:
        assert any(q['category'] == 'disjoint_optimum_sets' and q['task'] == family for q in quartets)
        assert any(q['category'] == 'same_optimum_set' and q['task'] == family for q in quartets)
    return payloads, labels, quartets, cases, dict(categories)
