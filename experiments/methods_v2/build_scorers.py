"""Create explicit V2 copies of audited V1 scoring, preserving V1 byte-for-byte."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / 'methods_v1'


def replace(source, old, new):
    assert old in source, old
    return source.replace(old, new)


def main():
    main = (V1 / 'score_prompt_methods.py').read_text(encoding='utf-8')
    main = replace(main, 'import math\n', 'import math\nimport os\n')
    main = replace(main, 'METHODS = ("baseline", "generic", "social_audit", "ledger_original", "ledger_only")',
        'METHODS = ("baseline", "canonical_modules", "canonical_profile")')
    main = replace(main, 'LORA_METHODS = ("sft", "sft_consistency")',
        'LORA_METHODS = ("anchor_sft", "anchor_js", "anchor_kl", "anchor_js_kl")')
    main = replace(main, 'INVAR_sampled_methods_score_v2_20261025', 'INVAR_normalization_anchor_score_v1_20261026')
    main = replace(main, 'Path(__file__).with_name("run_prompt_methods.py")',
        'Path(os.environ.get("INVAR_METHODS_V1", Path(__file__).resolve().parent.parent / "methods_v1")) / "run_prompt_methods.py"')
    main = replace(main, '# INVAR 提示与账本架构开发筛选结果', '# INVAR 第二轮：保守归一化与条件内锚定')
    main = replace(main, '只比较相同抽样接口、相同冻结题目的提示方法', '只比较相同生成接口、相同冻结题目的输入处理与参数学习方法')
    main = replace(main, 'LoRA在非HPVBG来源的事实/明示目标及部分等价表达上训练；下面全1,378输入混合汇总包含训练接触，只能描述开发表现，不能称未见数据泛化。自由选择标签未参与训练，也不能据此称相同情境材料未见。',
        '本轮LoRA在非HPVBG来源、排除V4/V6的材料上训练；自由行为还参与等价与条件内软分布约束。没有人类行为标签，但确实有训练接触。全1,378输入为接触混合的开发汇总，不能称未见泛化；主迁移分区另列。')
    main = replace(main, 'additionally supports sft,sft_consistency', 'additionally supports four anchor learning arms')
    main = replace(main, '"behavior_label_policy": "自由选择标签未训练；相同情境可通过事实/目标题接触"',
        '"behavior_label_policy": "未提供自由行为硬标签；参数方法可接受等价与冻结教师软分布约束，仍有训练接触"')
    main = replace(main, '来源/变体已见；事实/目标题可能训练接触，不是未见迁移',
        '来源/变体已见；事实、目标及自由行为约束可能训练接触，不是未见迁移')
    (HERE / 'score_methods.py').write_text(main, encoding='utf-8')
    aux = (V1 / 'finite_sample_audit.py').read_text(encoding='utf-8')
    aux = replace(aux, 'import math\n', 'import math\nimport os\n')
    aux = replace(aux, 'DEFAULT_METHODS = ("baseline", "generic", "social_audit", "ledger_original", "ledger_only")',
        'DEFAULT_METHODS = ("baseline", "canonical_modules", "canonical_profile")')
    aux = replace(aux, 'ALLOWED_METHODS = DEFAULT_METHODS + ("sft", "sft_consistency")',
        'ALLOWED_METHODS = DEFAULT_METHODS + ("anchor_sft", "anchor_js", "anchor_kl", "anchor_js_kl")')
    aux = replace(aux, 'INVAR_finite_sample_squared_audit_v1_20261025', 'INVAR_finite_sample_squared_audit_v2_20261026')
    aux = replace(aux, 'Path(__file__).with_name("run_prompt_methods.py")',
        'Path(os.environ.get("INVAR_METHODS_V1", Path(__file__).resolve().parent.parent / "methods_v1")) / "run_prompt_methods.py"')
    (HERE / 'finite_sample_audit.py').write_text(aux, encoding='utf-8')


if __name__ == '__main__':
    main()
