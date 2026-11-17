"""Create the scientific freeze exactly once, after engineering acceptance."""
from datetime import datetime,timezone
from campaign_common import R,MODELS,SEED,jl,load,save,sha
from science_worker import plan


def main():
    assert not (R/'freeze.json').exists()
    assert load(R/'preflight_acceptance.json')['status']=='passed_cpu_all_inputs_all_checkpoints'
    engineering=load(R/'engineering_acceptance.json')
    assert all(engineering[j]['status']=='engineering_completed_not_scientific' for j in MODELS)
    protocol=dict(seed=SEED,repeats=dict(behavior=8,goal=2,changed_fact=1,stable_fact=1,parameter=2),
        sampling=dict(temperature=.7,top_p=.95,top_k=-1,max_tokens=4096),max_model_len=8192,
        campaign_hours=24,mc_resamples=500,
        job_order=['qwen3_8b','gpt_oss_20b','llama31_8b','glm4_9b','mistral7b_v01','sotopia_mistral7b','socialr1_8b','qwen3_8b_base'],
        stop_rule='24h GPU campaign hard limit; preserve partial attempts; no score-based reruns or replacement selection',
        inference='Fixed-panel descriptive contrasts; separate native toggle, effort and prompt interventions; base separate',
        invalid_rule='Retain in all-attempt Q/U; conditional behavior with conservative missingness bounds',
        translation='AI-assisted Chinese derivative; structured review, no independent human certification')
    ps=jl(R/'frozen/model_inputs.jsonl');rows={j:len(plan(j,ps,protocol)) for j in MODELS}
    assert set(rows.values())=={11760} and sum(rows.values())==94080
    estimated={j:engineering[j]['generation_seconds']/engineering[j]['rows']*rows[j]/3600 for j in MODELS}
    files=[p for p in (R/'frozen').rglob('*') if p.is_file()]
    files += [p for p in R.iterdir() if p.is_file() and (p.suffix in ['.py','.md'] or p.name in [
        'panel_selection.json','translation_units.json','translation_map.json','translation_review.json',
        'parameter_translation_units.json','parameter_translation_map.json','parameter_translation_review.json',
        'preparation_audit.json','hardware_and_models.json','engineering_acceptance.json','preflight_acceptance.json',
        'final_model_inventory.json','test_acceptance.json'])]
    save(R/'freeze.json',dict(status='frozen_before_scientific_generations',frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol=protocol,models=MODELS,expected_rows_by_job=rows,total_expected_rows=sum(rows.values()),
        pilot_scaled_generation_hours=estimated,pilot_scaled_total_hours=sum(estimated.values()),
        conservative_1_5x_plus_1h_estimate=sum(estimated.values())*1.5+1,
        forecast_limitations='32 synthetic goal attempts per checkpoint; mixed real tasks/token-ID input/batching can change throughput; hard cutoff controls cost, not completion guarantee',
        files={str(p.relative_to(R)).replace('\\','/'):sha(p) for p in sorted(files)},
        metric_provenance=dict(source='experiments/shortpaper_20261111/score_joint_reasoning_v2.py',
            source_sha256=sha(R.parent/'shortpaper_20261111/score_joint_reasoning_v2.py'),
            note='w1,w1_check,distance,metrics,aggregate extracted unchanged; independent vector CDF validation')))
    print('scientific freeze created: 94080 planned attempts')


if __name__=='__main__':main()
