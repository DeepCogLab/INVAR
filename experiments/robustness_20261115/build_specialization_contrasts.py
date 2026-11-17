"""Descriptive matched-checkpoint differences from independently accepted scores."""
import csv
from pathlib import Path
from campaign_common import R,load,save,sha

MIRROR=Path('artifacts/robustness_20261115/remote')
PAIRS=[('mistral7b_v01','sotopia_mistral7b'),('qwen3_8b','socialr1_8b')]


def main():
    out=R/'results';selection=load(out/'SCORING_VERSION_SELECTION.json')
    with (out/'summary_corrected.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows={(v['model'],v['language'],v['arm']):v for v in reader}
    metrics=[k for k in fields if k not in ['model','family','kind','role','language','arm']]
    table=[];checks=[];pending=[]
    for base,special in PAIRS:
        if not all(j in selection['selected'] for j in [base,special]):pending.append([base,special]);continue
        identities=[load(MIRROR/'runs'/j/'identity.json') for j in [base,special]]
        check=dict(base=base,specialized=special,
            identities_sha256={j:sha(MIRROR/'runs'/j/'identity.json') for j in [base,special]},
            equal_fields={k:identities[0][k]==identities[1][k] for k in ['weight_sha256','config_sha256','rendered_sha256','prompt_token_ids_sha256','software','gpu_uuid']},
            selected_scores={j:selection['selected'][j] for j in [base,special]})
        if special=='sotopia_mistral7b':
            assert all(check['equal_fields'].values())
            adapter=load(MIRROR/'runs'/special/'adapter_loading.json')
            assert adapter['status']=='engine_add_lora_returned_true' and adapter['lora_int_id']==1
            assert adapter['adapter_sha256']==identities[1]['adapter_sha256']
            assert identities[0]['adapter_sha256'] is None
            check['adapter_loading_sha256']=sha(MIRROR/'runs'/special/'adapter_loading.json')
            check['adapter_runtime_evidence']='Frozen worker asserts each output LoRA request ID=1; successful completion and load receipt preserved. No per-row adapter ID field stored.'
        checks.append(check)
        for key,b in rows.items():
            if key[0]!=base:continue
            s=rows[special,key[1],key[2]]
            row=dict(base=base,specialized=special,language=key[1],arm=key[2])
            for metric in metrics:
                row[metric+'_special_minus_base']=float(s[metric])-float(b[metric]) if s[metric] and b[metric] else None
            row['E_missing_delta_lower']=float(s['E_missing_low'])-float(b['E_missing_high'])
            row['E_missing_delta_upper']=float(s['E_missing_high'])-float(b['E_missing_low'])
            table.append(row)
    if table:
        with (out/'matched_specialization_contrasts.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    save(out/'MATCHED_SPECIALIZATION_PROVENANCE.json',dict(checks=checks,pending=pending,
        summary_sha256=sha(out/'summary_corrected.csv'),selection_sha256=sha(out/'SCORING_VERSION_SELECTION.json'),
        builder_sha256=sha(Path(__file__)),scope='Post-generation descriptive differences for predeclared checkpoint matches; not pooled specialization effects or calibrated intervals. Common interface SOTOPIA is not the full interactive system. Independent model sampling streams.'))
    print('Matched specialization pairs:',len(checks),'pending:',pending)


if __name__=='__main__':main()
