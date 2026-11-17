"""CPU synthetic test of exactly the sequential PEFT switching used by proxy.py."""
import copy
import tempfile
from pathlib import Path
from common import save,R


def main():
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from peft import LoraConfig,PeftModel,get_peft_model
    torch.manual_seed(20261113)
    base=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2)).eval()
    ids=torch.tensor([[1,2,3,4]]);base_logits=base(ids).logits.detach()
    with tempfile.TemporaryDirectory() as tmp:
        paths=[];expected=[]
        for n in (1,2):
            model=get_peft_model(copy.deepcopy(base),LoraConfig(r=8,lora_alpha=16,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
            with torch.no_grad():
                for name,value in model.named_parameters():
                    if 'lora_B' in name:value.fill_(.1*n)
            model.eval();expected.append(model(ids).logits.detach())
            path=Path(tmp)/str(n);model.save_pretrained(path);paths.append(path)
        model=PeftModel.from_pretrained(copy.deepcopy(base),paths[0],adapter_name='current',torch_device='cpu').eval()
        assert torch.allclose(model(ids).logits,expected[0],atol=1e-6)
        model.delete_adapter('current');model.load_adapter(paths[1],adapter_name='current',is_trainable=False,torch_device='cpu')
        model.set_adapter('current');model.eval()
        assert torch.allclose(model(ids).logits,expected[1],atol=1e-6)
        with model.disable_adapter():assert torch.allclose(model(ids).logits,base_logits,atol=1e-6)
        assert not torch.allclose(expected[0],expected[1],atol=1e-6)
        assert not torch.cuda.is_initialized()
    save(R/'synthetic_adapter_validation.json',dict(status='passed_cpu_adapter_switch_and_disable',scientific_answers=0,no_CUDA=True))
    print('synthetic adapter switching passed')


if __name__=='__main__':main()
