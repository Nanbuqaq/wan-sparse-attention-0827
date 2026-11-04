"""Old ODE LoRA evaluation; never relabel this checkpoint as trained DMD."""
from h3_native_dmd_eval import main, ODE_LORA_CONTRACT

if __name__=='__main__':
    main(contract=ODE_LORA_CONTRACT,schema='h3_native_ode_lora_eval_v1')
