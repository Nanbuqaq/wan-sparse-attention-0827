"""Stage1B full-parameter checkpoints under the same request protocol as LoRA."""
from h3_native_dmd_eval import main

if __name__ == '__main__':
    main(contract='STAGE1B_CURRENT_ONLY_V1', schema='h3_stage1b_full_eval_v1')
