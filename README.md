# Pose-Guided Residual Refinement for Interpretable Text-to-Motion Generation and Editing
This is code repository for "Pose-Guided Residual Refinement for Interpretable Text-to-Motion Generation and Editing"

## Getting Started

### Dataset

For the Dataset(HumanML3D and KIT-ML datasets), please follow the instructions for downloading and preprocessing [here](https://github.com/EricGuo5513/HumanML3D)

The resulting file directory should look like this:

```
./dataset/[dataset_name]/
├── new_joint_vecs/
├── new_joints/
├── texts/
├── Mean.npy 
├── Std.npy 
├── train.txt
├── val.txt
├── test.txt
├── train_val.txt
└── all.txt
```

### Fine-grained Descriptions & Pose Codes

For the fine-grained description and pose codes, please follow the instructions for downloadling and preprocessing [here](https://github.com/yh2371/CoMo/tree/main)

### Pre-trained Models

```

```

## Dependencies

```
pip install -r requirements.txt
```

## Train PG(PoseGuided) Tokenizer

```
python train_pg_tokenizer.py \
--batch-size 256 \
--lr 1e-4 \
--total-iter 300000 \
--lr-scheduler 200000 \
--nb-code 392 \
--down-t 2 \
--depth 3 \
--eval-iter 1000 \
--warm-up-iter 1000 \
--dilation-growth-rate 3 \
--out-dir exp_result \
--dataname t2m \
--vq-act relu \
--loss-cfg-path ./configs/exp_dec/losses.yaml \
--exp-name exp_pg_tokenizer \
--use-keywords \
--output-emb-width 392 \
--rvq-num-quantizers 6 \
--rvq-quantize-dropout-prob 0.2 \
--rvq-quantize-dropout-cutoff-index 0 \
--rvq-nb-code 512 \
--rvq-quantizer-type soft \
--rvq-vq-loss-beta 0.25 \
--rvq-loss-weight 0.02 \
--detach-p-latent \
--params-soft-ent-loss 0.01 \
--pdrop-res 0.1 \
--unuse-ema
```

## Train Base Transformer

```
python train_base_trans.py \
--exp-name exp_base_trans \
--batch-size 64 \
--num-layers 9 \
--nb-code 392 \
--n-head-gpt 16 \
--block-size 62 \
--ff-rate 4 \
--out-dir exp_result \
--total-iter 300000 \
--lr-scheduler 150000 \
--lr 0.0001 \
--dataname t2m \
--down-t 2 \
--depth 3 \
--eval-iter 10000 \
--pkeep 0.5 \
--num_workers 16 \
--dilation-growth-rate 3 \
--output-emb-width 392 \
--val-shuffle \
--log-cat-right-num \
--eval-masking \
--min-sampling-prob 0.1 \
--dec-checkpoint-folder ./exp_result/exp_pg_tokenizer/{date1} \
--use-keywords

```

## Train Refine Transformer

```
python train_refine_trans.py \
--exp-name exp_refine_trans \
--batch-size 64 \
--num-layers 6 \
--nb-code 392 \
--n-head-gpt 8 \
--block-size 62 \
--ff-rate 4 \
--embed-dim-gpt 1024 \
--out-dir exp_result \
--total-iter 150000 \
--lr-scheduler 50000 \
--lr 0.0002 \
--dataname t2m \
--down-t 2 \
--depth 3 \
--eval-iter 10000 \
--pkeep 1.0 \
--num_workers 16 \
--dilation-growth-rate 3 \
--output-emb-width 392 \
--val-shuffle \
--loss-cfg-path ./configs/exp_rt2m/losses.yaml \
--log-cat-right-num \
--min-sampling-prob 0.1 \
--masking-prob 0.0 \
--drop-out-rate 0.2 \
--share-weight \
--warm-up-iter 5000 \
--start-warm-up \
--freeze-pose-code-emb \
--load-pretrained-pose-code-emb \
--dec-checkpoint-folder ./exp_result/exp_pg_tokenizer/{date1} \
--t2m-checkpoint-folder ./exp_result/exp_base_trans/{date2} \
--use-keywords
```

## Eval PG(PoseGuided) Tokenizer

```
python eval_pg_tokenizer.py \
--batch-size 256 \
--lr 2e-4 \
--total-iter 300000 \
--lr-scheduler 200000 \
--nb-code 392 \
--down-t 2 \
--depth 3 \
--dilation-growth-rate 3 \
--out-dir eval_output \
--dataname t2m \
--vq-act relu \
--exp-name Test/exp_pg_tokenizer \
--use-keywords \
--dec-checkpoint-folder ./exp_result/exp_pg_tokenizer/{date1} \
--output-emb-width 392
```

## Eval Base Transformer

```
python eval_base_trans.py \
--exp-name Test/exp_base_trans_eval_fast_mode \
--batch-size 256 \
--num-layers 9 \
--embed-dim-gpt 1024 \
--nb-code 392 \
--n-head-gpt 16 \
--block-size 62 \
--ff-rate 4 \
--drop-out-rate 0.1 \
--out-dir eval_output \
--total-iter 300000 \
--lr-scheduler 150000 \
--lr 0.0001 \
--dataname t2m \
--down-t 2 \
--depth 3 \
--eval-iter 10000 \
--pkeep 0.5 \
--dilation-growth-rate 3 \
--vq-act relu \
--eval-mode fast \
--mm_mode \
--output-emb-width 392 \
--dec-checkpoint-folder ./exp_result/exp_pg_tokenizer/{date1} \
--t2m-checkpoint-folder ./exp_result/exp_base_trans/{date2} \
--use-keywords
```

## Eval Refine Transformer

```
python eval_refine_trans.py \
--exp-name Test/exp_refine_trans \
--batch-size 32 \
--num-layers 6 \
--nb-code 392 \
--n-head-gpt 8 \
--block-size 62 \
--ff-rate 4 \
--embed-dim-gpt 1024 \
--drop-out-rate 0.1 \
--out-dir eval_output \
--total-iter 300000 \
--lr-scheduler 150000 \
--lr 0.0002 \
--dataname t2m \
--down-t 2 \
--depth 3 \
--eval-iter 10000 \
--pkeep 0.5 \
--num_workers 4 \
--dilation-growth-rate 3 \
--vq-act relu \
--share-weight \
--codes-folder-name codes \
--output-emb-width 392 \
--dec-checkpoint-folder ./exp_result/exp_pg_tokenizer/{date1} \
--t2m-checkpoint-folder ./exp_result/exp_base_trans/{date2} \
--residual-t2m-checkpoint-folder ./exp_result/exp_refine_trans/{date3} \
--use-keywords
```

## Motion Editing

## References

### Codes

