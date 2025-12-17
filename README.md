# Pose-Guided Residual Refinement for Interpretable Text-to-Motion Generation and Editing
This is official implementation for "Pose-Guided Residual Refinement for Interpretable Text-to-Motion Generation and Editing"

## Demo Pages

You can find demo page about motion generation and editing result in [here](https://jayze3736.github.io/PGR2M_Demo/)

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

### Dependencies

Run the following commands to download the required components
```
bash dataset/prepare/download_glove.sh
bash dataset/prepare/download_extractor.sh
bash dataset/prepare/download_smpl.sh
```

```
pip install -r requirements.txt
```

### Fine-grained Descriptions & Pose Codes

For the fine-grained description and pose codes, please follow the instructions for downloading and preprocessing [here](https://github.com/yh2371/CoMo/tree/main)   

You can use the following command to convert the motion into pose codes
```
bash dataset/prepare/parse_motion.sh
```

and you can use the following command to download the fine-grained keywords.
```
bash dataset/prepare/download_keywords.sh
```


### Pre-trained Models

If you want to download the pretrained model weights, run the following:

```
bash dataset/prepare/download_model.sh
```

then you can find pretrained model and some arguments in "pretrained/" folder and looks like

```
./pretrained/
├── exp_base_transformer/
│    ├── arguments.yaml
│    └── net_best_fid.pth
├── exp_pg_tokenizer/
│    ├── arguments.yaml
│    └── net_best_fid.pth
└── exp_refine_transformer/
     ├── arguments.yaml
     └── net_best_fid.pth
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

## Evaluate PG(PoseGuided) Tokenizer

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

## Evaluate Base Transformer

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

## Evaluate Refine Transformer

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

## Inference

You can run inference with the model by referring to "inference.ipynb". Please see the notebook for detailed instructions and code.

## Motion Editing

For code that performs motion editing using the ChatGPT API, please refer to "motion_editing_with_gpt.ipynb"

## References

```bibtex
@inproceedings{huang2024controllable,
  title={Como: Controllable motion generation through language guided pose code editing},
  author={Huang, Yiming and Wan, Weilin and Yang, Yue and Callison-Burch, Chris and Yatskar, Mark and Liu, Lingjie},
  booktitle={European Conference on Computer Vision},
  pages={180--196},
  year={2024},
  organization={Springer}
}
```

```bibtex
@inproceedings{guo2024momask,
  title={Momask: Generative masked modeling of 3d human motions},
  author={Guo, Chuan and Mu, Yuxuan and Javed, Muhammad Gohar and Wang, Sen and Cheng, Li},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1900--1910},
  year={2024}
}
```

```bibtex
@inproceedings{zhang2023generating,
  title={Generating human motion from textual descriptions with discrete representations},
  author={Zhang, Jianrong and Zhang, Yangsong and Cun, Xiaodong and Zhang, Yong and Zhao, Hongwei and Lu, Hongtao and Shen, Xi and Shan, Ying},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={14730--14740},
  year={2023}
}
```