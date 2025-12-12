import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for Amass',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ## cfg
    parser.add_argument('--cfg', type=str, default='./', help='dataset directory')

    ## dataloader
    parser.add_argument('--dataname', type=str, default='kit', help='dataset directory')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--fps', default=[20], nargs="+", type=int, help='frames per second')
    parser.add_argument('--seq-len', type=int, default=64, help='training motion length')
    parser.add_argument("--codes-folder-name", type=str, default='codes', help="code foler 이름")
    parser.add_argument('--soft-label-folder-name', type=str, default=None, help="soft label folder name")

    ## optimization
    parser.add_argument('--total-iter', default=100000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[60000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.5, type=float, help="learning rate decay")
    
    parser.add_argument('--weight-decay', default=1e-6, type=float, help='weight decay') 
    parser.add_argument('--decay-option',default='all', type=str, choices=['all', 'noVQ'], help='disable weight decay on codebook')
    parser.add_argument('--optimizer',default='adamw', type=str, choices=['adam', 'adamw'], help='disable weight decay on codebook')
    
    ## encdec arch
    parser.add_argument("--code-dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=512, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=3, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    

    ## transformer arch
    parser.add_argument("--block-size", type=int, default=62, help="seq len")
    parser.add_argument("--embed-dim-gpt", type=int, default=1024, help="embedding dimension")
    parser.add_argument("--clip-dim", type=int, default=512, help="latent dimension in the clip feature")
    parser.add_argument("--num-layers", type=int, default=2, help="nb of transformer layers")
    parser.add_argument("--n-head-gpt", type=int, default=8, help="nb of heads")
    parser.add_argument("--ff-rate", type=int, default=4, help="feedforward size")
    parser.add_argument("--drop-out-rate", type=float, default=0.1, help="dropout ratio in the pos encoding")


    ## resume
    # parser.add_argument("--resume-pth", type=str, default=None, help='resume vq pth')
    parser.add_argument("--resume-trans", type=str, default=None, help='resume gpt pth')
    parser.add_argument("--resume-r-trans", type=str, default=None, help='resume residual transformer pth')
    
    
    ## output directory 
    parser.add_argument('--out-dir', type=str, default='output_GPT_Final/', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')
    parser.add_argument('--vq-name', type=str, default='exp_debug', help='name of the generated dataset .npy, will create a file inside out-dir')
    
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=5000, type=int, help='evaluation frequency')
    parser.add_argument('--eval-loss-iter', default=1000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training. ')
    parser.add_argument("--if-maxtest", action='store_true', help="test in max")
    parser.add_argument('--pkeep', type=float, default=1.0, help='keep rate for gpt training')
    parser.add_argument('--min-sampling-prob', type=float, default=0.1, help='minimum sampling probability for masking')
    parser.add_argument('--num_workers', default=16, type=int, help='number of workers')

    parser.add_argument('--use-keywords', action='store_true', help='whether to use fine-grained keyword')
    parser.add_argument('--use-word-only', action='store_true', help='whether to use POS in word embedding')
    parser.add_argument('--cfg-cla-path', type=str, default=None, help='Configuration file path of category aggregator')
    parser.add_argument('--mm_mode', action='store_true', help="Whether to evlalute MModality metrics")
    parser.add_argument('--val-shuffle', action='store_true', help='whether to shuffle validation set')
    parser.add_argument("--cat-mode", type=str, default=None, help='')
    parser.add_argument('--loss-cfg-path', type=str, help='loss config file path')
    parser.add_argument('--log-cat-right-num', action='store_true', help='whether to log accuracy by category')
    parser.add_argument('--token-emb-layer', default='default', choices=['default', 'group-aware-v1', 'group-aware-v2'], help='Methods about group-aware')
    parser.add_argument('--disable-pos-emb-additional', action='store_true', help='Methods about group-aware')
    parser.add_argument('--eval-masking', action='store_true', help='Enable Masking on Validation')
    parser.add_argument('--start-warm-up', action='store_true', help='Enable Warm Up')
    parser.add_argument('--scheduled-sampling', action='store_true', help='Enable scheduled-sampling-prob')

    parser.add_argument('--residual-t2m-checkpoint-folder', type=str, default=None, help='Path to the checkpoint folder for the base T2M Transformer')
    parser.add_argument('--t2m-checkpoint-folder', type=str, default=None, help='Path to the checkpoint folder for the base T2M Transformer')
    parser.add_argument("--dec-checkpoint-folder", type=str, default=None, help='./')

    parser.add_argument('--eval-cache-file-name', type=str, help='file name ofsa evaluation cache file', default=None)
    parser.add_argument('--train-cache-file-name', type=str, help='file name of training cache file', default=None)

    parser.add_argument('--mask-residual-code', action='store_true', help='Enable masking of residual codes')
    parser.add_argument('--masking-prob', type=float, default=0.15, help='masking probability for residual codes')
    parser.add_argument('--schedule-masking-prob', action='store_true', help='Enable scheduled masking probability')
    parser.add_argument('--schedule-pkeep', action='store_true', help='Enable scheduled pose code masking probability')
    parser.add_argument('--clip-grad', action='store_true', help='Enable gradient clipping')
    parser.add_argument('--share-weight', action='store_true', help='Enable shared projection weight')
    # parser.add_argument("--loss-name", type=str, default=None, choices=[None, 'cace'], help='cace:Catogory Aware Cross Entropy Loss')
    parser.add_argument('--freeze-pose-code-emb', action='store_true', help='Freeze the pose code embedding layer during training')
    parser.add_argument('--load-pretrained-pose-code-emb', action='store_true', help='Load pretrained pose code embedding from the base T2M model')
    parser.add_argument("--eval-reference", type=str, default='net_best_fid', choices=['net_best_fid', 'net_best_matching', 'net_best_top1'], help='cace:Catogory Aware Cross Entropy Loss')
    parser.add_argument("--eval-mode", type=str, default='standard', choices=['standard', 'fast'], help='Evaluation mode: standard or fast')

    
    return parser.parse_args()
    # return parser.parse_known_args()