import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for Amass',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ## dataloader
    
    parser.add_argument('--dataname', type=str, default='t2m', help='dataset directory')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--fps', default=[20], nargs="+", type=int, help='frames per second')
    parser.add_argument('--seq-len', type=int, default=64, help='training motion length')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--dist_eval', action='store_true', default=True,
                        help='Enabling distributed evaluation')
    
    ## optimization
    parser.add_argument('--total-iter', default=100000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[60000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--lr-scheduler-type', default='MultiStepLR', type=str, choices=['MultiStepLR', 'CosineDecayScheduler'], help='learning rate scheduler type')
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    
    parser.add_argument('--weight-decay', default=1e-6, type=float, help='weight decay') 
    parser.add_argument('--decay-option',default='all', type=str, choices=['all', 'noVQ'], help='disable weight decay on codebook')
    parser.add_argument('--optimizer',default='adamw', type=str, choices=['adam', 'adamw'], help='disable weight decay on codebook')
    
    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=512, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--causal', action='store_true', help='whether use causal conv')

    ## gpt arch
    parser.add_argument("--block-size", type=int, default=201, help="seq len")
    parser.add_argument("--embed-dim-gpt", type=int, default=1024, help="embedding dimension")
    parser.add_argument("--clip-dim", type=int, default=512, help="latent dimension in the clip feature")
    parser.add_argument("--num-layers", type=int, default=9, help="nb of transformer layers")
    parser.add_argument("--n-head-gpt", type=int, default=16, help="nb of heads")
    parser.add_argument("--ff-rate", type=int, default=4, help="feedforward size")
    parser.add_argument("--drop-out-rate", type=float, default=0.1, help="dropout ratio in the pos encoding")
    parser.add_argument("--tie-weights", action='store_true', help="tie the weights of the lm head and the transformer")
    

    ## text encoder 
    parser.add_argument("--text_encode", type=str, default='clip', choices = ['clip', 'flan-t5-xxl', 'flan-t5-xl'], help="eps for optimal transport")
    parser.add_argument("--text_sum_way", type=str, default=None, choices = ['cls', 'mean', 'sum'], help="eps for optimal transport")
    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset', 'FSQ'], help="eps for optimal transport")
    parser.add_argument('--quantbeta', type=float, default=1.0, help='dataset directory')

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume vq pth')
    parser.add_argument("--resume-trans", type=str, default=None, help='resume gpt pth')
    
    
    ## output directory 
    parser.add_argument('--out-dir', type=str, default='output_GPT_Final/', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')
    parser.add_argument('--vq-name', type=str, default='exp_debug', help='name of the generated dataset .npy, will create a file inside out-dir')
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-metric-iter', default=10000, type=int, help='evaluation frequency')
    parser.add_argument('--eval-loss-iter', default=10000, type=int, help='evaluation frequency')
    parser.add_argument('--save-iter', default=10000, type=int, help='save frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training. ')
    parser.add_argument("--if-maxtest", action='store_true', help="test in max")
    parser.add_argument('--pkeep', type=float, default=0.5, help='keep rate for gpt training')

    # mask for cfg
    parser.add_argument('--root_cond_prob', type=float, default=0.1, help='mask for cfg root')
    parser.add_argument('--text_cond_prob', type=float, default=0.1, help='mask for cfg text')

    # debug
    parser.add_argument('--debug', action='store_true', help='debug mode')

    # llama args
    parser.add_argument('--pretrained_llama', type=str, default='7B', choices=['44M', '111M', '343M', '775M', '1B', '3B', '7B', '13B', '30B', '65B'], help='pretrained llama model')

    ## motionx
    parser.add_argument('--motion_type', type=str, default='vector_263', help='motion type')
    parser.add_argument('--text_type', type=str, default='texts', help='text type')
    parser.add_argument('--version', type=str, default='version1', help='version')


    ## loss type
    parser.add_argument('--loss_type', type=str, default='ce', help='loss type')

    # other
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'], help='mixed precision')
    parser.add_argument('--checkpoint', type=str, default='60000', help='mixed precision')
    
    return parser.parse_args()