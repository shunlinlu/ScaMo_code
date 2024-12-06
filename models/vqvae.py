import torch.nn as nn
from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
from models.LFQ import LFQ
from models.FSQ import FSQ
# from models.causal_cnn import CausalEncoder, CausalDecoder


class VQVAE_251(nn.Module):
    def __init__(self,
                 args,
                 nb_code=1024,
                 code_dim=512, # 512
                 output_emb_width=512, # 512
                 down_t=3, # 2
                 stride_t=2, # 2
                 width=512, # 512
                 depth=3, # 3
                 dilation_growth_rate=3, # 3
                 activation='relu',
                 norm=None):
        
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = args.quantizer
        
        # import pdb; pdb.set_trace()
        if args.causal:
            print('use causal conv !!!')
            self.encoder = CausalEncoder(251 if args.dataname == 'kit' else 263, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
            self.decoder = CausalDecoder(251 if args.dataname == 'kit' else 263, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        else:
            self.encoder = Encoder(251 if args.dataname == 'kit' else 263, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
            self.decoder = Decoder(251 if args.dataname == 'kit' else 263, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

        if args.quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, args)
        elif args.quantizer == "orig":
            self.quantizer = Quantizer(nb_code, code_dim, 1.0)
        elif args.quantizer == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim, args)
        elif args.quantizer == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim, args)
        elif args.quantizer == "LFQ":
            self.quantizer = LFQ(codebook_size=args.nb_code, dim=code_dim)
        elif args.quantizer == 'BSQ':
            self.quantizer = LFQ(codebook_size=args.nb_code, dim=code_dim, spherical=True)
        elif args.quantizer == 'FSQ':
            if args.nb_code == 256:
                levels = [8, 6, 5]
            elif args.nb_code == 512:
                levels = [8, 8, 8]
            elif args.nb_code == 1024:
                levels = [8, 5, 5, 5]
            elif args.nb_code == 2048:
                levels = [8, 8, 6, 5]
            elif args.nb_code == 4096:
                levels = [7, 5, 5, 5, 5]
            elif args.nb_code == 16384:
                levels = [8, 8, 8, 6, 5]
            elif args.nb_code == 65536:
                levels = [8, 8, 8, 5, 5, 5]
            else:
                raise ValueError('Unsupported number of codebooks')
            self.quantizer = FSQ(levels=levels, dim=code_dim)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x


    def encode(self, x):
        # import pdb; pdb.set_trace()
        if self.quant in ["LFQ", "BSQ", "FSQ"]:
            N, T, _ = x.shape
            x_in = self.preprocess(x)
            x_encoder = self.encoder(x_in)
            _, code_idx, _, _, _, _ = self.quantizer(x_encoder)
            code_idx = code_idx.view(N, -1)
            return code_idx
        else:
            N, T, _ = x.shape
            x_in = self.preprocess(x)
            x_encoder = self.encoder(x_in)
            x_encoder = self.postprocess(x_encoder)
            x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
            code_idx = self.quantizer.quantize(x_encoder)
            code_idx = code_idx.view(N, -1)
            return code_idx


    def forward(self, x):
        
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in) # 256, 512, 16
        # import pdb; pdb.set_trace() 
        ## quantization
        if self.quant in ["LFQ", "BSQ"]:
            x_quantized, _, loss, perplexity, activate, indices = self.quantizer(x_encoder)
        elif self.quant == "FSQ":
            x_quantized, _, loss, perplexity, activate, indices = self.quantizer(x_encoder)
        else:
            x_quantized, loss, perplexity, activate, indices  = self.quantizer(x_encoder) # (256, 512, 16)
        # import pdb; pdb.set_trace()
        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)

        return x_out, loss, perplexity, activate, indices


    def forward_decoder(self, x):
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        
        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out



class HumanVQVAE(nn.Module):
    def __init__(self,
                 args,
                 nb_code=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        
        self.nb_joints = 21 if args.dataname == 'kit' else 22
        self.vqvae = VQVAE_251(args, nb_code, code_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

    def encode(self, x):
        b, t, c = x.size()
        quants = self.vqvae.encode(x) # (N, T)
        return quants

    def forward(self, x):

        x_out, loss, perplexity, activate, indices = self.vqvae(x)
        
        return x_out, loss, perplexity, activate, indices

    def forward_decoder(self, x):
        x_out = self.vqvae.forward_decoder(x)
        return x_out
        