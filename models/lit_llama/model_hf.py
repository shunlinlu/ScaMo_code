"""Full definition of a LLaMA Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
"""
# mypy: ignore-errors
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self
from typing import Optional
from transformers.modeling_utils import PreTrainedModel
from torch.distributions import Categorical
import torch.nn.functional as F

@dataclass
class LLaMAHFConfig:
    block_size: int = 4096
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(**llama_configs[name])


llama_configs = {
    "44M": dict(n_layer=8, n_head=8, n_embd=512),
    "111M": dict(n_layer=12, n_head=12, n_embd=768),
    "343M": dict(n_layer=24, n_head=16, n_embd=1024),
    "775M": dict(n_layer=36, n_head=20, n_embd=1280),
    "1B": dict(n_layer=48, n_head=24, n_embd=1536),
    "3B": dict(n_layer=24, n_head=32, n_embd=3200),
    "7B": dict(n_layer=32, n_head=32, n_embd=4096),
    "13B": dict(n_layer=40, n_head=40, n_embd=5120),
    "30B": dict(n_layer=60, n_head=52, n_embd=6656),
    "65B": dict(n_layer=80, n_head=64, n_embd=8192)
}


class LLaMAHF(nn.Module):
# class LLaMA(PreTrainedModel):
    def __init__(self, config: LLaMAHFConfig) -> None:
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size-1, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd),
            )
        )
        
        # self.llama_proj = nn.Sequential(
        #     nn.Linear(256, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, config.n_embd)
        # )

        self.llama_proj = nn.Linear(config.clip_dim, config.n_embd)
        # import pdb; pdb.set_trace()
        # self.motion_proj = nn.Sequential(
        #     nn.Linear(config.n_embd, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 256)
        # )
        # self.motion_proj = nn.Linear(config.n_embd, 512)
        if config.tie_weights:
            self._tie_or_clone_weights(self.lm_head, self.transformer.wte)

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights depending of whether we are using TorchScript or not"""
        # import pdb; pdb.set_trace()
        output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (
                    0,
                    output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                ),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings

    def get_input_embeddings(self):
        return self.transformer.wte
    
    def set_input_embeddings(self, value):
        self.transformer.wte = value

    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))
            
    @torch.no_grad()
    def sample(self, clip_feature, y_mask, if_categorial=False):
        # import pdb; pdb.set_trace()
        for k in range(51):
            if k == 0:
                x = []
            else:
                x = xs
            # import pdb; pdb.set_trace()
            logits = self.forward_sample(x, clip_feature, y_mask)
            # import pdb; pdb.set_trace()
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            if if_categorial:
                dist = Categorical(probs)
                idx = dist.sample()
                if idx == self.config.vocab_size -2:
                    break
                idx = idx.unsqueeze(-1)
            else:
                _, idx = torch.topk(probs, k=1, dim=-1)

                if idx[0] == self.config.vocab_size - 2:
                    break

            # append to the sequence and continue
            if k == 0:
                xs = idx
            else:
                xs = torch.cat((xs, idx), dim=1)
            
            if k == 50:
                return xs[:, :-1]
        
        if k == 0:
            return torch.ones(1,1).to(clip_feature.device).long()
        else:
            return xs

    def forward_sample(self, idx: torch.Tensor, clip_feature: torch.Tensor, y_mask) -> torch.Tensor:
        # idx (bs, 50)
        # clip_feature (bs, text_length, clip_dim)
        # import pdb; pdb.set_trace()
        text_length = clip_feature.shape[1]
        if len(idx) == 0:
            x = self.llama_proj(clip_feature)[:, :int(y_mask[0].sum()), :]
        else:
            _, t = idx.size()
            assert (
                t <= self.config.block_size
            ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            # forward the LLaMA model itself
            x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

            # replace text_length tokens with clip_feature
            # expanded_mask = y_mask.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            # result = torch.where(expanded_mask == 1, self.llama_proj(clip_feature), x[:, :text_length, :])
            # result = torch.cat((result, x[:, text_length:, :]), dim=1)
            # x = result
            # import pdb; pdb.set_trace()
            x = torch.cat((self.llama_proj(clip_feature)[:, :int(y_mask[0].sum()), :],x), dim=1)

        for block in self.transformer.h:
            x = block(x, y_mask)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)  # (b, t, vocab_size)

        return logits

    def forward(self, idx: torch.Tensor, clip_feature: torch.Tensor, y_mask) -> torch.Tensor:
        # idx (bs, 50)
        # clip_feature (bs, text_length, clip_dim)
        # import pdb; pdb.set_trace()
        text_length = clip_feature.shape[1]
        if len(idx) == 0:
            x = self.llama_proj(clip_feature)[:, :int(y_mask[0].sum()), :]
        else:
            _, t = idx.size()
            assert (
                t <= self.config.block_size
            ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            # forward the LLaMA model itself
            x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

            # replace text_length tokens with clip_feature
            # import pdb; pdb.set_trace()
            expanded_mask = y_mask.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            result = torch.where(expanded_mask == 1, self.llama_proj(clip_feature), x[:, :text_length, :])
            result = torch.cat((result, x[:, text_length:, :]), dim=1)
            x = result

        for block in self.transformer.h:
            x = block(x, y_mask)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)  # (b, t, vocab_size)

        return logits
    
    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None, using_old_initilization: bool = False
    ) -> nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The new number of tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value.If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc

        Return:
            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.
        """
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # Update base model and current model config
        self.config.vocab_size = model_embeds.weight.shape[0]
        self.vocab_size = model_embeds.weight.shape[0]

        # Tie weights again if needed
        # self.tie_weights()

        return model_embeds
    
    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens, pad_to_multiple_of)
        old_embeddings_requires_grad = old_embeddings.weight.requires_grad
        new_embeddings.requires_grad_(old_embeddings_requires_grad)
        self.set_input_embeddings(new_embeddings)

        # Update new_num_tokens with the actual size of new_embeddings
        if pad_to_multiple_of is not None:
            # if is_deepspeed_zero3_enabled():
            #     import deepspeed

            #     with deepspeed.zero.GatheredParameters(new_embeddings.weight, modifier_rank=None):
            #         new_num_tokens = new_embeddings.weight.shape[0]
            # else:
            new_num_tokens = new_embeddings.weight.shape[0]

        # if word embeddings are not tied, make sure that lm head is resized as well
        # if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
        if self.get_output_embeddings() is not None and not False:
            old_lm_head = self.get_output_embeddings()
            new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
            # if hasattr(old_lm_head, "_hf_hook"):
            #     hook = old_lm_head._hf_hook
            #     add_hook_to_module(new_lm_head, hook)
            old_lm_head_requires_grad = old_lm_head.weight.requires_grad
            new_lm_head.requires_grad_(old_lm_head_requires_grad)
            self.set_output_embeddings(new_lm_head)

        return self.get_input_embeddings()
    
    def _get_resized_embeddings(
        self,
        old_embeddings: nn.Embedding,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (`torch.nn.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value. If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc


        Return:
            `torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
            `new_num_tokens` is `None`
        """

        if pad_to_multiple_of is not None:
            if not isinstance(pad_to_multiple_of, int):
                raise ValueError(
                    f"Asking to pad the embedding matrix to a multiple of `{pad_to_multiple_of}`, which is not and integer. Please make sure to pass an integer"
                )
            if new_num_tokens is None:
                new_num_tokens = old_embeddings.weight.shape[0]
            new_num_tokens = ((new_num_tokens + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        else:
            print(
                "You are resizing the embedding layer without providing a `pad_to_multiple_of` parameter. This means that the new embedding"
                f" dimension will be {new_num_tokens}. This might induce some performance reduction as *Tensor Cores* will not be available."
                " For more details about this, or help on choosing the correct value for resizing, refer to this guide:"
                " https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc"
            )

        if new_num_tokens is None:
            return old_embeddings

        # if is_deepspeed_zero3_enabled():
        if False:
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=None):
                old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        else:
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        # if old_num_tokens == new_num_tokens and not is_deepspeed_zero3_enabled():
        if old_num_tokens == new_num_tokens and not False:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. You"
                " should either use a different resize function or make sure that `old_embeddings` are an instance of"
                f" {nn.Embedding}."
            )

        # Build new embeddings

        # When using DeepSpeed ZeRO-3, we shouldn't create new embeddings with DeepSpeed init
        # because the shape of the new embedding layer is used across various modeling files
        # as well as to update config vocab size. Shape will be 0 when using DeepSpeed init leading
        # to errors when training.
        new_embeddings = nn.Embedding(
            new_num_tokens,
            old_embedding_dim,
            device=old_embeddings.weight.device,
            dtype=old_embeddings.weight.dtype,
        )

        # initialize all new embeddings (in particular added tokens)
        self._init_weights(new_embeddings)

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)

        # if is_deepspeed_zero3_enabled():
        if False:
            import deepspeed

            params = [old_embeddings.weight, new_embeddings.weight]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
        else:
            new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        return new_embeddings


    def _get_resized_lm_head(
        self, old_lm_head: nn.Linear, new_num_tokens: Optional[int] = None, transposed: Optional[bool] = False
    ) -> nn.Linear:
        """
        Build a resized Linear Module from a provided old Linear Module. Increasing the size will add newly initialized
        vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_lm_head (`torch.nn.Linear`):
                Old lm head liner layer to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the linear matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `torch.nn.Linear` module of the model without doing anything. transposed (`bool`, *optional*, defaults
                to `False`): Whether `old_lm_head` is transposed or not. If True `old_lm_head.size()` is `lm_head_dim,
                vocab_size` else `vocab_size, lm_head_dim`.

        Return:
            `torch.nn.Linear`: Pointer to the resized Linear Module or the old Linear Module if `new_num_tokens` is
            `None`
        """
        if new_num_tokens is None:
            return old_lm_head

        # if is_deepspeed_zero3_enabled():
        if False:
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_lm_head.weight, modifier_rank=None):
                old_num_tokens, old_lm_head_dim = (
                    old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
                )
        else:
            old_num_tokens, old_lm_head_dim = (
                old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
            )

        # if old_num_tokens == new_num_tokens and not is_deepspeed_zero3_enabled():
        if old_num_tokens == new_num_tokens and not False:
            return old_lm_head

        if not isinstance(old_lm_head, nn.Linear):
            raise TypeError(
                f"Old language model head is of type {type(old_lm_head)}, which is not an instance of {nn.Linear}. You"
                " should either use a different resize function or make sure that `old_lm_head` are an instance of"
                f" {nn.Linear}."
            )

        # Build new lm head
        new_lm_head_shape = (old_lm_head_dim, new_num_tokens) if not transposed else (new_num_tokens, old_lm_head_dim)
        has_new_lm_head_bias = old_lm_head.bias is not None

        # When using DeepSpeed ZeRO-3, we shouldn't create new embeddings with DeepSpeed init
        # because the shape of the new embedding layer is used across various modeling files
        # as well as to update config vocab size. Shape will be 0 when using DeepSpeed init leading
        # to errors when training.
        new_lm_head = nn.Linear(
            *new_lm_head_shape,
            bias=has_new_lm_head_bias,
            device=old_lm_head.weight.device,
            dtype=old_lm_head.weight.dtype,
        )

        # initialize new lm head (in particular added tokens)
        self._init_weights(new_lm_head)

        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)

        # if is_deepspeed_zero3_enabled():
        if False:
            import deepspeed

            params = [old_lm_head.weight, old_lm_head.bias, new_lm_head.weight, new_lm_head.bias]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                self._copy_lm_head_original_to_resized(
                    new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
                )
        else:
            self._copy_lm_head_original_to_resized(
                new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
            )

        return new_lm_head

    def _copy_lm_head_original_to_resized(
        self, new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
    ):
        # Copy old lm head weights to new lm head
        if not transposed:
            new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
        else:
            new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[:, :num_tokens_to_copy]

        # Copy bias weights to new lm head
        if has_new_lm_head_bias:
            new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(LLaMAHFConfig.from_name(name))


class Block(nn.Module):
    def __init__(self, config: LLaMAHFConfig) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        # self.attn = CausalSelfAttention(config)
        self.attn = LengthCausalSelfAttention(config)
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.rms_1(x), y_mask)
        x = x + self.mlp(self.rms_2(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LLaMAHFConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.rope_cache = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)

        if self.rope_cache is None:
            # cache for future forward calls
            self.rope_cache = build_rope_cache(
                seq_len=self.block_size,
                n_elem=self.n_embd // self.n_head, 
                dtype=x.dtype,
                device=x.device,
            )

        q = apply_rope(q, self.rope_cache)
        k = apply_rope(k, self.rope_cache)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        #  att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #  att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        #  att = F.softmax(att, dim=-1)
        #  y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)

        return y

class LengthCausalSelfAttention(nn.Module):
    def __init__(self, config: LLaMAHFConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.rope_cache = None

    def forward(self, x: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
        # import pdb; pdb.set_trace()
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)

        if self.rope_cache is None:
            # cache for future forward calls
            self.rope_cache = build_rope_cache(
                seq_len=self.block_size,
                n_elem=self.n_embd // self.n_head, 
                dtype=x.dtype,
                device=x.device,
            )

        # import pdb; pdb.set_trace()
        # q: 1, 16, 40 ,64
        # q: 128, 16, 106, 64
        q = apply_rope(q, self.rope_cache)
        k = apply_rope(k, self.rope_cache)

        # 创建注意力掩码    
        attn_mask = torch.ones(T, T, dtype=torch.bool, device=x.device)
        attn_mask = torch.tril(attn_mask)  # 创建下三角矩阵
        attn_mask = attn_mask.unsqueeze(0).expand(B, -1, -1)
        # import pdb; pdb.set_trace()

        text_mask = y_mask.unsqueeze(2)*y_mask.unsqueeze(1)
        text_mask = F.pad(text_mask, (0, T-y_mask.shape[1], 0, T-y_mask.shape[1]), mode='constant', value=0)
        # import pdb; pdb.set_trace()
        attn_mask = torch.logical_or(attn_mask, text_mask)
        # import pdb; pdb.set_trace()

        # 使用Flash Attention CUDA kernels进行高效注意力计算
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask.unsqueeze(1), dropout_p=0.0, is_causal=False)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 重新组合所有头的输出

        # 输出投影
        y = self.c_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, config: LLaMAHFConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3)
        N = 256
        # ensure n_hidden is multiple of N
        n_hidden = ((n_hidden - 1) // N) * N + N

        self.c_fc1 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        # norm_x = x.norm(2, dim=self.dim, keepdim=True)
        # rms_x = norm_x * d_x ** (-1. / 2)
        # x_normed = x / (rms_x + self.eps)
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed


def build_rope_cache(seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000) -> torch.Tensor:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)

    # Compute cache. Because polar only takes float32 or float64, we need to cast
    # when working with 16 bit floats (float16 or bfloat16)
    dtypes_requiring_casting = [torch.float16, torch.bfloat16, torch.int8]
    working_dtype = (
        torch.float32 if dtype in dtypes_requiring_casting else dtype
    )
    complex_dtype = (
        torch.complex32 if dtype in dtypes_requiring_casting else torch.complex64
    )
    cache = torch.polar(
        torch.ones_like(idx_theta).to(working_dtype), idx_theta.to(working_dtype)
    ).to(complex_dtype)
    return cache


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    x = x.transpose(1, 2)

    # truncate to support variable sizes
    T = x.size(1)
    rope_cache = rope_cache[:T]
    # import pdb; pdb.set_trace()
    # cast because `view_as_complex` does not support 16 bit tensors
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rope_cache = rope_cache.view(1, xc.size(1), 1, xc.size(3))
    x_out = torch.view_as_real(xc * rope_cache).flatten(3)
    return x_out.transpose(1, 2).type_as(x)
