import os
import urllib.request
import torch
import torch.nn as nn
from safetensors.torch import load_model

from config import Config
from train import Model

MODEL_URL = "https://huggingface.co/VOICEVOX/kanalizer-model/resolve/v5/model/c2k.safetensors"
MODEL_PATH = "c2k.safetensors"

CONFIG_DICT = {
    "train_data": "./vendor/v3.jsonl",
    "test_ratio": 0.01,
    "eval_data": "./vendor/unidic_words.jsonl",
    "eval_max_words": None,
    "dim": 256,
    "max_epochs": 15,
    "num_last_models_to_keep": 2,
    "num_best_models_to_keep": 2,
    "seed": 0,
    "optimizer_lr": 0.001,
    "use_layernorm": True,
    "weight_decay": 0.2
}

class CustomDynamicMHA(nn.Module):
    def __init__(self, mha_module):
        super().__init__()
        self.in_proj_weight = mha_module.in_proj_weight
        self.in_proj_bias = mha_module.in_proj_bias
        self.out_proj = mha_module.out_proj
        self.embed_dim = mha_module.embed_dim
        self.num_heads = mha_module.num_heads
        self.head_dim = self.embed_dim // self.num_heads

    def forward(self, query, key, value):
        B, Tq, D = query.size()
        B, Ts, D = key.size()

        q_w, k_w, v_w = self.in_proj_weight.chunk(3, dim=0)
        q_b, k_b, v_b = self.in_proj_bias.chunk(3, dim=0)

        q = nn.functional.linear(query, q_w, q_b)
        k = nn.functional.linear(key, k_w, k_b)
        v = nn.functional.linear(value, v_w, v_b)

        q = q.view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Ts, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Ts, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Tq, D)

        return self.out_proj(attn_output)


class EncoderWrapper(nn.Module):
    def __init__(self, model: Model):
        super().__init__()
        self.model = model

    def forward(self, src):
        src_emb = self.model.e_emb(src)
        enc_out, _ = self.model.encoder(src_emb)
        enc_out = self.model.encoder_norm(enc_out)
        enc_out = self.model.encoder_fc(enc_out)
        return enc_out


class DecoderStepWrapper(nn.Module):
    def __init__(self, model: Model):
        super().__init__()
        self.model = model
        self.attn = CustomDynamicMHA(model.attn)

    def forward(self, dec_input, enc_out, h1, h2):
        dec_emb = self.model.k_emb(dec_input)
        dec_out, h1_new = self.model.pre_decoder(dec_emb, h1)
        dec_out = self.model.pre_dec_norm(dec_out)
        
        attn_out = self.attn(dec_out, enc_out, enc_out)
        attn_out = self.model.attn_norm(attn_out)
        
        x = torch.cat([dec_out, attn_out], dim=-1)
        x, h2_new = self.model.post_decoder(x, h2)
        x = self.model.post_dec_norm(x)
        logits = self.model.fc(x)
        
        return logits, h1_new, h2_new


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model from {MODEL_URL}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")

    config = Config.from_dict(CONFIG_DICT)
    model = Model(config)
    load_model(model, MODEL_PATH)
    model.eval()

    batch_size = 1
    seq_len = 5
    dim = config.dim
    
    dummy_src = torch.zeros((batch_size, seq_len), dtype=torch.long)
    dummy_dec_input = torch.zeros((batch_size, 1), dtype=torch.long)
    dummy_enc_out = torch.zeros((batch_size, seq_len, dim), dtype=torch.float32)
    dummy_h1 = torch.zeros((1, batch_size, dim), dtype=torch.float32)
    dummy_h2 = torch.zeros((1, batch_size, dim), dtype=torch.float32)

    print("Exporting Encoder...")
    with torch.no_grad():
        torch.onnx.export(
            EncoderWrapper(model),
            (dummy_src,),
            "kanalizer_encoder.onnx",
            input_names=["src"],
            output_names=["enc_out"],
            dynamic_axes={
                "src": {0: "batch_size", 1: "seq_len"},
                "enc_out": {0: "batch_size", 1: "seq_len"}
            },
            opset_version=14,
        )

    print("Exporting Decoder Step...")
    with torch.no_grad():
        torch.onnx.export(
            DecoderStepWrapper(model),
            (dummy_dec_input, dummy_enc_out, dummy_h1, dummy_h2),
            "kanalizer_decoder_step.onnx",
            input_names=["dec_input", "enc_out", "h1", "h2"],
            output_names=["logits", "h1_new", "h2_new"],
            dynamic_axes={
                "dec_input": {0: "batch_size"},
                "enc_out": {0: "batch_size", 1: "seq_len"},
                "h1": {1: "batch_size"},
                "h2": {1: "batch_size"},
                "logits": {0: "batch_size", 1: "dec_len"},
                "h1_new": {1: "batch_size"},
                "h2_new": {1: "batch_size"}
            },
            opset_version=14,
        )
    print("Done! Generated 'kanalizer_encoder.onnx' and 'kanalizer_decoder_step.onnx'.")

if __name__ == "__main__":
    main()
