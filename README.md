# kanalizer-onnx

英単語から読みを推測するライブラリ。

[VOICEVOX/kanalizer](https://github.com/VOICEVOX/kanalizer) のフォークです。

`ort` を使った推論コードの例は、[ここ](https://github.com/o24s/haqumei/tree/main/haqumei-kanalizer) にあります。

関連Issue：[VOICEVOX/voicevox_project#65](https://github.com/VOICEVOX/voicevox_project/issues/65)

## リポジトリ構造

このリポジトリは以下の構造になっています。

- `infer/`：読みを推測するためのコード。
- `train/`：モデルを学習・ONNXエクスポートするためのコード。
- `dataset/`：データセットを生成するためのコード。

## ベンチマーク

バックエンドは ndarray (kanalizer-rs) と ONNX Runtime の2種類。

| 入力 | ndarray | ONNX Runtime | Speedup |
|---|---|---|---|
| `hi` | 863 µs | 386 µs | 2.24x |
| `hello` | 1,079 µs | 478 µs | 2.26x |
| `international` | 3,157 µs | 1,493 µs | 2.11x |

ベンチマークの実行：

```bash
cd train
uv sync
uv run src/export_onnx.py
cd ../bench
cargo bench
```

## Hugging Face

変換済み ONNX モデルは [fulmo/kanalizer-model-onnx](https://huggingface.co/fulmo/kanalizer-model-onnx) で公開しています。

## ライセンス

このリポジトリのコードはMITライセンスのもとで公開されています。
