# 進捗まとめ (2026-04-09)

## 環境クリーンアップ (RunPod)
- コンテナディスク `/` が 16G/20G (77%) で逼迫していた
- 原因: `/root/.cache/huggingface` が 16G (parameter-golf データセットの重複DL)
- 対処: HFキャッシュを削除 (データ本体は `/workspace/parameter-golf/data` に既存)
- 結果: `/` = **695M / 20G (4%)** に回復

## Git 状態 (`/workspace/parameter-golf`)
- ブランチ: `main` (origin/main と一致)
- 変更:
  - `train_gpt.py` modified — `+6 / -8` (14行)
  - `final_model.int8.ptz` (untracked)
  - `final_model.pt` (untracked)
- 未コミット。セーブしたい場合は commit & push が必要。

## ベースライン実行結果 (`logs/baseline_sp1024.log`)

### 設定
- attention: GQA (heads=8, kv_heads=4), flash SDPA
- tie_embeddings: True
- LR: embed=0.05, matrix=0.04, scalar=0.04, head=0.0
- batch_tokens=524288, seq_len=1024, iters=20000 (cap)
- wallclock cap: **600s**
- seed: 1337

### 学習
- 早期停止: wallclock cap で **step 587 / 20000** で停止
- step time: ~1023 ms/step
- step 0  : val_loss=6.9344 / val_bpb=4.1069
- step 587: val_loss=**2.4190** / val_bpb=**1.4327**
- peak mem: 13169 MiB allocated / 14482 MiB reserved

### 提出物サイズ
| 項目 | サイズ |
|---|---|
| Serialized model | 67,224,578 B |
| Code | 47,819 B |
| **Total submission** | **67,272,397 B (~64 MiB)** |
| Serialized model int8+zlib | 10,584,970 B (3.91x圧縮) |
| **Total submission int8+zlib** | **10,632,789 B (~10.1 MiB)** |

### int8+zlib ラウンドトリップ評価
- val_loss = **2.43199382**
- val_bpb  = **1.44036373**
- 量子化劣化: bpb +0.0077 (1.4327 → 1.4404)
- eval_time: 32,104 ms

## 次にやること候補
1. `train_gpt.py` の変更 (+6/-8) を commit & push してセーブポイント化
2. ベースライン bpb 1.4404 を基準に改善実験
3. 既存記録 `val_bpb 1.11473` (AR Self-Gen GPTQ + XSA + BigramHash) との差分分析
