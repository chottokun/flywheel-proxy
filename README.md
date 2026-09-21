# Flywheel Proxy

Logit-Router 統合型自立進化 LLM 透過プロキシ。

## 特徴
- **Stage 1**: Prefill Sliced LM-Head によるゼロショット高速ルーティング（Qwen2.5-1.5B / 0.5B）
- **Stage 2**: 収集データによる Ruri-v3-30m（ModernBERT-ja）+ SentencePiece Lite の完全 CPU 完結ルーティング
- **ストリーミング Tee 収集**: 低遅延なクライアント生中継と事後品質評価用テキスト収集を両立
- **クライアント切断遮断**: クライアント中断検知時に上流接続を即座に破棄（`aclose()`）
- **高耐久 SQLite ストレージ**: WAL モード + `asyncio.Queue` + バッチバルク書き込みによるロック競合 0% 設計
- **3 層コスト最適化**: アクティブサンプリング（5%）× カスケード判定 × Batch API による LLM-as-a-Judge コスト抑制

詳細は [`plan/flywheel-proxy.md`](plan/flywheel-proxy.md) および [`plan/implementation-steps-and-critique.md`](plan/implementation-steps-and-critique.md) を参照。
