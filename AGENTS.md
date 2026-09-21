# AGENTS.md

## Strict Constraints & Rules

1. **Package Manager & Execution**:
   - Python環境の構築・パッケージ管理・コマンド実行には **必ず `uv`** を使用すること（`uv run ...`, `uv pip ...`, `uv sync` 等）。
   - `pip` や `conda` の直接使用は禁止。

2. **Security & Data Privacy (GitHub Leak Prevention)**:
   - **機微情報（APIキー、トークン、秘密鍵、パスワード）やローカル環境の絶対パス・個人情報をGitHubに絶対にコミット/プッシュしないこと**。
   - `.env` や秘密情報は `.gitignore` で確実に除外されていることをコミット前に必ず確認すること（環境変数の雛形はダミー値の `.env.example` のみ共有）。
   - ログファイル、ローカルモデルファイル（`models/`）、SQLiteデータベースファイル（`*.db`、`*.db-wal` 等）を追跡対象に含めないこと。

3. **Jules CLI Operation**:
   - Julesのセッション状態監視ポーリングは、サーバー負荷やトークン消費を抑えるため、**5分程度の間隔**で行うこと。
