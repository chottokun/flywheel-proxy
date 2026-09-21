---
name: llm-wiki-okf
description: docs/ 配下の OKF (Open Knowledge Format v0.2) 形式ナレッジベースを探索・構築・維持し、コードと同期させるプロシージャ
---

# LLM-Wiki (OKF v0.2) 運用プロシージャ

## 0. ナレッジベースの初期化 (Init Protocol)
「ナレッジベースを初期化して」「Wikiをセットアップして」などの指示を受けた場合、または `docs/` が存在しない場合：
1. リポジトリルートに `docs/`, `docs/raw/`, `docs/architecture/`, `docs/domain/`, `docs/infrastructure/`, `docs/references/` ディレクトリを作成する。
2. スキル配下の `templates/` から以下をコピーして配置する：
   - `templates/README.md.template` -> `docs/README.md`
   - `templates/log.md.template` -> `docs/log.md`
3. コピー完了後、`docs/log.md` の日付（`YYYY-MM-DD`）を本日の日付に置き換える。

## 1. 知識の参照 (Query Protocol)
ユーザーから実装・設計・調査の指示を受けたら、コードを変更する前に以下の順序でナレッジをロードします：
1. `docs/README.md`（または `docs/index.md`）を読み、該当するサブカテゴリを特定する。
2. 必要な概念ページ（例: `docs/domain/payment.md`）にアクセスし、仕様やデータ構造を理解する。
3. ナレッジに記載がない場合のみ `docs/raw/` のソースを参照する。

### 1.1 Trust Tier に基づく信頼度判断
概念ページを読む際、`verified` フィールドから以下の信頼度ティアを判別し、意思決定の参考にする：
- **human-reviewed**: `verified` に `human:<id>` のアクターが含まれる → 最も信頼度が高い。
- **machine-confirmed**: `verified` に非 `human:` アクターのみ含まれる → 中程度の信頼度。
- **unverified**: `verified` フィールドが存在しない → 信頼度は低いが、参照・利用は可能（リジェクト不可）。

### 1.2 鮮度チェック
概念ページの `stale_after` フィールドが存在し、現在日時がその値以降である場合、その概念は古くなっている可能性がある旨をユーザーに通知する。

## 2. ソース取り込みと計画見直し (Ingest Protocol)
`docs/raw/` に新しい仕様書や改訂された実装計画が追加された場合：
1. 追加された raw ソースを解析し、抽出された概念ごとに 1トピック=1ファイル で概念ページを作成・更新する。
2. 概念ページの YAML フロントマターには以下を記述する（OKF v0.2 準拠）：

   **必須フィールド:**
   - `type`: 概念の種別（例: `Concept`, `Architecture Decision`, `Data Model`, `Configuration`, `Attested Computation` など）

   **推奨フィールド:**
   - `title`: 人間が読みやすい表示名
   - `description`: 概念の1行要約。インデックス生成や検索スニペットに利用される
   - `resource`: 概念が説明する対象アセットの正規URI（具体的なリソースに紐づく場合のみ。抽象的な概念には不要）
   - `tags`: クロスカッティングな分類用タグのリスト

   **生成情報（generated）:**
   - `generated`: `{ by: <actor>, at: <ISO8601> }`（生成主体と日時）
     - Actor 命名規則: エージェントの場合は `<producer>/<version>`（例: `reference_agent/gemini-2.5-pro`）、人間の場合は `human:<id>`、自動処理の場合は `process:<id>`

   **検証情報（verified）:**
   - `verified`: 検証イベントのリスト。各エントリは `{ by: <actor>, at: <ISO8601> }` 形式
     - 単一の検証者の場合、リストのダッシュなしのベアマッピング形式 `verified: { by: ..., at: ... }` も許容する
     - 複数の独立した検証（人間のサインオフ + 夜間バッチなど）はリスト形式で記録する
   - 例：
     ```yaml
     verified:
       - { by: human:ahormati, at: 2026-06-25T09:00:00Z }
       - { by: process:finance-nightly, at: 2026-06-26T02:00:00Z }
     ```

   **ライフサイクル:**
   - `status`: ライフサイクル状態（`draft` | `stable` | `deprecated`。省略時は `stable`）
   - `stale_after`: 概念の鮮度期限（ISO 8601 絶対日時）。`now >= stale_after` のとき古いと判断される

   **出典情報（sources）:**
   - `sources`: 参照した出所情報のリスト。各エントリに以下を含める：
     - `resource`（必須）: ソースのURIまたはバンドル内相対パス（例: `/docs/raw/xxx.md`）
     - `id`（任意）: 本文で脚注引用する場合のキー
     - `title`（任意）: ソースの人間可読名
     - 信頼性シグナル（すべて任意）：
       - `author`: ソースの著者（Actor 命名規則に従う）。権威性の指標
       - `usage_count`: `usage_window` 期間内のリソースの利用回数。活性度の指標
       - `last_modified`: ソース自体の最終更新日時。鮮度の指標
   - `usage_window`: `sources` と同レベルに記述。`usage_count` の計測期間を `{ from, to }` で示す
   - 例：
     ```yaml
     sources:
       - id: ga4-schema
         resource: https://developers.google.com/analytics/bigquery/export-schema
         title: GA4 BigQuery Export schema
         author: team:ga4-docs
         usage_count: 5000
         last_modified: 2026-05-30T00:00:00Z
     usage_window: { from: 2026-06-01T00:00:00Z, to: 2026-06-30T00:00:00Z }
     ```

3. **本文内の Per-Claim Attribution（脚注引用）**:
   本文中で特定のクレームをソースに帰属させる場合、`sources[].id` をキーとした Markdown 脚注を使用する：
   ```markdown
   `events_` テーブルは日次シャーディングされている。[^ga4-schema]

   [^ga4-schema]: GA4 BigQuery Export schema
   ```
   脚注ラベルは `sources` エントリへの結合キーとなる。位置ベースの参照（`sources[0]`）は使用しない（エージェントによるリスト並べ替え時に誤帰属が発生するため）。

4. **差分・矛盾検出 (Contradiction Check)**:
   既存の概念ページと新しい raw ソースの間に仕様の矛盾がある場合、サイレントに上書きせず、人間に対して差分（diff）と変更箇所を明示して確認を取る。

5. `docs/raw/` には、ユーザから新たな実装計画を示された場合にのみ追加し、それ以外の目的で追加・編集・削除をしない。

## 3. コード開発とナレッジ更新 (Update Protocol)
ソースコードの生成・修正が完了した際：
1. 実装によって確定した詳細（インターフェース、例外型、関連コンポーネント等）を、対応する概念ページ（`docs/` 配下）に書き戻す。
2. 更新時は `generated`（または検証時は `verified`）を更新する。
3. 廃止された仕様には `status: deprecated` を設定し、代替先があれば `superseded_by: <path>` を付与する。
4. `docs/log.md` の最上部に、`## YYYY-MM-DD` 形式で変更内容の要約を追記する。エントリの先頭には `* **Update**: ...`, `* **Creation**: ...`, `* **Deprecation**: ...` などのプレフィックスを用いる。

## 4. ナレッジの静的検証 (Lint Protocol)
ドキュメントの更新後、または独立した要求時に以下をチェック・自動修復する：
1. **予約ファイル名の保護**: `README.md`（または `index.md`）および `log.md` は予約ファイル名であり、個別の概念ページ名として使用されていないことを確認する。
2. **リンク切れの防止**: 存在しない Markdown ファイルへの参照（デッドリンク）を修正する。
3. **孤立ページの修正**: どのページからも参照されていない概念ページがあれば、適切な `README.md` や関連概念ページからリンクを繋ぐ。
4. **フロントマター検証**: すべての概念ファイルに `type` および有効な YAML 構文が存在することを確認する。
5. **鮮度チェック**: `stale_after` が設定されている概念のうち、期限を超過しているものを報告する。
6. **verified 形式の正規化**: `verified` がベアマッピング形式（非リスト）の場合、コンシューマ互換のためリスト形式への正規化を推奨として報告する（自動修正はしない）。

## 5. Attested Computation の運用 (Attested Computation Protocol)
`type: Attested Computation` は、値の「意味」だけでなく、その値を算出する**承認済み計算方法**を記録する特殊な概念タイプである（OKF v0.2 §10）。

### 5.1 使用判断基準
以下の条件を満たす場合に `Attested Computation` 概念を作成する：
- 特定の数値やメトリクスが、定義された計算方法で再現可能でなければならない場合
- エージェントが独自のクエリを即興で書くのではなく、承認済みの計算を使用していることを保証する必要がある場合

### 5.2 フロントマターの記述ルール
通常の概念フィールド（§2）に加えて、以下のフィールドを記述する：
- `runtime`（必須）: 計算の実行環境を示す（例: `bigquery`, `postgres`, `dbt`, `python`, `Looker`）
- `parameters`: 型付き・名前付きのパラメータリスト。各エントリは `{ name, type, required }`
- `computation`（任意）: 計算ファイルへのパス。省略時は本文の `# Computation` セクション内のフェンスコードブロックが計算定義となる
- `executor`: 計算の実行方法。`resource` に実行手順やコードを指し、`receipt` に実行結果として返すべきフィールドを列挙する
- `attester`: 決定論的な検証コード。`resource` に検証コードを指す（LLMなし）

### 5.3 計算の本文記述
計算は以下のいずれかの方法で提供する：
- **インライン**: 本文の `# Computation` 見出し配下にフェンスコードブロックとして記述（短い計算の場合に推奨）
- **外部ファイル**: `computation` フィールドにパスを設定（長い/生成された計算、または非OKFツールと共有するファイルの場合に推奨）

### 5.4 計算を利用する概念からの参照
メトリクスや概要などの概念が Attested Computation の値を利用する場合、通常の Markdown リンクで参照する：
```markdown
[収益計算](../computations/revenue.md) で算出された認識収益を使用する。
```

## 6. ドキュメント記法・フォーマット規約
1. **GitHubフレンドリーなインデックス（README.md）の扱い**:
   - OKF仕様では `index.md` が標準ですが、GitHub上での閲覧性・レンダリング互換性を最優先するため、本リポジトリでは各ディレクトリのインデックスとして **`README.md`** を主軸（Primary）として運用します。
   - `index.md` が存在する場合は `README.md` への参照または同一内容として扱います。
2. **Mermaid ダイアグラムの安全な記述 (パースエラー防止)**:
   - 日本語、半角空白、および記号（丸括弧 `()`、角括弧 `[]`、不等号 `<>`、コロン `:`、スラッシュ `/` 等）を含む場合、GitHub やパーサーの誤認を防ぐため**ノード名 `["..."]` およびエッジラベル `|"..."|` は必ずダブルクォーテーションで囲む**。
3. **パスと環境情報の抽象化 (プライバシー・移植性ガードレール)**:
   - 特定の個人環境に依存した絶対パス（`/home/username/...`）や実ローカル IP（`10.x.x.x`）の記述は厳禁。常に相対パス（`./docs/...`）またはダミー値（`/path/to/...`, `192.168.1.100`）を使用する。
4. **クロスリンクのパス形式**:
   - バンドル内の概念間リンクは、移動に強い**バンドル相対パス**（`/` 始まり、バンドルルート基準）を推奨する。
   - 同一ディレクトリ内のリンクには相対パス（`./other.md`）も許容する。
5. **`references/` サブディレクトリの慣例**:
   - 外部資料のミラー、実行手順、検証コード等は `docs/references/` 配下に配置する。
   - `sources`、`executor`、`attester` の `resource` フィールドからこのディレクトリを参照する。
6. **タイムスタンプの形式**:
   - OKF のすべてのタイムスタンプは ISO 8601 形式かつ明示的な UTC オフセット付きとする（例: `2026-06-30T14:00:00Z`）。
