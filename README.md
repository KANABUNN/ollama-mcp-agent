# Ollama + MCP ローカルファイルエージェント

ローカルの Ollama モデルから、stdio 接続の MCP サーバーを介して
`C:\LLM-Workspace` 配下のテキストファイルだけを一覧・読み取り・確認付きで
書き込む、最小権限の Windows 向けエージェントです。

任意コマンド実行、ファイル削除、GUI 操作、管理者操作、外部サービスへの送信は
実装していません。

## アーキテクチャ

```text
利用者
  │ 対話CLI / 書き込み承認 (y/N)
  ▼
ollama_mcp_agent.agent
  ├── loopback HTTP ──► Ollama
  ├── ツール名・JSON Schema・反復上限の検証
  ├── 承認処理
  └── 監査ログ
          │ stdio (MCP)
          ▼
ollama_mcp_agent.pc_tools_server
          │ 安全な相対パスだけ
          ▼
C:\LLM-Workspace
```

Ollama はツールを選択しますが、実際のファイル操作は MCP サーバーだけが行います。
書き込み承認はモデルの申告ではなく、エージェント側のコードで強制されます。

## 前提条件

- Windows 10 または Windows 11
- Python 3.11 以上（動作確認環境は Python 3.14）
- Ollama がインストール済みで、ローカルで起動できること
- Ollama に tool calling 対応モデルが導入済みであること
- 一般ユーザー権限の PowerShell

MCP Python SDK は 1.30.0 以上の1系を使用します。CVE-2026-116686 の修正により、
ツール結果のスキーマ検証で外部URLやローカルファイルを参照しません。これはstdio接続にも
関係するSDK側の保護です。本アプリの標準構成は同梱のローカルサーバーに接続し、
HTTP MCP通信・HTTP bearer認証は使用しません。既存の仮想環境も更新する場合は、
プロジェクトのPythonで `python -m pip install --upgrade -e ".[dev]"` を実行してください。

このセットアップはモデルを自動ダウンロードしません。

## Ollama とモデルの準備

別の PowerShell で Ollama の状態と導入済みモデルを確認します。

```powershell
ollama --version
ollama list
```

Ollama が起動していなければ、通常の Ollama アプリを起動してください。必要なモデルの
導入は、容量と利用条件を確認したうえで利用者自身が行います。

## セットアップ

```powershell
cd C:\ollama-mcp-agent
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

スクリプトはプロジェクト内に `.venv` を作成し、開発用依存関係を導入します。また、
存在しない場合だけ `C:\LLM-Workspace` と `.env` を作成します。管理者権限は要求しません。

続いて `.env` を開き、導入済みの tool calling 対応モデル名を設定します。

```dotenv
OLLAMA_MODEL=your-tool-capable-model
```

`OLLAMA_MODEL` は空のままでは起動できません。既定の `OLLAMA_HOST` は
`http://127.0.0.1:11434` です。エージェントは誤った外部送信を防ぐため、Ollama の接続先を
`127.0.0.1`、`localhost`、`::1` のループバックホストに限定します。

## 起動

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-agent.ps1
```

起動スクリプトは、`.env` と仮想環境を確認し、Ollama の `/api/tags` へループバックで
疎通確認してから対話 CLI を開始します。

使用例:

```text
> C:\LLM-Workspace のファイル一覧を表示して
> notes.txtを読んで要約して
> output\summary.mdに要約を書き込んで
```

パスは、会話中でも `C:\LLM-Workspace` からの相対パスとしてツールへ渡されます。
書き込み前には、ツール名、対象パス、新規作成か上書きか、文字数、短いプレビューと
`y/N` が表示されます。`y` または `yes` 以外は拒否です。

対話 CLI の特殊コマンド:

- `/help`: コマンド一覧
- `/tools`: MCP から取得した利用可能ツール
- `/config`: 秘密値を含まない有効設定
- `/exit`: 終了

## MCP サーバーだけを確認する

stdio サーバーは次で起動できます。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-server.ps1
```

正常時に通常ログやプロンプトは表示されず、MCP の JSON-RPC 入力を待ちます。
終了は `Ctrl+C` です。MCP Inspector から接続する場合は以下を指定します。

```text
Command:   C:\ollama-mcp-agent\.venv\Scripts\python.exe
Arguments: -I -m ollama_mcp_agent.pc_tools_server
Working directory: C:\ollama-mcp-agent
```

公開ツールは `list_files`、`read_text_file`、`write_text_file` の3つだけです。

`list_files` は指定ディレクトリ直下だけを返し、再帰しません。ドットで始まる名前や
Windows の hidden 属性を持つ項目も、通常の項目と同様に一覧へ含めます。

## テスト

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\verify.ps1
```

このスクリプトは `ruff check .` と `pytest` を順番に実行します。テストは pytest の
一時ディレクトリを使い、実際の `C:\LLM-Workspace` を変更しません。

## 設定

`.env.example` の項目:

| 変数 | 既定値 | 説明 |
|---|---:|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | ループバックの Ollama API |
| `OLLAMA_MODEL` | 空 | 利用者が指定するモデル名 |
| `ALLOWED_ROOT` | `C:\LLM-Workspace` | MCP がアクセスできる唯一のローカルルート（UNC不可） |
| `AUDIT_LOG_PATH` | `logs\audit.jsonl` | 監査ログ |
| `MAX_READ_BYTES` | `1048576` | 1回の最大読み取りバイト数 |
| `MAX_WRITE_BYTES` | `1048576` | 1回の最大書き込みバイト数 |
| `MAX_TOOL_ITERATIONS` | `10` | 1入力あたりの最大ツール反復回数 |

相対的な監査ログパスはプロジェクトルートから解決されます。監査ログはプロジェクト外を
指定できません。`.env` は Git 管理対象外です。

## 監査ログ

既定では `logs\audit.jsonl` に UTF-8 の JSON Lines として記録されます。各行には
タイムゾーン付き時刻、UUID、ツール名、サニタイズした引数、承認要否・結果、成否、
所要時間、エラー種別が入ります。

ファイル本文全体、書き込み本文全体、パスワード、トークン、環境変数値は保存しません。
書き込み内容は文字数と SHA-256 だけに縮約されます。
ログを書けなかった場合は再試行で操作を重複させず、エージェントの標準エラーへ警告します。

## 安全上の制限

- MCP ツールは相対パスだけを受け取ります。
- パスを正規化し、`..`、絶対パス、似た名前の隣接ディレクトリ、シンボリックリンクや
  ジャンクション経由のルート脱出を拒否します。
- 起動時の許可ルートのファイルシステム識別子を保持し、実行中のルート差し替えを拒否します。
- 複数ハードリンクを持つファイルは、許可ルート外データの別名になり得るため読みません。
- 読み書きサイズを処理前に検査します。
- 読み取りは UTF-8 テキストだけです。NUL を含むバイナリや不正 UTF-8 を拒否します。
- 書き込みは UTF-8・LF に正規化し、同一ディレクトリ内の一時ファイルから原子的に
  置換します。
- 書き込みは対話承認がない限り MCP に送信されません。
- 承認表示後に対象の新規作成／既存状態が変わった場合、その承認を破棄します。
- Ollama の未知ツール、不正引数、JSON Schema 不一致は実行されません。
- ツール反復回数には上限があります。
- 親エージェントで検証したルートとサイズ上限を子 MCP プロセスへ明示的に固定します。
- Ollama 通信はループバックだけで、HTTP リダイレクトと環境プロキシを無効にします。
- モデル回答や承認項目に含まれる端末制御文字は、表示前に可視エスケープへ変換します。
- MCP の stdio 標準出力はプロトコル専用です。
- ファイル削除、任意コマンド、`subprocess` を使った一般プログラム実行、GUI 操作、
  管理者操作、外部 API 送信はありません。

MCP サーバーを単体で公開すると、書き込み確認を担当するエージェントを迂回できます。
エージェント側の監査処理も通らないため、Inspector からの直接呼び出しは
`logs\audit.jsonl` に記録されません。単体起動は Inspector による検証用途に限り、
信頼できない MCP クライアントへ接続しないでください。

`ALLOWED_ROOT` は重要な信頼境界です。`read_text_file` はその配下にあるUTF-8ファイルを
読み取れるため、ユーザープロファイル、`.ssh`、ブラウザプロファイル、資格情報保管場所を
指定せず、許可ルート内にもパスワードやトークン等の秘密ファイルを置かないでください。
UNC、Windowsデバイス名前空間、ネットワークドライブは許可ルートとして拒否されます。

この設計は、モデルが作るパスや事前に存在するリンクからの脱出を防ぎます。一方、同じユーザー
権限の別プロセスが検査とファイルオープンのごく短い間に NTFS エントリを継続的に差し替える
競合までは、Python の移植可能なファイルAPIだけで完全には排除できません。信頼できない
ローカルプロセスに `C:\LLM-Workspace` の変更権限を同時に与えないでください。

## トラブルシューティング

### `OLLAMA_MODEL` が未設定

`.env` の `OLLAMA_MODEL` に `ollama list` で表示される tool calling 対応モデル名を設定します。

### Ollama に接続できない

Ollama を起動し、`http://127.0.0.1:11434/api/tags` がローカルで応答することを確認します。
ポートを変更した場合も、`OLLAMA_HOST` のホスト部分はループバックでなければなりません。

### モデルがツールを呼ばない、または引数を誤る

tool calling 対応モデルか確認してください。小規模モデルでは複数段階のツール利用が不安定な
場合があります。不正なツール呼び出しはエージェントが拒否し、その理由をモデルへ返します。

### パスが拒否される

絶対パスではなく `notes\sample.txt` のような相対パスを指定してください。許可ルート外を
指すリンクやジャンクションも拒否されます。

### 文字コードエラー

読み取り対象を BOM なしまたは BOM 付き UTF-8 で保存してください。バイナリや他の文字
コードは初期版の対象外です。

## 将来拡張する際の注意

拡張はファイル検索、読み取り専用 Git 状態、読み取り専用システム情報などを別の限定 MCP
ツールとして追加してください。`run_command(command: str)` のような汎用実行機能は追加せず、
読み取りと変更を分離し、変更・外部送信には人間の確認を必須にします。ブラウザや GUI を
扱う場合も別サーバー・別プロファイル・最小権限で隔離してください。
