Always respond in Japanese.

## 環境設定
- **パッケージ管理**: uv でパッケージを管理する
- **パッケージの追加**: uv add を使用する
- **コマンドの実行**: uv run を使用する
- **プロジェクトルート**: rootutils により自動検出（`.project-root` マーカー使用）

## プロジェクト構造
- **設定管理**: Hydra による階層的設定（configs/）
- **実験管理**: experiment-centric design（configs/experiment/）
- **スクリプト**: scripts/ 配下に train.py, eval.py
  - rootutils.setup_root() により、どのディレクトリからでも実行可能
  - Pythonパスの自動設定により、インポートエラーを回避

## 実装
- t-wadaのTDDを意識してコードを実装する
- 適切なMCPを活用する。

## 注意
- 必要な場合を除いて、Pythonなどのコード実行を禁止します。
