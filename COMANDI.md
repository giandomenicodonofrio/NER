uv run python .\src\analysis\data_analysis.py 

uv run python .\scripts\download_embeddings.py --out-dir embeddings --models all

uv run python .\src\analysis\embedding_visualization.py 