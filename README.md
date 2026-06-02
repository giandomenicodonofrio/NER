# Named Entity Recognition sul dataset KIND

Questo repository contiene un sistema supervisionato di Named Entity Recognition
(NER) sviluppato sul dataset KIND di EVALITA 2023.

Il progetto usa word embedding open-source pre-addestrati e modelli neurali
addestrati da zero sui dati annotati. Non utilizza Transformer pre-addestrati
come BERT, RoBERTa o GPT.

Sono previsti due casi d'uso:

- **modello generalista**: addestrato sui domini ADG, FIC e WN;
- **modello specifico**: addestrato esclusivamente sul dominio Wikinews (WN).

La relazione completa con analisi, scelte progettuali e risultati sperimentali
si trova in [`RelazioneTecnica.docx`](RelazioneTecnica.docx).

## Dataset

KIND contiene testi appartenenti a tre domini:

| Codice | Dominio |
| --- | --- |
| `ADG` | Alcide De Gasperi |
| `FIC` | Fiction |
| `WN` | Wikinews |

Le entità annotate seguono lo schema BIO:

- `PER`: persone;
- `ORG`: organizzazioni;
- `LOC`: luoghi;
- `O`: token esterni alle entità.

I file del dataset non sono versionati nel repository. Devono essere inseriti
manualmente nella seguente struttura:

```text
data/
`-- raw/
    |-- ADG/
    |   |-- ADG_train.tsv
    |   |-- ADG_dev.tsv
    |   |-- ADG_test.tsv
    |   `-- ADG_test_nolabel.tsv
    |-- FIC/
    |   |-- FIC_train.tsv
    |   |-- FIC_dev.tsv
    |   |-- FIC_test.tsv
    |   `-- FIC_test_nolabel.tsv
    `-- WN/
        |-- WN_train.tsv
        |-- WN_dev.tsv
        |-- WN_test.tsv
        `-- WN_test_nolabel.tsv
```

## Architettura

L'architettura principale è una **CharCNN + BiLSTM + CRF**:

```text
Token
  |-- Word embedding pre-addestrato
  `-- Character embedding -> CharCNN -> max pooling
                  |
       concatenazione delle feature
                  |
                BiLSTM
                  |
                 CRF
                  |
          sequenza di label BIO
```

I componenti svolgono ruoli distinti:

- i **word embedding** forniscono rappresentazioni semantiche iniziali;
- la **CharCNN** cattura informazioni morfologiche utili per nomi propri,
  parole rare e token fuori vocabolario;
- la **BiLSTM** modella il contesto a sinistra e a destra di ogni token;
- il **CRF** apprende dipendenze tra label consecutive e rende più coerenti le
  sequenze BIO prodotte dal modello.

Il codice supporta anche architetture più semplici usate negli studi di
ablazione:

- `BiLSTM + Softmax`;
- `BiLSTM + CRF`;
- `CharCNN + BiLSTM + CRF`.

## Pipeline

Il flusso principale di training esegue questi passaggi:

1. carica una configurazione YAML e risolve i riferimenti alle configurazioni di
   modello, preprocessing ed embedding;
2. legge gli split `train`, `dev` e `test`;
3. normalizza i token e applica l'eventuale rimozione delle stopword;
4. costruisce i vocabolari di token, caratteri, label e dataset;
5. carica gli embedding testuali e costruisce la matrice iniziale;
6. addestra il modello con `AdamW`, gradient clipping ed early stopping sulla
   macro-F1 token-level NERMuD del development set;
7. salva il checkpoint migliore;
8. valuta il checkpoint e genera metriche, predizioni e matrici di confusione.

La metrica primaria segue lo scorer ufficiale NERMuD: rimuove i prefissi BIO e
calcola macro precision, macro recall e macro F1 token-level sulle classi `PER`,
`LOC` e `ORG`. Vengono salvate anche la micro-F1 NERMuD e le metriche
entity-level calcolate con `seqeval`:

- precision;
- recall;
- F1-score.

Viene salvata anche l'accuracy token-level e, per i modelli generalisti, il
dettaglio delle metriche per dominio.

## Preprocessing

Le configurazioni di preprocessing si trovano in
[`configs/preprocessing`](configs/preprocessing).

| Configurazione | Descrizione |
| --- | --- |
| `p0_base.yaml` | baseline con normalizzazione delle cifre |
| `p1_normalized_embeddings.yaml` | baseline e normalizzazione L2 degli embedding |
| `p2_no_stopwords_normalized_embeddings.yaml` | normalizzazione L2 e rimozione delle stopword |

La configurazione `P2` usa la lista locale
[`resources/stopwords_it.txt`](resources/stopwords_it.txt). Quando la rimozione
di una stopword lascia un tag `I-*` senza un precedente `B-*` o `I-*`
compatibile, il dataset lo converte in `B-*` per preservare la validità della
sequenza BIO.

## Esperimenti

Le configurazioni degli esperimenti si trovano in
[`configs/experiment`](configs/experiment) e sono organizzate per fase:

| Directory | Scopo |
| --- | --- |
| `all_datasets/` | ablazione preprocessing sui tre domini |
| `wn_only/` | ablazione preprocessing sul solo dominio WN |
| `architecture/` | confronto tra architetture |
| `embeddings/` | confronto tra embedding pre-addestrati |
| `freeze/` | confronto tra embedding congelati e fine-tuning |
| `balancing/` | sampling bilanciato per dataset |
| `tuning/` | tuning degli iperparametri |
| `post_tuning/` | modello generalista selezionato dopo il tuning |
| `error_analysis/` | snapshot storici usati per l'analisi degli errori |

Ogni configurazione di esperimento compone tre file:

```yaml
model_config: configs/model/...
embedding_config: configs/embeddings/...
preprocessing_config: configs/preprocessing/...
```

In questo modo architettura, embedding e preprocessing possono essere modificati
indipendentemente.

## Configurazione di Riferimento

La configurazione scelta al termine degli esperimenti usa:

- embedding `NLPL Word2Vec IT` da 100 dimensioni;
- embedding congelati durante il training;
- CharCNN con 50 filtri;
- BiLSTM bidirezionale con hidden size 256;
- CRF per il decoding;
- word dropout pari a `0.10`;
- optimizer `AdamW` con learning rate `0.001`.

Il confronto finale è definito in
[`scripts/final_evaluation.py`](scripts/final_evaluation.py):

| Modello | Training | Valutazione |
| --- | --- | --- |
| generalista | `ADG`, `FIC`, `WN` | test set completo |
| specifico | `WN` | test set completo con dettaglio sul dominio `WN` |

La relazione tecnica registra il seguente snapshot sperimentale storico:

| Modello | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| generalista | 0.7977 | 0.7785 | 0.7880 |
| specifico su WN | 0.7958 | 0.7913 | 0.7935 |

Questi valori descrivono gli artefatti prodotti durante lo sviluppo con la
precedente metrica entity-level. Non sono risultati NERMuD ufficiali: prima di
usarli come benchmark è necessario rigenerare gli esperimenti con lo scorer
attuale.

## Requisiti

- Python `>= 3.13`;
- [`uv`](https://docs.astral.sh/uv/) come package manager;
- spazio su disco sufficiente per dataset, embedding e checkpoint;
- GPU CUDA opzionale ma consigliata per il training.

Eseguire i comandi dalla root del repository.

Installazione delle dipendenze:

```bash
uv sync
```

## Download Embedding

Gli embedding non sono versionati. Lo script di download li salva in
`embeddings/vec/` e genera `embeddings/embeddings_manifest.json`.

Per scaricare soltanto NLPL Word2Vec IT, usato dalla configurazione finale:

```bash
uv run python scripts/download_embeddings.py --out-dir embeddings --models nlpl_it_word2vec
```

Per scaricare tutti gli embedding usati nelle ablazioni:

```bash
uv run python scripts/download_embeddings.py --out-dir embeddings --models all
```

Per elencare i modelli disponibili:

```bash
uv run python scripts/download_embeddings.py --list
```

## Training

È possibile lanciare un singolo esperimento passando direttamente la config:

```bash
uv run python -m src.scripts.train \
  --config configs/experiment/post_tuning/post_tuning_all_datasets_word_dropout_010.yaml
```

Su PowerShell il comando equivalente su una singola riga è:

```powershell
uv run python -m src.scripts.train --config configs/experiment/post_tuning/post_tuning_all_datasets_word_dropout_010.yaml
```

Per forzare il device:

```powershell
uv run python -m src.scripts.train --config configs/experiment/tuning/tuning_wn_word_dropout_010.yaml --device cuda
```

Lo script [`main.py`](main.py) consente anche di eseguire una sequenza di
esperimenti. La variabile `EXPERIMENTS` è vuota per impostazione predefinita:
la sequenza da eseguire viene passata con `--only`, senza modificare il file:

```powershell
uv run python main.py --only configs/experiment/architecture/wn_a0_bilstm_softmax.yaml configs/experiment/architecture/wn_a1_bilstm_crf.yaml configs/experiment/architecture/wn_a2_charcnn_bilstm_crf.yaml
```

Per valutare un checkpoint `best.pt` già esistente senza ripetere il training:

```powershell
uv run python -m src.scripts.train --config configs/experiment/tuning/tuning_wn_word_dropout_010.yaml --eval-only
```

## Valutazione Finale

Lo script [`scripts/final_evaluation.py`](scripts/final_evaluation.py) confronta
il modello generalista e quello specifico sugli stessi dataset di test:

```powershell
uv run python scripts/final_evaluation.py
```

Le configurazioni e i checkpoint valutati sono dichiarati nella costante
`EVALUATIONS` all'interno dello script.

## Analisi

Analisi esplorativa dei dataset:

```powershell
uv run python src/analysis/data_analysis.py --dataset_dir data/raw
```

Visualizzazione PCA degli embedding:

```powershell
uv run python src/analysis/embedding_visualization.py --dataset-dir data/raw --method pca
```

Visualizzazione t-SNE:

```powershell
uv run python src/analysis/embedding_visualization.py --dataset-dir data/raw --method tsne
```

Confronto tra sampling casuale e bilanciato:

```powershell
uv run python -m src.scripts.compare_sampling
```

Analisi degli errori sullo snapshot `error_analysis`:

```powershell
uv run python -m src.scripts.error_analysis
```

Per analizzare un altro output:

```powershell
uv run python -m src.scripts.error_analysis --predictions outputs/post_tuning/post_tuning_all_datasets_word_dropout_010/predictions/test_errors.tsv --out-dir outputs/analysis/error_analysis/post_tuning
```

Generazione dei grafici della relazione:

```powershell
uv run python -m src.scripts.build_report_figures
```

## Output

Gli artefatti generati sono salvati sotto `outputs/` e non vengono versionati.
Ogni esperimento produce una struttura simile:

```text
outputs/<gruppo>/<esperimento>/
|-- checkpoints/
|   `-- best.pt
|-- confusion_matrices/
|-- metrics/
|   |-- history.json
|   |-- test_metrics.json
|   |-- test_metrics_by_dataset.json
|   |-- test_classification_report.json
|   `-- test_span_classification_report.json
|-- predictions/
|   |-- test_predictions.tsv
|   `-- test_errors.tsv
|-- vocabs/
`-- config_resolved.json
```

`config_resolved.json` è importante per la riproducibilità: contiene la
configurazione effettivamente usata dopo il merge dei file YAML.

## Struttura Repository

```text
.
|-- configs/
|   |-- embeddings/       # configurazioni degli embedding
|   |-- experiment/       # configurazioni degli esperimenti
|   |-- model/            # configurazioni delle architetture
|   `-- preprocessing/    # configurazioni del preprocessing
|-- resources/
|   `-- stopwords_it.txt
|-- scripts/
|   |-- download_embeddings.py
|   `-- final_evaluation.py
|-- src/
|   |-- analysis/         # analisi esplorative e visualizzazioni
|   |-- data/             # dataset, collate e dataloader
|   |-- evaluation/       # metriche e salvataggio output
|   |-- models/           # modelli neurali
|   |-- preprocessing/    # normalizzazione token ed embedding
|   |-- scripts/          # entry point per training e analisi
|   |-- training/         # trainer e seed
|   `-- utils/            # config, reader e vocabolari
|-- main.py               # runner per più esperimenti
`-- RelazioneTecnica.docx
```
