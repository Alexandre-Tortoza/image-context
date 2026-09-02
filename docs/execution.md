# Execution

## CLI

O projeto expõe quatro modos principais:

```bash
image-context sample --config config.yaml
image-context run --config config.yaml
image-context reprocess --config config.yaml --source-run runs/<run-id>
image-context analyze --config config.yaml --image path/to/image.png --strategy all
```

## Fluxo dos comandos

```mermaid
flowchart TD
    CLI[image-context] --> Cmd{Command}

    Cmd -->|sample| Sample[Extract reproducible ROS bag images]
    Cmd -->|run| Run[Execute baseline over sampled frames]
    Cmd -->|reprocess| Re[Reuse persisted VLM responses]
    Cmd -->|analyze| Analyze[Analyze one common image]

    Analyze --> Strategy{strategy}
    Strategy -->|baseline| B[Baseline only]
    Strategy -->|region-first| R[Region-first only]
    Strategy -->|all| Both[Run both independently]
```

## `sample`

Extrai imagens do ROS bag conforme `sample_size` e `seed`, sem carregar os modelos pesados.

```mermaid
sequenceDiagram
    participant CLI
    participant Sampler as RosbagImageSampler
    participant Disk

    CLI->>Sampler: sample(size, seed)
    Sampler-->>CLI: ImageSample[]
    CLI->>Disk: image files
    CLI->>Disk: selected_frames.json
```

## `run`

Executa o baseline sobre as imagens amostradas do dataset configurado.

A ordem de residência dos modelos é deliberada:

```mermaid
flowchart LR
    Q[Load Qwen] --> QRun[Run VLM passes]
    QRun --> QClose[Unload Qwen]
    QClose --> G[Load Grounding DINO]
    G --> GRun[Run grounding]
    GRun --> GClose[Unload Grounding DINO]
    GClose --> S[Load SAM2]
    S --> SRun[Run segmentation]
    SRun --> SClose[Unload SAM2]
```

## `reprocess`

`reprocess` evita executar novamente o Qwen quando as respostas VLM persistidas já são adequadas para um novo experimento de parsing, thresholds, grounding ou segmentação.

```mermaid
flowchart TD
    Old[Existing run] --> VLM[Persisted VLM responses]
    VLM --> Parse[Parse again]
    Parse --> Consolidate[Consolidate concepts]
    Consolidate --> Grounding[Grounding DINO]
    Grounding --> SAM[SAM2]
    SAM --> New[New run directory]
```

## `analyze`

`analyze` normaliza uma imagem comum para `input.png` e executa uma ou duas estratégias.

```bash
image-context analyze \
  --config config.yaml \
  --image path/to/image.png \
  --strategy all \
  --run-id comparison-01
```

Valores válidos para `--strategy`:

- `baseline`
- `region-first`
- `all`

## Fingerprints

Cada execução usa fingerprints derivados da configuração e da entrada. Eles impedem reutilização silenciosa de artefatos incompatíveis.

```mermaid
flowchart LR
    Config[Configuration] --> Hash[SHA-256 fingerprint]
    Image[Input image fingerprint] --> Hash
    Prompt[Prompt/model settings] --> Hash
    Hash --> Manifest[manifest.json]

    Manifest --> Check{Same fingerprint?}
    Check -->|Yes| Resume[Reuse compatible cache]
    Check -->|No| Stop[Require overwrite or new run-id]
```

Na estratégia region-first também existem fingerprints por estágio, como:

- `region-discovery`
- `dense-features`
- `vlm-global-local`
- `fusion`

Isso permite invalidar apenas o estágio cuja configuração mudou.

## Retomada

A execução consulta artefatos persistidos antes de processar novamente etapas caras.

```mermaid
flowchart TD
    Stage[Pipeline stage] --> Cache{Valid cached artifact?}
    Cache -->|Yes| Read[Read persisted result]
    Cache -->|No| Compute[Run model / compute stage]
    Compute --> Persist[Persist result + marker]
    Read --> Next[Next stage]
    Persist --> Next
```

No baseline, isso inclui respostas VLM, detecções e máscaras. No region-first, inclui regiões descobertas, embeddings e respostas VLM globais/locais.

## Isolamento de falhas

Os pipelines tentam manter falhas localizadas.

```mermaid
flowchart TD
    Work[Process item] --> Error{Error?}
    Error -->|No| Continue[Persist and continue]
    Error -->|Yes| Scope{Pipeline}
    Scope -->|baseline| Frame[Record frame/stage failure]
    Scope -->|region-first| Region[Record failed region]
    Frame --> Continue2[Continue when possible]
    Region --> Continue2
```

A estratégia region-first, em particular, pode produzir resultado parcial quando uma interpretação local falha, sem necessariamente descartar todas as demais regiões.

## Qualidade

```bash
pytest
ruff check .
mypy
```
