# Pipelines

## Baseline

A estratégia `baseline` é VLM-first. O VLM determina quais conceitos devem ser procurados antes do grounding e da segmentação.

```mermaid
flowchart TD
    I[Input image] --> O[Qwen: objects]
    O --> E[Qwen: environment]
    E --> R[Qwen: risks]

    R --> P[Parse VLM responses]
    P --> C[Consolidate concepts]

    C --> Empty{Accepted concepts?}
    Empty -->|No| F[Indoor/outdoor fallback queries]
    Empty -->|Yes| G[Grounding queries]
    F --> G

    G --> GD[Grounding DINO]
    GD --> D[Detections / bounding boxes]
    D --> SAM[SAM2 per detection]
    SAM --> M[Masks]

    M --> Result[result.json]
    M --> Overlay[overlays]
    R --> Result
    C --> Result
    D --> Result
```

### Sequência

```mermaid
sequenceDiagram
    participant CLI
    participant Pipeline as ImageContextPipeline
    participant Qwen
    participant Grounder as Grounding DINO
    participant SAM as SAM2
    participant Artifacts

    CLI->>Pipeline: analyze(ImageSample)

    Pipeline->>Qwen: create_vlm()
    loop objects, environment, risks
        Pipeline->>Artifacts: read cached VLM result
        alt cache miss
            Pipeline->>Qwen: generate(image, prompt)
            Qwen-->>Pipeline: raw response
            Pipeline->>Artifacts: persist raw + parsed result
        end
    end
    Pipeline->>Qwen: close()

    Pipeline->>Pipeline: consolidate concepts
    Pipeline->>Artifacts: persist consolidated context

    Pipeline->>Grounder: create_grounder()
    Pipeline->>Grounder: detect concept queries
    Grounder-->>Pipeline: detections
    Pipeline->>Artifacts: persist detections
    Pipeline->>Grounder: close()

    Pipeline->>SAM: create_segmenter()
    loop detections
        Pipeline->>SAM: segment(box)
        SAM-->>Pipeline: mask + confidence
        Pipeline->>Artifacts: persist mask
    end
    Pipeline->>SAM: close()

    Pipeline->>Artifacts: write result + overlays
```

### Característica principal

O recall depende inicialmente do que o VLM propõe como conceito. Se um objeto ou região não for proposto, ele não chega naturalmente às etapas de Grounding DINO e SAM2. O fallback reduz casos vazios, mas continua sendo um conjunto de consultas configuradas.

## Region-first

A estratégia `region-first` inverte a ordem. Primeiro descobre evidência visual; depois atribui semântica.

```mermaid
flowchart TD
    I[Input image] --> SAM[SAM2 automatic region discovery]
    SAM --> Filter[Area filtering + IoU deduplication]
    Filter --> VR[VisualRegion instances]

    VR --> DINO[DINOv2 dense features]
    DINO --> Pool[Mask pooling]
    Pool --> Emb[Embedding per region]

    I --> Global[Qwen global context]
    VR --> Composite[Region context image]
    Global --> Local[Qwen local interpretation]
    Composite --> Local

    Local --> Semantic[SemanticRegion]
    Emb --> Semantic
    VR --> Semantic

    Semantic --> JSON[semantic_regions.json]
    Semantic --> Overlay[region_overlay.png]
    Semantic --> Metrics[metrics.json]
```

### Sequência

```mermaid
sequenceDiagram
    participant CLI
    participant Pipeline as RegionFirstContextPipeline
    participant Discoverer as SAM2 Discoverer
    participant Features as DINOv2
    participant VLM as Qwen
    participant Disk as Artifact storage

    CLI->>Pipeline: analyze(ImageSample)

    Pipeline->>Disk: read discovered_regions cache
    alt cache miss
        Pipeline->>Discoverer: create_discoverer()
        Pipeline->>Discoverer: discover(image)
        Discoverer-->>Pipeline: VisualRegion[]
        Pipeline->>Discoverer: close()
        Pipeline->>Disk: persist region masks + metadata
    end

    Pipeline->>Disk: read dense feature cache
    alt cache miss
        Pipeline->>Features: create_feature_extractor()
        Pipeline->>Features: extract(image, regions)
        Features-->>Pipeline: embedding per region
        Pipeline->>Features: close()
        Pipeline->>Disk: persist .npy embeddings
    end

    Pipeline->>Disk: read global/local VLM cache
    alt semantic cache incomplete
        Pipeline->>VLM: create_vlm()
        Pipeline->>VLM: global scene prompt
        VLM-->>Pipeline: global context
        loop each region
            Pipeline->>Disk: create context.png
            Pipeline->>VLM: local region prompt
            VLM-->>Pipeline: region interpretation
        end
        Pipeline->>VLM: close()
    end

    Pipeline->>Pipeline: fuse geometry + embedding + semantics
    Pipeline->>Disk: semantic_regions.json + metrics + overlay
```

### Proveniência

Cada `SemanticRegion` mantém separadas as origens da informação:

```mermaid
flowchart LR
    G[Geometry] --> SR[SemanticRegion]
    E[Visual embedding] --> SR
    L[Local semantics] --> SR
    C[Global scene context] --> SR

    G -. source .-> GS[SAM2 region discovery]
    E -. source .-> DS[DINOv2 mask pooling]
    L -. source .-> QS[Qwen local]
    C -. source .-> QG[Qwen global]
```

Isso permite auditar qual modelo forneceu geometria, representação visual e interpretação textual.

## Comparação conceitual

```mermaid
flowchart LR
    subgraph Baseline
        BI[Image] --> BV[VLM concepts]
        BV --> BG[Grounding]
        BG --> BS[Segmentation]
        BS --> BR[Semantic output]
    end

    subgraph RegionFirst
        RI[Image] --> RS[Region discovery]
        RS --> RF[Dense features]
        RS --> RV[VLM interpretation]
        RF --> RR[Semantic regions]
        RV --> RR
    end
```

| Aspecto | Baseline | Region-first |
| --- | --- | --- |
| Primeiro sinal | texto/semântica | geometria visual |
| Descoberta | Grounding por conceito | regiões class-agnostic |
| Segmentação | depois da detecção | início do pipeline |
| Features densas | não | DINOv2 por máscara |
| VLM | 3 passagens globais | global + uma interpretação por região |
| Unidade final | detecção segmentada | `SemanticRegion` |
| Risco principal | conceito não proposto | excesso/fragmentação de regiões |

## Execução conjunta

Com `--strategy all`, as duas estratégias recebem a mesma imagem normalizada.

```mermaid
flowchart TD
    Source[Original image] --> Normalized[input.png]
    Normalized --> B[baseline/]
    Normalized --> R[region-first/]

    B --> BM[metrics.json]
    R --> RM[metrics.json]

    BM --> Compare[comparison.json]
    RM --> Compare
```

Essa execução não representa uma fusão entre as estratégias. Ela é uma comparação controlada entre dois pipelines independentes.
