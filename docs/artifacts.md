# Artifacts

Os artefatos persistidos fazem parte da arquitetura do projeto. Eles permitem auditoria, comparação, retomada e reprocessamento sem repetir etapas caras.

## Execução comparativa

Ao usar `image-context analyze --strategy all`, a estrutura conceitual é:

```text
runs/<run-id>/
├── input.png
├── comparison.json
├── baseline/
└── region-first/
```

```mermaid
flowchart TD
    Root[runs/<run-id>] --> Input[input.png]
    Root --> Comparison[comparison.json]
    Root --> Baseline[baseline/]
    Root --> Region[region-first/]

    Baseline --> BM[manifest.json]
    Baseline --> BMetrics[metrics.json]
    Baseline --> BComplete[complete.json]
    Baseline --> BFrames[frames/image/]

    Region --> RM[manifest.json]
    Region --> RMetrics[metrics.json]
    Region --> RComplete[complete.json]
    Region --> RSemantic[semantic_regions.json]
    Region --> RRegions[regions/]
```

## Baseline

A saída principal por imagem contém os resultados das três passagens VLM, conceitos consolidados, detecções e segmentações.

```text
baseline/
├── manifest.json
├── metrics.json
├── complete.json
├── input.png
└── frames/
    └── image/
        ├── vlm_objects.json
        ├── vlm_environment.json
        ├── vlm_risks.json
        ├── consolidated_context.json
        ├── detections.json
        ├── result.json
        ├── grounding_overlay.png
        ├── sam2_overlay.png
        ├── overlay.png
        └── masks/
```

### Relação dos dados

```mermaid
flowchart LR
    VO[vlm_objects.json] --> CC[consolidated_context.json]
    VE[vlm_environment.json] --> CC
    VR[vlm_risks.json] --> CC

    CC --> D[detections.json]
    D --> Masks[masks/*]

    VO --> Result[result.json]
    VE --> Result
    VR --> Result
    CC --> Result
    D --> Result
    Masks --> Result

    D --> GO[grounding_overlay.png]
    Masks --> SO[sam2_overlay.png]
    Masks --> O[overlay.png]
```

## Region-first

A estratégia region-first persiste a descoberta geométrica antes de semântica e mantém um subdiretório por região.

```text
region-first/
├── manifest.json
├── complete.json
├── metrics.json
├── input.png
├── discovered_regions.json
├── dense_features_complete.json
├── vlm_global_raw.txt
├── global_context.json
├── global_complete.json
├── semantic_regions.json
├── region_overlay.png
└── regions/
    └── <region-id>/
        ├── mask.png
        ├── visual_embedding.npy
        ├── context.png
        ├── vlm_raw.txt
        └── vlm_complete.json
```

### Ciclo de uma região

```mermaid
flowchart TD
    Discovery[Discovered VisualRegion] --> Mask[mask.png]
    Discovery --> Meta[discovered_regions.json]

    Mask --> Dense[DINOv2 mask pooling]
    Dense --> Emb[visual_embedding.npy]

    Discovery --> Context[context.png]
    Global[global_context.json] --> Local[Qwen local interpretation]
    Context --> Local
    Local --> Raw[vlm_raw.txt]

    Mask --> Semantic[semantic_regions.json]
    Emb --> Semantic
    Raw --> Semantic
    Global --> Semantic
```

## `semantic_regions.json`

Esse arquivo funciona como a representação estruturada principal do region-first. Cada região semântica combina:

- identificador estável da região;
- caminho da máscara;
- bounding box;
- confiança geométrica;
- label e descrição;
- atributos e condição;
- confiança semântica;
- caminho e dimensão do embedding visual;
- relações candidatas;
- proveniência.

```mermaid
flowchart LR
    Mask[Geometry] --> Region[SemanticRegion]
    Emb[Embedding] --> Region
    Sem[Local semantics] --> Region
    Global[Scene context] --> Region
    Prov[Provenance] --> Region
```

## Métricas

As duas estratégias produzem `metrics.json` para facilitar comparação experimental.

As métricas podem incluir:

- tempo total;
- tempo por estágio;
- número de regiões semânticas;
- regiões rotuladas;
- confiança média;
- chamadas ao VLM;
- pico de VRAM;
- regiões descobertas e descartadas no region-first;
- imagens concluídas/falhas no baseline.

```mermaid
flowchart TD
    Baseline[baseline/metrics.json] --> Compare[comparison.json]
    Region[region-first/metrics.json] --> Compare
    Compare --> Eval[Experimental evaluation]
```

## Manifest e cache

`manifest.json` registra a identidade da execução e sua estratégia. Marcadores adicionais registram fingerprints por estágio.

```mermaid
flowchart TD
    Manifest[manifest.json] --> Fingerprint[Strategy fingerprint]
    Fingerprint --> Resume{Compatible?}

    Resume -->|Yes| Cache[Reuse artifact]
    Resume -->|No| Recompute[Recompute / require overwrite]

    RegionFP[Stage fingerprint] --> RegionCache[Region-first stage cache]
    RegionCache --> Resume
```

## Auditoria

A persistência deliberada de respostas brutas, máscaras, embeddings, thresholds indiretos via manifest/configuração e overlays permite reconstruir por que uma região ou detecção apareceu no resultado final.

```mermaid
flowchart LR
    Raw[Raw model output] --> Parsed[Parsed representation]
    Parsed --> Intermediate[Intermediate artifacts]
    Intermediate --> Final[Final structured result]
    Final --> Overlay[Visual evidence]

    Raw -. audit .-> Final
    Intermediate -. audit .-> Final
    Overlay -. inspect .-> Final
```
