# Architecture

## Objetivo

`image-context` é um laboratório de percepção visual para comparar estratégias de enriquecimento contextual de imagens antes de integrar os resultados a pipelines maiores de mapeamento semântico/contextual.

A arquitetura separa:

- entrada e preparação da imagem;
- orquestração da estratégia;
- adaptadores de modelos pesados;
- contratos de dados;
- persistência de artefatos;
- comparação experimental.

## Arquitetura de alto nível

```mermaid
flowchart LR
    User[User / CLI] --> CLI[cli.py]
    CLI --> Config[config.py]
    CLI --> Input[ImageSample]

    subgraph Strategies
        Baseline[ImageContextPipeline]
        Region[RegionFirstContextPipeline]
    end

    CLI --> Baseline
    CLI --> Region

    subgraph BaselineAdapters[Baseline model adapters]
        Qwen[Qwen VLM]
        GD[Grounding DINO]
        SAMBox[SAM2 box segmentation]
    end

    subgraph RegionAdapters[Region-first model adapters]
        SAMAuto[SAM2 automatic discovery]
        DINO[DINOv2 dense features]
        QwenR[Qwen global/local]
    end

    Baseline --> Qwen
    Baseline --> GD
    Baseline --> SAMBox

    Region --> SAMAuto
    Region --> DINO
    Region --> QwenR

    Baseline --> ArtifactRepo[ArtifactRepository]
    Region --> RegionArtifacts[Region-first artifact persistence]

    ArtifactRepo --> Runs[runs/<run-id>/baseline]
    RegionArtifacts --> Runs2[runs/<run-id>/region-first]

    Runs --> Compare[comparison.json]
    Runs2 --> Compare
```

## Separação por responsabilidade

### CLI

`src/image_context/cli.py` é o ponto de entrada. Ele:

- carrega `config.yaml`;
- aplica overrides de linha de comando;
- prepara a imagem ou amostra do ROS bag;
- seleciona `baseline`, `region-first` ou ambas;
- cria diretórios independentes por estratégia;
- calcula fingerprints;
- escreve o resumo de comparação.

### Baseline

`ImageContextPipeline` implementa a estratégia `baseline`.

Responsabilidades principais:

1. executar três passagens VLM, `objects`, `environment` e `risks`;
2. consolidar conceitos;
3. aplicar fallback configurado quando nenhum conceito é aceito;
4. localizar conceitos com Grounding DINO;
5. segmentar detecções com SAM2;
6. persistir resultados e falhas por estágio.

### Region-first

`RegionFirstContextPipeline` implementa a estratégia `region-first`.

Responsabilidades principais:

1. descobrir regiões sem classe;
2. validar máscaras e IDs;
3. extrair um embedding visual por região;
4. obter contexto global da cena;
5. interpretar cada região localmente;
6. gerar `SemanticRegion` preservando proveniência;
7. persistir métricas, overlays e caches por estágio.

## Portas, adaptadores e modelos

```mermaid
flowchart TB
    P1[Pipeline orchestration] --> Ports[Protocols / ports]
    Ports --> Factory[Model factories]
    Factory --> Adapters[Transformers adapters]
    Adapters --> External[External models]

    External --> Q[Qwen]
    External --> G[Grounding DINO]
    External --> S[SAM2]
    External --> D[DINOv2]
```

A orquestração depende de interfaces e factories, não da implementação concreta dos modelos. Isso mantém os pipelines testáveis com doubles e permite trocar backends sem modificar o fluxo principal.

## Contratos de dados

Os contratos centrais vivem em `models.py` e `region_models.py`.

```mermaid
classDiagram
    class ImageSample {
      frame_id
      source_index
      timestamp_ns
      width
      height
      image_path
    }

    class VisualConcept {
      label
      detector_query
      confidence
      region_kind
      parser_notes
    }

    class Detection {
      detection_id
      bounding_box
      confidence
    }

    class VisualRegion {
      region_id
      mask
      bounding_box
      confidence
      source
    }

    class SemanticRegion {
      region_id
      mask_path
      bounding_box
      geometry_confidence
      label
      description
      attributes
      condition
      confidence
      visual_embedding_path
      candidate_relations
      provenance
    }

    ImageSample --> Detection
    VisualConcept --> Detection
    ImageSample --> VisualRegion
    VisualRegion --> SemanticRegion
```

A distinção principal é que o baseline começa por `VisualConcept`, enquanto a estratégia region-first começa por `VisualRegion`.

## Coexistência das estratégias

As duas abordagens não são variantes que sobrescrevem o mesmo resultado. Elas são tratadas como experimentos independentes.

```mermaid
flowchart TD
    A[One normalized input image]

    A --> BDir[baseline/]
    A --> RDir[region-first/]

    BDir --> BM[baseline manifest + metrics]
    BDir --> BR[result.json + masks + overlays]

    RDir --> RM[region-first manifest + metrics]
    RDir --> RR[semantic_regions.json + embeddings + overlays]

    BM --> Comparison[comparison.json]
    BR --> Comparison
    RM --> Comparison
    RR --> Comparison
```

Isso permite medir recall, número de regiões, confiança, chamadas VLM, tempo por estágio e pico de VRAM de forma comparável.

## Uso de memória dos modelos

Um requisito arquitetural importante é não manter todos os modelos pesados residentes simultaneamente.

```mermaid
sequenceDiagram
    participant Pipeline
    participant ModelA
    participant ModelB
    participant ModelC

    Pipeline->>ModelA: create()
    Pipeline->>ModelA: process stage
    Pipeline->>ModelA: close()

    Pipeline->>ModelB: create()
    Pipeline->>ModelB: process stage
    Pipeline->>ModelB: close()

    Pipeline->>ModelC: create()
    Pipeline->>ModelC: process stage
    Pipeline->>ModelC: close()
```

Essa estratégia reduz a pressão de VRAM e torna possível executar o laboratório em GPUs menores.
