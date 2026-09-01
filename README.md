# Image Context

Laboratorio independente para experimentar enriquecimento contextual de imagens antes de
portar os resultados para o `vlm-context-map`.

O fluxo inicial seleciona uma amostra reproduzivel do topico RGB do Corridor02 e executa:

```text
Qwen: objetos -> Qwen: ambiente -> Qwen: riscos
  -> consolidacao de conceitos
  -> Grounding DINO por conceito
  -> SAM2 por bounding box
  -> JSON, mascaras e overlay
```

Os tres modelos pesados nunca ficam residentes ao mesmo tempo. O Qwen processa todas as
imagens e e descarregado antes do Grounding DINO; o mesmo ocorre antes do SAM2. Isso permite
usar a GPU de referencia com 8 GB de VRAM.

O baseline permanece disponivel sem alteracao. Uma segunda estrategia parte da geometria
observada, antes de solicitar rotulos ao VLM:

```text
SAM2 automatico -> regioes class-agnostic
  -> DINOv2 denso com pooling por mascara
  -> Qwen global -> Qwen local por regiao
  -> regioes semanticas 2D, embeddings, proveniencia e overlay
```

Tambem nesse fluxo cada modelo e descarregado antes do seguinte ser criado.

## Preparacao

O projeto requer Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,models]"
```

Se o `python3.12` estiver sendo gerenciado pelo mise:

```bash
mise use python@3.12.14
python -m venv .venv
```

## Validar a amostragem

Este comando nao carrega modelos e apenas extrai as 10 imagens configuradas:

```bash
image-context sample --config config.yaml --overwrite
```

Para conferir reproducibilidade ou testar um caso pequeno:

```bash
image-context sample --config config.yaml \
  --sample-size 1 --seed 42 --run-id smoke-sample --overwrite
```

## Pipeline completo

```bash
image-context run --config config.yaml --overwrite
```

Uma execucao interrompida pode ser retomada sem `--overwrite`. Artefatos concluidos com a
mesma configuracao sao reutilizados. Uma alteracao de seed, prompt, checkpoint ou threshold
muda o fingerprint e exige outro `--run-id` ou `--overwrite`.

Para testar parser, queries, thresholds, DINO ou SAM2 sem executar novamente o Qwen:

```bash
image-context reprocess --config config.yaml \
  --source-run runs/corridor02-v5 \
  --run-id corridor02-v5-adaptive --overwrite
```

## Saida

```text
runs/<run-id>/
├── manifest.json
├── selected_frames.json
└── frames/
    └── frame-XXXXXX/
        ├── image.png
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

Respostas brutas da VLM, configuracao, seed, indices de origem, caixas, confiancas e mascaras
sao mantidos para auditoria.

Quando as tres passagens VLM nao produzem nenhum conceito aceito, o atributo ambiental escolhe
`indoor_fallback_queries` ou `outdoor_fallback_queries`. Essas consultas usam um threshold
proprio e registram a origem em `parser_notes`, mantendo a diferenca entre proposta VLM e
fallback.

## Comparar estrategias

O comando `analyze` recebe uma imagem comum e executa uma estrategia ou ambas sobre a mesma
entrada. Os artefatos e caches ficam separados, portanto executar ou sobrescrever uma estrategia
nao remove o resultado da outra.

```bash
image-context analyze --config config.yaml --image caminho/imagem.png \
  --strategy all --run-id comparison-01
```

Valores aceitos por `--strategy`: `baseline`, `region-first` e `all`. A raiz da execucao contem
`comparison.json`; o novo fluxo grava `region-first/semantic_regions.json`, `global_context.json`,
`metrics.json`, `region_overlay.png` e, por regiao, `mask.png`, `context.png`, resposta VLM bruta e
`visual_embedding.npy`.

O SAM2 automatico usa uma grade deterministica de pontos, filtra mascaras por area e remove
duplicatas por IoU. Os thresholds, limite de regioes e checkpoint DINOv2 ficam na secao
`region_first` de `config.yaml`.

## Qualidade

```bash
pytest
ruff check .
mypy
```
