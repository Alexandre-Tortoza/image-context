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
        ├── overlay.png
        └── masks/
```

Respostas brutas da VLM, configuracao, seed, indices de origem, caixas, confiancas e mascaras
sao mantidos para auditoria.

## Qualidade

```bash
pytest
ruff check .
mypy
```
