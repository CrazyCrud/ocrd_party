# OCR-D wrapper for the party text recognizer

ocrd_party is an OCR‑D compliant workspace processor that wraps the Party text‐recognizer to perform line‐level OCR on existing <TextLine> baselines in PAGE‑XML documents.     

*party* GitHub project: https://github.com/mittagessen/party

## Installation
`pip install . `

Make sure your model directory contains a model.safetensors file compatible with Party.
GitHub
## Quick Start

Assuming you have already run layout and line segmentation (e.g. via ocrd-kraken-segment), recognize all lines in your workspace:

```ocrd-party-recognize \
  -I OCR-D-SEG -O OCR-D-TXT \
  -P model_dir=/path/to/party/model \
  -P device=cuda:0 \
  -P batch_size=8
```

Parameters

    model_dir
    Directory containing model.safetensors.

    device
    Compute device for inference (cpu, cuda:0, etc.).

    batch_size
    Number of lines to process in parallel.