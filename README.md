# OCR-D wrapper for the party text recognizer

> ocrd_party is an OCR‑D compliant workspace processor that wraps the Party text‐recognizer to perform line‐level OCR on existing <TextLine> baselines in PAGE‑XML documents.     
> *party* GitHub project: https://github.com/mittagessen/party
     
*The project is still in progress*

## Installation

```commandline
git clone https://github.com/mittagessen/party.git <path-to>/party
cd <path-to>/party
pip install .

cd <path-to>/ocrd_party
pip install -r requirements.txt
pip install .
```     

Or install via Docker:
```
- docker compose build
- docker-compose run ocrd-party
```
For CPU only:
```
- docker compose build ocrd-party-cpu
- docker-compose run ocrd-party-cpu
```    

Make sure your model directory contains a model.safetensors file compatible with Party.

## Quick Start

Assuming you have already run layout and line segmentation (e.g. via ocrd-kraken-segment), recognize all lines in your workspace:

```ocrd-party-recognize \
  -I OCR-D-SEG -O OCR-D-TXT \
  -P model /path/to/party/model \
  -P device cuda \
  -P batch_size 8
```

Parameters

    model
    Directory containing model.safetensors.

    device
    Compute device for inference (cpu, cuda:0, etc.).

    batch_size
    Number of lines to process in parallel.
