# SGE-DETR: Reproducible Implementation

This repository provides the paper-specific implementation of **SGE-DETR**, an RT-DETR-based detector for small-object detection. The public code is organized around the **three paper-level enhancement designs**, rather than treating every internal operator as a separate contribution.

## Three paper-level contributions

### 1\. CPO — Collaborative P2-P3 Optimization

CPO is derived from the **SOEP-RFPN design** in the original experiment files. In the final SGE-DETR architecture, the CPO enhancement path combines **SPDConv**, **CSPOmniKernel**, and **SNI** for P2-P3 collaborative feature enhancement. SNI is additionally reused in the top-down feature aggregation path. The original `rtdetr-SOEP-RFPN.yaml` is retained as a reference configuration for the CPO-related SOEP-RFPN design; it is not an additional paper-level contribution.

### 2\. ETB — Entangled Transformer Block

ETB replaces the original AIFI operation and jointly models spatial and frequency-domain representations.

### 3\. GCConv — Global-Context Convolution

GCConv is inserted into the PAN bottom-up path to supplement local convolution with global contextual information.

## Repository structure

```text
JEI-SGE-DETR/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── SGE-DETR.yaml              # final model used in the paper
├── ultralytics/
│   ├── nn/modules/                # minimal RT-DETR base modules
│   ├── nn/extra\_modules/
│   │   ├── cpo.py                 # CPO implementation
│   │   ├── etb.py                 # ETB implementation
│   │   └── gcconv.py              # GCConv implementation
│   └── nn/tasks.py                # minimal model parser and RT-DETR model
├── scripts/
│   └── test\_modules.py
```

## Experimental setting

* Input resolution: **640 × 640**.
* Final model configuration: `configs/SGE-DETR.yaml`.
* The `1024` in `scales: l: \[1.00, 1.00, 1024]` is the **maximum channel-width parameter**, not the input image resolution.
* Dataset and pretrained weights are intentionally not redistributed.

## Installation

```bash
pip install -r requirements.txt
```

## Verify the proposed modules

```bash
PYTHONPATH=. python scripts/test\_modules.py
```

## Build the final model

```bash
PYTHONPATH=. python -c "from ultralytics.nn.tasks import RTDETRDetectionModel; m=RTDETRDetectionModel('configs/SGE-DETR.yaml', ch=3, nc=80, verbose=True); print(sum(p.numel() for p in m.parameters()))"
```

## Training / validation / inference

The scripts add the repository root to `PYTHONPATH` automatically, so they can be launched directly from the repository root. Prepare the dataset according to `configs/dataset\_template.yaml`, then use the provided scripts:

```bash
python scripts/train.py
python scripts/val.py --weights path/to/best.pt
python scripts/predict.py --weights path/to/best.pt --source path/to/image.jpg
```

The final paper configuration is evaluated at **640 × 640**.

## What is intentionally excluded

The repository does **not** include the VisDrone/dataset files, pretrained checkpoints, training logs, or unrelated experimental modules from the original development project. The release keeps only the framework code and method-specific components needed to construct the proposed network.

## License

The RT-DETR framework code is derived from Ultralytics YOLO and is distributed under the GNU Affero General Public License v3.0. See `LICENSE-AGPL-3.0.txt`.

