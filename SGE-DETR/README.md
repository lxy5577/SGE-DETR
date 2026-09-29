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
SGE-DETR/
├── README.md
├── requirements.txt
├── configs/
│   ├── SGE-DETR.yaml              # final model used in the paper
├── modules/
│   ├── modules/                # minimal RT-DETR base modules
│   ├── nn/modules/
│   │   ├── cpo.py                 # CPO implementation
│   │   ├── etb.py                 # ETB implementation
│   │   └── gcconv.py              # GCConv implementation
│   └── nn/tasks.py                # minimal model parser and RT-DETR model
├── scripts/
│   └── test\\\_modules.py
```

## Experimental setting

* Input resolution: **640 × 640**.
* Final model configuration: `configs/SGE-DETR.yaml`.
* The `1024` in `scales: l: \\\[1.00, 1.00, 1024]` is the **maximum channel-width parameter**, not the input image resolution.
* Dataset and pretrained weights are intentionally not redistributed.

## Datasets

The experiments can be conducted using publicly available aerial and small-object detection benchmarks. The corresponding datasets should be downloaded from their official project or dataset pages and prepared according to the annotation format required by the training configuration.

## VisDrone

The VisDrone benchmark provides aerial images and videos for object detection and related computer-vision tasks.

Official download page:

https://aiskyeye.com/download/

AI-TOD

AI-TOD (Tiny Object Detection) is a benchmark specifically designed for tiny-object detection in aerial imagery.

Official project repository:

https://github.com/jwwangchn/AI-TOD

UAVDT

UAVDT is an aerial video benchmark for object detection and tracking.

Official project page:

https://sites.google.com/view/grli-uavdt

These datasets are maintained by their respective authors or organizations. Please refer to the corresponding dataset websites and licenses for terms of use and citation requirements. The datasets themselves are not included in this repository.

## Installation

```bash
pip install -r requirements.txt
```

## Verify the proposed modules

```bash
PYTHONPATH=. python scripts/test\\\\\\\\\\\\\\\_modules.py
```

## Build the final model

```bash
PYTHONPATH=. python -c "from ultralytics.nn.tasks import RTDETRDetectionModel; m=RTDETRDetectionModel('configs/SGE-DETR.yaml', ch=3, nc=80, verbose=True); print(sum(p.numel() for p in m.parameters()))"
```What is intentionally excluded

The repository does **not** include the VisDrone/dataset files, pretrained checkpoints, training logs, or unrelated experimental modules from the original development project. The release keeps only the framework code and method-specific components needed to construct the proposed network.

## License

The RT-DETR framework code is derived from Ultralytics YOLO and is distributed under the GNU Affero General Public License v3.0. See `LICENSE-AGPL-3.0.txt`.

