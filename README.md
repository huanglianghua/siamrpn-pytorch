# SiamRPN - PyTorch

A clean PyTorch implementation of SiamRPN tracker described in paper [High Performance Visual Tracking with Siamese Region Proposal Network](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf). The code is evaluated on 7 tracking datasets ([OTB (2013/2015)](http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html), [VOT (2018)](http://votchallenge.net), [DTB70](https://github.com/flyers/drone-tracking), [TColor128](http://www.dabi.temple.edu/~hbling/data/TColor-128/TColor-128.html), [NfS](http://ci2cv.net/nfs/index.html) and [UAV123](https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx)), using the [GOT-10k toolkit](https://github.com/got-10k/toolkit).

## Performance

### GOT-10k

| Dataset | AO    | SR<sub>0.50</sub> | SR<sub>0.75</sub> |
|:------- |:-----:|:-----------------:|:-----------------:|
| GOT-10k | 0.462 | 0.556             | 0.218             |

The scores surpass the highest performance on [GOT-10k leaderboard](http://got-10k.aitestunion.com/leaderboard) (AO 0.374, SR<sub>0.50</sub> 0.404) by a large margin.

However, since SiamRPN is trained on 4 extra datasets (ILSVRC-VID, YouTube-BB, ImageNet Detection and COCO) and it does not follow the **one-shot principle** (zero-overlap between training and test object classes) of GOT-10k, the comparison may not be fair.

### OTB / UAV123 / DTB70 / TColor128 / NfS

| Dataset       | Success Score    | Precision Score |
|:-----------   |:----------------:|:----------------:|
| OTB2013       | 0.641            | 0.855            |
| OTB2015       | 0.629            | 0.837            |
| UAV123        | 0.599            | 0.770            |
| UAV20L        | 0.531            | 0.656            |
| DTB70         | 0.548            | 0.756            |
| TColor128     | 0.533            | 0.736            |
| NfS (30 fps)  | 0.453            | 0.529            |
| NfS (240 fps) | 0.589            | 0.706            |

### VOT2018

| Dataset       | Accuracy    | Robustness (unnormalized) |
|:-----------   |:-----------:|:-------------------------:|
| VOT2018       | 0.576       | 27.00                     |

## Dependencies

Install PyTorch, opencv-python and GOT-10k toolkit:

```bash
pip install torch
pip install opencv-python
pip install --upgrade git+https://github.com/got-10k/toolkit.git@master
```

[GOT-10k toolkit](https://github.com/got-10k/toolkit) is a visual tracking toolkit that implements evaluation metrics and tracking pipelines for 7 main datasets (GOT-10k, OTB, VOT, UAV123, NfS, etc.).

## Running the tracker

In the root directory of `siamrpn-pytorch`:

1. Download pretrained `model.pth` from [Baidu Yun](https://pan.baidu.com/s/1QYoQUNraPMUmFW6rp5PDFA) or [Google Drive](https://drive.google.com/open?id=1P0nshF9OjEJwuY9bScuLhPyA2CXSNB5f), and put the file under `pretrained/siamrpn`.

2. Create a symbolic link `data` to your datasets folder (e.g., `data/OTB`, `data/UAV123`, `data/GOT-10k`).

3. Run:

```
python run_tracking.py
```

By default, the tracking experiments will be executed and evaluated over all 7 datasets. Comment lines in `run_tracker.py` as you wish if you need to skip some experiments.
