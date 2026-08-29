# AFBSTSR: Adaptive Frequency Band Selection Transformer for Lightweight Image Super-Resolution

## Dependencies
- Python 3.9
- PyTorch 1.10.0

```
cd code
pip install -r requirements.txt
python setup.py develop
```
## Datasets
- AFBSTSR/datasets

|  Training Set   | Testing Set   |
|  ----  | ----  |
|  DIV2K | Set5 + Set14 + BSD100 + Urban100 + Manga109  |

Refer to the datasets folder for the complete data. Relevant configurations can be modified in options/train/AFBSTSR/AFBSTSR_xx.yml

## Implementation of AFBSTSR
### Train

```shell
#scale factor 2
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt options/train/AFBSTSR/AFBSTSR_x2.yml --launcher pytorch
#scale factor 3
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt options/train/AFBSTSR/AFBSTSR_x3.yml --launcher pytorch
#scale factor 4
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt options/train/EFATSR/AFBSTSR_x4.yml --launcher pytorch
```
### Test
```shell
#scale factor 2
python scripts/test_SISR.py --scale 2 --model_path './experiments/pretrained_models/AFBSTSR_x2.pth'
#scale factor 3
python scripts/test_SISR.py --scale 3 --model_path './experiments/pretrained_models/AFBSTSR_x3.pth'    
#scale factor 4
python scripts/test_SISR.py --scale 4 --model_path './experiments/pretrained_models/AFBSTSR_x4.pth'  
```

