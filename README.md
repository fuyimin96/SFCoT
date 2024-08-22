<p align="center">
  <h1 align="center">SFCoT: Transferable Adversarial Attacks for Remote Sensing Object Recognition via Spatial-Frequency Co-Transformation (TGRS'24)</h1>
  <p align="center">
    <a href="https://github.com/fuyimin96"><strong>Yimin Fu</strong></a>
    &nbsp;&nbsp;
    <strong>Zhunga Liu</strong></a>
    &nbsp;&nbsp;
    <a href="https://github.com/jialinlvcn"><strong>Jialin Lyu</strong></a>
  </p>
  <br>

Pytorch implementation for "[**Transferable Adversarial Attacks for Remote Sensing Object Recognition via Spatial-Frequency Co-Transformation**](https://ieeexplore.ieee.org/document/10636327)"

> **Abstract:** *Adversarial attacks serve as an efficient approach to investigating model robustness, providing insights into internal weaknesses. In real-world applications, the model deployment typically adheres to a black-box setting, necessitating the transferability of adversarial examples crafted on a source model to others. Attack methods in the general computer vision field often employ global input transformations in individual spatial or frequency domains to boost adversarial transferability. However, the recognition of remote sensing objects primarily relies on target-related discriminative regions, whose determination exhibits significant model specificity. Besides, the coupling between objects and background further exacerbates the gap between models. Consequently, the transferability of adversarial examples is limited due to overfitting to the source model. To tackle this problem, we propose a spatial-frequency cotransformation (SFCoT) to improve adversarial transferability for remote sensing object recognition. Specifically, the input image is decomposed into blocks and components in the spatial and frequency domains, respectively. Then, a selective frequency transformation is performed on the low-frequency components to narrow inter-model gaps. Subsequently, modular spatial transformations are adopted in blocks to enhance target-related diversity. Incorporating transformations across domains effectively mitigates the overfitting to model-specific information, leading to better adversarial transferability. Extensive experiments have been conducted on FGSCR-42 and MTARSI datasets, and the results demonstrate that the proposed method achieves state-of-the-art performance across various model architectures.*

:hammer: **We integrate our proposed method and the state-of-the-art adversarial attack methods involved in the comparison experiments into a toolbox to facilitate subsequent research.**

## Requirements
To install the requirements, you can run the following in your environment first:
```bash
pip install -r requirements.txt
```
To run the code with CUDA properly, you can comment out `torch` and `torchvision` in `requirement.txt`, and install the appropriate version of `torch` and `torchvision` according to the instructions on [PyTorch](https://pytorch.org/get-started/locally/).

## Datasets
For the dataset used in this paper, please download the following datasets [MTARSI](https://www.kaggle.com/datasets/aqibriaz/mtarsidataset) / [FGSCR42](https://github.com/DYH666/FGSCR-42) and move them to ```./dataset```.

## Run the code
You can also run the code with the following command:

First run the `train_mode.py` file to train the backbone model needed to counter the attack.
```bash
python train_model.py
```
Or download the [pretrained weights](https://drive.google.com/file/d/150Z3Zli-rxmpfdDCBIooJJx9uobeAgYz/view?usp=drive_link) we gave and move to `./checkpoints`

https://drive.google.com/file/d/150Z3Zli-rxmpfdDCBIooJJx9uobeAgYz/view?usp=drive_link

Then, generate the attack with the following command:
```bash
python main.py \
  --method sfcot --data_type [data_type]\ 
  --model_type [model_type] --batch_size [batch_size] \
  --alpha [alpha] --eps [alpha] --epochs [epochs]\
  --mu [mu] --num_blocks [num_blocks]\
  --num_copies [num_copies] --th [th]\
  --resize_max [resize_max]\
  --rholl [rholl] --wave [wave]
```
The script also supports the following options:
- `--data_type`:  The dataset used against the attack (default: "MTARSI")
- `--model_type`: Backbone as a white-box model (default: "resnet34")
- `--batch_size`: Batch size run no effect on results
- `--alpha`:  The step size of the iteration during the attack
- `--eps`:  Maximum attack strength
- `--epochs`:  Number of attack iterations
- `--mu`:  The Parameter of the momentum method
- `--num_blocks`:  Number of chunks, e.g. 3 means 3*3 chunks.
- `--num_copies`:  Number of replicated samples per iteration
- `--th`:  Thresholds for CAM
- `--resize_max`:  Maximum percentage of random scaling
- `--rholl`:  The Parameter of low-frequency random oscillations
- `--wave`:  Wavelet function name

Examples:
```bash
python main.py \
  --method sfcot --data_type MTARSI\ 
  --model_type resnet34 --batch_size 32 \
  --alpha 1.0 --eps 16.0 --epochs 30 \
  --mu 1.0 --num_blocks 3\
  --num_copies 5 --th 0.5\
  --resize_max 0.4\
  --rholl 0.1 --wave db3
```

## Results
We visualize for attention shift of the target model as below.
<p align="center">
    <img src=./images/show_result.png width="800">
</p>

## Citation
If you find our work and this repository useful. Please consider giving a star :star: and citation.
```bibtex
@article{fu2024transferable,
  title={Transferable Adversarial Attacks for Remote Sensing Object Recognition via Spatial-Frequency Co-Transformation},
  author={Fu, Yimin and Liu, Zhunga and Lyu, Jialin},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
```
