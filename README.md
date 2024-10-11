# VAT
This paper has been accepted by RecSys24. [Link to the paper on Arxiv](https://arxiv.org/pdf/2409.17476)

## Authors
- **Kaike Zhang**
- Qi Cao
- Yunfan Wu
- Fei Sun
- Huawei Shen
- Xueqi Cheng

## Abstract
Recommender systems play a pivotal role in mitigating information overload in various fields. Nonetheless, the inherent openness of these systems introduces vulnerabilities, allowing attackers to insert fake users into the system's training data to skew the exposure of certain items, known as poisoning attacks. Adversarial training has emerged as a notable defense mechanism against such poisoning attacks within recommender systems. Existing adversarial training methods apply perturbations of the same magnitude across all users to enhance system robustness against attacks. Yet, in reality, we find that attacks often affect only a subset of users who are vulnerable. These perturbations of indiscriminate magnitude make it difficult to balance effective protection for vulnerable users without degrading recommendation quality for those who are not affected. To address this issue, our research delves into understanding user vulnerability. Considering that poisoning attacks pollute the training data, we note that the higher degree to which a recommender system fits users' training data correlates with an increased likelihood of users incorporating attack information, indicating their vulnerability. Leveraging these insights, we introduce the Vulnerability-aware Adversarial Training (\textbf{VAT}), designed to defend against poisoning attacks in recommender systems. VAT employs a novel vulnerability-aware function to estimate users' vulnerability based on the degree to which the system fits them. Guided by this estimation, VAT applies perturbations of adaptive magnitude to each user, not only reducing the success ratio of attacks but also preserving, and potentially enhancing, the quality of recommendations. Comprehensive experiments confirm VAT's superior defensive capabilities across different recommendation models and against various types of attacks.


## Environment
- python >= 3.8
- numpy >= 1.22.2
- scikit-learn >= 1.0.2
- scipy >= 1.8.0
- torch >= 1.10.1


## Usage (Quick Start)
1. Install the required packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the main script with the desired backbone model and dataset:

    ```bash
    python main.py --model=<backbone model> --dataset=<dataset>
    ```

   Replace `<backbone model>` with the name of your model, and `<dataset>` with the name of your dataset.

## Citation
If you find our work useful, please cite our paper using the following BibTeX:

```bibtex
@article{zhang2024improving,
  title={Improving the Shortest Plank: Vulnerability-Aware Adversarial Training for Robust Recommender System},
  author={Zhang, Kaike and Cao, Qi and Wu, Yunfan and Sun, Fei and Shen, Huawei and Cheng, Xueqi},
  journal={arXiv preprint arXiv:2409.17476},
  year={2024}
}
