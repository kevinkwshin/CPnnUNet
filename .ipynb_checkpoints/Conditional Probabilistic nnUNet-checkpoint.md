  가장 직접적이고 명확한 제목 (Standard & Clear)


  이 카테고리의 제목들은 연구 내용을 가장 명확하게 전달하며, 검색 용이성이 높습니다.


   1. Probabilistic nnU-Net: A Framework for Modeling Inter-Observer Variability in Medical Image Segmentation
       * (가장 표준적이고 무난한 제목입니다. '무엇을' '어떻게' 했는지 명확히 보여줍니다.)
   2. Learning from Multiple Annotators with Probabilistic nnU-Net
       * (조금 더 간결하며, '다중 Annotator'라는 핵심 문제에 집중하는 제목입니다.)
   3. Modeling Annotator Variability in nnU-Net via Conditional Latent Space
       * (방법론의 핵심인 'Conditional Latent Space'를 강조한 제목입니다.)

  ---


  기술적 기여를 강조하는 제목 (Technically Focused)

  이 카테고리의 제목들은 우리가 구현한 '계층적 z 주입'이라는 독창적인 기술을 부각합니다.


   4. Hierarchical Latent Space Conditioning for Annotator-Specific Segmentation in nnU-Net
       * (핵심 기술인 'Hierarchical Conditioning'을 전면에 내세워, 기술적 깊이를 어필하는 제목입니다.)
   5. Disentangling Annotator Styles in Medical Segmentation using Multi-Scale Probabilistic Conditioning
       * ('Multi-Scale'과 'Disentangling(분리)'이라는 키워드를 사용하여, 모델이 스타일을 얼마나 정교하게 학습하는지
         강조합니다.)
   6. Probabilistic nnU-Net: Diverse and Style-Conditioned Segmentation via Hierarchical Latent Injection
       * (우리가 최종적으로 구현한 'Hierarchical Latent Injection'을 명시하여, 방법론의 독창성을 직접적으로
         보여줍니다.)

  ---

  결과 및 영향력을 강조하는 제목 (Impact-Oriented)

  이 카테고리의 제목들은 연구 결과가 가지는 의미와 영향력에 초점을 맞춥니다.


   7. Beyond a Single Ground Truth: Generating Diverse and Plausible Segmentations with Probabilistic nnU-Net
       * (기존 연구의 한계("Single Ground Truth")를 지적하며, 우리 연구의 패러다임 전환을 강조하는, 임팩트 있는
         제목입니다.)
   8. Quantifying Inter-Observer Variability in Medical Segmentation with a Probabilistic nnU-Net
       * (단순히 다양한 결과를 만드는 것을 넘어, '불확실성을 정량화'한다는 연구의 응용 및 기여도에 집중하는
         제목입니다.)

  ---

  최종 추천


  어떤 점을 가장 강조하고 싶으신지에 따라 선택하시면 됩니다. 제 개인적인 추천은 다음과 같습니다.


   - 가장 균형 잡힌 선택: 1번 또는 4번
       - 1번은 연구 내용을 가장 명확하게 전달하고, 4번은 기술적 독창성을 효과적으로 어필합니다.
   - 가장 인상적인 선택: 7번
       - "Beyond a Single Ground Truth"라는 문구는 리뷰어들에게 연구의 중요성과 독창성을 강하게 각인시킬 수 있습니다.


  이 제목들 중에서 마음에 드시는 것을 고르시거나, 여러 제목의 좋은 부분을 조합하여 최종 제목을 결정하시면 좋겠습니다.



### **Probabilistic nnU-Net: A Framework for Learning from Multiple Annotators and Generating Diverse Segmentations**

**Abstract**

Biomedical image segmentation datasets often contain significant variability due to the diverse annotation styles of multiple experts. Standard segmentation models, such as nnU-Net, are typically trained on a single consensus ground truth, failing to capture this rich inter-observer variability. This limitation can lead to models that are overconfident in a single, averaged prediction, neglecting the plausible range of alternative segmentations. To address this, we propose **Probabilistic nnU-Net**, an extension of the robust nnU-Net framework designed to learn from multiple annotators and generate diverse, realistic segmentation masks.

Our method introduces a conditional variational autoencoder (CVAE) architecture directly into the U-Net's bottleneck and decoder pathway. A low-dimensional latent vector, conditioned on the annotator's ID, is sampled from a learned prior distribution and injected into multiple layers of the network. Specifically, the latent vector is systematically integrated not only at the bottleneck to control global segmentation properties but also into the skip connections at various decoder stages. This hierarchical injection strategy allows the model to learn a rich, multi-scale representation of annotator-specific styles, from high-level structural variations to fine-grained boundary details.

During inference, by providing a specific annotator ID, our model can generate a segmentation that mimics that expert's unique style. Furthermore, by sampling different latent vectors from the learned prior distribution for a given annotator, the model can produce a diverse set of plausible segmentations, effectively capturing the inherent uncertainty and variability in the annotation process. We demonstrate that our framework successfully learns from multi-annotator datasets, generates diverse and realistic segmentations, and provides a more comprehensive understanding of segmentation uncertainty, making it a valuable tool for clinical and research applications.

---

## **3. Methods**

Our proposed framework, Probabilistic nnU-Net, extends the robust and automated nnU-Net architecture to explicitly model and reproduce the segmentation variability inherent in multi-annotator datasets. The core of our method is the integration of a conditional variational autoencoder (CVAE) into the U-Net's latent space and decoding pathway, enabling the generation of diverse, style-conditioned segmentations.

### **3.1. Baseline Architecture: nnU-Net**

We build upon the standard nnU-Net framework, which has demonstrated state-of-the-art performance across a wide range of medical segmentation tasks. The nnU-Net automatically configures its entire pipeline, including preprocessing, network architecture (such as patch size, network depth, and convolutional kernel sizes), and post-processing, based on the properties of a given dataset. Our work retains these automated configuration capabilities for preprocessing and base network topology, ensuring that our probabilistic extension benefits from the same robust foundation. The baseline is a deterministic U-Net that, given an input image, produces a single segmentation map.

### **3.2. Probabilistic Latent Space Modeling**

To capture inter-observer variability, we introduce a low-dimensional latent vector, **z**, which is learned to represent the unique stylistic characteristics of each annotator. The model learns a conditional distribution, *p*(*z*|**X**, *s*), where **X** is the input image and *s* is the annotator ID. This is achieved through a CVAE framework composed of a prior and a posterior network.

**Prior Network:** The prior network, *q<sub>φ</sub>*(*z*|**X**, *s*), aims to predict the distribution of styles for a given annotator *s*. It takes the U-Net's bottleneck feature map concatenated with a one-hot encoded annotator ID vector, *s*, as input. It then outputs the parameters (mean *μ<sub>prior</sub>* and log-variance *log(σ<sup>2</sup>)<sub>prior</sub>*) of a diagonal Gaussian distribution. During inference, we sample **z** from this learned prior, *N*(*μ<sub>prior</sub>*, *σ<sup>2</sup><sub>prior</sub>*), to generate segmentations in the style of annotator *s*.

**Posterior Network:** The posterior network, *p<sub>θ</sub>*(*z*|**X**, **Y**<sub>s</sub>), is used only during training to guide the learning of the latent space. It takes the bottleneck feature map and the corresponding ground truth segmentation, **Y**<sub>s</sub>, from annotator *s* as input. Its purpose is to estimate the "ideal" latent distribution for reconstructing that specific ground truth. It outputs the parameters (*μ<sub>posterior</sub>*, *log(σ<sup>2</sup>)<sub>posterior</sub>*) of the posterior distribution.

Both prior and posterior networks are implemented as lightweight convolutional layers with 1x1 kernels (`AxisAlignedConv`), preserving the spatial resolution of the bottleneck while effectively mapping features to the parameters of the latent distributions.

### **3.3. Hierarchical Latent Vector Injection**

A key contribution of our work is the method by which the sampled latent vector **z** modulates the segmentation output. Instead of a single injection at the bottleneck, we employ a hierarchical conditioning scheme to impose stylistic control across multiple scales of the U-Net decoder.

First, **z** is passed through a dedicated 1x1 convolutional layer (`z_to_features`) and added to the bottleneck feature map, influencing the global, high-level structure of the segmentation. Subsequently, to control finer, scale-specific details, **z** is injected into the skip-connection pathways of the decoder. For each decoder stage (except the final one closest to the output), a separate 1x1 convolutional layer transforms **z** into a feature map with the same channel dimension as the corresponding skip-connection feature map from the encoder. This transformed `z` feature map is then added element-wise to the skip-connection features before they are concatenated with the up-sampled features from the previous decoder stage. This multi-scale injection ensures that the annotator's style is reflected not just globally, but also in local boundary definitions and regional textures at each level of detail.

### **3.4. Loss Function**

The network is trained end-to-end by optimizing a composite loss function, *L<sub>total</sub>*, which consists of two components:

*L<sub>total</sub>* = *L<sub>recon</sub>* + *β* *L<sub>KL</sub>*

1.  **Reconstruction Loss (*L<sub>recon</sub>*):** This term measures the fidelity of the generated segmentation to the ground truth. It is a weighted sum of a batch-wise Dice loss and a Cross-Entropy loss, identical to the standard nnU-Net loss function. To place greater emphasis on precise boundary delineation, we increased the weight of the Cross-Entropy component (*weight_ce*=2).

2.  **KL Divergence Loss (*L<sub>KL</sub>*):** This term acts as a regularizer on the latent space. It minimizes the Kullback-Leibler (KL) divergence between the prior distribution *q<sub>φ</sub>*(*z*|**X**, *s*) and the posterior distribution *p<sub>θ</sub>*(*z*|**X**, **Y**<sub>s</sub>). This forces the prior network to learn to produce meaningful latent distributions conditioned on the annotator ID, without needing the ground truth segmentation as input. The hyperparameter *β* (controlled by `kl_weight` in our implementation) balances the trade-off between reconstruction accuracy and the regularization of the latent space. We employed an annealing scheme for *β*, starting from a small value and gradually increasing it during training. For numerical stability, a small epsilon was added to the variance terms within the KL divergence calculation.

### **3.5. Training and Hyperparameter Modifications**

The model was trained using the Adam optimizer. To ensure stable learning with our more complex, multi-injection architecture, the initial learning rate was reduced to `1e-3` from the nnU-Net default. Furthermore, to encourage greater diversity in the learned styles, we introduced a `variance_scaling_factor` as a command-line argument. This factor artificially scales the variance of the learned prior distribution during training, prompting the model to explore a wider range of stylistic variations.

---

## **4. Evaluation**

To assess the performance of our Probabilistic nnU-Net, we designed an evaluation protocol to measure not only segmentation accuracy but also the model's ability to capture annotator variability and generate diverse outputs.

### **4.1. Segmentation Accuracy**

First, we evaluate the model's overall segmentation accuracy. For each case in the test set, we generate a segmentation by sampling from the prior distribution conditioned on a specific annotator ID (e.g., `annotator_id=0`). This generated segmentation is then compared against the ground truth segmentation of that *same* annotator using the standard Dice Similarity Coefficient (DSC) and 95% Hausdorff Distance (HD95). This is repeated for all annotators, and the average performance is reported to demonstrate the model's ability to faithfully reconstruct the style of each expert.

### **4.2. Diversity and Specificity**

The primary goal of our model is to generate diverse and specific segmentations. We evaluate this in two ways:

1.  **Intra-Annotator Diversity:** For a single test case and a single annotator ID (e.g., `annotator_id=0`), we generate multiple (N=10) different segmentations by repeatedly sampling different latent vectors **z** from the learned prior distribution *q<sub>φ</sub>*(*z*|**X**, *s*=0). We then compute the average pairwise Dice score among these N samples. A lower pairwise Dice score indicates higher diversity, signifying that the model has learned a non-trivial distribution of styles for a single annotator.

2.  **Inter-Annotator Specificity:** We measure if the model correctly captures the distinct styles of different annotators. For a single test case, we generate one segmentation for each annotator ID available (e.g., *s*=0, 1, 2, 3). We then compare the segmentation generated for annotator *i* against the ground truth of annotator *j*. We expect the Dice score to be highest when *i*=*j* (e.g., prediction for annotator 0 vs. ground truth of annotator 0) and lower when *i*≠*j* (e.g., prediction for annotator 0 vs. ground truth of annotator 1). A significant drop in performance for the mismatched pairs demonstrates that the model has successfully learned annotator-specific styles rather than a single, generic mode.

### **4.3. Comparison to Baselines**

We compare our model against two baselines:
1.  **Standard nnU-Net:** Trained on a single consensus ground truth, generated by averaging the multi-annotator labels (e.g., using the STAPLE algorithm).
2.  **Multi-Head nnU-Net:** A deterministic nnU-Net with multiple segmentation heads, where each head is trained to predict the segmentation of a single, specific annotator.

This comparison will demonstrate the advantages of our probabilistic, latent-space approach in terms of generating a diverse and realistic spectrum of segmentations, compared to the single-output nature of the deterministic baselines.