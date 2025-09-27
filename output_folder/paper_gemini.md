Published as a conference paper at ICLR 2021

The MLP contains two layers with a GELU non-linearity.

$z_0 = [\textbf{x}_{class}; \textbf{x}_p^1\textbf{E}; \textbf{x}_p^2\textbf{E}; \cdots; \textbf{x}_p^N\textbf{E}] + \textbf{E}_{pos}$,  $\textbf{E} \in \mathbb{R}^{(P^2.C)\times D}$, $\textbf{E}_{pos} \in \mathbb{R}^{(N+1)\times D}$ (1)

$z'_\ell = MSA(LN(z_{\ell-1})) + z_{\ell-1}$, $\ell = 1 \dots L$ (2)

$z_\ell = MLP(LN(z'_\ell)) + z'_\ell$, $\ell = 1 \dots L$ (3)

$y = LN(z_L^0)$ (4)

***Inductive bias.*** We note that Vision Transformer has much less image-specific inductive bias than
CNNs. In CNNs, locality, two-dimensional neighborhood structure, and translation equivariance are
baked into each layer throughout the whole model. In ViT, only MLP layers are local and transla-
tionally equivariant, while the self-attention layers are global. The two-dimensional neighborhood
structure is used very sparingly: in the beginning of the model by cutting the image into patches and
at fine-tuning time for adjusting the position embeddings for images of different resolution (as de-
scribed below). Other than that, the position embeddings at initialization time carry no information
about the 2D positions of the patches and all spatial relations between the patches have to be learned
from scratch.

***Hybrid Architecture.*** As an alternative to raw image patches, the input sequence can be formed
from feature maps of a CNN (LeCun et al., 1989). In this hybrid model, the patch embedding
projection $\textbf{E}$ (Eq. 1) is applied to patches extracted from a CNN feature map. As a special case,
the patches can have spatial size 1x1, which means that the input sequence is obtained by simply
flattening the spatial dimensions of the feature map and projecting to the Transformer dimension.
The classification input embedding and position embeddings are added as described above.

### 3.2 FINE-TUNING AND HIGHER RESOLUTION

Typically, we pre-train ViT on large datasets, and fine-tune to (smaller) downstream tasks. For
this, we remove the pre-trained prediction head and attach a zero-initialized $D \times K$ feedforward
layer, where $K$ is the number of downstream classes. It is often beneficial to fine-tune at higher
resolution than pre-training (Touvron et al., 2019; Kolesnikov et al., 2020). When feeding images
of higher resolution, we keep the patch size the same, which results in a larger effective sequence
length. The Vision Transformer can handle arbitrary sequence lengths (up to memory constraints),
however, the pre-trained position embeddings may no longer be meaningful. We therefore perform
2D interpolation of the pre-trained position embeddings, according to their location in the original
image. Note that this resolution adjustment and patch extraction are the only points at which an
inductive bias about the 2D structure of the images is manually injected into the Vision Transformer.

### 4 EXPERIMENTS

We evaluate the representation learning capabilities of ResNet, Vision Transformer (ViT), and the
hybrid. To understand the data requirements of each model, we pre-train on datasets of varying size
and evaluate many benchmark tasks. When considering the computational cost of pre-training the
model, ViT performs very favourably, attaining state of the art on most recognition benchmarks at
a lower pre-training cost. Lastly, we perform a small experiment using self-supervision, and show
that self-supervised ViT holds promise for the future.

### 4.1 SETUP

***Datasets.*** To explore model scalability, we use the ILSVRC-2012 ImageNet dataset with 1k classes
and 1.3M images (we refer to it as ImageNet in what follows), its superset ImageNet-21k with
21k classes and 14M images (Deng et al., 2009), and JFT (Sun et al., 2017) with 18k classes and
303M high-resolution images. We de-duplicate the pre-training datasets w.r.t. the test sets of the
downstream tasks following Kolesnikov et al. (2020). We transfer the models trained on these
dataset to several benchmark tasks: ImageNet on the original validation labels and the cleaned-up
ReaL labels (Beyer et al., 2020), CIFAR-10/100 (Krizhevsky, 2009), Oxford-IIIT Pets (Parkhi et al.,
2012), and Oxford Flowers-102 (Nilsback & Zisserman, 2008). For these datasets, pre-processing
follows Kolesnikov et al. (2020).


{0}------------------------------------------------


Published as a conference paper at ICLR 2021

| Model     | Layers | Hidden size *D* | MLP size | Heads | Params |
| :-------- | :----- | :--------------- | :------- | :---- | :----- |
| ViT-Base  | 12     | 768              | 3072     | 12    | 86M    |
| ViT-Large | 24     | 1024             | 4096     | 16    | 307M   |
| ViT-Huge  | 32     | 1280             | 5120     | 16    | 632M   |

Table 1: Details of Vision Transformer model variants.

We also evaluate on the 19-task VTAB classification suite (Zhai et al., 2019b). VTAB evaluates
low-data transfer to diverse tasks, using 1 000 training examples per task. The tasks are divided into
three groups: ***Natural*** – tasks like the above, Pets, CIFAR, etc. ***Specialized*** – medical and satellite
imagery, and ***Structured*** – tasks that require geometric understanding like localization.

**Model Variants.** We base ViT configurations on those used for BERT (Devlin et al., 2019), as
summarized in Table 1. The “Base" and "Large" models are directly adopted from BERT and we
add the larger "Huge" model. In what follows we use brief notation to indicate the model size and
the input patch size: for instance, ViT-L/16 means the “Large" variant with 16 × 16 input patch size.
Note that the Transformer's sequence length is inversely proportional to the square of the patch size,
thus models with smaller patch size are computationally more expensive.

For the baseline CNNs, we use ResNet (He et al., 2016), but replace the Batch Normalization lay-
ers (Ioffe & Szegedy, 2015) with Group Normalization (Wu & He, 2018), and used standardized
convolutions (Qiao et al., 2019). These modifications improve transfer (Kolesnikov et al., 2020),
and we denote the modified model "ResNet (BiT)". For the hybrids, we feed the intermediate fea-
ture maps into ViT with patch size of one "pixel". To experiment with different sequence lengths,
we either (i) take the output of stage 4 of a regular ResNet50 or (ii) remove stage 4, place the same
number of layers in stage 3 (keeping the total number of layers), and take the output of this extended
stage 3. Option (ii) results in a 4x longer sequence length, and a more expensive ViT model.

**Training & Fine-tuning.** We train all models, including ResNets, using Adam (Kingma & Ba,
2015) with *β*₁ = 0.9, *β*₂ = 0.999, a batch size of 4096 and apply a high weight decay of 0.1, which
we found to be useful for transfer of all models (Appendix D.1 shows that, in contrast to common
practices, Adam works slightly better than SGD for ResNets in our setting). We use a linear learning
rate warmup and decay, see Appendix B.1 for details. For fine-tuning we use SGD with momentum,
batch size 512, for all models, see Appendix B.1.1. For ImageNet results in Table 2, we fine-tuned at
higher resolution: 512 for ViT-L/16 and 518 for ViT-H/14, and also used Polyak & Juditsky (1992)
averaging with a factor of 0.9999 (Ramachandran et al., 2019; Wang et al., 2020b).

**Metrics.** We report results on downstream datasets either through few-shot or fine-tuning accuracy.
Fine-tuning accuracies capture the performance of each model after fine-tuning it on the respective
dataset. Few-shot accuracies are obtained by solving a regularized least-squares regression problem
that maps the (frozen) representation of a subset of training images to {−1,1}^K target vectors. This
formulation allows us to recover the exact solution in closed form. Though we mainly focus on
fine-tuning performance, we sometimes use linear few-shot accuracies for fast on-the-fly evaluation
where fine-tuning would be too costly.

## 4.2 COMPARISON TO STATE OF THE ART

We first compare our largest models – ViT-H/14 and ViT-L/16 – to state-of-the-art CNNs from
the literature. The first comparison point is Big Transfer (BiT) (Kolesnikov et al., 2020), which
performs supervised transfer learning with large ResNets. The second is Noisy Student (Xie et al.,
2020), which is a large EfficientNet trained using semi-supervised learning on ImageNet and JFT-
300M with the labels removed. Currently, Noisy Student is the state of the art on ImageNet and
BiT-L on the other datasets reported here. All models were trained on TPUv3 hardware, and we
report the number of TPUv3-core-days taken to pre-train each of them, that is, the number of TPU
v3 cores (2 per chip) used for training multiplied by the training time in days.

Table 2 shows the results. The smaller ViT-L/16 model pre-trained on JFT-300M outperforms BiT-L
(which is pre-trained on the same dataset) on all tasks, while requiring substantially less computa-
tional resources to train. The larger model, ViT-H/14, further improves the performance, especially
on the more challenging datasets - ImageNet, CIFAR-100, and the VTAB suite. Interestingly, this

5


{1}------------------------------------------------


Published as a conference paper at ICLR 2021

|                     | Ours-JFT (ViT-H/14) | Ours-JFT (ViT-L/16) | Ours-I21k (ViT-L/16) | BiT-L (ResNet152x4) | Noisy Student (EfficientNet-L2) |
| ------------------- | -------------------- | -------------------- | -------------------- | ------------------- | -------------------------------- |
| ImageNet            | 88.55 ± 0.04         | 87.76 ± 0.03         | 85.30 ± 0.02         | 87.54 ± 0.02        | 88.4/88.5*                       |
| ImageNet ReaL       | 90.72 ± 0.05         | 90.54 ± 0.03         | 88.62 ± 0.05         | 90.54               | 90.55                            |
| CIFAR-10            | 99.50 ± 0.06         | 99.42 ± 0.03         | 99.15 ± 0.03         | 99.37 ± 0.06        | —                                |
| CIFAR-100           | 94.55 ± 0.04         | 93.90 ± 0.05         | 93.25 ± 0.05         | 93.51 ± 0.08        | —                                |
| Oxford-IIIT Pets    | 97.56 ± 0.03         | 97.32 ± 0.11         | 94.67 ± 0.15         | 96.62 ± 0.23        | —                                |
| Oxford Flowers-102  | 99.68 ± 0.02         | 99.74 ± 0.00         | 99.61 ± 0.02         | 99.63 ± 0.03        | —                                |
| VTAB (19 tasks)     | 77.63 ± 0.23         | 76.28 ± 0.46         | 72.72 ± 0.21         | 76.29 ± 1.70        | —                                |
| TPUv3-core-days     | 2.5k                 | 0.68k                | 0.23k                | 9.9k                | 12.3k                            |

Table 2: Comparison with state of the art on popular image classification benchmarks. We report mean and standard deviation of the accuracies, averaged over three fine-tuning runs. Vision Transformer models pre-trained on the JFT-300M dataset outperform ResNet-based baselines on all datasets, while taking substantially less computational resources to pre-train. ViT pre-trained on the smaller public ImageNet-21k dataset performs well too. *Slightly improved 88.5% result reported in Touvron et al. (2020).

**Figure 2: Breakdown of VTAB performance in *Natural, Specialized, and Structured* task groups.**

*The image is a composite bar chart showing a breakdown of VTAB performance across Natural, Specialized, and Structured task groups. The y-axis is labeled "Accuracy [%]" and ranges from 65 to 80. There are four task groups represented on the x-axis: VTAB (19 tasks), Natural (7 tasks), Specialized (4 tasks), and Structured (8 tasks). Each group contains four bars representing different models: ViT-H/14, BiT-L (R152x4), VIVI-Ex-100% (R50x3), and S4L (R50x1).*

model still took substantially less compute to pre-train than prior state of the art. However, we note that pre-training efficiency may be affected not only by the architecture choice, but also other parameters, such as training schedule, optimizer, weight decay, etc. We provide a controlled study of performance vs. compute for different architectures in Section 4.4. Finally, the ViT-L/16 model pre-trained on the public ImageNet-21k dataset performs well on most datasets too, while taking fewer resources to pre-train: it could be trained using a standard cloud TPUv3 with 8 cores in approximately 30 days.

Figure 2 decomposes the VTAB tasks into their respective groups, and compares to previous SOTA methods on this benchmark: BiT, VIVI – a ResNet co-trained on ImageNet and Youtube (Tschannen et al., 2020), and S4L – supervised plus semi-supervised learning on ImageNet (Zhai et al., 2019a). ViT-H/14 outperforms BiT-R152x4, and other methods, on the *Natural* and *Structured* tasks. On the *Specialized* the performance of the top two models is similar.

## 4.3 PRE-TRAINING DATA REQUIREMENTS

The Vision Transformer performs well when pre-trained on a large JFT-300M dataset. With fewer inductive biases for vision than ResNets, how crucial is the dataset size? We perform two series of experiments.

First, we pre-train ViT models on datasets of increasing size: ImageNet, ImageNet-21k, and JFT-300M. To boost the performance on the smaller datasets, we optimize three basic regularization parameters - weight decay, dropout, and label smoothing. Figure 3 shows the results after fine-tuning to ImageNet (results on other datasets are shown in Table 5)² . When pre-trained on the smallest dataset, ImageNet, ViT-Large models underperform compared to ViT-Base models, despite (moderate) regularization. With ImageNet-21k pre-training, their performances are similar. Only with JFT-300M, do we see the full benefit of larger models. Figure 3 also shows the performance

²Note that the ImageNet pre-trained models are also fine-tuned, but again on ImageNet. This is because the resolution increase during fine-tuning improves the performance.