# Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction

**Ziyi Yang¹·², Xinyu Gao¹, Wen Zhou², Shaohui Jiao², Yuqing Zhang¹, Xiaogang Jin¹†**

¹State Key Laboratory of CAD&CG, Zhejiang University
²ByteDance Inc.

arXiv:2309.13101v2 [cs.CV] 19 Nov 2023

## Abstract

Implicit neural representation has paved the way for new approaches to dynamic scene geometry and appearance rendering. Nonetheless, cutting-edge dynamic neural rendering methods rely heavily on these implicit representations, which frequently struggle to capture fine details of objects in the scene. Furthermore, implicit methods have difficulty achieving real-time rendering in general dynamic scenes, limiting their use in a variety of tasks. To address the issues, we propose a deformable 3D Gaussians Splatting method that reconstructs scenes using 3D Gaussians and learns them in canonical space with a deformation field to model monocular dynamic scenes. We also introduce an annealing smoothing training mechanism with no extra overhead, which can mitigate the impact of inaccurate poses on the dynamic scene reconstruction in real-world datasets. Through a differential Gaussian rasterizer, the deformable 3D Gaussians not only achieve higher rendering quality but also real-time rendering speed. Experiments show that our method outperforms existing methods significantly in terms of both rendering quality and speed, making it well-suited for tasks such as novel-view synthesis, time interpolation, and real-time rendering. Our code is available at https://github.com/ingra14m/Deformable-3D-Gaussians.

## 1. Introduction

High-quality reconstruction and photorealistic rendering of dynamic scenes from a set of input images is critical for a variety of applications, including augmented reality/virtual reality (AR/VR), 3D content production, and entertainment. Previously used methods for rendering these dynamic scenes relied heavily on mesh-based representations, as demonstrated by methods described in [7, 14, 18, 40]. However, these strategies frequently face inherent limitations, such as a lack of detail and realism, a lack of semantic information, and difficulties in accommodating topological changes. With the introduction of neural rendering techniques, this paradigm has undergone a significant shift. Implicit neural representations, particularly as implemented by NeRF [38], have demonstrated commendable efficacy in tasks such as novel-view synthesis, scene reconstruction, and light decomposition.

To improve inference efficiency in NeRF-based static scenes, researchers have developed a variety of acceleration methods, including grid-based structures [7, 46] and pre-computation approaches [14, 52]. Notably, by incorporating hash encoding, Instant-NGP [28] has achieved rapid training. In terms of quality improvement, zipNeRF [3] pioneered an effective anti-aliasing method, which was later incorporated into the grid-based approach by zipNeRF [4]. 3D-GS [15] recently extended the point-based approach with efficient CUDA implementation with 3D Gaussians, which has enabled real-time rendering while matches or exceeds the quality of Mip-NeRF [2]. However, this method is designed for representing static scenes, and its highly customized CUDA rasterization pipeline diminishes its scalability.

Implicit representations have been increasingly harnessed for modeling dynamic scenes. To handle the motion part in a dynamic scene, entangled methods [43, 49] conditioned the NeRF on a time variable. Conversely, disentangled methods [23, 30, 31, 34, 39] employ a deformation field to model a scene in canonical space by mapping point coordinates at a given time to this space. The decoupling modeling approach can effectively represent scenes with nondramatic action variations. However, irrespective of the categorization, adopting an implicit representation for dynamic scenes often proves both inefficient and ineffective, manifesting slow convergence rates coupled with a marked susceptibility to overfitting. Drawing inspiration from seminal NeRF acceleration research, numerous studies on dynamic scene modeling have integrated discrete structures, such as voxel-grids [11, 38], or planes [6, 36]. This integration amplifies both training speed and modeling accuracy. However, challenges remain. Techniques leveraging discrete structures and graphic with the help of constructions of achieving real-time rendering speeds and producing highquality outputs with adequate detail. Multiple works tackle derpin these challenges: Firstly, ray-casting, as a rendering modality, frequently becomes inefficient, especially when scaled to higher resolutions. Secondly, dynamic scenes, in comparison to static ones, exhibit a higher rank, which sets the upper limit of quality achievable by such approaches.

In this paper, to address the aforementioned challenges, we extend the static 3D-GS and propose a deformable 3D Gaussian framework for modeling dynamic scenes. To enhance the applicability of the model, we specifically focus on the modeling of monocular dynamic scenes. Rather than reconstructing the scene frame by frame [36], we condition the 3D Gaussians on time and jointly train a purely implicit deformation field with the learnable 3D Gaussians in canonical space. The gradients for these two components are derived from a customized differential Gaussian rasterizer pipeline. Furthermore, to solve the jitter in temporal sequences during the reconstruction process caused by inaccurate poses, we incorporate an annealing smoothing training (AST) mechanism. This strategy not only improves the smoothness between frames in the time interpolation task but also allows for greater detail to be rendered.

In summary, the major contributions of our work are:
- A deformable 3D-GS framework for modeling monocular dynamic scenes that can achieve real-time rendering and high-fidelity scene reconstruction.
- A novel annealing smoothing training mechanism that ensures temporal smoothness while preserving dynamic details without adding extra training or computational complexity.
- The first framework to extend 3D-GS for dynamic scenes through a deformation field, enabling the learning of 3D Gaussians in canonical space.

## 2. Related Work

### 2.1. Neural Rendering for Dynamic Scenes

Neural rendering, due to its unparalleled capability to generate photorealistic images, has seen an uptick in scholarly interest. Recently, NeRF [38] facilitates photorealistic novel view synthesis through the use of MLPs. Subsequent research has expanded the utility of NeRF for various applications, encompassing tasks such as mesh reconstruction from a collection of images [50, 43], inverse rendering [5, 25, 54], optimization of camera parameters [21, 48], and fewshot learning [10, 51].

Constructing radiance fields for dynamic scenes is a critical branch in the advancement of NeRF, with significant implications for real-world applications. A cardinal challenge in rendering these dynamic scenes lies in the encoding and effective utilization of temporal information, especially when addressing the reconstruction of monocular dynamic scenes, a task inherently involves sparse reconstruction from a single viewpoint. One class of dynamic NeRF approaches models time by deformation by adding time t as an additional input to the radiance field. However, this strategy couples the positional variations induced by temporal changes with the radiance field, lacking the geometric prior information regarding the influence of time on the scene. Consequently, substantial regularization [9, 33] is required to ensure temporal consistency in the rendering results. Another category of methods [23, 30, 31, 34] introduces a deformation field to decouple time and the radiance field, mapping point coordinates to the canonical space corresponding to time t through the deformation field. This decoupled approach is conducive to the learning of pronounced rigid motions and is versatile enough to cater to scenes undergoing topological shifts. Other methods seek to enhance the quality of dynamic neural rendering from various aspects, including segmenting static and dynamic objects in the scene [39, 42], incorporating depth information [1] to introduce geometric prior, introducing 2D CNN to encode scene priors [22, 33], and leveraging the redundant information in multi-view videos [19] to up keyframe compression storage, thereby accelerating the rendering speed.

However, the rendering quality of existing dynamic scene modeling based on MLP (Multilayer Perceptron) remains unsatisfactory. In this work, we will focus on the reconstruction of monocular dynamic scenes. We decouple the deformation field and the radiance field. To enhance the editability and rendering quality of intermediate steps in dynamic scenes, we have adapted this modeling approach to fit within the framework of differentiable pointbased rendering.

### 2.2. Acceleration of Neural Rendering

Real-time rendering has long been a pivotal objective in the field of computer graphics, a goal that is also pursued in the domain of neural rendering. Numerous studies dedicated to NeRF acceleration have meticulously navigated the tradeoff between spatial and temporal efficiency.

Pre-computed methods [12, 35] utilize spatial acceleration structures such as spherical harmonics coefficients [52] and feature vectors [13], cached or distilled from implicit neural representations themselves, as opposed to directly representing the neural representations themselves. A prominent technique [8] in this category transforms NeRF scenes into an amalgamation of coarse meshes and feature textures, thereby enhancing rendering velocity in contemporary mobile graphics pipelines. However, this pre-computed approach may necessitate significant storage capacities for individual scenes. While it offers advantages in terms of inference speed, it demands protracted training durations and exhibits considerable overhead.

Hybrid methods [4, 7, 24, 27, 41, 46] incorporate a neural component within the explicit grid. The hybrid approaches confer the dual benefits of expediting both training and inference phases while producing outcomes on par with advanced frameworks [2, 3]. This is primarily attributed to the robust representational capabilities of the grid. This grid or plane-based strategy has been extended to the acceleration [11] or representation of time-conditioned 4D feature [6, 36, 38] in dynamic scene modeling and time-conditioned compact 4D dynamic scene modeling.

Recently, several studies [16, 35] have evolved the continuous radiance field from implicit representations to differentiable point-based radiance fields, markedly enhancing the rendering speed. 3D-GS [15] further innovates point-based rendering by introducing a customized CUDAbased differential Gaussian rasterization pipeline. This approach not only achieves superior outcomes in tasks like novel-view synthesis and scene modeling, but also facilitates rapid training times on the order of minutes, and supports real-time rendering surpassing 100 FPS. However, the method employs a customized differential Gaussian rasterization pipeline, which complicates its direct extension to dynamic scenes. In this work, our will leverage the point-based rendering framework, 3D-GS, to expedite both the training and rendering speeds for dynamic modeling.

## 3. Method

The overview of our method is illustrated in Fig. 2. The input to our method is a set of images of a monocular dynamic scene, together with the time label and the corresponding camera poses calibrated by SfM [37] which also produces a sparse point cloud. From these points, we create a set of 3D Gaussians $G(x, r, s, \sigma)$ defined by a center position $x$, opacity $\sigma$, and 3D covariance matrix $\Sigma$ from quaternion $r$ and scaling $s$. The view-dependent appearance of each 3D Gaussian is represented via spherical harmonics (SH). To model the dynamic 3D Gaussians that vary over time, we decode the 3D Gaussians via a deformation field. The deformation field takes the positions of the 3D Gaussians and the current time $t$ as inputs, outputting the offsets for the position. Subsequently, we map the deformed 3D Gaussians $\hat{G}(x + \delta x, r + \delta r, s + \delta s, \sigma)$ into the efficient differential Gaussian rasterization pipeline, which is a tile-based rasterizer that allows $\alpha$-blending of anisotropic splats. The 3D Gaussians and deformation network are optimized jointly through the fast backward pass by backpropagating the loss from the rasterized image. Experimental results show that after 30k training iterations, the shape of the 3D Gaussians stabilizes, as does the canonical space, which indirectly proves the efficacy of our design.

### 3.1. Differentiable Rendering Through 3D Gaussians Splatting in Canonical Space

To optimize the parameters of 3D Gaussians in canonical space, it is imperative to differentiably render to 2D pixels from these 3D Gaussians. In this work, we employ the differential Gaussian rasterization pipeline proposed by [15]. Following [55], the 3D Gaussians can be projected to 2D and rendered for each pixel using the following 2D covariance matrix:

$$\Sigma' = J V \Sigma V^T J^T,$$
(1)

where $J$ is the Jacobian of the affine approximation of the projective transformation, $V$ symbolizes the view matrix, transitioning from world to camera coordinates, and $\Sigma$ denotes the 3D covariance matrix.

To make learning the 3D Gaussians easier, $\Sigma$ is divided into two learnable components: the quaternion $r$ representing rotation, and the 3D-vector $s$ represents scaling. These components are then transformed into the corresponding rotation and scaling matrices $R$ and $S$. The reasoning $\Sigma$ can be expressed as:

$$\Sigma = RSS^T R^T.$$
(2)

The color of the pixel on the image plane, denoted by $\mathbf{p}$, is rendered sequentially with point-based volume rendering technique:

$$C(\mathbf{p}) = \sum_{i \in N} T_i \alpha_i c_i,$$
$$\alpha_i = \sigma_i e^{-\frac{1}{2}(\mathbf{p} - \mu_i)^T \Sigma_i'^{-1}(\mathbf{p} - \mu_i)},$$
(3)

where $T_i$ is the transmittance defined by $\Pi_{j=1}^{i-1}(1 - \alpha_j)$, $c_i$ signifies the color of the Gaussians along the ray, and $\mu_i$ represents the $uv$ coordinates of the 3D Gaussians projected onto the image plane.

During the optimization, adaptive density control emerges as a pivotal component, enabling the rendering of 3D Gaussians to achieve desirable outcomes. This control serves a dual purpose: firstly, it mandates the pruning of transparent Gaussians based on opacity. Secondly, it necessitates the densification of Gaussian distribution. This densification lifts regions with explicit geometric intricacies, while simultaneously subdividing areas where Gaussians are large and exhibit significant overlaps. Notably, opacity tends to display pronounced positional gradients. Following [15], we discern the 3D Gaussians that demand adjustments using a threshold set by $t_{pos} = 0.0002$. For diminutive Gaussians inadequate for capturing geometric details, we clone the Gaussians and move them in a certain distance in the direction of the positional gradients. Conversely, for those that are conspicuously large and overlapping, we split them and divide their scale by a hyperparameter $\xi = 1.6$.

It is clear that 3D Gaussians are only appropriate for representing static scenes. Applying a time-conditioned learnable parameter for each 3D Gaussian not only contradicts the original intent of the differentiable Gaussian rasterization pipeline, but also results in the loss of spatiotemporal continuity of motion. To enable 3D Gaussians to represent dynamics while retaining the natural physical meaning of their individual learnable components, we decided to learn 3D Gaussians in canonical space and use an additional deformation field to learn the position and shape variations of the 3D Gaussians.

### 3.2. Deformable 3D Gaussians

An intuitive solution to model dynamic scenes using 3D Gaussians is to separately train 3D-GS on each timedependent view collection and then perform interpolation between these sets as a post-processing step. While such an approach is feasible [9] (e.g., Stereo Multi-View Stereo (MVS) captures at discrete time, it falls short for continuous monocular captures within extended sequences. To deal with the latter, a more general case, we jointly learn a deformation field along with 3D Gaussians.

We decouple the motion and geometry structure by leveraging a deformation network alongside 3D Gaussians, converting the learning process into a canonical space to obtain time-independent 3D Gaussians. This decoupling approach introduces geometric priors of the scene, associating the changes in the positions of the 3D Gaussians with both time and coordinates. The core of the deformation network is an MLP. In our study, we did not employ the grid/plane structures applied in static NeRF that can accelerate rendering and enhance its quality. This is because such methods operate on a low-rank assumption, whereas dynamic scenes possess a higher rank. Explicit point-based rendering further elevates the rank of the scene.

Given center position $x$ of 3D Gaussians as inputs, the deformation MLP produces offsets, which subsequently transform the canonical 3D Gaussians to the deformed space:

$$(\delta x, \delta r, \delta s) = \mathcal{F}_{\theta}(\gamma(\text{sg}(x)), \gamma(t)),$$
(4)

where $\text{sg}(\cdot)$ indicates a stop-gradient operation, $\gamma$ denotes the positional encoding:

$$\gamma(p) = (\sin(2^k \pi p), \cos(2^k \pi p))_{k=0}^{L-1},$$
(5)

where $L = 10$ for $x$ and $L = 6$ for $t$ in synthetic scenes, while $L = 10$ for both $x$ and $t$ in real scenes. We set the depth of the deformation network $D = 8$ and the dimension of the hidden layer $W = 256$. Experiments demonstrate that applying positional encoding to the input of the deformation network can enhance the details in rendering results.

### 3.3. Annealing Smooth Training

A prevalent challenge with numerous real-world datasets is the inaccuracies in pose estimation, a phenomenon markedly evident in dynamic scenes. Training under imprecise poses can lead to overfitting on the training data. Moreover, as also mentioned in HyperNeRF [31], the imprecise poses from colmap for real datasets can cause spatial jitter between each frame w.r.t. the test train set, resulting in a noticeable deviation when rendering the test image compared to the ground truth. Previous methods that used implicit representations benefited from the MLP's inherent smoothness, making the impact of such minor offsets on the final rendering results relatively inconspicuous. However, for the task involving interpolated time, this kind of inconsistent scene at different times can lead to irregular rendering jitters.

To address this issue, we propose a novel annealing smooth training (AST) mechanism specifically designed for real-world dynamic scenes:

$$\Delta = \mathcal{F}_{\theta}(\gamma(\text{sg}(x)), \gamma(t) + \mathcal{X}(i)),$$
$$\mathcal{X}(i) = \mathbb{N}(0, 1) \cdot \beta \cdot \Delta t \cdot (1 - i/\tau),$$
(6)

where $\mathcal{X}(i)$ represents the linearly decaying Gaussian noise at the $i$-th training iteration, $\mathbb{N}(0, 1)$ denotes the standard Gaussian distribution, $\beta$ is an empirically determined scaling factor with a value of 0.1, $\Delta t$ represents the mean time interval, and $\tau$ is the threshold iteration for annealing smooth training (empirically set to 20k).

Compared to the smooth loss introduced by methods of [24, 53], our approach does not incur any additional computational overhead. It can enhance the model's temporal generalization in the early stages of training, as well as prevent excessive smoothing in the later stages, thereby preserving the details of objects in dynamic scenes. Concurrently, it reduces the jitter observed in real datasets during time interpolation tasks.

## 4. Experiment

In this section, we present the experimental evaluation of our method. To give proof of effectiveness, we evaluate our method on the benchmark which consists of the synthetic dataset from D-NeRF [34] and real-world datasets derived from HyperNeRF [31] and NeRF-DS [30]. The division on training and testing part, as well as the image resolution, aligns perfectly with the original paper.

### 4.1. Implementation Details

We implement our framework using PyTorch [32] and modify the differentiable Gaussian rasterization by incorporating depth visualization. For training, we conducted training for a total of 40k iterations. During the initial 3k iterations, we solely trained the 3D Gaussians to attain relatively stable positions and shapes. Subsequently, we jointly train the 3D Gaussians and the deformation field. For optimization, a single Adam optimizer [17] is used with a different learning rate for each component: the learning rate of 3D Gaussians is exactly the same as the official implementation, while the learning rate of the deformation network undergoes exponential decay, ranging from 8e-4 to 1.6e-6. Adam's $\beta$ value range is set to (0.9, 0.999). Experiments with synthetic datasets were all conducted against a black background and at a full resolution of 800x800. All the experiments were done on an NVIDIA RTX 3090.

### 4.2. Results and Comparisons

**Comparisons on synthetic dataset.** In our experiments, we benchmarked our method against several baselines using the monocular synthetic dataset introduced by D-NeRF [34]. The quantitative evaluation, presented in Tab. 1, offers compelling evidence of the superior performance of our approach over the current state-of-the-art. Notably, metrics pertinent to structural consistency, such as LPIPS and SSIM, demonstrate a marked superiority.

For a more visual assessment, we provide qualitative results in Fig. 3. These visual comparisons underscore the capability of our method in delivering high-fidelity dynamic scene modeling. It's evident from the results that our approach ensures enhanced consistency and captures intricate rendering details in novel-view renderings.

**Comparisons on real-world dataset.** We compare our method with the baselines using the monocular real-world dataset from NeRF-DS [30] and HyperNeRF [31]. It should be noted that the camera poses for some of the HyperNeRF datasets are very inaccurate. Given that metrics like PSNR, designed to assess image rendering quality, are inherently penalize slight deviations more than blurring, we have refrained from comparing HyperNeRF in our quantitative analysis. For a qualitative analysis of HyperNeRF, please refer to the supplementary materials. The quantitative and qualitative evaluations for the NeRF-DS dataset are featured in Tab. 2 and Fig. 5, respectively. These results attest to the robustness of our method when applied to real-world scenes, even when the associated poses are not perfectly accurate.

**Rendering Efficiency.** The rendering speed is correlated with the quantity of 3D Gaussians. Overall, when the number of 3D Gaussians is below 250k, our method can achieve real-time rendering over 30 FPS on an NVIDIA RTX 3090. Detailed results can be found in the supplementary material.

**Depth Visualization.** We visualized the depth of synthetic classes scenes in Fig. 6 to demonstrate that our deformation network is well fitted to produce temporal transformation rather than relying on color-based hard-coding. The precise depth underscores the accuracy of our position reconstruction, proving highly advantageous for the novelview synthesis task.

### 4.3. Ablation Study

**Annealing smooth training.** As illustrated in Fig. 4 and Tab. 2, this mechanism fosters improved convergence towards intricate details, effectively mitigating the overfitting tendencies in real-world datasets. Furthermore, it is unequivocally clear from our observations that this strategy significantly bolsters the temporal smoothness of the deformation field. See more ablations in supplementary materials.

## 5. Limitations

Through our experimental evaluations, we observed that the convergence of 3D Gaussians is profoundly influenced by the diversity of perspectives. As a result, datasets characterized by sparse viewpoints and limited viewpoints coverage may lead our method to encounter overfitting challenges. Additionally, the efficacy of our approach is contingent upon the accuracy of pose estimation. This dependency was evident when our method did not achieve optimal PSNR scores on the NeRF-DS/HyperNeRF dataset, attributable to deviations in pose estimation via COLMAP. Furthermore, the temporal complexity of our approach is directly proportional to the quantity of 3D Gaussians. In scenarios with an extensive array of 3D Gaussians, there is a potential escalation in both training duration and memory consumption. Lastly, our evaluations have predominantly revolved around scenes with moderate motion dynamics. The method's adeptness at handling extreme human motions, such as nuanced facial expressions, remains an open question. We perceive these constraints as promising directions for subsequent research endeavors.

## 6. Conclusions

We introduce a novel deformable 3D Gaussian splatting method, specifically designed for monocular dynamic scene modeling, which surpasses existing methods in both quality and speed. By learning the 3D Gaussians in canonical space, we enhance the versatility of the 3D-GS differentiable rendering pipeline for dynamically captured monocular scenes. It's crucial to recognize that point-based methods, in comparison to implicit representations, are notably better suited for post-production tasks. Additionally, our method incorporates an annealing smooth training strategy, aimed at reducing overfitting associated with time encoding while maintaining intricate scene details, without adding any extra training overhead. Experimental results demonstrate that our method not only achieves superior rendering effects but is also capable of real-time rendering.

## References

[References section with numbered citations 1-55]

---

## Appendix

### A. Overview

The appendix provides some implementation details and further results that accompany the paper.
- Section B introduces the implementation details of the network architecture for our approach.
- Section C provides additional results, including more visualizations, rendering efficiency, more comparisons, and more ablations.
- Section D discusses the failure cases of our method.

### B. Implementation Details

#### B.1. Network Architecture of the Deformation Field

We learn the deformation field with an MLP network $\mathcal{F}_{\theta}: (\gamma(x), \gamma(t)) \rightarrow (\delta x, \delta r, \delta s)$, which maps from each coordinate of 3D Gaussians and time to their corresponding deviations in position, rotation, and scaling. The parameters $\theta$ of the MLP are optimized through this mapping. As shown in Fig. 7, our MLP $\mathcal{F}_{\theta}$ initially processes the input through eight fully connected layers that include ReLU activations and feature 256-dimensional hidden layers, and outputs a 256-dimensional feature vector. This feature vector is subsequently passed through three additional fully connected layers (without activation) to separately output the offsets over time for position, rotation, and scaling. It should be noted that similar to NeRF, we concatenate the feature vector and the input in the fourth layer. Due to the compact structure of MLP, our additional storage compared to 3D Gaussians is only 2MB.

#### B.2. Optimization Loss

During the training of our deformable Gaussians, we deform the 3D Gaussians at each timestep into the canonical space. We then jointly optimize both the deformation network and the 3D Gaussians using a combination of $\mathcal{L}_1$ loss and D-SSIM loss [19]:

$$\mathcal{L} = (1 - \lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{D\text{-SSIM}},$$
(7)

where $\lambda = 0.2$ is used in all our experiments.

### C. Additional Results

#### C.1. Per-Scene Results on the NeRF-DS Dataset

In Tab. 3, we provide the results for individual scenes associated with scenes 4-6 of the original paper. It can be observed that our method achieved superior metrics in almost every scene compared to those without AST, underscoring the generalizability of AST in real datasets where the pose is not perfectly accurate. Overall, our method outperforms baselines on the NeRF-DS Dataset.

#### C.2. Results on the HyperNeRF Dataset

We visualize the results of the HyperNeRF dataset in Fig. 9. Notably, metrics designed to assess image rendering quality, such as PSNR, tend to penalize minor offsets more heavily than blurring. Therefore, for datasets with less accurate camera poses, like HyperNeRF, our method's quantitative metrics might not consistently outperform those of methods yielding blurred outputs when faced with imprecise camera poses. Despite that, our method still often exhibit fewer artifacts and greater clarity. This phenomenon aligns with observations reported in Mip-NeRF [2] and HyperNeRF [31].

#### C.3. Results on Rendering Efficiency

In our research, we present comprehensive Frames Per Second (FPS) testing results in Tab. 4. Tests were conducted on one NVIDIA RTX 3090. It is observed that when the number of point clouds remains below ~250k, our method can achieve real-time rendering at rates greater than 30 FPS.

#### C.4. More Ablations

**Network architecture.** We present ablation experiments on the architecture of our purely implicit network, as shown in Tab. 7. The results of these experiments suggest that the structure within our pipeline is optimal. Notably, we did not adopt feature-based structures because dynamic scenes do not conform to the low-rank assumption. Furthermore, the explicit point-based rendering of 3D-GS exacerbates the rank of dynamic scenes. Our early experimental validations have corroborated this assertion.

#### C.5. Background color

In the research of Neural Rendering, it's common to use a black or white background for rendering scenes without a background. In our experiments, we found that the background color has an impact on certain scenes in the D-NeRF dataset. The experimental results are shown in Tab. 8. Overall, a black background yields better rendering results. For the sake of consistency in our experiments, we uniformly used a black background in our main text experiments. However, for the bouncing and trex scenes, using a white background can produce better results.

#### C.6. Deformation using SE(3) Field

Drawing inspiration from Nerfies [30], we applied a 6-DOF SE(3) field that accounts for rotation to the transformation of 3D Gaussian positions. The experimental results, presented in Tab. 5 and Tab. 6, indicate that this constraint offers a minor improvement on the D-NeRF dataset. However, it appears to diminish the quality on the more complex real-world NeRF-DS dataset. Moreover, the additional computational overhead introduced by the SE(3) Field approximately increases 50 % of the training time and results in about a 20% decrease in FPS during rendering. Consequently, we opted to utilize a direct addition without imposing SE(3) constraints on the transformation of position.

### D. Failure Cases

**Inaccurate poses.** In our research, we find that inaccurate poses can lead to the failure of the convergence of deformable-gs, as illustrated in Fig. 8. For implicit representations, their inherent smoothness can maintain robustness in the face of minor deviations in pose. However, for the explicit point-based rendering, such inaccuracies are particularly detrimental, resulting in inconsistencies in the scene at different moments.

**Few training viewpoints.** In our study, a notable scarcity of training views presents a dual challenge: both few-shot learning and a limited number of viewpoints. Either aspect can lead to overfitting in deformable-gs and even in 3D-GS on the training set. As demonstrated in Fig. 10, significant overfitting is evident in the DeVRF [25] dataset. The training set for this scene contains 100 images, but the viewpoints for training are limited to only four. However, by swapping the training and test sets, where the test set contained an equal number of 100 images and viewpoints, we obtained markedly better results.
