# OMNIRE: OMNI URBAN SCENE RECONSTRUCTION

**Ziyu Chen¹·⁶\*, Jiawei Yang¹, Jiahui Huang⁵, Riccardo de Lutio⁵, Janick Martinez Esturo⁵, Boris Ivanovic⁷, Or Litany²·⁵, Zan Gojcic⁶, Sanja Fidler⁶·⁵, Marco Pavone⁶·⁵, Li Song¹, Yue Wang⁶·⁷**

¹Shanghai Jiao Tong University, ²Technion, ³University of Toronto
⁴Stanford University, ⁵NVIDIA Research, ⁶University of Southern California

arXiv:2408.16760v2 [cs.CV] 19 Apr 2025

\* Work done during a research internship at University of Southern California.
✉ Ziyu Chen <ziyu.sjtu@gmail.com> Yue Wang <yue.w@usc.edu>

## ABSTRACT

We introduce OmniRe, a comprehensive system for efficiently creating highfidelity digital twins of dynamic real-world scenes from on-device logs. Recent methods using neural fields or Gaussian Splatting primarily focus on vehicles, hindering a holistic framework for all dynamic elements demanded by downstream applications, e.g., the simulation of human behavior. OmniRe extends beyond vehicle modeling to enable accurate, full-fledged reconstruction of diverse dynamic objects in urban scenes. Our approach builds scene graphs on 3DGS and constructs multiple Gaussian representations for canonical and model various dynamic actors, including vehicles, pedestrians, cyclists, and others. OmniRe allows holistically reconstructing any dynamic object in the scene, enabling advanced simulations (~60 Hz) that include human-participated scenarios, such as pedestrian behavior simulation and human-vehicle interaction. This comprehensive simulation capability is unmatched by existing methods. Extensive evaluations on the Waymo dataset show that our approach outperforms prior state-of-the-art methods quantitatively and qualitatively by a large margin. We further extend our results to 5 additional popular driving datasets to demonstrate its generalizability on common urban scenes. Code and results are available at omnire.

## 1. INTRODUCTION

Creating photorealistic digital twins of 4D real-world is valuable for enabling high-fidelity simulation, robust algorithm training and evaluation. As autonomous driving algorithms increasingly adopt end-to-end models, the need for scalable and high-fidelity simulation environments, where these systems can be evaluated in closed-loop, is becoming more evident. While traditional artist-generated assets are reaching their limits in scale, learned and data-driven methods offer a strong alternative by creating realistic digital twins directly from real-world sensor data. Indeed, neural radiance fields (NeRFs) (Mildenhall et al., 2020; Barron et al., 2022; 2021; Yang et al., 2023b; Guo et al., 2023; Yang et al., 2023a; Wu et al., 2023b) and Gaussian Splatting (GS) (Kerbl et al., 2023; Yan et al., 2024) have emerged as powerful tools for reconstructing scenes with high levels of visual and geometric fidelity. However, accurately and holistically reconstructing dynamic urban scenes remains a significant challenge, especially due to the diverse dynamic actors and their complex rigid and non-rigid motions in urban environments.

Several works have already tried to tackle this challenge. Early methods typically ignore dynamic actors and reconstruct only static parts of the scene (Tancik et al., 2022; Martin-Brualla et al., 2021; Rematas et al., 2022; Guo et al., 2023). Subsequent works aim to reconstruct the dynamic scenes by either (i) modeling the scenes as a combination of a static and time-dependent dynamic neural field (Wu et al., 2022), where the static-dynamic decomposition is an emergent property (Yang et al., 2023a; Turki et al., 2023), or (ii) building a scene graph, in which dynamic actors and the static background are represented as nodes and reconstructed in their canonical frame. The nodes of the scene graph are connected with edges that encode relative transformation representing the motion of each actor through time (Ost et al., 2021; Kundu et al., 2022; Yang et al., 2023b; Wu et al., 2023b; Tonderski et al., 2024; Fischer et al., 2024b). But both approaches fall short of meeting the requirements for comprehensive and interactive digital twins: while providing a more general formulation, methods of (i) lack editability and cannot be directly controlled with classical behavior models. Previous methods following (ii) still focus primarily on representing rigid bodies, thereby largely neglecting other vulnerable road users (VRUs) such as pedestrians and cyclists that are fundamental and critical in urban scenes.

To fill this critical gap, our work aims to model all dynamic actors, including vehicles, pedestrians, and cyclists, and many others, in a manner that allows for interactive simulation. This leads to two primary challenges: (i) developing a holistic framework for modeling diverse non-rigid dynamic actors, given the wide range of non-rigid categories in real-world scenes; (ii) giving specific focus on humans, as their behavior is critical for decision-making (Lei et al., 2023; Jiang et al., 2022; Kocabas et al., 2024) where pedestrian actions directly impact safety. Thus, precise joint-level reconstruction (Lei et al., 2023; Jiang et al., 2022; Kocabas et al., 2024) is crucial for fine control of human behavior in the simulator. To address the specific challenge of modeling human motion dynamics due to unavoidable sensor observations and the limitations of data collected in the wild (Wang et al., 2024; Yang et al., 2021; Wang et al., 2023). Furthermore, reconstructing high-fidelity human appearance from sparse sensor data under more geometry adds additional complexity. Lastly, interactions with large equipment, such as wheelchairs or strollers, which cannot be represented by explicit templates (e.g., SMPL), further compound the geometry complexity.

To address these challenges, we propose an "omni" system capable of handling diverse actors for urban digital twins. Our method OmniRe efficiently reconstructs high-fidelity urban scenes that include static backgrounds, driving vehicles, and non-rigidly moving dynamic actors (see Fig. 1). Specifically, we construct a dynamic neural scene graph (Ost et al., 2021) based on 3D Gaussian Splatting (Kerbl et al., 2023), with dedicated Gaussian representations for different fields of dynamic actors in their local canonical spaces. In our framework, backgrounds and vehicles are represented as static Gaussians, while vehicles undergo rigid body transformations to simulate their motion over time. For non-rigid actors, we incorporate the SMPL model to enable joint-level control for pedestrians using dynamic Gaussians, as SMPL provides a prior template geometry for 3DGS initialization and explicit control for modeling desired human behaviors, which is advantageous for downstream simulation applications. To extract SMPL parameters for human motion modeling, we designed a novel human body pose estimation pipeline dedicated to driving logs with multi-camera setups and severe in-the-wild occlusions. For other template-less dynamic actors, we propose a shared deformation field approach in a similar manner. This framework enables a unified representation of all non-rigid categories and achieves specialized joint-level control for pedestrians. Thus, OmniRe allows for accurate representation and controllable reconstruction of most objects of interest in the scene. Notably, our representation is directly amenable to behavior and animation models that are commonly used in AV simulation (e.g., Fig. 1-(c)).

To summarize, we make the following contributions:

- We introduce OmniRe, a holistic framework for dynamic urban scene reconstruction that embodies the "omni" principle of dynamic category coverage and representation flexibility. OmniRe leverages dynamic neural scene graphs based on Gaussian representations to unify the reconstruction of static backgrounds, driving vehicles, and non-rigidly moving dynamic actors (§ 4). It enables high-fidelity scene reconstruction and human-centered simulation, including pedestrian behavior and human-vehicle interaction—capabilities unmatched by existing methods. (§ 5).
- We address the challenges of modeling humans and dynamic actors from logs such as occlusion, cluttered environments, and the limitations of existing human pose prediction models (§ 4.2). We demonstrate our method on 5 additional popular driving datasets (project page). While our findings are based on AV scenarios, they can generalize to other domains.
- We perform extensive experiments and ablations to demonstrate the benefits of our holistic framework. OmniRe achieves state-of-the-art performance in scene reconstruction and novel view synthesis (NVS), significantly outperforming previous methods in terms of full image metrics (+1.88 PSNR for reconstruction and +2.38 PSNR for NVS). The differences are pronounced for dynamic actors, such as vehicles (+1.18 PSNR), and humans (+4.09 PSNR for reconstruction and +3.06 PSNR for NVS) (Tab. 1).

## 2. RELATED WORK

### Dynamic Scene Modeling.

Neural representations are dominating novel view synthesis (Mildenhall et al., 2020; Barron et al., 2022; 2021; Müller et al., 2022; Fridovich-Keil et al., 2022; Kerbl et al., 2023). These have been extended in different ways to enable dynamic scene reconstruction. Deformation-based approaches (Pumarola et al., 2021a; Park et al., 2021a; Tretschk et al., 2021; Park et al., 2021b; Cai et al., 2022) and recently DeformableGS (Yang et al., 2023c) and (Wu et al., 2023a) propose to model dynamic scenes using a canonical space, coupled with a deformation network mapping time-dependent observations to canonical deformations. These are generally limited to small scenes with limited movement, making them impractical for challenging urban dynamic scenes. Modulation-based techniques operate by directly feeding the image timestamps (or latent codes) as an additional input to neural representations (Xian et al., 2021; Li et al., 2021; 2022; Luiten et al., 2024). However, this generally results in an underconstrained formulation, therefore requiring additional supervision, such as optical flow (Xian et al., 2021), or multi-view inputs captured from synchronized cameras (Li et al., 2022; Luiten et al., 2024). D²NeRF (Wu et al., 2022) proposed to expand on this formulation by partitioning the scene into static and dynamic fields. Following this, SUDS (Turki et al., 2023) and EmerNeRF (Yang et al., 2023a) have shown impressive reconstruction ability for dynamic autonomous driving scenes. However, they model all dynamic elements using a single dynamic field, which are modeled separately, thus they lack controllability, limiting their practicality as sensor simulators. Explicit decomposition of the scene into separate agents enables scene graphs can be represented as bounding boxes in a scene graph as in Neural Scene Graphs (NSG) (Ost et al., 2021) that is widely adopted in UniSim (Yang et al., 2023b), MARS (Wu et al., 2023b), NeRF-DS (Yan et al., 2024), ML-NSG (Fischer et al., 2024b) and recent Gaussian-based works StreetGaussians (Yan et al., 2024), DrivingGaussians (Zhou et al., 2023), and HUGS (Zhou et al., 2024). However, these approaches handle only rigid objects due to limitations of time-independent representations (Ost et al., 2021; Wu et al., 2023b; Yang et al., 2023b; Zhou et al., 2023; 2024; Yan et al., 2024; Tonderski et al., 2024; Fischer et al., 2024b) or limitations of deformation-based techniques (Yang et al., 2023c; Huang et al., 2023). A recent concurrent work Fischer et al. (2024a) also considers non-rigid modeling using a deformation field, addressing a subset of the challenges in modeling holistic dynamics, but does not address fine-grained human models that allow flexible control. To address them, OmniRe proposes a Gaussian scene graph that incorporates various Gaussian representations for both rigid and non-rigid objects, providing extra flexibility and controllability for diverse actors.

### Human Modeling.

Human bodies have variable appearance and complex motions, calling for dedicated modeling techniques. NeuMan (Jiang et al., 2022) proposes to employ the SMPL body model (Loper et al., 2015) to warp raw points from 2015) to warp raw points in canonical space. This approach enables the reconstruction of non-rigid human bodies and warrants fine control. Similarly, recent works such as GART (Lei et al., 2023), GauHuman (Hu & Liu, 2023) and HumanGaussians (Kocabas et al., 2023) have combined the Gaussian representation and the SMPL model. However, these methods are not directly applicable in-the-wild. As for recovering human dynamics in driving scenes, Yang et al. (2021) focuses on shape and pose reconstruction of LIDAR data and LIDAR simulation, while Wang et al. (2023; 2024) aim to recreate natural and accurate human motion from partial observations. However, these methods focus solely on body on shape and pose reconstruction and appearance modeling. In contrast, our method not only models human appearance but also integrates this modeling within a holistic scene framework, to achieve comprehensive solution. Urban scenes typically involve numerous pedestrians, with sparse observation, often accompanied by severe occlusion. We analyze these challenges in detail and address them in § 4.2.

## 3. PRELIMINARIES

### 3D Gaussian Splatting.

First introduced in Kerbl et al. (2023), 3D Gaussian Splatting (3DGS) represents scenes via a set of colored blobs $\mathcal{G} = \{g\}$ whose intensity distribution is a Gaussian. Each Gaussian (blob) $g = (o, \mu, \mathbf{q}, s, c)$ is parameterized by the following attributes: opacity $o \in (0, 1)$, mean position $\mu \in \mathbb{R}^3$, rotation $\mathbf{q} \in \mathbb{R}^4$ represented as a quaternion, anisotropic scaling factors $s \in \mathbb{R}^3_+$, and view-dependent colors $c \in \mathbb{R}^f$ represented as spherical harmonics (SH) coefficients. To compute the color $C$ of a pixel, Gaussians overlapping with this pixel are sorted by their distance to the camera center (sorted by $i \in \mathcal{N}$) and $\alpha$-blended: $C = \sum_{i \in \mathcal{N}} \prod_{j=1}^{i-1}(1 - \alpha_j) \cdot c_i$, where $\alpha_i = o_i \exp(-\frac{1}{2}(\mathbf{p} - \mu_i)^T \boldsymbol{\Sigma}_i^{-1}(\mathbf{p} - \mu_i))$, $\boldsymbol{\Sigma}_i$ is the 2D projection covariance. We further define the application of a rigid (affine) transformation $\mathbf{T} = (\mathbf{R}, \mathbf{t}) \in \mathbb{SE}(3)$ to all Gaussians in the set as: $\mathbf{T} \otimes \mathcal{G} = (o, \mathbf{R}\mu + \mathbf{t}, \text{Rot}(\mathbf{R}, \mathbf{q}), s, c)$, where $\text{Rot}(\cdot)$ denotes rotating the quaternion by the rotation matrix.

### Skinned Multi-Person Linear (SMPL) Model.

SMPL (Loper et al., 2015) is a parametric human body model that combines the advantages of a triangle mesh with linear blending skinning (LBS) to manipulate body shape and pose. At its core, SMPL uses a template mesh $\mathcal{M}_h = (\mathcal{V}, \mathcal{F})$ defined in a canonical rest pose, parameterized by $n_v$ vertices $\mathcal{V} \in \mathbb{R}^{n_v \times 3}$. The template mesh can be shaped and transformed using shape parameters $\boldsymbol{\theta}$: $V_S = \mathcal{V} + B_S(\boldsymbol{\beta}) + B_P(\boldsymbol{\theta})$, where $B_S(\boldsymbol{\beta}) \in \mathbb{R}^{n_v \times 3}$ and $B_P(\boldsymbol{\theta}) \in \mathbb{R}^{n_v \times 3}$ are the $xyz$ offsets to individual vertices (Kocabas et al., 2024) and $V_S$ are the vertex locations in the shaped space.

To further deform the vertices $V_S$ to achieve the desired pose $\boldsymbol{\theta}'$, SMPL utilizes pre-defined LBS weights $W \in \mathbb{R}^{n_v \times n_k}$ and the joint transformations $G$ to define the deformation of each vertex $v_i: v_i' = (\sum_k W_{k,i}G_k) v_i$, where $n_k$ denotes the number of joints, and the joint transformations $G$ are derived from the source pose $\boldsymbol{\theta}$, the target pose $\boldsymbol{\theta}'$ and shape $\boldsymbol{\beta}$. The pose parameters include the body pose component $\boldsymbol{\theta}_b \in \mathbb{R}^{23 \times 3 \times 3}$ and the global pose (translation) $\boldsymbol{\theta}_t \in \mathbb{R}^{3 \times 3}$. For more details of SMPL, we refer readers to Loper et al. (2015). Our method obtains pose parameters $\boldsymbol{\theta}$ for each pedestrian across all frames, as well as the individual body shape parameter $\boldsymbol{\beta}^{\langle h \rangle}$, these pose sequences initialize the non-rigid dynamics of pedestrians. The detailed process is described in § 4.2.

## 4. METHOD

As overviewed in Fig. 2, we build a comprehensive 3DGS framework that holistically reconstructs both the static background and diverse movable entities. We discuss our systematic approach that represents different semantic classes with diverse Gaussian representations in § 4.1, highlighting that this complex yet efficient system-level framework is one of our primary contributions. Modeling humans in unconstrained environments is particularly challenging due to the complexity of human motions and the difficulty of accurately modeling geometry and appearance due to severe occlusions in the wild. We present our approach to this problem in § 4.2, which significantly expands our framework from previous scene-graph based methods. We describe how to optimize the complete scene in § 4.3. After training, we obtain faithful and controllable reconstructions of all movable elements in the scene, enabling advanced simulation and interactive scenarios for downstream applications.

### 4.1. DYNAMIC GAUSSIAN SCENE GRAPH MODELING

**Gaussian Scene Graph.** To allow for flexible control of diverse movable objects in the scene without sacrificing reconstruction quality, we opt for a Gaussian Scene Graph representation. Our scene graph is composed of the following nodes: (1) a Sky Node representing the sky that is far away from the ego-car, (2) a Background Node representing the static scene background such as buildings, roads, and vegetation, (3) a set of Rigid Nodes, each representing a rigidly movable object such as a vehicle, (4) a set of Non-rigid Nodes that include both pedestrians and cyclists. Nodes of type (2,3,4) can be converted directly into world-space Gaussians which we will introduce next. These Gaussians are concatenated and rendered jointly using the rasterizer proposed in Kerbl et al. (2023). The Sky Node is represented by an optimizable environment texture map, similar to Chen et al. (2023), rendered separately, and composited with the rasterized Gaussian image with simple alpha blending.

**Background Node.** The background node is represented by a set of static Gaussians $\mathcal{G}^{\text{bg}}$. These Gaussians are initialized by accumulating 3DGS points and additional points generated randomly in accordance with the strategy described in Chen et al. (2023).

**Rigid Nodes.** Gaussians representing the vehicles (e.g. cars or trucks) are defined as $\mathcal{G}_v^{\text{rigid}}$ in the object's local space (denoted by the upper left element in the index of the vehicle/node). While the Gaussians within a vehicle will not change over time in the local space, the positions of Gaussians in world space will change according to the vehicle's pose $\mathbf{T}_v \in \mathbb{SE}(3)$. At a given time $t \in \mathbb{R}$, the Gaussians are transformed into world space by simply applying the pose transformation:

$$\mathcal{G}_v^{\text{rigid}}(t) = \mathbf{T}_v(t) \otimes \mathcal{G}_v^{\text{rigid}}.$$
(1)

**Non-Rigid Nodes.** Non-rigid individuals are often overlooked by previous methods (Zhou et al., 2023; Yan et al., 2024; Zhou et al., 2024) due to the modeling complexity, despite their importance for human-centered simulation. Unlike rigid vehicles, non-rigid dynamic classes such as pedestrians and cyclists, require extra consideration of both their global movements in world space and their continuous deformations to accurately reconstruct their dynamics. To enable a reconstruction that fully explains the underlying geometry, we further subdivide the non-rigid nodes into two categories: SMPL Nodes for walking or running pedestrians with templates that enable joint-level control and Deformable Nodes for out-of-distribution non-rigid instances (such as cyclists and other template-less dynamic entities).

**Non-Rigid SMPL Nodes.** As introduced in § 3, SMPL provides a parametric way of representing human poses and deformations, and we hence use the model parameters $(\boldsymbol{\theta}(t), \boldsymbol{\beta})$ to drive the 3D Gaussians within the nodes. Here $\boldsymbol{\theta}(t) \in \mathbb{R}^{24 \times 3 \times 3}$ represents the human posture that changes over time $t$. For each node indexed by $h$, We tessellate the SMPL template mesh $\mathcal{M}_h$ instantiated from the resting pose (the 'Da' pose) with 3D Gaussians $\mathcal{G}_h^{\text{SMPL}}$ using a strategy similar to Lei et al. (2023), where each Gaussian is binded to its corresponding vertex of $\mathcal{M}_h$. The world-space Gaussians for each node can be then computed as:

$$\mathcal{G}_h^{\text{SMPL}}(t) = \mathbf{T}_h(t) \otimes \text{LBS}(\boldsymbol{\theta}(t), \mathcal{G}_h^{\text{SMPL}}).$$
(2)

Here $\mathbf{T}_h(t) \in \mathbb{SE}(3)$ is the global pose of the node at time $t$, and $\text{LBS}(\cdot)$ is the linear blend skinning operation that deforms the Gaussians in $\mathcal{G}_h^{\text{SMPL}}$ w.r.t. the SMPL key joints. Once $\boldsymbol{\theta}$ changes over time, the key joints' transformations are updated and linearly interpolated onto the Gaussians to obtain the deformed positions and rotations, while other attributes in the Gaussian remain unchanged. Crucially, it is highly challenging to accurately optimize the SMPL poses $\boldsymbol{\theta}(t)$ from scratch just based on sparse multi-person or indoor scenarios (Jiang et al., 2022; Lei et al., 2023; Kocabas et al., 2024). Hence a rough initialization of $\boldsymbol{\theta}(t)$ is typically needed, whose details are dedicated to a dedicated section § 4.2.

**Non-Rigid Deformable Nodes.** These nodes act as a unified representation for other significant non-rigid instances, including those that fall beyond the scope of SMPL modeling, such as extremely faraway pedestrians for which even state-of-the-art 3D/4D-humans estimators cannot provide accurate estimations, or out-of-distribution, template-less non-rigid individuals. Hence, we propose to use a general deformation network $\mathcal{F}_\varphi$ with parameter $\varphi$ to learn the non-rigid motions within the nodes. Specifically, for node $h$, the world-space Gaussians are defined as:

$$\mathcal{G}_h^{\text{deform}}(t) = \mathbf{T}_h(t) \otimes (\mathcal{G}_h^{\text{deform}} \oplus \mathcal{F}_\varphi(\mathcal{G}_h^{\text{deform}}, \mathbf{e}_h, t)),$$
(3)

where the deformation network generates the changes of the Gaussian attributes from time $t$ to the canonical space Gaussians $\mathcal{G}_h^{\text{deform}}$, outputting the changes in position $\delta \mu_h(t)$, rotation $\delta \mathbf{q}_h(t)$, and the scaling factors $\delta s_h(t)$. The changes are applied back to $\mathcal{G}_h^{\text{deform}}$ with the $\oplus$ operator that internally performs a simple arithmetic addition that results in $(\mu, r + \delta \mu(t), \mathbf{q} + \delta \mathbf{q}(t), s + \delta s(t), c)$. Notably, previous approaches such as Yang et al. (2023c) utilizes a single deformation network for the entire scene, and usually fail in highly complex scenes with many movements. On the contrary, in our work, we define a per-node deformation field which has much more representation power. To maintain computational efficiency, the network weights share their identities of the nodes are disambiguated via an instance embedding parameter $\mathbf{e}_h$. Experimental results in § 5.2 show that deformable Gaussians are essential for achieving good reconstruction quality.

**Sky Node.** We use a separate optimizable environmental map to fit the sky color from viewing directions. Compositing the sky image $C_{\text{sky}}$ with the rendered Gaussians $C_G$ consisting of $(\mathcal{G}^{\text{bg}}, \{\mathcal{G}_v^{\text{rigid}}\}, \{\mathcal{G}_h^{\text{SMPL}}\}, \{\mathcal{G}_h^{\text{deform}}\})$, we obtain the final rendering as:

$$C = C_G + (1 - O_G)|C_{\text{sky}},$$
(4)

where $O_G = \sum_{i=1}^N \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$ is the rendered opacity mask of Gaussians.

### 4.2. RECONSTRUCTING IN-THE-WILD HUMANS

Reconstructing humans from driving logs faces challenges as in-the-wild pose estimators (Goel et al., 2023; Rajasegaran et al., 2022) are typically designed for single video input and often miss predictions in occlusion cases. We designed a pipeline that addresses these limitations to predict accurate and temporally consistent human poses from multi-view videos with frequent occlusions.

Formally, given a set of 3D tracklets for $N$ Humans $\{\mathbf{T}_h, \mathbf{b}_h\}_{h=1}^{N-1}$ from the dataset, our goal is to obtain the corresponding SMPL pose sets: $\boldsymbol{\theta} = \{\boldsymbol{\theta}_h\}_{h=1}^{N-1}$. Here, $\mathbf{T}_h$ and $\boldsymbol{\theta}_h$ (For brevity, $\langle h \rangle$ is omitted) represent the boxes sequence and body pose sequence of the $h$-th human. We apply 4D-Humans (Goel et al., 2023) to each camera's video independently in our multi-camera setup. This yields separately processed results of human tracklets and poses: $\mathbf{T} = \bigcup_{c=0}^{n_c-1} \mathbf{T}^c$ and $\boldsymbol{\theta} = \bigcup_{c=0}^{n_c-1} \boldsymbol{\theta}^c$, where $\mathbf{T}^c = \{\mathbf{T}^c_j\}_{j \in \mathcal{D}^c}$ and $\boldsymbol{\theta}^c = \{\boldsymbol{\theta}^c_j\}_{j \in \mathcal{D}^c}$ represent the predicted tracklets and poses from camera $c$, respectively. Here, $\mathcal{D}^c$ is the set of detected human indices in camera $c$. Our task is to reconstruct $\boldsymbol{\theta}$ using $\hat{\boldsymbol{\theta}}$. We achieve this through the following steps:

**Tracklet Matching:** We define a matching function $\mathcal{M}$ that finds the most similar predicted tracklets for each ground truth tracklet by computing the maximum mean IoU of their 2D projections:

$$\hat{\boldsymbol{\theta}}_h = \mathcal{M}(h, \hat{\boldsymbol{\theta}}, \mathbf{T}, \hat{\mathbf{T}}).$$
(5)

This function learns a matching between ground truth tracklets and predicted tracklets, then outputs the corresponding matched pose sequences. Consider 3-camera setup as an example (Fig. 3(a)), if the $h$-th ground truth tracklet matches with predicted tracklets $j_0, j_1, j_2$ in cameras 0 to 2 respectively, then $\hat{\boldsymbol{\theta}}_h = \{\boldsymbol{\theta}_{j_0}^0, \boldsymbol{\theta}_{j_1}^1, \boldsymbol{\theta}_{j_2}^2\}$, where $\boldsymbol{\theta}_{j_c}^c$ is the pose sequence from camera $c$ for the detected tracklet $j_c$.

**Pose Completion:** As visualized in Fig. 3(b), 4D-Humans (Goel et al., 2023) fails to predict SMPL poses for occluded individuals in driving scenarios, we design a process to recover missing poses:

$$\boldsymbol{\theta}_h = \mathcal{H}(\hat{\boldsymbol{\theta}}_h, \mathbf{T}, \hat{\mathbf{T}}).$$
(6)

Here, function $\mathcal{H}$ identifies missing detections by comparing the ground truth and predicted tracklets, and interpolates missing poses to complete $\boldsymbol{\theta}_h$ from $\hat{\boldsymbol{\theta}}_h$.

### 4.3. OPTIMIZATION

We simultaneously optimize all the parameters as mentioned in § 4.1 in a single stage to reconstruct the entire scene. These parameters include Gaussian attributes (opacity, mean positions, scaling, rotation, and appearance) in their local spaces, namely $\mathcal{G}^{\text{bg}}, \{\mathcal{G}_v^{\text{rigid}}\}, \{\mathcal{G}_h^{\text{SMPL}}\}, \{\mathcal{G}_h^{\text{deform}}\}$, (2) the poses of both rigid and non-rigid nodes for each frame $t$, i.e., $\{\mathbf{T}_v(t)\}, \{\mathbf{T}_h(t)\}$, (3) the human poses of all the SMPL nodes for each frame $t$: $\{\boldsymbol{\theta}_h(t)\}$, and the corresponding skinning weights, (4) the weight $\varphi$ of the deformation network $\mathcal{F}_\varphi$ (5) the weight of the sky model.

We use the following objective function for optimization:

$$\mathcal{L} = (1 - \lambda_s) \mathcal{L}_1 + \lambda_s \mathcal{L}_{\text{SSIM}} + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}} + \lambda_{\text{opacity}} \mathcal{L}_{\text{opacity}} + \mathcal{L}_{\text{reg}},$$
(7)

where $\mathcal{L}_1$ and $\mathcal{L}_{\text{SSIM}}$ are the L1 and SSIM losses on rendered images, $\mathcal{L}_{\text{depth}}$ compares the rendered depth of Gaussians with sparse depth signals from LiDAR, $\mathcal{L}_{\text{opacity}}$ encourages the opacity of the Gaussians to align with the non-sky mask, and $\mathcal{L}_{\text{reg}}$ represents various regularization terms applied to different Gaussian representations. Detailed descriptions of loss terms are provided in the Appendix.

## 5. EXPERIMENTS

**Dataset.** We conduct experiments on the Waymo Open Dataset (Sun et al., 2020), which comprises real-world driving logs. We tested up to 32 dynamic scenes in Waymo, including eight highly complex dynamic scenes that, in addition to typical vehicles, also contain diverse dynamic classes such as pedestrians and cyclists. Each selected segment contains approximately 150 frames. The segment IDs are listed in Tab. 12 and Tab. 6. To further demonstrate our effectiveness on common driving scenes, we extend our results to 5 additional popular driving datasets: NuScenes (Caesar et al., 2020), Argoverse2 (Wilson et al., 2023), PandaSet (Xiao et al., 2021), KITTI (Geiger et al., 2012), and NuPlan (Caesar et al., 2021).

**Baselines.** We compare our method against several Gaussian Splatting approaches: 3DGS (Kerbl et al., 2023), DeformableGS (Yang et al., 2023c), StreetGS (Yan et al., 2024), HUGS (Zhou et al., 2024), and PVG (Chen et al., 2023). Additionally, we compare our method with NeRF-based approach EmerNeRF (Yang et al., 2023a). Our own reimplementation. For 3DGS (Kerbl et al., 2023) and DeformableGS (Yang et al., 2023c), we use the implementation that come with LiDAR depth supervision to ensure the comparison fairness. For other methods, we use their official code. For training, we utilize data from the three front-facing cameras, resized to a resolution of 640×960 for all methods, along with LiDAR data for supervision. We utilize the instance bounding boxes provided by the dataset to transform objects and refine them via pose optimization during training. For further implementation details, please refer to Appendix.

### 5.1. MAIN RESULTS

**Appearance.** We evaluate our method on scene reconstruction and novel view synthesis (NVS) tasks, using every 10th frame as the held-out set for NVS. We report PSNR and SSIM scores for full images, as well as human-related and vehicle-related regions, to assess dynamic reconstruction capabilities. The quantitative results in Tab. 1 show that OmniRe outperforms all other methods, with a significant margin in human-related regions, validating our holistic modeling of dynamic actors. Additionally, while StreetGS (Yan et al., 2024) and our method model vehicles in a similar way, we observe that OmniRe is slightly better than StreetGS even in vehicle regions. This is due to the absence of human modeling in StreetGS, which allows supervision signals from human regions (e.g., colors, LiDAR depth) to incorrectly influence vehicle modeling. The issues StreetGS faces are one of our motivations for modeling almost everything in a scene holistically, aiming to eliminate erroneous supervision and unintended gradient propagation.

In addition, we show visualizations in Fig. 4 to assess model performance qualitatively. Although PVG (Chen et al., 2023) performs well on the scene reconstruction task, it struggles with the novel view synthesis task in highly dynamic scenes, resulting in blurry dynamic objects in novel views (Fig. 4-(f)). HUGS (Zhou et al., 2024) (Fig. 4-(e)), StreetGS (Yan et al., 2024)(Fig. 4-(d)) and 3DGS (Kerbl et al., 2023) (Fig. 8-(a)) fail to reconstruct the details because they are not capable of modeling non-rigid objects. DeformableGS (Yang et al., 2023c) (Fig. 8-(g)) suffers from extreme motion blur for outdoor scenes with significant motion, despite achieving reasonable performance for indoor scenes and cases with small motion. EmerNeRF (Yang et al., 2023a) reconstructs coarse structures for pedestrians and vehicles at a distance, but struggles with fine-grained details (Fig. 4-(c)). In contrast to all these methods in comparison, our method faithfully reconstructs fine details for any part of the scene, handling occlusion, deformation, and extreme motion. Video comparisons are included in the project page.

**Geometry.** In addition to appearance, we also investigate whether our method can reconstruct fine geometry of urban scenes. We evaluate Root Mean Squared Error (RMSE) and two-way Chamfer Distances (CD) for LiDAR depth reconstruction on both training frames and novel frames. Details about evaluation procedures are provided in Appendix. Tab. 1's reported results show that our method outperforms others by a large margin. Fig. 5 illustrates the accurate reconstruction of dynamic actors achieved by our method in comparison to other approaches.

### 5.2. ABLATION STUDIES & APPLICATIONS

**SMPL Modeling.** SMPL modeling is important to model the local, continuous movements of humans. We study its impact by disabling the human pose deformation enabled by SMPL and report the results in Tab. 2 ((a) v.s. (b)) and illustrate these effects in Fig. 7-(B). Without template-based modeling, the reconstructed human actions appear blurry, particularly around the legs, thus failing to accurately reconstruct human body movements. This contrasts sharply with the precise leg reconstruction observed in our default setting. Moreover, SMPL modeling provides joint-level control, improving the controllability (Fig. 1-(c,3), (c,4)).

**Human Body Pose Refinement.** The human body poses extracted as described in (§ 4.2) exhibit prediction errors and scale ambiguity, which subsequently lead to pose errors that degrade reconstruction quality, as shown in Fig. 6 (Noisy). We improve this by jointly optimizing the human poses and Gaussians via the same reconstruction loss. Fig. 6-(a) illustrates this design choice, and Fig. 6 showcases the refined poses. These results verify the effectiveness of our refinement strategy.

**Deformable Nodes.** Deformable nodes are important for accurately reconstructing out-of-distribution or template-less actors. Our approach tackles this challenge by learning a self-supervised deformation field that transforms Gaussians from their canonical space to the shape space. Tab. 2 ((a) v.s. (d)) proves the importance of this component. Fig. 7-(A) shows that without deformable nodes, some dynamic actors are either ignored or incorrectly blended into the background.

**Boxes Refinement.** In practice, we observe that the instance bounding boxes provided by the dataset are imprecise. These noisy ground truth boxes can be harmful to rendering quality. To address this, we jointly refine the bounding box parameters during training. Tab. 4 and Fig. 12 show the practical benefits of this refinement.

**Applications to Simulation.**
Thanks to the decomposition nature of OmniRe, each instance is modeled separately. After joint training, we obtain holistically reconstructed assets that can be flexibly edited in terms of position and rotation. Beyond editing within a single scene, we can also transfer assets from one scene to another, adding variety and complexity to the reconstructed environments. Fig. 1-(c,left) demonstrates a swap of the black vehicle originally in the scene (inset) with a reconstructed vehicle from another scene; and (c,right) an insertion of a pedestrian from one scene in the inset to the street to be meet by a moving car. Additional use case edits are shown in Fig. 11. Through explicit modeling of pedestrians and other non-rigid individuals, we achieve the simulation of reenacted scenarios involving detailed pedestrian-vehicle interaction. As demonstrated in Fig. 9, we simulate a moving vehicle stopping at a crossing, waiting for a pedestrian who slowly crosses. The pedestrian is reconstructed with precision from another scene. This level of precise control over reconstructed photorealistic assets opens up possibilities for interactive simulation with previous simulators for automated simulation (Wang et al., 2022; Wei et al., 2024)

## 6. CONCLUSION

Our method, OmniRe, tackles comprehensive urban scene modeling using Gaussian Scene Graphs. It achieves fast, high-quality reconstruction and rendering, suggesting promise for driving and robotics simulation. We also present solutions for human modeling in complex environments. Future work includes self-supervised learning, improved multi-stage training, and safety/privacy considerations. To ensure reproducibility, the code is available at link.

**Broader impact.** Our method aims to address a significant problem in autonomous driving—simulation. This approach has the potential to aid in the development and testing of autonomous vehicles, potentially leading to safer and more efficient AV systems. Simulation, in a safe and controllable manner, remains an open and fundamental research question.

**Limitations.** While enabling holistic scene modeling, OmniRe still has certain limitations. First, our method does not explicitly model lighting effects, which may lead to visual harmony issues during simulations, particularly when combining elements reconstructed under varying lighting conditions. Addressing this non-trivial challenge requires dedicated efforts beyond the scope of our current work. Further research into modeling light effects and enhancing simulation realism remains crucial for achieving more convincing and harmonious results. Second, similar to other per-scene optimization methods, OmniRe produces less satisfactory novel views when the camera deviates significantly from the training trajectories. Future works to address this issue include incorporating data-driven priors, such as image or video generative models, and optimizing camera poses jointly.

## 7. ETHICS STATEMENT

Our work does not involve the collection or annotation of new data. We utilize well-established public datasets that adhere to strict ethical guidelines. These datasets ensure that sensitive information, including identifiable human features, is blurred or anonymized to protect individual privacy. We are committed to ensuring that our method, as well as future applications, are employed responsibly and ethically to maintain safety and preserve privacy.

## 8. ACKNOWLEDGEMENTS

This work is supported by funding from Toyota Research Institute, Dolby, and Google DeepMind. Yue Wang is also supported by a Powell Faculty Research Award. We also thank Jiageng Mao, Junjie Ye, Ziyi Yang, Haozhe Lou, and Yifan Lu for their valuable discussions during the project, which helped us resolve issues and improve the methods.

## REFERENCES

[Long list of academic references follows...]

---

## Supplemental Material

### A. IMPLEMENTATION DETAILS

**Initialization:** For the background model, we refer to PVG (Chen et al., 2023), combining $6 \times 10^5$ LiDAR points with $4 \times 10^5$ random samples, which are divided into $2 \times 10^5$ near samples uniformly distributed by distance to the scene's origin and $2 \times 10^5$ far samples uniformly distributed by inverse distance. To initialize the background, we filter out the LiDAR samples of dynamic objects. For rigid nodes and non-rigid deformable nodes, we utilize accumulated LiDAR points, while for non-rigid SMPL nodes, we initialize the Gaussians on the template mesh in their canonical space. To determine the initial value of Gaussians that project onto the image plane, whereas random samples are initialized with random colors. The initial human body pose sequences of non-rigid SMPL Nodes are acquired from process described in § 4.2.

**Training:** Our method trains for 30,000 iterations with all scene nodes optimized jointly. The learning rate for Gaussian properties aligns with the default settings of 3DGS (Kerbl et al., 2023), but varies slightly across different node types. Specifically, we set the learning rate for the rotation of Gaussians to $5 \times 10^{-5}$ for non-rigid SMPL nodes and $1 \times 10^{-5}$ for other nodes. The degrees of spherical harmonics are set to 3 for background, rigid nodes, and non-rigid deformable nodes, while it is set to 1 for non-rigid SMPL nodes. The learning rate for the rotation of instance boxes is $1 \times 10^{-5}$, decreasing exponentially to $5 \times 10^{-6}$. The learning rate for the translation of instance boxes is $5 \times 10^{-2}$, decreasing exponentially to $1 \times 10^{-5}$. The learning rate for human body poses of non-rigid SMPL nodes is $5 \times 10^{-5}$, decreasing exponentially to $1 \times 10^{-7}$. For the Gaussian densification strategy, we utilize the absolute gradient Gaussians as introduced in Ye et al. (2024) to control memory usage. We set the densification threshold of position gradient to $3 \times 10^{-4}$. This use of absolute gradient has a minimal impact on performance, as shown in Appendix D.4. The densification threshold for scaling is $3 \times 10^{-3}$. Our method runs on a single NVIDIA RTX 4090 GPU, with training for each scene taking about 1 hour. Training time varies with different training settings.

**Optimization:** We utilize the loss function introduced in Eq (7) to jointly optimize all learnable parameters. The image loss is computed as:

$$\mathcal{L}_{\text{image}} = (1 - \lambda_s) \mathcal{L}_1 + \lambda_s \mathcal{L}_{\text{SSIM}}$$
(8)

due to sparse temporal-spatial observation of the dynamic part, its supervision signal is insufficient. To address this, we apply a higher image loss weight to the dynamic regions identified by the rendered dynamic mask. This weight is set to 5. The depth loss is computed as:

$$\mathcal{L}_{\text{depth}} = \frac{1}{hwc} \sum ||\mathcal{D}^s - \hat{\mathcal{D}}||_1$$
(9)

where $\mathcal{D}^s$ is the inverse of the sparse depth map. We project LiDAR points onto the image plane to generate the sparse LiDAR map, and $\hat{\mathcal{D}}$ is the inverse of the predicted depth map.

The mask loss $\mathcal{L}_{\text{opacity}}$ is computed as:

$$\mathcal{L}_{\text{opacity}} = -\frac{1}{hw} \sum O_G \cdot \log O_G - \frac{1}{hw} \sum M_{\text{sky}} \cdot \log(1 - O_G)$$
(10)

where $M_{\text{sky}}$ is the sky mask, and $O_G$ is the rendered opacity map.

In addition to the reconstruction losses, we introduce various regularization terms for different Gaussian representations to improve quality. Among these, an important regularization term is $\mathcal{L}_{\text{pose}}$, designed to ensure smooth human body poses over time. This term is defined as:

$$\mathcal{L}_{\text{pose}} = \frac{1}{2} ||\boldsymbol{\theta}(t - \delta) + \boldsymbol{\theta}(t + \delta) - 2\boldsymbol{\theta}(t)||_1$$
(11)

where $\delta$ is a randomly chosen integer from $\{1, 2, 3, 4, 5\}$. We set the weight of the SSIM loss, $\lambda_s$, to 0.2, the depth loss, $\lambda_{\text{depth}}$, to 0.1, the opacity loss, $\lambda_{\text{opacity}}$, to 0.05, and the pose smoothness loss, $\lambda_{\text{pose}}$, to 0.01.

### B. BASELINES

[Detailed descriptions of baseline methods EmerNeRF, DeformableGS, StreetGS, HUGS, and PVG]

### C. EVALUATION

**Appearance.** For the Novel View Synthesis task, we select every 10th frame from the original sequence as the test set. We use PSNR and SSIM to evaluate the quality of the rendered images. Since we focus on dynamic scenes, we also compute PSNR and SSIM for regions with vehicles and humans. To identify regions of vehicles and humans, we use Segformer (Xie et al., 2021) to obtain semantic masks. We further identify dynamic objects using dynamic masks obtained by projecting bounding boxes of moving object bounding boxes, utilizing their velocity information. One example of dynamic masks can be seen in Fig. 10.

**Geometry.** Our method uses LiDAR data to initialize Gaussians and supervise scene depth by comparing the rendered depth map with the sparse LiDAR depth map. Post-training, Gaussians typically deviate from their initial state during reconstruction or optimization. Therefore, comparing the LiDAR depth reconstruction is still a valid comparison. We follow the depth evaluation method of StreetSurf (Guo et al., 2023): render the 3D points, unproject depth pixels to 3D LiDAR rays. For Chamfer Distance, re-project the predicted depth to 3D using the LiDAR ray direction and origin. For RMSE, compare the GT and predicted ranges for LiDAR rays.

### D. ADDITIONAL RESULTS

#### D.1. QUALITATIVE COMPARISON

We recommend readers to our project page for video comparisons of the methods.

#### D.2. QUANTITATIVE COMPARISON

To further validate our method's effectiveness, we tested our method against StreetGS (Yan et al., 2024) and EmerNeRF (Yang et al., 2023a) on 32 dynamic scenes from the Waymo dataset, with results reported in Tab. 5.

[Tables 5-14 with detailed quantitative results and ablation studies follow]

### E. OMNIRE IN PRACTICE

**Bounding Boxes.** Similar to other scene-graph-based approaches (Ost et al., 2021; Yang et al., 2023b; Tonderski et al., 2024; Fischer et al., 2024b; Zhou et al., 2023; 2024; Yan et al., 2024), we utilize bounding boxes for driving scene modeling.

**How to Determine Gaussian Representations for Humans?** We categorize pedestrians into two groups for modeling. Near-range pedestrians, detected by our human pose processing module introduced in § 4.2, are modeled using deformable nodes. This approach naturally distinguishes between near and far-range pedestrians based on human detection capability.

[Detailed explanations of implementation choices and practical considerations follow...]
