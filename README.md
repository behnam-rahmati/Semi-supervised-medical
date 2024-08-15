This repository includes the paper "Semi-Supervised Segmentation of Medical Images Focused on the Pixels with Unreliable Predictions"

Some useful introduction about the paper: 
Pseudo-labeling is a well-studied approach in semi-supervised learning. However, unreliable or potentially incorrect pseudo-labels can accumulate training errors during iterative self-training steps, leading to unstable performance. Addressing this challenge typically involves either discarding unreliable pseudo-labels, resulting in the loss of important data, or attempting to refine them, risking the possibility of worsening the pseudo-labels in some cases/pixels. In this paper, we propose a novel method based on pseudo-labeling for semi-supervised segmentation of medical images. Unlike existing approaches, our method neither discards any data nor worsens reliable pseudo-labels. Our approach generates uncertainty maps for the predictions, utilizing reliable pixels without any modification as ground truths and modifying the unreliable ones rather than discarding them. Furthermore, we introduce a novel loss function that incorporates both mentioned parts by multiplying each term by its corresponding uncertainty mask, encompassing reliable and unreliable pixels. The reliable pixels are addressed using a masked cross-entropy loss function, while the modification of the unreliable pixels is performed through a deep-learning-based adaptation of active contours. The entire process is solved within a single loss function without the need to solve traditional active contour equations. We evaluated our approach on three publicly available datasets, including MRI and CT images from cardiac structures and lung tissue. Our core idea is presented in Figure1.

<div align="center">
    <img src="https://github.com/user-attachments/assets/e176909d-b472-475c-b817-129daa0a113b" alt="image" width="350" height="auto">
</div>
The repository is organized as follows:

Models:
This folder includes different models (architectures) that you can test within our framework. To use any model, simply import and call it in the training and testing scripts. Various loss functions are defined in the models, including the standard ones, "CE" for Cross Entropy and "Dice" for Dice Loss, as well as our method-specific loss functions, "AC" and "AC2," which implement our novel loss function. This is an adaptation of active contour models, solved using deep learning rather than traditional partial differential equations.

Test:
This folder includes the module for testing our proposed method. It is tested on all three datasets (uncomment the one you wish to test).

Traditional Active Contours:
This folder includes the implementation of traditional active contours. These methods are not used in our implementation (our approach includes a novel deep learning-based adaptation of them), but we included the traditional methods for readers who may be interested in implementing them.

Training:
This file includes the training modules for all three datasets.

Utilities:
This folder includes utility and helper functions.

The repository paths are shown in the figure below.
<div align="center">
    <img width="200" alt="image" src="https://github.com/user-attachments/assets/3547274c-a277-491d-8b91-416d646d34f9">
</div>


