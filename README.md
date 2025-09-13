# ResNet-to-ViT Feature Distillation for Qwen-VL
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-experimental-red.svg)


This project implements a pipeline to replace the Vision Transformer (ViT) encoder of the Qwen-VL model with a ResNet-50 encoder, using a multi-stage adapter for feature space translation. The goal is to create a more computationally efficient Vision-Language Model (VLM) while maintaining good text generation capabilities.

## Technical Concept

The main challenge is to bridge the architectural mismatch between a CNN encoder (ResNet-50) and an LLM that expects input from a ViT.

  * **ResNet-50** produces hierarchical, spatial feature maps at different stages of depth (e.g., `(B, 256, 96, 96)`, `(B, 512, 48, 48)`, etc.).
  * **Qwen-VL's ViT** outputs a flat sequence of patch embeddings (e.g., `(B, 196, 2048)`).

The solution is a **Multi-Stage Adapter** trained to map the hierarchical output of ResNet-50 into the sequential latent space of the original ViT, by minimizing the Mean Squared Error (MSE) between the produced and target embeddings.

## Repository Structure

The project is organized to separate model definitions, training/inference logic, and necessary assets.

```
.
├── assets/                  # Config files, dataset splits, and other non-code data
├── requirements.txt         # Project Python dependencies
├── src/
│   ├── models/              # PyTorch module definitions for the adapters
│   │   ├── MultiStage_Composite_V1.py
│   │   ├── MultiStage_Composite_V2.py
│   │   ├── MultiStage_Composite_V3.py
│   │   └── MultiStage_Composite_V4.py
│   ├── util.py              # Utility functions (e.g., data loading, transforms)
│   ├── qwen_extract_embeddings.ipynb  # 1. Extracts target embeddings from the original ViT
│   ├── composite_model_training.ipynb     # 2. Base training for the adapters
│   ├── composite_model_finetuning.ipynb   # 3. Fine-tuning (joint and sequential)
│   ├── composite_model_test.ipynb         # 4. Quantitative evaluation (MSE) on the test set
│   ├── qwen_standard_inference.ipynb    # A. Inference with the standard Qwen-VL model
│   └── qwen_custom_encoder_inference.ipynb # B. Inference with the ResNet+Adapter encoder
└── test_images/             # Example images for inference
```

### Component Breakdown

  * **`src/models/`**: Contains the 4 versions of the adapter architecture.

      * `MultiStage_Composite_V1.py`: Baseline with a fusion block based on 1x1 convolutions.
      * `MultiStage_Composite_V2.py`: Adds a Transformer Encoder after the fusion block to capture global dependencies via self-attention.
      * `MultiStage_Composite_V3.py`: Enhances the fusion block with a 3x3 convolution for better local spatial context learning.
      * `MultiStage_Composite_V4.py`: A hybrid architecture combining both the enhanced fusion block (from V3) and the Transformer Encoder (from V2).

  * **Jupyter Notebooks**: The notebooks guide you through the entire project pipeline.

      * `qwen_extract_embeddings.ipynb`: Pre-processes the miniImageNet dataset and uses the original Qwen-VL ViT encoder to generate "ground truth" embeddings. These are saved to disk and will serve as the target during adapter training.
      * `composite_model_training.ipynb`: Loads the target embeddings and trains one of the adapter models defined in `src/models/` from scratch. The objective is to minimize the MSE loss.
      * `composite_model_finetuning.ipynb`: Implements advanced training strategies. It loads a pre-trained adapter, unfreezes the ResNet-50 weights, and trains them jointly with the adapter (using differential learning rates), followed by a final fine-tuning phase for the adapter only.
      * `composite_model_test.ipynb`: Performs the final evaluation of the trained model on the test set, calculating the MSE loss for a quantitative performance estimate.
      * `qwen_custom_encoder_inference.ipynb`: The final notebook that assembles the hybrid model. It replaces the Qwen-VL encoder with the trained ResNet-50 + adapter pair and generates text descriptions for arbitrary images.

## Workflow and Usage

To replicate the results, follow these steps in order.

### 1\. Environment Setup

Clone the repository and install the dependencies. Make sure you have the miniImageNet dataset available.

```bash
git clone https://github.com/fabb-24/resnet-to-vit-feature-distillation.git
cd resnet-to-vit-feature-distillation
pip install -r requirements.txt
```

### 2\. Ground Truth Embedding Extraction

Run the `src/qwen_extract_embeddings.ipynb` notebook. This script iterates over the miniImageNet training set, passes each image through the Qwen-VL ViT encoder, and saves the output tensors.

  * **Input**: Image dataset.
  * **Output**: A directory containing the target embedding tensors, one for each image.

### 3\. Adapter Training

Open and run `src/composite_model_training.ipynb`.

  * Select the adapter version from `src/models/` you wish to train.
  * The notebook will dynamically load images, pass them through the ResNet-50 to get the multi-stage feature maps, and use the target embeddings from the previous step to compute the MSE loss.
  * **Output**: Trained adapter model weights (`.pth`).

### 4\. Fine-Tuning (Optional but Recommended)

Use `src/composite_model_finetuning.ipynb` to further improve performance.

  * Load the weights of a model trained in step 3.
  * The notebook will first perform joint fine-tuning of the ResNet and adapter, and then sequential fine-tuning of the adapter alone.
  * **Output**: Optimized adapter model weights.

### 5\. Evaluation and Inference

  * **Quantitative Evaluation**: Run `src/composite_model_test.ipynb` to calculate the final MSE loss on the test set.
  * **Qualitative Inference**: Run `src/qwen_custom_encoder_inference.ipynb`. This notebook loads the Qwen-LLM, the ResNet-50, and the weights of your best adapter. Provide an image (e.g., from `test_images/`) and a prompt to generate a text description.
