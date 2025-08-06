# 🚗 Simple StyleGAN for CIFAR-10 Cars

A simplified, from-scratch implementation of a StyleGAN-like architecture in TensorFlow/Keras. This project is designed to generate 32x32 images of cars, trained on the "automobile" class from the CIFAR-10 dataset.

This repository was built as a learning exercise to understand and implement the core concepts that make StyleGAN so powerful, even on a small scale.

## 🖼️ Generated Results

## ✨ Key Features

This model is not a full replica of the original StyleGAN paper but implements its most crucial architectural innovations:

  * **🎨 Mapping Network:** A deep Fully Connected network that transforms the initial latent vector `z` into a disentangled intermediate latent space `w`.
  * **💉 Style Injection (AdaIN):** Uses Adaptive Instance Normalization to control the style of the generated image at each resolution level based on the `w` vector.
  * **🔊 Noise Injection:** Adds per-pixel noise at each block to generate stochastic details, making the images look more realistic.
  * **👁️‍🗨️ Powerful Discriminator:** Utilizes a pre-trained ResNet50 backbone as a powerful and stable feature extractor for the discriminator.
  * **⚖️ R1 Regularization:** Employs R1 regularization in a custom training loop for more stable convergence, which is a standard practice for modern GANs.

## 🛠️ Tech Stack

  * **TensorFlow / Keras:** For building and training the neural network models.
  * **NumPy:** For numerical operations and data manipulation.
  * **Matplotlib:** For generating and saving image samples during training.

## ⚙️ Installation

To get this project running on your local machine, follow these steps.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Villeneve/Simple_StyleGAN.git
    cd Simple_StyleGAN
    ```

2.  **Create a Python virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    *(It's recommended to create a `requirements.txt` file with the content below for easy installation)*

    ```bash
    pip install tensorflow numpy matplotlib
    ```

    **requirements.txt:**

    ```
    tensorflow
    numpy
    matplotlib
    ```

## 🚀 Usage

The entire model definition and training loop are contained within the Jupyter Notebook `stylegan_.ipynb`.

1.  **Start Jupyter Notebook:**

    ```bash
    jupyter notebook
    ```

2.  **Run the Notebook:** Open the `stylegan_.ipynb` file and run the cells in order.

The training process will start, and you will see the Generator and Discriminator losses printed periodically. Image samples will be saved to the root directory as `fig_XXX.png` every 50 epochs, and the generator model will be saved as `style.keras`.