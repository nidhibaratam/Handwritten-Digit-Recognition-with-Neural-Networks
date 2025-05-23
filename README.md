# From Pixels to Predictions: Our First Dive into Image Recognition


**By ,
BITAN DAS, BARATAM NIDHISHRI, BHAVYE GARG
from The LNM Institute of Information Technology**

Have you ever wondered how your phone recognizes faces or how computers distinguish between different objects? It’s all thanks to the magic of **image recognition**! As a team of college students eager to dive into the world of **computer vision**, we decided to build our very own model to identify handwritten digits — a classic challenge that taught us the ropes.

Explore the code: [GitHub](https://github.com/nidhibaratam/Handwritten-Digit-Recognition-with-Neural-Networks/blob/main/Handwritten_Digit_Recognition_with_Neural_Networks.ipynb)
### The “Why”: Seeing Beyond the Pixels

As beginners, we wanted to understand how machines “see.” Unlike humans, computers view images as grids of numbers (pixel values). Teaching a machine to understand a squiggly handwritten ‘7’ as distinct from a ‘1’ is a fascinating problem that perfectly introduced us to **Artificial Neural Networks (ANNs)**. Our goal was to build a model that could accurately interpret these numerical grids.

### Data at Hand: The Famous MNIST Dataset

For this project, we leveraged the widely-used **MNIST dataset**. It’s a collection of thousands of 28x28 pixel grayscale images of handwritten digits, from 0 to 9. It’s the “Hello World” of deep learning for a reason — perfect for learning without getting overwhelmed by huge, complex datasets.

### Preparing Our “Eyes”: Data Preprocessing

Before our model could learn, we had to prepare the image data. This involved a couple of crucial steps:

* **Normalization:** We scaled down the pixel values from a range of 0–255 to 0–1. Think of it like standardizing units so the model doesn’t get confused by large numbers — it helps the learning process be smoother and faster.
* **Flattening:** Our 28x28 pixel images were “unrolled” into a single, long list of 784 numbers. This is because our type of neural network expects a flat input, not a 2D grid.

### Building Our Brain: The Keras Model Architecture

We chose **TensorFlow’s Keras API** for its user-friendliness, which felt like building with LEGOs! Our Artificial Neural Network (ANN) architecture consisted of:

* **An Input Layer:** Taking our flattened 784-pixel data.
* **Two Hidden Layers (128 & 32 neurons):** These are the “thinking” layers where the model learns complex patterns and features within the digits. We used the **ReLU activation function** for efficient processing.
* **An Output Layer (10 neurons):** One neuron for each digit (0–9). The **Softmax activation** function here gave us probabilities for each digit, allowing the model to “guess” the most likely number.

### The Learning Process: Training & Evaluation

After defining our model, we compiled it with an **Adam optimizer** (our learning engine) and a **sparse categorical crossentropy loss function** (how we measure “wrongness”).

We then trained our model for **20 epochs**, feeding it thousands of handwritten digits. It rapidly learned, achieving impressive accuracy. Our learning journey was visualized through:

* **Accuracy & Loss Graphs:** Showing how our model improved over time, and also highlighting the importance of validation data to catch nuances like overfitting (where the model learns the training data *too* well).
* **Confusion Matrix:** A visual heatmap showing exactly where our model made correct predictions and where it got “confused” between similar-looking digits (e.g., mistaking a ‘4’ for a ‘9’).

Ultimately, our model achieved a fantastic **~97.4% accuracy** on unseen handwritten digits!

### Putting It to the Test: Real-World Predictions

The most exciting part? Watching our model predict! We fed it brand new, unseen handwritten digits from our test set and saw it correctly identify them.

### Our Deep Learning Debut: What We Learned

This project was an incredible deep dive into fundamental deep learning concepts. We learned about:

* The structure and function of Artificial Neural Networks.
* The critical role of data preprocessing in machine learning.
* How to train and evaluate a deep learning model using Keras.
* The power of tools like Google Colab for collaborative projects.

This project was about more than just recognizing numbers; it was about demystifying the incredible potential of computer vision and deep learning for us as students. We hope our journey inspires you to explore this exciting field too!

Explore the code on [GitHub](https://github.com/nidhibaratam/Handwritten-Digit-Recognition-with-Neural-Networks/blob/main/Handwritten_Digit_Recognition_with_Neural_Networks.ipynb). We welcome your feedback and ideas!
