# CarGenerator

Car Generator:
The car generator is a neural network model trained using DC-GAN principles.
It takes random noise as input and generates car images that resemble real car images.
The generator network consists of multiple convolutional layers followed by upsampling layers to increase the spatial dimensions of the generated image.

Super Resolution Model:
The super-resolution model is designed to enhance the resolution and details of low-resolution car images.
It utilizes deep convolutional layers to extract and learn high-frequency information from the input image.
The model takes a low-resolution car image as input and produces a high-resolution version of the image.

Training Process:
The car generator and super-resolution model are trained using appropriate datasets.
For the car generator, a dataset of real car images is used as the target distribution.
For the super-resolution model, a dataset of low-resolution and high-resolution car image pairs is required for training.
The training process involves iteratively updating the generator and discriminator networks using adversarial loss and other appropriate loss functions.

Try out a live demo on Hugging face [here](https://huggingface.co/spaces/AyanCh0w/CarImageGenerator)
