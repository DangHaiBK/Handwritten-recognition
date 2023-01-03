# Handwritten-recognition
This project used the MNIST dataset to build the model. Ten numbers of this dataset were classified using an MLP model. Then, the model was quantized to reduce the size to embed into an MCU.
+ MLP model: 2 layers: input layer with 784 nodes and output layer with 10 nodes.
+ Model was converted into integer quantization model. 
+ Quantized model and original model were compared to evaluate the performance.
+ The gui_for_handwritten.py file contains the source code to read the model and perform the classification on the computer. It also contains the code for GUI.
+ The mnist_model.h5 contains the source code of the quantized model.
+ The stm32_ai contains the MCU project that can be run on STM32L476.
* Note that: One or many hidden layers and the dropping-out rate were not applied. Therefore, these can affect directly to the classification performance.
