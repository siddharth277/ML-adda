# GAN-based Art Generator 

---

## Goal
The aim of this project is to develop a GAN-based model that generates unique art-style images, demonstrating the potential of GANs in artistic image generation.

##  Dataset
This project uses a synthetic dataset of images generated from random noise to simulate artistic patterns, forming the training base for the GAN.

##  Description
The project implements a GAN with a generator to create art-like images and a discriminator to evaluate them. Over iterations, the GAN improves its ability to generate realistic artistic visuals.

## What I Had Done!
- Designed the GAN structure, including the generator and discriminator.
- Trained the GAN model on the synthetic dataset.
- Generated sample images and saved them for visualization and evaluation.

## Models Implemented
- VGG16
- ResNet50
- Inception
- MobileNet


### Performance of the Models based on Accuracy Scores

```text
=================================================
Model           Accuracy        Loss    Rank
=================================================
VGG16           0.7124          0.2091  #1
ResNet50        0.5860          0.4233  #2
MobileNet       0.5824          0.4980  #3
Inception       0.5650          0.6521  #4
=================================================

Best model     : VGG16 (0.7124)
Worst model    : Inception (0.5650)
Mean accuracy  : 0.6115
Std deviation  : 0.0679
```

<img src="model_comparison.png" alt="Model Comparison" width="600">


## 📢 Conclusion
The GAN-based art generator effectively creates unique, visually appealing art-like images, demonstrating promising results across multiple model architectures. Fine-tuning the model could lead to even more sophisticated outputs.

--- 
**Upgraded work**

We can Use real CIFAR-10 or any colour based images dataset instead of random noise to get better accuracy.



**AUTHOR**
```
SIDDHARTH SHUKLA

github.com/siddharth277
```



