Neural Style Transfer using pretrained VGG19

This code uses a pretrained VGG19 network which has been trianed on the ImageNet data (very large dataset) to learn low and high level features.

The inputs are the content image and style image which are appropriately sized. The code outputs the content image styled using features learnt from the style image.

This code is based on the work done by Gatys et al. (A Neural Algorithm of Artistic Style). The basic principal of this approach is that the representations of content and style in the Convolutional Neural Network are separable. This is done by minimizing a loss function that includes weighted terms for the style and content image. Increased weight on the style will result in images that match the appearance of the artwork, effectively giving a texturised version of it, but hardly show any of the photograph’s content. When placing strong emphasis on content, one can clearly identify the photograph, but the style of the painting is not as well-matched. For a specific pair of source images one can adjust the trade-off between content and style to create visually appealing images.
