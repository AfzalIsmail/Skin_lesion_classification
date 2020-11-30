# Skin_lesion_classification

Skin cancer is a disease that affects a lot of people worldwide every year. It is the most
commonly diagnosed cancer in the UK. Malignant skin cancer, like Melanoma, can be very
dangerous and even fatal if not cured rapidly, as the cancer can spread to vital organs. The
number of melanoma cases increase every year. It currently accounts for 4% of the annual
total number of new cancer diagnosis. This paper proposes to use the advances made in
deep learning in recent years to come up with a classification algorithm. This can be very
useful to areas where resources are limited. It can be of great help to doctors for their
diagnosis and can give them a head start in providing treatment for the patients. This project
aims to investigate the performances of pre-trained convolutional neural networks and the
technique of transfer learning to classify skin lesion images. These neural networks have
proven to be very performant in image feature extraction. An image dataset, with 7 different
skin lesion classes and a total of 10015 images, was divided into 2 main classes. The 7
types of skin lesions were categorized as being malignant or benign. Several versions of this
dataset were generated, and this included balanced, imbalanced or augmented datasets.
These datasets, each containing malignant and benign images, were trained on several pretrained
neural networks, namely ResNet50, VGG16, VGG19, Inception V3, MobileNet and
DenseNet. The results were recorded and based on them an algorithm was proposed. The
algorithm proposed to combine the output the 3 best performing pre-trained network to
improve the accuracy of the classification. The models combined were ResNet50, DenseNet
and VGG16. The accuracy of this model was further increased with the addition trainable
convolution layers with each model. The overall accuracy achieved for this model was
84.01%.
