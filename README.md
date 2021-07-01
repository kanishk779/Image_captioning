# Image_captioning
SMAI(Statistical Methods in Artificial Intelligence) Final Project.
We had to work in teams of 3 and implement the Image Captioning with Semantic
Attention.
<href>https://arxiv.org/pdf/1603.03925.pdf </href>

This paper was based on combining two approaches prevalent before this paper.
Top down approach dealt with starting with gist of the image and generating the
caption. While the bottom up approach first comes up with words describing
various aspects of an image and then combines them. Language models are formed
in both paradigms to form coherent sentence.

### Data Loader
First the Vocabulary Class was created, which has methods for converts words to
indices. It also adds special tokens like SOS, EOS, PAD, UNK to the vocabulary.
For each image, we load the related caption in the dataset and convert it into
an array of indices in the vocabulary. This data is returned from the
__getitem__(index) function of FlickrDataset class. A custom collate function is
also created to pad the captions of all images in a batch to have same size.
Only then it is possible to feed the data to the model. Finally the dataloader
is created using the above dataset and collate function.

### Relevant words
To extract the visual concepts from an image we have used KNN (K Nearest
Neighbour) approach. We first pass the image through the **GoogleNet** model to
extract an embedding. In this embedding space we find the 10-Nearest neighbour
and take their captions. From all these captions we select few high frequency
words. There are other complex ways to detect these visual concepts as well, for
example building a multi-label classifier as in <href>https://arxiv.org/pdf/1312.4894.pdf </href>.

### Encoder CNN


### Decoder RNN
