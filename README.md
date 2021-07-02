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
We used pretrained **GoogleNet** for extracting the embedding from a given
image. The last two fully connected layers were not considered.

### Decoder RNN
Glove embeddings of 300 dimension were used to convert the one hot vocabulary to
dense representation. Utilising the concept of attention to focus on relevant
information for generation of meaningful sentences was achieved. There were two
types of attention, input model for attention and output model for attention.
To generate the next input, scores are calculated for each word in the
vocabulary by using the previous output. The input vector is constructed by
averaging the previous output and weighted words in the vocabulary (weights are
the scores calculated above).

The output model calculates the score for each word in the vocabulary using the
hidden vector produced by RNN. The average vector is calculated as in the case
of input attention.
In addition to the cross-entropy loss, we also add regularisation to the
attention scores, so that they penalise excessive attention paid to any single
attribute over the entire sentence and sparsity of the attention score at a
given time.
