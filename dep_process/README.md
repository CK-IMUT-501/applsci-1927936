dataset_splite.py splits the entire dataset into three parts: training set, test set and validation set.

dep.py is used to extract the dependency syntax information from the dataset(training set) and to construct and save a dependency syntax matrix for each sentence.

dep_save.py is used to split and save all the dependency syntax matrixs in groups of batch_size.

As the Traditional Mongolian-Chinese dataset in this paper is private, an open source English-Chinese dataset news_en_zh_shuffle_final.txt is provided here.
