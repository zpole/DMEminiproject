# DMEminiproject
## [THE 4 UNIVERSITIES DATASET](http://www-2.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/)

[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

- **Description:** This data set contains WWW-pages collected from computer science departments of various universities in January 1997 by the World Wide Knowledge Base (WebKb) project of the CMU text learning group. The 8,282 pages were manually classified into 7 classes: 1) student, 2) faculty, 3) staff, 4) department, 5) course, 6) project and 7) other. For each class the data set contains pages from the four universities: Cornell, Texas, Washington, Wisconsin and 4,120 miscellaneous pages from other universities. The files are organized into a directory structure, one directory for each class. Each of these seven directories contains 5 subdirectories, one for each of the 4 universities and one for the miscellaneous pages. These directories in turn contain the Web-pages.
- Size:
  - 8,282 webspages, 7 classes
  - 60.8 MB
- References:
  - *Text Classification from Labeled and Unlabeled Documents using EM (2000)* by Kamal Nigam, Andrew McCallum, Sebastian Thrun and Tom Mitchell.
- **Task**: Prepare the data for mining and perform an exploratory data analysis (these steps will probably not be independent). The data mining task is to classify the texts according to the 7 classes. You should compare at least 2 different classifiers. Since each university's web pages have their own idiosyncrasies, it is not recommended to do training and testing on pages from the same university. We recommend training on three of the universities plus the misc collection, and testing on the pages from a fourth, held-out university (four-fold cross validation). An additional topic might be to look at labelled/unlabelled data, as in the reference.
- **Challenges**: An important challenge from web mining point of view will be the preprocessing of the dataset. Since the data are html files you have to remove all the irrelevant text information, such as html commands etc. and convert the rest of the text into a bag-of-words format. See help on the [4 Universities Data Set](http://www-2.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/) web page about doing this with rainbow.