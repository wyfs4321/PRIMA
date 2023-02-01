We make use of two real datasets and two synthetic datasets in our experiments:
|Dataset|Description|
|---|---
|**IPUMS**|A US census dataset from the IPUMS repository, which contains around 3 million records.
|**Adult**|A dataset from the UCI machine learning repos- itory. After removing missing values, the dataset contains around 50 thousands records.
|**Normal**|A dataset which is synthesized from the multivari- ate normal distribution with mean 0, standard deviation 1, contains 50 ordinal attributes. The covariance between every two attributes is 0.5.
|**Random**|A dataset which is synthesized from the uniform distribution, contains 50 ordinal attributes.

To experiment with different sizes of data table, we generate multiple test datasets from the four datasets with the number of records ranging from 10k to 1M. In addition, for evaluating different numbers of attributes and domain sizes, we generate multiple versions of these four datasets with the domain sizes of attributes sizes ranging from 10 to 50.

These datasets are named in the following format：
XXXX_AA-MM: 
|Field|Description|
|---|---
|**XXXX**|The source of the dataset.
|**AA**|The number of dimensions in the dataset.
|**MM**|The domain of the attributes in the dataset.
