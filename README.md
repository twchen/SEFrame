# SEFrame
This repository contains the code for the paper "An Efficient and Effective Framework for Session-based Social Recommendation".

## Requirements
- Python 3.8
- CUDA 10.2
- PyTorch 1.7.1
- DGL 0.5.3
- NumPy 1.19.2
- Pandas 1.1.3

## Usage
1. Install all the requirements.

2. Download the datasets:
   - [Gowalla](https://snap.stanford.edu/data/loc-gowalla.html)
   - [Delicious](https://grouplens.org/datasets/hetrec-2011/)
   - [Foursquare](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)

3. Create a folder called `datasets` and extract the raw data files to the folder.  
   The folder should include the following files for each dataset:
   - Gowalla: `loc-gowalla_totalCheckins.txt` and `loc-gowalla_edges.txt`
   - Delicious: `user_taggedbookmarks-timestamps.dat` and `user_contacts-timestamps.dat`
   - Foursquare: `dataset_WWW_Checkins_anonymized.txt` and `dataset_WWW_friendship_new.txt`

4. Preprocess the datasets using the Python script [preprocess.py](preprocess.py).  
   For example, to preprocess the *Gowalla* dataset, run the following command:
   ```bash
   python preprocess.py --dataset gowalla
   ```
   The above command will create a folder `datasets/gowalla` to store the preprocessed data files.  
   Replace `gowalla` with `delicious` or `foursquare` to preprocess other datasets.

   To see the detailed usage of `preprocess.py`, run the following command:
   ```bash
   python preprocess.py -h
   ```

5. Train and evaluate a model using the Python script [run.py](run.py).  
   For example, to train and evaluate the model NARM on the *Gowalla* dataset, run the following command:
   ```bash
   python run.py --model NARM --dataset-dir datasets/gowalla
   ```
   Other available models are NextItNet, STAMP, SRGNN, SSRM, SNARM, SNextItNet, SSTAMP, SSRGNN, SSSRM, DGRec, and SERec.  
   You can also see all the available models in the [srs/models](srs/models) folder.

   To see the detailed usage of `run.py`, run the following command:
   ```bash
   python run.py -h
   ```

## Dataset Format
You can train the models using your datasets. Each dataset should contain the following files:

- `stats.txt`: A TSV file containing three fields, `num_users`, `num_items`, and `max_len` (the maximum length of sessions). The first row is the header and the second row contains the values.

- `train.txt`: A TSV file containing all training sessions, where each session has three fileds, namely, `sessionId`, `userId`, and `items`. Both `sessionId` and `userId` should be integers. A session with a larger `sessionId` means that it was generated later (this requirement can be ignored if the used models do not care about the order of sessions, i.e., when the models are not DGRec). The `userId` should be in the range of `[0, num_users)`. The `items` field of each session contains the clicked items in the session which is a sequence of item IDs separated by commas. The item IDs should be in the range of `[0, num_items)`.

- `valid.txt` and `test.txt`: TSV files containing all validation and test sessions, respectively. Both files have the same format as `train.txt`. Note that the session IDs in `valid.txt` and `test.txt` should be larger than those in `train.txt`.

- `edges.txt`: A TSV file containing the relations in the social network. It has two columns, `follower` and `followee`. Both columns contain the user IDs.

You can see [datasets/delicious](datasets/delicious) for an example of the dataset.

## Citation
If you use this code for your research, please cite our [paper](http://home.cse.ust.hk/~raywong/paper/wsdm21-SEFrame.pdf):
```
@inproceedings{chen2021seframe,
   title="An Efficient and Effective Framework for Session-based Social Recommendation",
   author="Tianwen {Chen} and Raymond Chi-Wing {Wong}",
   booktitle="Proceedings of the Fourteenth ACM International Conference on Web Search and Data Mining (WSDM '21)",
   pages="400--408",
   year="2021"
}
```
