import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.preprocessing import LabelEncoder


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, ratio=0.7, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_) # classes_:  0번부터 순서대로 변화된 인코딩 값에 대한 원본값을 가지고 있음

    def __preprocessing(self, df, is_train=True):
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        # LabelEncoder() 객체 생성 후 fit()과 transform()을 적용해서 인코딩. ( 디코딩하려면 inverse_transfrom() )
        # assessmentItemID_classes.npy, testId_classes.npy, KnowledgeTag_classes.npy에 인코딩 값에 대한 원본값을 저장
        for col in cate_cols:

            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"] # 912명의 unique 유저 list에 'unknown'추가 -> len(a)=913
                le.fit(a)
                self.__save_labels(le, col) # 인코딩 값에 대한 원본값을 저장
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )

            # 모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        def convert_time(s):
            timestamp = time.mktime(
                datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
            )
            return int(timestamp)

        df["Timestamp"] = df["Timestamp"].apply(convert_time)

        return df

    def __feature_engineering(self, df):
        return df

    def load_data_from_file(self, file_name, is_train=True):
        """
        IN:[[0, 'A080038004', 'A080000038', 1, '2020-06-07 23:07:23', 4687],
            [...],
            [...],
            [...],
            [...]],
            ....

        OUT: user당 시퀀스
        [
        (
          array([5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 9, 9, 9, ....]),
          array([602, 602, 602, 602, 602, 602, 603, 603, 603, 603, 603, 603, 603, 60....]),
          array([3836, 3837, 3838, 3839, 3840, 3841, 3842, 3843, 3844, 476, 477, 478, ....]),
          array([1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, ....]),
        )
        (
          array([5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 9, 9, 9, ....]),
          array([602, 602, 602, 602, 602, 602, 603, 603, 603, 603, 603, 603, 603, 60....]),
          array([3836, 3837, 3838, 3839, 3840, 3841, 3842, 3843, 3844, 476, 477, 478, ....]),
          array([1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, ....]),
        )
        ....
        ]

        """
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용

        self.args.n_questions = len(
            np.load(os.path.join(self.args.asset_dir, "assessmentItemID_classes.npy"))
        )
        self.args.n_test = len(
            np.load(os.path.join(self.args.asset_dir, "testId_classes.npy"))
        )
        self.args.n_tag = len(
            np.load(os.path.join(self.args.asset_dir, "KnowledgeTag_classes.npy"))
        )

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        columns = ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag"]
        group = (
            df[columns]
            .groupby("userID")
            .apply(
                lambda r: (
                    r["testId"].values,
                    r["assessmentItemID"].values,
                    r["KnowledgeTag"].values,
                    r["answerCode"].values,
                )
            )
        )

        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train=False)

# torch의 dataset은 2가지 스타일이 존재함
# 1. Map-style dataset
#   : index가 존재하여 data[index]로 데이터를 참조할 수 있음
#   : __getitem__과 __len__ 선언 필요
# 2. Iterable-style dataset
#   : random으로 읽기에 어렵거나, data에 따라 batch size가 달라지는 데이터(dynamic batch size)에 적합
#   : 비교하자면 stream data, real-time log 등에 적합
#   : __iter__ 선언 필요
class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        """
        row: 한 유저의 시퀀스 (4개의 feature 각각)
            (
            array([5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 9, 9, 9, ....]),
            array([602, 602, 602, 602, 602, 602, 603, 603, 603, 603, 603, 603, 603, 60....]),
            array([3836, 3837, 3838, 3839, 3840, 3841, 3842, 3843, 3844, 476, 477, 478, ....]),
            array([1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, ....]),
            )
        """
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        # testId, assessmentItemID, KnowledgeTag, answerCode
        test, question, tag, correct = row[0], row[1], row[2], row[3]

        cate_cols = [test, question, tag, correct]

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len :]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)

        return cate_cols

    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence

# map-style 데이터셋에서 sample list를 batch 단위로 바꾸기 위해 필요한 기능
# zero-padding이나 variable size 데이터 등 데이터 사이즈를 맞추기 위해 많이 사용함
# 보통의 경우 batch(samples)를 input으로 받아 x_feature_list와 y_label_list를 return함
def collate(batch):
    # 기존 코드
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col):] = col
            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])

    """
    # 강의에 나온 코드
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]

    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            col_list[i].append(col)

    # 각 column의 값들을 대상으로 padding 진행
    # pad_sequence([[1,2,3], [3,4]]) -> [[1,2,3],
    #                                    [3,4,0]]
    # 뒤에 붙이거나 앞에 붙이거나 하면됨(성능의 차이가 있는지 확인 해볼 것)
    for i, col_batch in enumerate(col_list):
        col_list[i] = pad_sequence(col_batch, batch_first=True) # padding

    # mask의 경우 max_seq_len을 기준으로 길이가 설정되어 있다.
    # 만약 다른 column들의 seq_len이 max_seq_len보다 작다면
    # 이 길이에 맞추어 mask의 길이도 조절해준다.
    col_seq_len = col_list[0].size(1) # col_list[0]의 1차원 size 확인
    mask_seq_len = col_list[-1].size(1) # col_list[-1]의 1차원 size 확인
    if col_seq_len < mask_seq_len:
        col_list[-1] = col_list[-1][:, :col_seq_len]
    """
    return tuple(col_list)


def get_loaders(args, train, valid):
    # num_workers = 0인 경우, pin_memory = True로 설정하면 많은 프로세스가 분기되기 때문에
    # pin_memory = False가 훨씬 빠르다고 함
    # True로 설정하면, 데이터로더는 Tensor를 CUDA 고정 메모리에 올림
    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        # 앞서 만들었던 dataset을 input으로 넣어주면 여러 옵션(데이터 묶기, 섞기, 알아서 병렬처리)을 통해 batch를 만들어줌
        # 서버에서 돌릴 때는 num_worker를 조절해서 load 속도를 올릴 수 있지만,
        # PC에서는 default로 설정해야 오류가 안난다.
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )

    return train_loader, valid_loader

# pytorch는 torch.utils.data.Dataset과 torch.utils.data.DataLoader의 두 가지 도구를 제공한다.
#
# Dataset은 Data를 가지고 있는 객체로써 input feature x와 label y를 input으로 받아 저장하며, __len__, __getitem__을 구현해야 한다.
# DataLoader는 batch 기반으로 모델을 학습시키기 위해 dataset을 input으로 받아 batch size로 슬라이싱하는 역할을 한다.
# DataLoader의 기본 옵션은 다음과 같다.
# DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
#            batch_sampler=None, num_workers=0, collate_fn=None,
#            pin_memory=False, drop_last=False, timeout=0,
#            worker_init_fn=None)
#####
# 이때, batch_size가 2이상일 경우
# dataset이 variable length(input의 size가 data마다 다른 dataset)면 바로 못묶이고 에러가 나므로 collate_fn을 만들어서 넘겨줘야한다.
# ex)   tensor([[0.]])
#       tensor([[1., 1.]])
#       tensor([[2., 2., 2.]])
#       =>
#       tensor([[0., 0., 0.],
#               [1., 1., 0.],
#               [2., 2., 2.]]) tensor([0., 1., 2.])
#####
# 데이터 프로세싱에 무조건 많은 CPU 코어를 할당해주는 것이 좋은것만은 아니기 때문에
# 가장 적합한 num_workers 수치를 찾아 내는 것도 하이퍼 파라미터 튜닝으로 볼 수 있음
# num_workers 튜닝을 위해 고려해야 하는 것은 학습 환경의 GPU개수, CPU개수, I/O 속도, 메모리 등이 있음

