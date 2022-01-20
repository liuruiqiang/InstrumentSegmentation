from prepare_data import data_path
import random

def get_split(fold,gap=1,clip=3):
    folds = {0: [1,2],
             1: [3,4],
             2: [5,8],
             #3:[6,7]
             3: [9,10,11,12,13,14,15,16,17,18]}


    train_path = data_path /'cropped_train'
    print(train_path)
    train_file_names = []
    val_file_names = []
    #for instrument_id in range(1,9):#[1,9]:#
    for instrument_id in [1,2,3,4,5,8,9,10,11,12,13,14,15,16,17,18]:
        if instrument_id in folds[fold]:
            val_file_names += list((train_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*'))
        else:
            train_file_names += list((train_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*'))

    clip_len=clip # num of image in each clip of video (also batch size)
    train_file_names=sorted(train_file_names)
    # print(train_file_names)
    val_file_names = sorted(val_file_names)

    train=[]
    val=[]
    for i in range(0,len(train_file_names)-clip_len,gap):
        # print(train_file_names[i:i+clip_len])
        train.append(train_file_names[i:i+clip_len:5])
        tempt = train_file_names[i:i + clip_len:5]
        tempt.reverse()
        train.append(tempt)
        # train_file_names[i:i + clip_len::]

    # train_file_names.reverse()
    # # val_file_names.reverse()
    # for i in range(0,len(train_file_names)-clip_len,gap):
    #     train.append(train_file_names[i:i+clip_len])

    for i in range(0,len(val_file_names)//5):
        tem = []
        for i in range(0,5):
            tem.append(val_file_names.pop(0))
        val.append(tem)


    random.shuffle(train)
    random.shuffle(val)
    train_file_names=train
    val_file_names=val
    return train_file_names, val_file_names

train_names,val_names = get_split(3,gap=50,clip=25)
# print(sorted(train_names),)
# for i in range(0,25,5):
#     print(i)
# print(len(val_names[0]),len(val_names))
# print(len(train_names))
# files=[]
# for i in train_names:
#     files=files+i
# print(sorted(files))
# print(len(files))
# print(len(sorted(train_names)),len(train_names[0]))
# for (train,val) in zip(sorted(train_names),val_names):
#     print(train)
# print(len(train_names),len(val_names))
# for i in range(0,10,1):
#     print(i)
# data = [1,2,3,4,5,6,7,8,9,10]
# data2 = data[1:8:3]
# # data2.reverse()
# print(data2)

