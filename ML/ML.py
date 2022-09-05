import os
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score

from utils import to_int, label_D, label_A, label_L, get_model


np.random.seed(42)
fold = 5
each = True
D_acc_list = []
A_acc_list = []
L_acc_list = []
D_rec_list = []
A_rec_list = []
L_rec_list = []
D_f1_list = []
A_f1_list = []
L_f1_list = []
final_D = None
final_A = None
final_L = None


def read(path):
    df = pd.read_csv(path)
    label = np.array(df[['Label_D', 'Label_A', 'Label_L']].values.tolist())
    df[['Label_D', 'Label_A', 'Label_L']] = df[['Label_D', 'Label_A', 'Label_L']].apply(
        {'Label_D': label_D, 'Label_A': label_A, 'Label_L': label_L})
    df = np.array(df.values.tolist())
    return df, label


def solve_one(df_train, df_test, x_index, y_index,model_name):
    X_train = df_train[:, x_index].astype(float)
    Y_train = df_train[:, y_index].astype(float)
    X_train = X_train.reshape(X_train.shape[0], len(x_index))
    Y_train = Y_train.reshape(Y_train.shape[0], 1)
    Y_train = Y_train.ravel()

    X_test = df_test[:, x_index].astype(float)
    Y_test = df_test[:, y_index].astype(float)
    X_test = X_test.reshape(X_test.shape[0], len(x_index))
    Y_test = Y_test.reshape(Y_test.shape[0], 1)
    Y_test = Y_test.ravel()

    model = get_model(model_name)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    return np.squeeze(pred)


def solve(in_path, out_dir,model_name):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df, origin_labels = read(in_path)
    KF = KFold(fold)
    K = 1
    for train_index, test_index in KF.split(df):
        # print(train_index)
        res = []

        label = origin_labels
        df_train = df[train_index]
        df_test = df
        x_index = [1, 2, 3]  # D A L
        for model_iscale in range(8):  # number of model
            for y_index in [-3, -2, -1]:  # D A L
                x = x_index
                x = list(map(lambda x: x + model_iscale * 3, x))
                pred = solve_one(df_train, df_test, x, y_index, model_name)
                # print(pred)
                pred = to_int(pred, y_index)
                res.append(pred)
        label = label.T
        test(res, df_test)

        res.append(label[0])
        res.append(label[1])
        res.append(label[2])
        res = np.array(res)
        res = res.T
        res = pd.DataFrame(res, index=df_test[:, 0],
                           columns=['resnet18_d', 'resnet18_a', 'resnet18_l', 'mobilenet_d', 'mobilenet_a',
                                    'mobilenet_l', 'squeezenet_d', 'squeezenet-a'
                               , 'squeezenet_l', 'densenet_d', 'densenet_a', 'densenet_l', 'vgg11_d', 'vgg11_a',
                                    'vgg11_l', 'vgg13_d', 'vgg13_a', 'vgg13_l',
                                    'vgg16_d', 'vgg16_a', 'vgg16_l', 'vgg19_d', 'vgg19_a', 'vgg19_l', 'D', 'A', 'L'])
        # psychology res
        # fold_out_dir = out_dir + '/' + str(K) + '.csv'
        # res.to_csv(fold_out_dir)L
        K += 1

    temp_D=np.stack((D_acc_list,D_rec_list,D_f1_list),axis=1)
    temp_A = np.stack((A_acc_list, A_rec_list, A_f1_list), axis=1)
    temp_L = np.stack((L_acc_list,L_rec_list, L_f1_list), axis=1)

    global final_D
    global final_A
    global final_L
    if final_D is None:
        final_D = temp_D
    else:
        final_D = np.concatenate((final_D,temp_D),axis=-1)
    if final_A is None:
        final_A = temp_A
    else:
        final_A = np.concatenate((final_A,temp_A),axis=-1)
    if final_L is None:
        final_L = temp_L
    else:
        final_L = np.concatenate((final_L,temp_L),axis=-1)
    print(final_A.shape)



def test(res, df):
    res = np.array(res)
    D_label = to_int(df[:, -3].astype(float), -3)
    A_label = to_int(df[:, -2].astype(float), -2)
    L_label = to_int(df[:, -1].astype(float), -1)
    D_acc = []
    D_rec = []
    D_f1 = []
    for i in range(0, res.shape[0], 3):
        D_acc.append(accuracy_score(D_label, res[i]))
        D_rec.append(recall_score(D_label, res[i], average='macro'))
        D_f1.append(f1_score(D_label, res[i], average='macro'))
    A_acc = []
    A_rec = []
    A_f1 = []
    for i in range(1, res.shape[0], 3):
        A_acc.append(accuracy_score(A_label, res[i]))
        A_rec.append(recall_score(A_label, res[i], average='macro'))
        A_f1.append(f1_score(A_label, res[i], average='macro'))
    L_acc = []
    L_rec = []
    L_f1 = []
    for i in range(2, res.shape[0], 3):
        L_acc.append(accuracy_score(L_label, res[i]))
        L_rec.append(recall_score(L_label, res[i], average='macro'))
        L_f1.append(f1_score(L_label, res[i], average='macro'))
    D_acc_list.append(D_acc)
    A_acc_list.append(A_acc)
    L_acc_list.append(L_acc)
    D_rec_list.append(D_rec)
    A_rec_list.append(A_rec)
    L_rec_list.append(L_rec)
    D_f1_list.append(D_f1)
    A_f1_list.append(A_f1)
    L_f1_list.append(L_f1)
    return


if __name__ == '__main__':
    in_dir = '5times_emoldanet'
    in_path = 'result_clear.csv'
    out_dir = './5fold'
    model_list = ['svm','tree','forest']
    final_D_list = []
    final_A_list = []
    final_L_list = []
    cnt = 1
    for file in os.listdir(in_dir):
        final_D = None
        final_A = None
        final_L = None
        in_path = os.path.join(in_dir,file)
        print(in_path)
        for model_name in model_list:
            np.random.seed(42)
            print(model_name)
            D_acc_list = []
            A_acc_list = []
            L_acc_list = []
            D_rec_list = []
            A_rec_list = []
            L_rec_list = []
            D_f1_list = []
            A_f1_list = []
            L_f1_list = []
            # out_dir_temp = os.path.join(out_dir, str(cnt))  # 5 fold respective res
            out_dir_temp = os.path.join(out_dir, str(cnt),model_name) # psychology res
            solve(in_path, out_dir_temp, model_name)

        final_D = final_D.reshape((fold,len(model_list)*3, 8))
        if each:
            final_D_each = final_D.transpose(0,2,1)
            seed_dir = os.path.join('./D', str(cnt))
            if not os.path.exists(seed_dir):
                os.makedirs(seed_dir)
            for i in range(5):
                df_D = pd.DataFrame(final_D_each[i],
                                    index=['ResNet18', 'MobileNet', 'SqueezeNet', 'DenseNet', 'VGG11', 'VGG13', 'VGG16',
                                           'VGG19'],
                                    columns=pd.MultiIndex.from_product([['Accuracy', 'Recall', 'F1-score'],
                                                                        ['SVM', 'DT', 'RF']]))
                df_D.to_csv(f'{seed_dir}/{i+1}.csv')
        final_D = np.mean(final_D,0)
        final_D = final_D.T
        final_D = np.around(np.multiply(100, final_D),1)
        final_D_list.append(final_D)

        final_A = final_A.reshape((fold,len(model_list) * 3, 8))
        if each:
            final_A_each = final_A.transpose(0,2,1)
            seed_dir = os.path.join('./A', str(cnt))
            if not os.path.exists(seed_dir):
                os.makedirs(seed_dir)
            for i in range(5):
                df_A = pd.DataFrame(final_A_each[i],
                                    index=['ResNet18', 'MobileNet', 'SqueezeNet', 'DenseNet', 'VGG11', 'VGG13', 'VGG16',
                                           'VGG19'],
                                    columns=pd.MultiIndex.from_product([['Accuracy', 'Recall', 'F1-score'],
                                                                        ['SVM', 'DT', 'RF']]))
                df_A.to_csv(f'{seed_dir}/{i+1}.csv')
        final_A = np.mean(final_A, 0)
        final_A = final_A.T
        final_A = np.around(np.multiply(100, final_A), 1)
        final_A_list.append(final_A)
        final_L = final_L.reshape((fold,len(model_list) * 3, 8))
        if each:
            final_L_each = final_L.transpose(0,2,1)
            seed_dir = os.path.join('./L', str(cnt))
            if not os.path.exists(seed_dir):
                os.makedirs(seed_dir)
            for i in range(5):
                df_L = pd.DataFrame(final_L_each[i],
                                    index=['ResNet18', 'MobileNet', 'SqueezeNet', 'DenseNet', 'VGG11', 'VGG13', 'VGG16',
                                           'VGG19'],
                                    columns=pd.MultiIndex.from_product([['Accuracy', 'Recall', 'F1-score'],
                                                                        ['SVM', 'DT', 'RF']]))
                df_L.to_csv(f'{seed_dir}/{i+1}.csv')
        final_L = np.mean(final_L, 0)
        final_L = final_L.T
        final_L = np.around(np.multiply(100, final_L), 1)
        final_L_list.append(final_L)

        df_D = pd.DataFrame(final_D,
                            index=['ResNet18', 'MobileNet', 'SqueezeNet', 'DenseNet', 'VGG11', 'VGG13', 'VGG16',
                                   'VGG19'], columns=pd.MultiIndex.from_product([['Accuracy', 'Recall', 'F1-score'],
                                                                                 ['SVM', 'DT', 'RF']]))

        df_A = pd.DataFrame(final_A,
                            index=['ResNet18', 'MobileNet', 'SqueezeNet', 'DenseNet', 'VGG11', 'VGG13', 'VGG16',
                                   'VGG19'],
                            columns=pd.MultiIndex.from_product([['Accuracy', 'Recall', 'F1-score'],
                                                                ['SVM', 'DT', 'RF']]))
        df_L = pd.DataFrame(final_L,
                            index=['ResNet18', 'MobileNet', 'SqueezeNet', 'DenseNet', 'VGG11', 'VGG13', 'VGG16',
                                   'VGG19'],
                            columns=pd.MultiIndex.from_product([['Accuracy', 'Recall', 'F1-score'],
                                                                ['SVM', 'DT', 'RF']]))

        cnt+=1

    f = lambda x: x+'$\\pm$'
    f = np.vectorize(f)
    final_D_list = np.array(final_D_list)
    print(final_D_list.shape)
    final_D_max = np.max(final_D_list,axis=0)
    final_D_min = np.min(final_D_list,axis=0)
    final_D_mean = np.around(np.mean(final_D_list,axis=0),1)
    final_D_delta = np.around(np.maximum(np.abs(final_D_max-final_D_mean),np.abs(final_D_min-final_D_mean)),1).astype(str)
    final_D_mean = final_D_mean.astype(str)
    final_D = np.char.add(f(final_D_mean),final_D_delta)
    # final_D = final_D_mean

    final_A_list = np.array(final_A_list)
    final_A_max = np.max(final_A_list, axis=0)
    final_A_min = np.min(final_A_list, axis=0)
    final_A_mean = np.around(np.mean(final_A_list, axis=0),1)
    final_A_delta = np.around(np.maximum(np.abs(final_A_max - final_A_mean), np.abs(final_A_min - final_A_mean)),1).astype(str)
    final_A_mean = final_A_mean.astype(str)
    final_A = np.char.add(f(final_A_mean.astype(str)), final_A_delta)
    # final_A = final_A_mean

    final_L_list = np.array(final_L_list)
    final_L_max = np.max(final_L_list, axis=0)
    final_L_min = np.min(final_L_list, axis=0)
    final_L_mean = np.around(np.mean(final_L_list, axis=0),1)
    final_L_delta = np.around(np.maximum(np.abs(final_L_max - final_L_mean), np.abs(final_L_min - final_L_mean)),1).astype(str)
    final_L_mean = final_L_mean.astype(str)
    final_L = np.char.add(f(final_L_mean.astype(str)), final_L_delta)
    # final_L = final_L_mean

    df_D = pd.DataFrame(final_D,index = ['ResNet18', 'MobileNet', 'SqueezeNet', 'DenseNet', 'VGG11', 'VGG13', 'VGG16','VGG19'],columns=pd.MultiIndex.from_product([['Accuracy', 'Recall','F1-score'],
                                                    ['SVM', 'DT','RF']]))

    df_A = pd.DataFrame(final_A,
                        index=['ResNet18', 'MobileNet', 'SqueezeNet', 'DenseNet', 'VGG11', 'VGG13', 'VGG16', 'VGG19'],
                        columns=pd.MultiIndex.from_product([['Accuracy', 'Recall', 'F1-score'],
                                                            ['SVM', 'DT', 'RF']]))
    df_L = pd.DataFrame(final_L,
                        index=['ResNet18', 'MobileNet', 'SqueezeNet', 'DenseNet', 'VGG11', 'VGG13', 'VGG16', 'VGG19'],
                        columns=pd.MultiIndex.from_product([['Accuracy', 'Recall', 'F1-score'],
                                                            ['SVM', 'DT', 'RF']]))
    # 5 seed 5 fold mean
    df_D.to_csv('D_mean.csv')
    df_A.to_csv('A_mean.csv')
    df_L.to_csv('L_mean.csv')

