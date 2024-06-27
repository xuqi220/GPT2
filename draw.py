import matplotlib.pyplot as plt

def read_data(data_path="log.txt"):
    loss_train, loss_val = [], []
    with open(data_path, "r",encoding="utf-8") as fi:
        for line in fi.readlines():
            items = line.split(" ")
            if items[1] == "train":
                loss_train.append(float(items[-1]))
            elif items[1] == "val":
                loss_val.append(float(items[-1]))
    return loss_train, loss_val


def draw():
    loss_train, loss_val = read_data()
    fig, axes = plt.subplots(1,2)
    