import prepare_data as data
import numpy as np
import pandas
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


def plot_same_vectors(vectors, indexes, labels, figname, label_names):
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(vectors)

    for ti in range(labels.shape[1]):
        print(ti)
        df = pandas.DataFrame({
            'X': Y[:, 0],
            'Y': Y[:, 1],
            'index': indexes,
            'label': labels[:, ti]
        })

        categories = np.sort(np.unique(df['label']))
        colors = np.linspace(0, 1, len(categories))
        colordict = dict(zip(categories, colors))
        c_map = plt.get_cmap("viridis")

        done_labels = set()
        df['color'] = df['label'].apply(lambda x: colordict[x])
        for idx in df['index'].unique():
            c = df[df['index'] == idx]['color'].tolist()[0]
            l = df[df['index'] == idx]['label'].tolist()[0]
            color = c_map(c)
            plt.plot(df[df['index'] == idx]['X'], df[df['index'] == idx]['Y'], c=color, label=l if l not in done_labels else "")
            done_labels.add(l)
        plt.legend(loc='upper right')

        plt.title(figname + "-" + label_names[ti])
        plt.savefig(figname + "-" + label_names[ti] + ".png")
        plt.close()


def plot_by_label(vectors, indexes, labels, figname):
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(vectors)

    df = pandas.DataFrame({
        'X': Y[:, 0],
        'Y': Y[:, 1],
        'index': indexes,
        'label': labels
    })

    categories = np.sort(np.unique(df['label']))
    colors = np.linspace(0, 1, len(categories))
    colordict = dict(zip(categories, colors))
    c_map = plt.get_cmap("viridis")

    done_labels = set()
    df['color'] = df['label'].apply(lambda x: colordict[x])
    for idx in df['label'].unique():
        c = df[df['label'] == idx]['color'].tolist()[0]
        l = df[df['label'] == idx]['label'].tolist()[0]
        color = c_map(c)
        plt.plot(df[df['label'] == idx]['X'], df[df['label'] == idx]['Y'], c=color, label=l if l not in done_labels else "")
        done_labels.add(l)
    plt.legend(loc='upper right')

    plt.title(figname)
    plt.savefig(figname + ".png")
    plt.close()


def reduce_dim(x, y, ids):
    vectors = []
    indexes = []
    labels = []
    patient_icu = []
    tsne_x = []
    tsne_y = []
    # random_idx = list(range(x.shape[0]))
    # np.random.shuffle(random_idx)
    # n = 500000
    # for pid, picu, xp, yp in zip(ids[random_idx][:n], icus[random_idx][:n], x[random_idx][:n], y[random_idx][:n]):
    tsne = TSNE(n_components=2, random_state=42)
    for pid, xp, yp in zip(ids, x, y):
        try:
            Y = tsne.fit_transform(xp)
            tsne_x += Y[:, 0].tolist()
            tsne_y += Y[:, 0].tolist()

            indexes += ([pid] * len(xp))
            labels += ([yp] * len(xp))
        except Exception as e:
            print(e)

    # np.set_printoptions(suppress=True)
    df = pandas.DataFrame({
        'X': np.asarray(tsne_x),
        'Y': np.asarray(tsne_y),
        'index': indexes,
        'label': labels
    })    

    return df


def heatmap(vectors, labels, figname, label_names):
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(vectors)

    n_labels = 1 if len(labels.shape) == 1 else labels.shape[1]
    for ti in range(n_labels):
        labels_i = labels if n_labels == 1 else labels[:, ti]
        df = pandas.DataFrame({
            'X': Y[:, 0],
            'Y': Y[:, 1],
            'label': labels_i
        })

        cmap = plt.get_cmap("coolwarm")
        cmap.set_under('w')
        fig, ax = plt.subplots()
        hmap, x, y = create_heatmap(df)
        for xi in x:
            for yi in y:
                if hmap[xi, yi] == 0:
                    hmap[xi, yi] = -10
        ax.pcolor(x, y, np.asarray(hmap).T, cmap=cmap, vmin=-1, vmax=1, linewidth=0)
        # ax.pcolor(x, y, hmap, cmap='seismic', vmin=-1, vmax=1)
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        # ax.colorbar()

        # plt.show()

        ax.set_title(figname)
        plt.savefig(figname + "-" + label_names[ti] + ".png")
        plt.close()


def create_heatmap(df):
    x_min = df.X.min()
    x_max = df.X.max()
    y_min = df.Y.min()
    y_max = df.Y.max()
    n_steps = 40
    x_step = (x_max - x_min) / n_steps
    y_step = (y_max - y_min) / n_steps
    hmap = np.zeros((n_steps, n_steps))
    for i, xi in enumerate(np.arange(x_min, x_max, x_step)):
        for j, yi in enumerate(np.arange(y_min, y_max, y_step)):
            x_range = (df.X > xi) & (df.X < (xi + x_step))
            y_range = (df.Y > yi) & (df.Y < (yi + y_step))
            hmap[i, j] = np.nan_to_num(df.label[(x_range & y_range)].mean())
    x = y = np.asarray(list(range(0, n_steps)))
    return hmap, x, y


def heatmap_withpatient(vectors, labels, figname, label_name):
    df = pandas.DataFrame({
        'X': vectors[:, 0],
        'Y': vectors[:, 1],
        'label': labels
    })

    fig, ax = plt.subplots()
    hmap, x, y = create_heatmap(df)

    # hmap_df = pandas.DataFrame(hmap)
    # hmap_df.to_csv(figname + ".csv")

    cmap = plt.get_cmap("coolwarm")
    cmap.set_under('w')
    for xi in x:
        for yi in y:
            if hmap[xi, yi] == 0:
                hmap[xi, yi] = -10
    ax.pcolor(x, y, np.asarray(hmap).T, cmap=cmap, vmin=-1, vmax=1, linewidth=0)

    ax.set_title(figname)
    fig.savefig(figname + "-" + label_name + ".png")
    fig.close()


def heatmap_withpatient_gif(di, vectors, labels, patient_vectors):
    df = pandas.DataFrame({
        'X': vectors[:, 0],
        'Y': vectors[:, 1],
        'label': labels
    })

    max_d = len(patient_vectors[0])
    max_s = len(patient_vectors[1])
    max_i = np.max([max_d, max_s])
    cmap = plt.get_cmap("coolwarm")
    cmap.set_under('w')
    for i in range(max_i):
        try:
            i_d = np.min([i, max_d])
            i_s = np.min([i, max_s])

            fig, ax = plt.subplots()
            hmap, x, y = create_heatmap(df)
            # hcolors = []
            for xi in x:
                for yi in y:
                    if hmap[xi, yi] == 0:
                        hmap[xi, yi] = -10
            #         elif hmap[xi, yi] < 0:
            #             hmap[xi, yi] = 0
            #         else:
            #             hmap[xi, yi] = 3
            ax.pcolor(x, y, np.asarray(hmap).T, cmap=cmap, vmin=-1, vmax=1, linewidth=0)

            x_min = df.X.min()
            x_max = df.X.max()
            y_min = df.Y.min()
            y_max = df.Y.max()
            n_steps = 40
            x_step = (x_max - x_min) / n_steps
            y_step = (y_max - y_min) / n_steps

            vxq = ((patient_vectors[0][i_d, 0] - x_min) / x_step)
            vyq = ((patient_vectors[0][i_d, 1] - y_min) / y_step)
            ax.scatter(vxq, vyq, c='r', s=60, linewidths=15)
            # ax.annotate("Non-Survival", (vxq+0.5, vyq+1), color='y', bbox=dict(facecolor='black', edgecolor='y'))

            vxq = ((patient_vectors[1][i_s, 0] - x_min) / x_step)
            vyq = ((patient_vectors[1][i_s, 1] - y_min) / y_step)
            ax.scatter(vxq, vyq, c='#00f2ff', s=60, linewidths=15)
            # ax.annotate("Survival", (vxq+0.5, vyq+1), color='y', bbox=dict(facecolor='black', edgecolor='y'))

            ax.axis([x.min(), x.max(), y.min(), y.max()])

            ax.set_title("ICU 1 Death Heatmap - Patient Trajectory")
            fig.savefig("trajectory/" + str(di) + "-trajectory-" + str(i) + ".png")
            # ax.close()
        except Exception as e:
            print(e)


def plot_heatmap(filename, x, y):
    vectors = []
    indexes = []
    labels = []
    random_idx = list(range(x.shape[0]))
    np.random.shuffle(random_idx)
    n = 400
    last_v = []
    for i, p in enumerate(zip(x[random_idx][:n], y[random_idx][:n])):
        for v in p[0]:
            if not np.array_equal(last_v, v):
                vectors.append(v)
                indexes.append(i)
                labels.append(p[1])
                last_v = v
    heatmap(vectors, np.asarray(labels), filename, data.targets)


def plot_mean(filename, x, y):
    from keras.preprocessing import sequence

    x = sequence.pad_sequences(x)
    for yt in range(y.shape[1]):
        vectors = []
        indexes = []
        labels = []

        for c in np.unique(y[:, yt]):
            x_c = x[y[:, yt] == c]
            mxc = np.mean(x_c, axis=0)
            for i in range(mxc.shape[0]):
                vectors.append(mxc[i])
                indexes.append(yt)
                labels.append(c)
        plot_by_label(vectors, indexes, labels, filename + "-" + data.targets[yt])


def plot_all(filename, x, y):
    vectors = []
    indexes = []
    labels = []
    random_idx = list(range(x.shape[0]))
    np.random.shuffle(random_idx)
    n = 50
    last_v = []
    for i, p in enumerate(zip(x[random_idx][:n], y[random_idx][:n])):
        for v in p[0]:
            if not np.array_equal(last_v, v):
                vectors.append(v)
                indexes.append(i)
                labels.append(p[1])
                last_v = v
    plot_same_vectors(vectors, indexes, np.asarray(labels), filename, data.targets)


def main():
    # for k in ['a', 'b', 'c']:
    # for icu in [1, 2, 3, 4]:
    # for icu in [1, 2, 3, 4]:
    for icu in [1]:
        filename = "uti-"+str(icu)
        print("Target: "+filename)
        x, y, icu_ids = data.load_icus("60", [icu], data.targets[0])

        # x = np.asarray(x)
        last_x = []
        for xi in x:
            last_x.append(xi[len(xi)-1, :])
        last_x = np.asarray(last_x)
        y = np.asarray(y)
        y[y == 0] = -1

        # plot_heatmap(filename, last_x, y)
        heatmap(last_x, y, "Coronary ICU - Raw Death Heatmap", data.targets[0])
    # plot_mean(filename, x, y)
    # plot_all(filename, x, y)


if __name__ == '__main__':
    main()
