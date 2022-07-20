import numpy as np

def permutation_groups(xs, group_size):
    """
    Yields group_size-sized batches of xs indefinitely, by cycling through
    all objects before repeating one"""
    while 1:
        yield [xs[i] for i in np.random.permutation(group_size)]


def create_numpy_dataset_from_dataloaders(trainloaders):
    """Takes a list of torch trainloaders, returns the content in
    numpy arrays with env variables too"""
    data = []
    labels = []
    env_n = []
    for i, trainloader in enumerate(trainloaders):
        data.append(trainloader.dataset.tensors[0].numpy())
        labels.append(trainloader.dataset.tensors[1].numpy())
        env_n.append(i * data[0].shape[0])

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    env_n = np.stack(env_n, axis=0)

    return data, labels, env_n
