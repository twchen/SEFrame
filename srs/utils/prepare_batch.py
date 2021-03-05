def prepare_batch_factory(device):
    def prepare_batch(batch):
        inputs, labels = batch
        inputs_gpu = [x.to(device) for x in inputs]
        labels_gpu = labels.to(device)
        return inputs_gpu, labels_gpu

    return prepare_batch


def prepare_batch_factory_recursive(device):
    def prepare_batch_recursive(batch):
        if type(batch) is list or type(batch) is tuple:
            return [prepare_batch_recursive(x) for x in batch]
        elif type(batch) is dict:
            return {k: prepare_batch_recursive(v) for k, v in batch.items()}
        else:
            return batch.to(device)

    return prepare_batch_recursive
