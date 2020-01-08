import DataGenerators as Gen
from BatchGenerator import flow_to_data


def init_data(num_datas):
    Gen.set_making_seed(0)
    train_datas = [flow_to_data(Gen.make_batch_generator()) for _ in range(0, num_datas)]
    val_data = flow_to_data(Gen.make_batch_generator())

    return train_datas, val_data
