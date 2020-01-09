from Flows import *


def _set_making_seed(seed):
    np.random.seed(seed)


def _make_normal_flow_randomized(length):
    random_factors = np.random.random(3) * 0.1

    comp_const = ConstFlow(length, 1 * random_factors[0])
    comp_sin = SinFlow(length,
                           random_factors[1], np.random.ranf() * length,
                           np.random.ranf() * length / 2 - length / 4)
    comp_noise = PerlinFlow(length, random_factors[2], 200,  np.random.randint(4) + 1,
                                seed=np.random.randint(length))

    return SumFlow([comp_const, comp_sin, comp_noise])


def _make_hazard_flow_randomized(length, start):
    random_grow = np.random.randint(3)
    if random_grow == 0:
        return HazardSinGrow(length, start, length, 1)
    if random_grow == 1:
        return HazardParabolicGrow(length, start, length, 1)
    if random_grow == 2:
        return HazardLinearGrow(length, start, length, 1)


def _make_flow():
    n1 = _make_normal_flow_randomized(1000)
    n2 = _make_normal_flow_randomized(1000)
    h1 = _make_hazard_flow_randomized(1000, 550)
    h2 = _make_hazard_flow_randomized(1000, 550 + np.random.randint(50))
    hb = HazardTrueFalse(1000, 550, 1000, 1)
    return VectorFlow([hb, SumFlow([n1, h1]), SumFlow([n2, h2])])


def _make_sample(flow, beg):
    sample = [[flow[beg + of][1], flow[beg + of][2]] for of in range(0, 100)]
    label = flow[beg + 99][0]
    return sample, label


def init_data(count_flows_test, count_flows_val, samples_per_flow_test=10, samples_per_flow_val=10):
    _set_making_seed(0)
    samples_train = []
    for _ in range(0, count_flows_test):
        flow = _make_flow()
        for _ in range(0, samples_per_flow_test):
            samples_train += [_make_sample(flow, np.random.randint(0, 900))]

    samples_val = []
    for _ in range(0, count_flows_val):
        flow = _make_flow()
        for _ in range(0, samples_per_flow_val):
            samples_val += [_make_sample(flow, np.random.randint(0, 900))]

    return samples_train, samples_val


def init_test_samples():
    _set_making_seed(1)

    samples = []
    labels = []
    flow = _make_flow()
    for beg in range(0, 900):
        samples += [_make_sample(flow, beg)[0]]
        labels += [flow[beg + 99][0]]

    return samples, labels
