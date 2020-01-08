import numpy as np

from matplotlib import pyplot as plt

from Data import DataGenerators as Gen

def show_flow_vector(flow):
    plt.figure()
    t = np.arange(0, len(flow), 1)
    plt.plot(flow[0:len(flow) - 1])
    plt.title('flow')
    plt.legend()
    plt.show(block=True)


np.random.seed(0)

flow1 = Gen.ConstFlow(1000, 0.1)
flow2 = Gen.SinFlow(1000, 0.1, 200, 50)
flow3 = Gen.PerlinFlow(1000, 0.1, 200, 5)

flow4 = Gen.SinFlow(1000, 0.1, 500, 125)
flow5 = Gen.SinFlow(1000, 0.01, 75, 125)
flow6 = Gen.SumFlow([flow4, flow5])

flow7 = Gen.HazardLinearGrow(1000, 500, 1000, 2)
flow8 = Gen.HazardSinGrow(1000, 500, 1000, 2)
flow9 = Gen.HazardParabolicGrow(1000, 500, 1000, 2)
flow10 = Gen.HazardTrueFalse(1000, 500, 1000, 2)

# show_flow_vector(Gen.VectorFlow([flow1, flow2, flow3, flow4, flow5, flow6, flow7, flow8, flow9, flow10]))


# show_flow_vector(Gen.VectorFlow([make_normal_flow_randomized(1000), Gen.ConstFlow(1000, 1)]))
show_flow_vector(Gen.make_batch_generator())