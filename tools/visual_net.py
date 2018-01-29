import mxnet as mx
data_shape = (1, 3, 224, 224)

sym = mx.sym.load('./resnet18-symbol.json')
mx.viz.plot_network(symbol=sym, shape={"data": data_shape}, node_attrs={"shape": 'oval', "fixedsize": 'false'}).view()
