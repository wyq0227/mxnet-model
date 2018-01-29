import sys
sys.path.append('/home/user/workspace/py-RFCN-priv/caffe-priv/python')
import caffe
import mxnet as mx
import numpy as np

gpu_id = 4
caffe.set_mode_gpu()
caffe.set_device(gpu_id)

mx_prefix   = 'resnext101-32x4d'
mx_sym      = 'resnext101-32x4d-symbol.json'
cf_model    = 'resnext101-32x4d.caffemodel'
cf_prototxt = 'deploy_resnext101-32x4d.prototxt'

input_dim = (1, 3, 224, 224)

# load models
net = caffe.Net(cf_prototxt, cf_model, caffe.TEST)
sym = mx.sym.load(mx_sym)
args = sym.list_arguments()
# print args
auxs = sym.list_auxiliary_states()
# print auxs
model = mx.mod.Module(symbol=sym, label_names=['softmax_label', ], context=mx.gpu(gpu_id))
model.bind(data_shapes=[('data', tuple(input_dim))])

arg_params = {}
aux_params = {}

# Convert
layer_names = net._layer_names
def convert_params_conv(layer_name):
  print
  print layer_name
  arg_params[layer_name + '_weight'] = mx.nd.zeros(net.params[layer_name][0].data.shape)
  arg_params[layer_name + '_weight'][:] = np.array(net.params[layer_name][0].data)
  if np.sum(arg_params[layer_name + '_weight'].asnumpy() - net.params[layer_name][0].data) != 0:
     print layer_name, ',res weight:', np.sum(arg_params[layer_name + '_weight'].asnumpy() - net.params[layer_name][0].data)

def convert_params_bn(layer_name):
  print
  print layer_name
  aux_params[layer_name +'_moving_mean'] = mx.nd.zeros(net.params[layer_name][0].data.shape)
  aux_params[layer_name +'_moving_mean'][:] = np.array(net.params[layer_name][0].data)
  if np.sum(aux_params[layer_name + '_moving_mean'].asnumpy() - net.params[layer_name][0].data) != 0:
    print layer_name, ',res moving_mean:', np.sum(aux_params[layer_name + '_moving_mean'].asnumpy() - net.params[layer_name][0].data)
  aux_params[layer_name +'_moving_var'] = mx.nd.zeros(net.params[layer_name][1].data.shape)
  aux_params[layer_name +'_moving_var'][:] = np.array(net.params[layer_name][1].data)
  if np.sum(aux_params[layer_name + '_moving_var'].asnumpy() - net.params[layer_name][1].data) != 0:
    print layer_name, ',res moving_var:', np.sum(aux_params[layer_name + '_moving_var'].asnumpy() - net.params[layer_name][1].data)

def convert_params_scale(layer_name):
  print
  print layer_name
  # print layer_name.split('_scale')[0] + '_bn_gamma'
  arg_params[layer_name.split('_scale')[0] + '_bn_gamma'] = mx.nd.zeros(net.params[layer_name][0].data.shape)
  arg_params[layer_name.split('_scale')[0] + '_bn_gamma'][:] = np.array(net.params[layer_name][0].data)
  if np.sum(arg_params[layer_name.split('_scale')[0] + '_bn_gamma'].asnumpy() - net.params[layer_name][0].data) != 0:
    print layer_name, ',res gamma:', np.sum(arg_params[layer_name.split('_scale')[0] + '_bn_gamma'].asnumpy() - net.params[layer_name][0].data)
  arg_params[layer_name.split('_scale')[0] + '_bn_beta'] = mx.nd.zeros(net.params[layer_name][1].data.shape)
  arg_params[layer_name.split('_scale')[0] + '_bn_beta'][:] = np.array(net.params[layer_name][1].data)
  if np.sum(arg_params[layer_name.split('_scale')[0] + '_bn_beta'].asnumpy() - net.params[layer_name][1].data) != 0:
    print layer_name, ',res beta:', np.sum(arg_params[layer_name.split('_scale')[0] + '_bn_beta'].asnumpy() - net.params[layer_name][1].data)

for layer_name in layer_names:
    # print layer_name

    if layer_name == 'input':
        pass
    elif '_relu' in layer_name:
        pass
    elif 'prob' in layer_name:
        pass
    elif 'pool' in layer_name:
        pass

    ########## conv1 ##########
    elif layer_name == 'conv1':
        print layer_name
        arg_params[layer_name+'_weight'] = mx.nd.zeros(net.params[layer_name][0].data.shape)
        print 'net'
        print net.params[layer_name][0].data
        net.params[layer_name][0].data[:, [0, 2], :, :] = net.params[layer_name][0].data[:, [2, 0], :, :]
        arg_params[layer_name+'_weight'][:] = np.array(net.params[layer_name][0].data)
        print 'mx'
        print net.params[layer_name][0].data
        print np.sum(arg_params[layer_name+'_weight'].asnumpy() - net.params[layer_name][0].data)

    ########## classifier ##########
    elif layer_name == 'classifier':
        print
        print layer_name
        arg_params['classifier'+ '_weight'] = mx.nd.zeros(net.params[layer_name][0].data.shape)
        arg_params['classifier'+ '_weight'][:] = np.array(net.params[layer_name][0].data)
        arg_params['classifier' + '_bias'] = mx.nd.zeros(net.params[layer_name][1].data.shape)
        arg_params['classifier' + '_bias'][:] = np.array(net.params[layer_name][1].data)

    elif '_bn' in layer_name:
        convert_params_bn(layer_name)
    elif '_scale' in layer_name:
        convert_params_scale(layer_name)

    else:
        if '_conv' in layer_name:
            convert_params_conv(layer_name)
        else:
            print
            print 'no param layer name---------', layer_name

model.init_params(arg_params=arg_params, aux_params=aux_params)
model.save_checkpoint(mx_prefix, 0)


print("\n- Finished.\n")
















