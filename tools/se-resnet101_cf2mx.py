import sys
sys.path.append('/home/user/workspace/py-RFCN-priv/caffe-priv/python')
import caffe
import mxnet as mx
import numpy as np

gpu_id = 1
caffe.set_mode_gpu()
caffe.set_device(gpu_id)

mx_prefix   = 'se-resnet101'
mx_sym      = 'se-resnet101-symbol.json'
cf_model    = 'se-resnet101.caffemodel'
cf_prototxt = 'deploy_se-resnet101.prototxt'

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
name = ''
def convert_name(layer_name):
    global name
    layer_name = layer_name.split('/')[0]
    if 'conv2' in layer_name:
        name = 'seres' + layer_name[6] + layer_name[7:]
    elif 'conv3' in layer_name:
        name = 'seres' + str(3+ int(layer_name[6])) + layer_name[7:]
    elif 'conv4' in layer_name:
        name = 'seres' + str(7+ int(layer_name.split('_')[1])) + '_' + '_'.join(layer_name.split('_')[2:])
    elif 'conv5' in layer_name:
        name = 'seres' + str(30 + int(layer_name[6])) + layer_name[7:]
    return name

def convert_params_conv(new_name, layer_name):
  print
  print layer_name
  print new_name + ' conv'
  arg_params[new_name + '_weight'] = mx.nd.zeros(net.params[layer_name][0].data.shape)
  arg_params[new_name + '_weight'][:] = np.array(net.params[layer_name][0].data)
  if np.sum(arg_params[new_name + '_weight'].asnumpy() - net.params[layer_name][0].data) != 0:
     print layer_name, ',res weight:', np.sum(arg_params[new_name + '_weight'].asnumpy() - net.params[layer_name][0].data)

def convert_params_bn(new_name, layer_name):
  print
  print layer_name
  print new_name + ' bn'
  aux_params[new_name +'_bn_moving_mean'] = mx.nd.zeros(net.params[layer_name][0].data.shape)
  aux_params[new_name +'_bn_moving_mean'][:] = np.array(net.params[layer_name][0].data)
  if np.sum(aux_params[new_name + '_bn_moving_mean'].asnumpy() - net.params[layer_name][0].data) != 0:
    print layer_name, ',res moving_mean:', np.sum(aux_params[new_name + '_bn_moving_mean'].asnumpy() - net.params[layer_name][0].data)
  aux_params[new_name +'_bn_moving_var'] = mx.nd.zeros(net.params[layer_name][1].data.shape)
  aux_params[new_name +'_bn_moving_var'][:] = np.array(net.params[layer_name][1].data)
  if np.sum(aux_params[new_name + '_bn_moving_var'].asnumpy() - net.params[layer_name][1].data) != 0:
    print layer_name, ',res moving_var:', np.sum(aux_params[new_name + '_bn_moving_var'].asnumpy() - net.params[layer_name][1].data)

def convert_params_scale(new_name, layer_name):
  print
  print layer_name
  print new_name + ' scale'
  # print layer_name.split('_scale')[0] + '_bn_gamma'
  arg_params[new_name + '_bn_gamma'] = mx.nd.zeros(net.params[layer_name][0].data.shape)
  arg_params[new_name + '_bn_gamma'][:] = np.array(net.params[layer_name][0].data)
  if np.sum(arg_params[new_name + '_bn_gamma'].asnumpy() - net.params[layer_name][0].data) != 0:
    print layer_name, ',res gamma:', np.sum(arg_params[new_name + '_bn_gamma'].asnumpy() - net.params[layer_name][0].data)
  arg_params[new_name + '_bn_beta'] = mx.nd.zeros(net.params[layer_name][1].data.shape)
  arg_params[new_name + '_bn_beta'][:] = np.array(net.params[layer_name][1].data)
  if np.sum(arg_params[new_name + '_bn_beta'].asnumpy() - net.params[layer_name][1].data) != 0:
    print layer_name, ',res beta:', np.sum(arg_params[new_name + '_bn_beta'].asnumpy() - net.params[layer_name][1].data)



for layer_name in layer_names:
    # print layer_name
    if layer_name == 'input':
        pass
    elif 'relu' in layer_name:
        pass
    elif 'prob' in layer_name:
        pass
    elif 'split' in layer_name:
        pass
    elif 'pool' in layer_name:
        pass

    elif layer_name == 'conv1/7x7_s2':
        print layer_name
        new_name = 'conv1'
        arg_params[new_name+'_weight'] = mx.nd.zeros(net.params[layer_name][0].data.shape)
        net.params[layer_name][0].data[:, [0, 2], :, :] = net.params[layer_name][0].data[:, [2, 0], :, :]
        arg_params[new_name+'_weight'][:] = np.array(net.params[layer_name][0].data)
        # print arg_params[new_name+'_weight'].asnumpy()
    elif layer_name == 'conv1/7x7_s2/bn':
        new_name = 'conv1'
        convert_params_bn(new_name, layer_name)
    elif layer_name == 'conv1/7x7_s2/bn/scale':
        new_name = 'conv1'
        convert_params_scale(new_name, layer_name)
    elif layer_name == 'classifier':
        print
        print layer_name
        arg_params['classifier'+ '_weight'] = mx.nd.zeros(net.params[layer_name][0].data.shape)
        arg_params['classifier'+ '_weight'][:] = np.array(net.params[layer_name][0].data)
        arg_params['classifier' + '_bias'] = mx.nd.zeros(net.params[layer_name][1].data.shape)
        arg_params['classifier' + '_bias'][:] = np.array(net.params[layer_name][1].data)
    else:
        if 'bn' in layer_name:
            new_name = convert_name(layer_name)
            if 'scale' in layer_name:
                convert_params_scale(new_name, layer_name)
            else:
                convert_params_bn(new_name, layer_name)
        else:
            if 'conv' in layer_name:
                if 'x' not in layer_name:
                    pass
                else:
                    if 'down' in layer_name:
                        new_name = convert_name(layer_name)
                        print 'down'
                        print layer_name
                        arg_params[new_name + '_weight'] = mx.nd.zeros(net.params[layer_name][0].data.shape)
                        arg_params[new_name + '_weight'][:] = np.array(net.params[layer_name][0].data)
                        arg_params[new_name + '_bias'] = mx.nd.zeros(net.params[layer_name][1].data.shape)
                        arg_params[new_name + '_bias'][:] = np.array(net.params[layer_name][1].data)
                    elif 'up' in layer_name:
                        new_name = convert_name(layer_name)
                        print 'up'
                        print layer_name
                        arg_params[new_name + '_weight'] = mx.nd.zeros(net.params[layer_name][0].data.shape)
                        arg_params[new_name + '_weight'][:] = np.array(net.params[layer_name][0].data)
                        arg_params[new_name + '_bias'] = mx.nd.zeros(net.params[layer_name][1].data.shape)
                        arg_params[new_name + '_bias'][:] = np.array(net.params[layer_name][1].data)
                    else:
                        new_name = convert_name(layer_name)
                        convert_params_conv(new_name, layer_name)
            else:
                print
                print 'no param layer name---------', layer_name
model.init_params(arg_params=arg_params, aux_params=aux_params)
model.save_checkpoint(mx_prefix, 0)


print("\n- Finished.\n")



