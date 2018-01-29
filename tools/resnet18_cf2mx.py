import sys
sys.path.append('/home/user/workspace/py-RFCN-priv/caffe-priv/python')
import caffe
import mxnet as mx
import numpy as np

gpu_id = 4
caffe.set_mode_gpu()
caffe.set_device(gpu_id)

mx_prefix   = 'resnet18-priv'
mx_sym      = 'resnet18-priv-symbol.json'
cf_model    = 'resnet18-priv.caffemodel'
cf_prototxt = 'deploy_resnet18-priv.prototxt'

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
def convert_params_conv(layer_name, new_layer_name):
  print
  print layer_name
  arg_params[new_layer_name + '_weight'] = mx.nd.zeros(net.params[layer_name][0].data.shape)
  arg_params[new_layer_name + '_weight'][:] = np.array(net.params[layer_name][0].data)
  if np.sum(arg_params[new_layer_name + '_weight'].asnumpy() - net.params[layer_name][0].data) != 0:
     print layer_name, ',res weight:', np.sum(arg_params[new_layer_name + '_weight'].asnumpy() - net.params[layer_name][0].data)

def convert_params_bn(layer_name, new_layer_name):
  print
  print layer_name
  aux_params[new_layer_name +'_moving_mean'] = mx.nd.zeros(net.params[layer_name][0].data.shape)
  aux_params[new_layer_name +'_moving_mean'][:] = np.array(net.params[layer_name][0].data)
  if np.sum(aux_params[new_layer_name + '_moving_mean'].asnumpy() - net.params[layer_name][0].data) != 0:
    print layer_name, ',res moving_mean:', np.sum(aux_params[new_layer_name + '_moving_mean'].asnumpy() - net.params[layer_name][0].data)
  aux_params[new_layer_name +'_moving_var'] = mx.nd.zeros(net.params[layer_name][1].data.shape)
  aux_params[new_layer_name +'_moving_var'][:] = np.array(net.params[layer_name][1].data)
  if np.sum(aux_params[new_layer_name + '_moving_var'].asnumpy() - net.params[layer_name][1].data) != 0:
    print layer_name, ',res moving_var:', np.sum(aux_params[new_layer_name + '_moving_var'].asnumpy() - net.params[layer_name][1].data)

def convert_params_scale(layer_name, new_layer_name):
  print
  print layer_name
  # print layer_name.split('_scale')[0] + '_bn_gamma'
  arg_params[new_layer_name.split('_scale')[0] + '_bn_gamma'] = mx.nd.zeros(net.params[layer_name][0].data.shape)
  arg_params[new_layer_name.split('_scale')[0] + '_bn_gamma'][:] = np.array(net.params[layer_name][0].data)
  if np.sum(arg_params[new_layer_name.split('_scale')[0] + '_bn_gamma'].asnumpy() - net.params[layer_name][0].data) != 0:
    print layer_name, ',res gamma:', np.sum(arg_params[new_layer_name.split('_scale')[0] + '_bn_gamma'].asnumpy() - net.params[layer_name][0].data)
  arg_params[new_layer_name.split('_scale')[0] + '_bn_beta'] = mx.nd.zeros(net.params[layer_name][1].data.shape)
  arg_params[new_layer_name.split('_scale')[0] + '_bn_beta'][:] = np.array(net.params[layer_name][1].data)
  if np.sum(arg_params[new_layer_name.split('_scale')[0] + '_bn_beta'].asnumpy() - net.params[layer_name][1].data) != 0:
    print layer_name, ',res beta:', np.sum(arg_params[new_layer_name.split('_scale')[0] + '_bn_beta'].asnumpy() - net.params[layer_name][1].data)  

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

    elif layer_name == 'conv1':
        print layer_name
        arg_params['conv1_weight'] = mx.nd.zeros(net.params[layer_name][0].data.shape)
        print 'net'
        print net.params[layer_name][0].data
        net.params[layer_name][0].data[:, [0, 2], :, :] = net.params[layer_name][0].data[:, [2, 0], :, :]
        arg_params['conv1_weight'][:] = np.array(net.params[layer_name][0].data)
        print 'mx'
        print net.params[layer_name][0].data
        print np.sum(arg_params['conv1_weight'].asnumpy() - net.params[layer_name][0].data)


    elif layer_name == 'bn_conv1':
        print
        print layer_name
        aux_params['conv1_bn' +'_moving_mean'] = mx.nd.zeros(net.params[layer_name][0].data.shape)
        aux_params['conv1_bn' +'_moving_mean'][:] = np.array(net.params[layer_name][0].data)
        if np.sum(aux_params['conv1_bn' + '_moving_mean'].asnumpy() - net.params[layer_name][0].data) != 0:
            print layer_name, ',res moving_mean:', np.sum(aux_params['conv1_bn' + '_moving_mean'].asnumpy() - net.params[layer_name][0].data)
        aux_params['conv1_bn' +'_moving_var'] = mx.nd.zeros(net.params[layer_name][1].data.shape)
        aux_params['conv1_bn' +'_moving_var'][:] = np.array(net.params[layer_name][1].data)
        if np.sum(aux_params['conv1_bn' + '_moving_var'].asnumpy() - net.params[layer_name][1].data) != 0:
            print layer_name, ',res moving_var:', np.sum(aux_params['conv1_bn' + '_moving_var'].asnumpy() - net.params[layer_name][1].data)

    elif layer_name == 'scale_conv1':
        print
        print layer_name
        # print layer_name.split('_scale')[0] + '_bn_gamma'
        arg_params['conv1_scale'.split('_scale')[0] + '_bn_gamma'] = mx.nd.zeros(net.params[layer_name][0].data.shape)
        arg_params['conv1_scale'.split('_scale')[0] + '_bn_gamma'][:] = np.array(net.params[layer_name][0].data)
        if np.sum(arg_params['conv1_scale'.split('_scale')[0] + '_bn_gamma'].asnumpy() - net.params[layer_name][0].data) != 0:
            print layer_name, ',res gamma:', np.sum(arg_params['conv1_scale'.split('_scale')[0] + '_bn_gamma'].asnumpy() - net.params[layer_name][0].data)
        arg_params['conv1_scale'.split('_scale')[0] + '_bn_beta'] = mx.nd.zeros(net.params[layer_name][1].data.shape)
        arg_params['conv1_scale'.split('_scale')[0] + '_bn_beta'][:] = np.array(net.params[layer_name][1].data)
        if np.sum(arg_params['conv1_scale'.split('_scale')[0] + '_bn_beta'].asnumpy() - net.params[layer_name][1].data) != 0:
            print layer_name, ',res beta:', np.sum(arg_params['conv1_scale'.split('_scale')[0] + '_bn_beta'].asnumpy() - net.params[layer_name][1].data)

    elif layer_name == 'fc1000':
        print
        print layer_name
        arg_params['classifier'+ '_weight'] = mx.nd.zeros(net.params[layer_name][0].data.shape)
        arg_params['classifier'+ '_weight'][:] = np.array(net.params[layer_name][0].data)
        arg_params['classifier' + '_bias'] = mx.nd.zeros(net.params[layer_name][1].data.shape)
        arg_params['classifier' + '_bias'][:] = np.array(net.params[layer_name][1].data)

########## res1 ##########
    elif layer_name == 'res2a_branch2a':
        convert_params_conv(layer_name, 'res1_conv1')

    elif layer_name == 'bn2a_branch2a':
        convert_params_bn(layer_name,'res1_conv1_bn')

    elif layer_name == 'scale2a_branch2a':
        convert_params_scale(layer_name,'res1_conv1_scale')

    elif layer_name == 'res2a_branch2b':
        convert_params_conv(layer_name,'res1_conv2')

    elif layer_name == 'bn2a_branch2b':
        convert_params_bn(layer_name,'res1_conv2_bn')

    elif layer_name == 'scale2a_branch2b':
        convert_params_scale(layer_name,'res1_conv2_scale')

########## res2 ##########  
    elif layer_name == 'res2b_branch2a':
        convert_params_conv(layer_name, 'res2_conv1')

    elif layer_name == 'bn2b_branch2a':
        convert_params_bn(layer_name,'res2_conv1_bn')

    elif layer_name == 'scale2b_branch2a':
        convert_params_scale(layer_name,'res2_conv1_scale')

    elif layer_name == 'res2b_branch2b':
        convert_params_conv(layer_name,'res2_conv2')

    elif layer_name == 'bn2b_branch2b':
        convert_params_bn(layer_name,'res2_conv2_bn')

    elif layer_name == 'scale2b_branch2b':
        convert_params_scale(layer_name,'res2_conv2_scale')

########## res3 ##########  
    elif layer_name == 'res3a_branch2a':
        convert_params_conv(layer_name, 'res3_conv1')

    elif layer_name == 'bn3a_branch2a':
        convert_params_bn(layer_name,'res3_conv1_bn')

    elif layer_name == 'scale3a_branch2a':
        convert_params_scale(layer_name,'res3_conv1_scale')

    elif layer_name == 'res3a_branch2b':
        convert_params_conv(layer_name,'res3_conv2')

    elif layer_name == 'bn3a_branch2b':
        convert_params_bn(layer_name,'res3_conv2_bn')

    elif layer_name == 'scale3a_branch2b':
        convert_params_scale(layer_name,'res3_conv2_scale')

    elif layer_name == 'res3a_branch1':
        convert_params_conv(layer_name,'res3_match_conv')

    elif layer_name == 'bn3a_branch1':
        convert_params_bn(layer_name,'res3_match_conv_bn')

    elif layer_name == 'scale3a_branch1':
        convert_params_scale(layer_name,'res3_match_conv_scale')

########## res4 ##########  
    elif layer_name == 'res3b_branch2a':
        convert_params_conv(layer_name, 'res4_conv1')

    elif layer_name == 'bn3b_branch2a':
        convert_params_bn(layer_name,'res4_conv1_bn')

    elif layer_name == 'scale3b_branch2a':
        convert_params_scale(layer_name,'res4_conv1_scale')

    elif layer_name == 'res3b_branch2b':
        convert_params_conv(layer_name,'res4_conv2')

    elif layer_name == 'bn3b_branch2b':
        convert_params_bn(layer_name,'res4_conv2_bn')

    elif layer_name == 'scale3b_branch2b':
        convert_params_scale(layer_name,'res4_conv2_scale')
########## res5 ##########  
    elif layer_name == 'res4a_branch2a':
        convert_params_conv(layer_name, 'res5_conv1')

    elif layer_name == 'bn4a_branch2a':
        convert_params_bn(layer_name,'res5_conv1_bn')

    elif layer_name == 'scale4a_branch2a':
        convert_params_scale(layer_name,'res5_conv1_scale')

    elif layer_name == 'res4a_branch2b':
        convert_params_conv(layer_name,'res5_conv2')

    elif layer_name == 'bn4a_branch2b':
        convert_params_bn(layer_name,'res5_conv2_bn')

    elif layer_name == 'scale4a_branch2b':
        convert_params_scale(layer_name,'res5_conv2_scale')

    elif layer_name == 'res4a_branch1':
        convert_params_conv(layer_name,'res5_match_conv')

    elif layer_name == 'bn4a_branch1':
        convert_params_bn(layer_name,'res5_match_conv_bn')

    elif layer_name == 'scale4a_branch1':
        convert_params_scale(layer_name,'res5_match_conv_scale')

########## res6 ##########  
    elif layer_name == 'res4b_branch2a':
        convert_params_conv(layer_name, 'res6_conv1')

    elif layer_name == 'bn4b_branch2a':
        convert_params_bn(layer_name,'res6_conv1_bn')

    elif layer_name == 'scale4b_branch2a':
        convert_params_scale(layer_name,'res6_conv1_scale')

    elif layer_name == 'res4b_branch2b':
        convert_params_conv(layer_name,'res6_conv2')

    elif layer_name == 'bn4b_branch2b':
        convert_params_bn(layer_name,'res6_conv2_bn')

    elif layer_name == 'scale4b_branch2b':
        convert_params_scale(layer_name,'res6_conv2_scale')

########## res7 ##########  
    elif layer_name == 'res5a_branch2a':
        convert_params_conv(layer_name, 'res7_conv1')

    elif layer_name == 'bn5a_branch2a':
        convert_params_bn(layer_name,'res7_conv1_bn')

    elif layer_name == 'scale5a_branch2a':
        convert_params_scale(layer_name,'res7_conv1_scale')

    elif layer_name == 'res5a_branch2b':
        convert_params_conv(layer_name,'res7_conv2')

    elif layer_name == 'bn5a_branch2b':
        convert_params_bn(layer_name,'res7_conv2_bn')

    elif layer_name == 'scale5a_branch2b':
        convert_params_scale(layer_name,'res7_conv2_scale')

    elif layer_name == 'res5a_branch1':
        convert_params_conv(layer_name,'res7_match_conv')

    elif layer_name == 'bn5a_branch1':
        convert_params_bn(layer_name,'res7_match_conv_bn')

    elif layer_name == 'scale5a_branch1':
        convert_params_scale(layer_name,'res7_match_conv_scale')

########## res8 ##########  
    elif layer_name == 'res5b_branch2a':
        convert_params_conv(layer_name, 'res8_conv1')

    elif layer_name == 'bn5b_branch2a':
        convert_params_bn(layer_name,'res8_conv1_bn')

    elif layer_name == 'scale5b_branch2a':
        convert_params_scale(layer_name,'res8_conv1_scale')

    elif layer_name == 'res5b_branch2b':
        convert_params_conv(layer_name,'res8_conv2')

    elif layer_name == 'bn5b_branch2b':
        convert_params_bn(layer_name,'res8_conv2_bn')

    elif layer_name == 'scale5b_branch2b':
        convert_params_scale(layer_name,'res8_conv2_scale')

    else:
      print
      print 'no param layer name---------', layer_name


model.init_params(arg_params=arg_params, aux_params=aux_params)
model.save_checkpoint(mx_prefix, 0)


print("\n- Finished.\n")













