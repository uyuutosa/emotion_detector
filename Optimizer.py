import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import theano.tensor.signal as signal 
import numpy as np
import matplotlib.pyplot as plt
import copy as cp
from theano.printing import pydotprint
from PIL import Image
import os
import matplotlib as mpl
from theano.tensor.shared_randomstreams import RandomStreams
from sklearn.datasets import *
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
#mpl.rc("savefig", dpi=1200)

#%config InlineBackend.rc = {'font.size': 20, 'figure.figsize': (12.0, 8.0),'figure.facecolor': 'white', 'savefig.dpi': 72, 'figure.subplot.bottom': 0.125, 'figure.edgecolor': 'white'}
#%matplotlib inline

class optimizer:
    def __init__(self, x_arr=None, y_arr=None, 
                 out=None, thetalst=None, nodelst=None,
                 test_size=0.1, n_batch=500):
        
        self.n_batch = theano.shared(int(n_batch))
        
        if x_arr is not None and y_arr is not None:
            self.set_data(x_arr, y_arr, test_size)
            self.set_variables()
        
        
        
        self.thetalst = [] #if thetalst is None else thetalst
        
        self.n_view = None
        self.updatelst = []
        self.tmplst = []
    
    def set_data(self, x_arr, y_arr, test_size=0.1):
        self.x_train_arr, \
        self.x_test_arr,\
        self.y_train_arr,\
        self.y_test_arr,\
        = train_test_split(x_arr.astype(theano.config.floatX),
                           y_arr.astype(theano.config.floatX),
                           test_size = test_size)
        
        self.nodelst = [[int(np.prod(self.x_train_arr.shape[1:]))]] # if nodelst is None else nodelst
        
    
    def set_variables(self):
        if self.n_batch.get_value() > self.x_train_arr.shape[0]: 
            self.n_batch.set_value(int(self.x_train_arr.shape[0]))
        self.n_data = self.x_train_arr.shape[0]
        n_xdim = self.x_train_arr.ndim
        n_ydim = self.y_train_arr.ndim
        if  n_xdim == 0:
            self.x = T.scalar()
        if  n_xdim == 1:
            self.x_train_arr = self.x_train_arr[:,None]
            self.x_test_arr = self.x_test_arr[:,None]
            self.x = T.matrix()
        elif n_xdim == 2:
            self.x = T.matrix()
        elif n_xdim == 3:
            self.x = T.tensor3()
        else:
            self.x = T.tensor4()
            
        if n_ydim == 0:
            self.y = T.scalar()
        if n_ydim == 1:
            self.y_train_arr = self.y_train_arr[:,None]
            self.y_test_arr = self.y_test_arr[:,None]
            self.y = T.matrix()
        elif n_ydim == 2:
            self.y = T.matrix()
        elif n_ydim == 3:
            self.y = T.tensor3()
        else:
            self.y = T.tensor4()
            
        self.out = self.x  #if out is None else out
        self.batch_shape_of_C = T.concatenate([T.as_tensor([self.n_batch]), theano.shared(np.array([3]))], axis=0)
        
    def set_datasets(self, data="mnist", data_home="data_dir_for_optimizer", is_one_hot=True):
        
        if data == "mnist":
            data_dic = fetch_mldata('MNIST original', data_home=data_home)
            if is_one_hot == True:
                idx = data_dic["target"]
                arr = np.zeros((idx.shape[0],10)).flatten()
                arr[idx.flatten().astype(int) + np.arange(idx.shape[0]) * 10]  = 1
                data_dic["target"] = arr.reshape(idx.shape[0], 10)
        elif data == "boston":
            data_dic = load_boston()   
        elif data == "digits":
            data_dic = load_digits()
        elif data == "iris":
            data_dic = load_iris()
        elif data == "linnerud":
            data_dic = load_linnerud()
        elif data == "xor":
            data_dic = {"data": np.array([[0,0], [0,1], [1,0], [1,1]]),##.repeat(20, axis=0),
                        "target": np.array([0,1,1,0])}#.repeat(20, axis=0)}
            #data_dic = {"data": np.array([[0,0], [0,1], [1,0], [1,1]]).repeat(20, axis=0),
            #            "target": np.array([0,1,1,0]).repeat(20, axis=0)}
        elif data == "serial":
            data_dic = {"data": np.array(np.arange(20).reshape(5,4)).repeat(20, axis=0),
                        "target": np.arange(5).repeat(20, axis=0)}
        elif data == "sin":
            data_dic = {"data": np.arange(0,10,0.01)[:,None],
                        "target": np.sin(np.arange(0,10,0.01) * np.pi)}
            
        self.set_data(data_dic["data"], data_dic["target"])
        self.set_variables()
    
    def copy(self):
        return cp.copy(self)
    
    def update_node(self, n_out):
        self.nodelst = self.nodelst + [n_out]
        
    def get_curr_node(self):
        return list(self.nodelst[-1])
    
    def dropout(self, rate=0.5, seed=None):
        obj = self.copy()
        
        srng = RandomStreams(seed)
        obj.out = T.where(srng.uniform(size=obj.out.shape) > rate, obj.out, 0)
        
        return obj
    
        
    def dense(self, n_out):
        obj = self.copy()
        n_in = obj.get_curr_node()[-1]
        
        #theta =  theano.shared(np.random.rand(n_in, n_out).astype(theano.config.floatX))
        #b  =     theano.shared(np.random.rand(n_out).astype(theano.config.floatX))
        theta =  theano.shared(np.zeros((n_in, n_out)).astype(theano.config.floatX))
        #b     =  theano.shared(np.zeros((n_out)).astype(theano.config.floatX))
        b = theano.shared(np.random.rand(1).astype(dtype=theano.config.floatX)[0])
        
        obj.out = obj.out.dot(theta) + b
        #obj.out   = theta.dot(obj.out.flatten())+b
        obj.thetalst += [theta,b]
        
        obj.update_node([n_out])
        
        return obj
    
    def lstm(self):
        obj = self.copy()
        
        curr_shape = obj.get_curr_node()
        n_in = n_out = curr_shape[-1]
        
        #batch_shape_of_h = T.concatenate(
        #batch_shape_of_C = T.concatenate(, axis=0)
#        h = T.ones(theano.sharedl())
        h = T.zeros([obj.n_batch, *curr_shape], dtype=theano.config.floatX)
        C = T.zeros([obj.n_batch, n_out], dtype=theano.config.floatX)
        
        Wi =  theano.shared(np.random.rand(n_in, n_out).astype(theano.config.floatX))
        Wf =  theano.shared(np.random.rand(n_in, n_out).astype(theano.config.floatX))
        Wc =  theano.shared(np.random.rand(n_in, n_out).astype(theano.config.floatX))
        Wo =  theano.shared(np.random.rand(n_in, n_out).astype(theano.config.floatX))
        bi =  theano.shared(np.random.rand(n_out).astype(theano.config.floatX))
        bf =  theano.shared(np.random.rand(n_out).astype(theano.config.floatX))
        bc =  theano.shared(np.random.rand(n_out).astype(theano.config.floatX))
        bo =  theano.shared(np.random.rand(n_out).astype(theano.config.floatX))
        Ui =  theano.shared(np.random.rand(n_in, n_out).astype(theano.config.floatX))
        Uf =  theano.shared(np.random.rand(n_in, n_out).astype(theano.config.floatX))
        Uc =  theano.shared(np.random.rand(n_in, n_out).astype(theano.config.floatX))
        Uo =  theano.shared(np.random.rand(n_in, n_out).astype(theano.config.floatX))
        
        i = nnet.sigmoid(obj.out.dot(Wi) + h.dot(Ui) + bi)
        
        C_tilde = T.tanh(obj.out.dot(Wc) + h.dot(Uc) + bc)
        
        f = nnet.sigmoid(obj.out.dot(Wf) + h.dot(Uf) + bf)
        
        tmp = (i * C_tilde + f * C).reshape(C.shape)
        
        obj.tmplst += [(C, tmp)]
        
        C = tmp
        
        o = nnet.sigmoid(obj.out.dot(Wo) + h.dot(Uo) + bo)
        
        tmp = (o * T.tanh(C)).reshape(h.shape)
        
        obj.tmplst += [(h, tmp)]
        
        obj.out =  tmp
        
        obj.thetalst += [Wi, bi, Ui, Wf, bf, Uf, Wc, bc, Uc, Wo, bo, Uo]
        
        obj.update_node([n_out])
        
        return obj

    
    def conv2d(self, kshape=(1,1,3,3), mode="full", reshape=None):
        obj = self.copy()
        
        n_in = obj.get_curr_node()
        
        if reshape is not None:
            obj.out = obj.out.reshape(reshape)
            n_in = reshape
            
        if obj.out.ndim == 2:
            obj.out = obj.out[None, :, :]
            n_in = (1, n_in[1], n_in[2])
        elif obj.out.ndim == 1:
            obj.out = obj.out[None, None, :]
            n_in = [1, 1] + list(n_in)
            
        if mode == "full":
            n_out = [kshape[0], n_in[-2] + (kshape[-2] - 1), n_in[-1] + (kshape[-2] - 1)]
            #n_out = [n_in[0], kshape[0], n_in[-2] + (kshape[-2] - 1), n_in[-1] + (kshape[-2] - 1)]
        elif mode == "valid":
            n_out = [kshape[0], n_in[-2] - (kshape[-2] - 1), n_in[-1] - (kshape[-2] - 1)]
        
        theta = theano.shared(np.random.rand(*kshape).astype(dtype=theano.config.floatX))
        b = theano.shared(np.random.rand(1).astype(dtype=theano.config.floatX)[0])
        obj.b = b
            
        
        obj.out = nnet.conv2d(obj.out, theta, border_mode=mode) + b
        obj.thetalst += [theta, b]
        
        obj.update_node(n_out)
        
        return obj
    
    def conv_and_pool(self, fnum, height, width, mode="full", ds=(2,2)):
        obj = self.copy()
        n_in = obj.get_curr_node()
        kshape = (fnum, n_in[0], height, width)
        n_batch = obj.n_batch.get_value()
        obj = obj.conv2d(kshape=kshape)
        #.reshape((n_batch, np.array(n_in, dtype=int).sum()))\
        
        if mode == "full":
            obj = obj.relu().pool(ds=ds)
            n_in = obj.get_curr_node()
            obj = obj.reshape((kshape[0], *n_in[-2:]))
            #obj = obj.reshape((kshape[0], M[0]+(m[0]-1),M[1]+(m[1]-1)))
        elif mode == "valid":
            n_in = obj.get_curr_node()
            obj = obj.reshape((kshape[0], *n_in[-2:]))
        return obj
            
    
    def pool(self, ds=(2,2)):
        obj = self.copy()
        n_in = obj.get_curr_node()
        
        obj.out = signal.pool.pool_2d(obj.out, ds=ds, ignore_border=True)
        
        n_in[-2] = n_in[-2] // ds[0] #+ (1 if (n_in[-2] % ds[0]) else 0) 
        n_in[-1] = n_in[-1] // ds[1] #+ (1 if (n_in[-1] % ds[1]) else 0) 
        
        obj.update_node(n_in)
        #print(obj.nodelst)
        
        return obj
    
    def mean(self, axis):
        obj = self.copy()
        
        n_in = obj.get_curr_node()
        obj.out = obj.out.mean(axis=axis)
        obj.update_node(np.ones(n_in).mean(axis=axis).shape)
        return obj
    
    def reshape(self, shape):
        obj = self.copy()
        obj.out = obj.out.reshape([obj.n_batch, *shape])
        obj.update_node(shape)
        return obj
    def reshape_(self, shape):
        obj = self.copy()
        obj.out = obj.out.reshape(shape)
        obj.update_node(shape[1:])
        return obj
    
    def flatten(self):
        obj = self.copy()
        n_in = obj.get_curr_node()
        last_ndim = np.array(n_in).prod()
        obj.out = obj.out.reshape((obj.n_batch, last_ndim))
        obj.update_node([last_ndim])
        return obj
    
    def taylor(self, M, n_out):
        obj = self.copy()
        n_in = int(np.asarray(obj.get_curr_node()).sum())
        
        x_times = T.concatenate([obj.out, T.ones((obj.n_batch, 1)).astype(theano.config.floatX)],axis=1)
        for i in range(M-1):
            idx = np.array([":,"]).repeat(i+1).tolist()
            a = dict()
            exec ("x_times = x_times[:," + "".join(idx) + "None] * x_times", locals(), a)
            x_times = a["x_times"]

        x_times = x_times.reshape((obj.n_batch, -1))
        #x_times = x_times.reshape(self.n_batch, (n_in+1) ** M)

        theta = theano.shared(np.ones(((n_in+1) ** M, n_out)).astype(theano.config.floatX))
        obj.theta = theano.shared(np.ones(((n_in+1) ** M, n_out)).astype(theano.config.floatX))
        #theta = theano.shared(np.random.rand(n_out, (n_in+1) ** M).astype(theano.config.floatX))
        
        obj.out = x_times.dot(theta.astype(theano.config.floatX))
        obj.thetalst += [theta]
        
        obj.update_node([n_out])
        
        return obj

        
    def relu(self, ):
        obj = self.copy()
        obj.out = nnet.relu(obj.out)
        return obj
    
    def tanh(self, ):
        obj = self.copy()
        obj.out = T.tanh(obj.out)
        return obj
    
    def sigmoid(self, ):
        obj = self.copy()
        obj.out = nnet.sigmoid(obj.out)
        return obj
    
    def softmax(self, ):
        obj = self.copy()
        obj.out = nnet.softmax(obj.out)
        return obj
        
    def loss_msq(self):
        obj = self.copy()
        obj.loss =  T.mean((obj.out - obj.y) ** 2)
        return obj
    
    def loss_cross_entropy(self):
        obj = self.copy()
        obj.loss =  -T.mean(obj.y * T.log(obj.out + 1e-4))
        return obj
    
    def loss_softmax_cross_entropy(self):
        obj = self.copy()
        obj.out = nnet.softmax(obj.out)
        tmp_y = T.cast(obj.y, "int32")
        obj.loss  = -T.mean(T.log(obj.out)[T.arange(obj.y.shape[0]), tmp_y.flatten()])
        obj.out = obj.out.argmax(axis=1)[:,None]
        #obj.out2 = obj.out.argmax(axis=1)
        #obj.out2 = obj.out[T.arange(obj.y.shape[0]), tmp_y.flatten()]
        
        
#        obj.loss2 = T.log(obj.out)[T.arange(obj.y.shape[0]), tmp_y.flatten()]
        return obj
    
    def opt_sgd(self, alpha=0.1):
        obj = self.copy()
        obj.updatelst = []
        for theta in obj.thetalst:
            obj.updatelst += [(theta, theta - (alpha * T.grad(obj.loss, wrt=theta)))]
            
        obj.updatelst += obj.tmplst
        return obj
    
    def opt_RMSProp(self, alpha=0.1, gamma=0.9, ep=1e-8):
        obj = self.copy()
        obj.updatelst = []
        obj.rlst = [theano.shared(x.shape).astype(theano.config.floatX) for x in obj.thetalst]
        
        for r, theta in zip(obj.rlst, obj.thetalst):
            g = T.grad(obj.loss, wrt=theta)
            obj.updatelst += [(r, gamma * r + (1 - gamma) * g ** 2),\
                              (theta, theta - (alpha / (T.sqrt(r) + ep)) * g)]
        obj.updatelst += obj.tmplst
        return obj
                               
    def opt_AdaGrad(self, ini_eta=0.001, ep=1e-8):
        obj = self.copy()
        obj.updatelst = []
                               
        obj.hlst = [theano.shared(ep*np.ones(x.get_value().shape, theano.config.floatX)) for x in obj.thetalst]
        obj.etalst = [theano.shared(ini_eta*np.ones(x.get_value().shape, theano.config.floatX)) for x in obj.thetalst]
        
        for h, eta, theta in zip(obj.hlst, obj.etalst, obj.thetalst):
            g   = T.grad(obj.loss, wrt=theta)
            obj.updatelst += [(h,     h + g ** 2),
                              (eta,   eta / T.sqrt(h+1e-4)),
                              (theta, theta - eta * g)]
            
        obj.updatelst += obj.tmplst
        return obj
    
    def opt_Adam(self, alpha=0.001, beta=0.9, gamma=0.999, ep=1e-8, t=3):
        obj = self.copy()
        obj.updatelst = []
        obj.nulst = [theano.shared(np.zeros(x.get_value().shape, theano.config.floatX)) for x in obj.thetalst]
        obj.rlst = [theano.shared(np.zeros(x.get_value().shape, theano.config.floatX)) for x in obj.thetalst]
                               
        for nu, r, theta in zip(obj.nulst, obj.rlst, obj.thetalst):
            g = T.grad(obj.loss, wrt=theta)
            nu_hat = nu / (1 - beta)
            r_hat = r / (1 - gamma)
            obj.updatelst += [(nu, beta * nu + (1 - beta) * g),\
                              (r, gamma * r +  (1 - gamma) * g ** 2),\
                              (theta, theta - alpha*(nu_hat / (T.sqrt(r_hat) + ep)))]
            
        obj.updatelst += obj.tmplst
        return obj
                               
    
    
    def optimize(self, n_epoch=10, n_view=1000):
        obj = self.copy()
        obj.dsize = obj.x_train_arr.shape[0]
        if obj.n_view is None: obj.n_view = n_view  
        
        x_train_arr_shared = theano.shared(obj.x_train_arr).astype(theano.config.floatX)
        y_train_arr_shared = theano.shared(obj.y_train_arr).astype(theano.config.floatX)

        #rng = T.shared_randomstreams.RandomStreams(seed=0)
        #obj.idx = rng.permutation(size=(1,), n=obj.x_train_arr.shape[0]).astype("int64")
        #obj.idx = rng.permutation(size=(1,), n=obj.x_train_arr.shape[0])[0, 0:obj.n_batch].astype("int64")
        #idx = obj.idx * 1
        i = theano.shared(0).astype("int32")
        idx = theano.shared(np.random.permutation(obj.x_train_arr.shape[0]))
        
        obj.loss_func = theano.function(inputs=[i],
                                      outputs=obj.loss,
                                      givens=[(obj.x,x_train_arr_shared[idx[i:obj.n_batch+i],]), 
                                              (obj.y,y_train_arr_shared[idx[i:obj.n_batch+i],])],
                                      updates=obj.updatelst,
                                      on_unused_input='warn')
        
        i = theano.shared(0).astype("int32")
        idx = theano.shared(np.random.permutation(obj.x_train_arr.shape[0]))
        
        
        obj.acc_train = theano.function(inputs=[i],
                                      #outputs=((T.eq(obj.out2[:,None],obj.y))),
                                      outputs=((T.eq(obj.out,obj.y)).sum().astype(theano.config.floatX) / obj.n_batch),
                                      givens=[(obj.x,x_train_arr_shared[idx[i:obj.n_batch+i],]), 
                                              (obj.y,y_train_arr_shared[idx[i:obj.n_batch+i],])],
                                      on_unused_input='warn')
        #idx = rng.permutation(size=(1,), a=obj.x_train_arr.shape[0])#.astype("int8")
        
        #idx = rng.permutation(size=(1,), n=500, ndim=None)[0,0:10]#.astype("int8")
        
        c = T.concatenate([obj.x,obj.y],axis=1)
        obj.test_func = theano.function(inputs=[i],
                               outputs=c,
                               givens=[(obj.x,x_train_arr_shared[idx[i:obj.n_batch+i],]), 
                                       (obj.y,y_train_arr_shared[idx[i:obj.n_batch+i],])],
                               on_unused_input='warn')
        
#        obj.test_func = theano.function(inputs=[i],
#                               outputs=c,
#                               givens=[(obj.x,x_train_arr_shared[idx[i:obj.n_batch+i],]), 
#                                       (obj.y,y_train_arr_shared[idx[i:obj.n_batch+i],])],
#                               on_unused_input='warn')
        
        obj.pred_func = theano.function(inputs  = [obj.x],
                                        outputs = obj.out,
                                        updates = obj.tmplst,
                                        on_unused_input='warn')
            
        obj.v_loss = []
        try:
            for epoch in range(n_epoch):
                idx.set_value(np.random.permutation(obj.x_train_arr.shape[0]))
                mean_acc = 0.
                mean_loss = 0.
                N = obj.dsize-obj.n_batch.get_value() + 1
                step = obj.n_batch.get_value()
                for i in range(0, N, step):
                    mean_acc  += obj.acc_train(i)
                    mean_loss += obj.loss_func(i) 
                mean_acc  /= N / step
                mean_loss /= N / step
                if not (epoch % n_view):
                    print("Epoch. %s: loss = %s, acc = %s." %(epoch, mean_loss, mean_acc))
                
                obj.v_loss += [mean_loss]
            
            
        except KeyboardInterrupt  :
            print ( "KeyboardInterrupt\n" )
            return obj
        return obj
    
    def view_params(self):
        for theta in self.thetalst:
            print(theta.get_value().shape)
            print(theta.get_value())
            
        
    def view(self, yscale="log"):
        if not len(self.v_loss):
            raise ValueError("Loss value is not be set.")
        plt.clf()
        
        plt.xlabel("IterNum [x%s]" %self.n_view)
        plt.ylabel("Loss")
        plt.yscale(yscale)
        plt.plot(self.v_loss)
        plt.show()
    
    def view_graph(self, width='100%', res=60):
        path = 'examples'; name = 'mlp.png'
        path_name = path + '/' + name 
        if not os.path.exists(path):
            os.mkdirs(path)
        pydotprint(self.loss, path_name)
        plt.figure(figsize=(res, res), dpi=80)
        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, hspace=0.0, wspace=0.0)
        plt.axis('off')
        plt.imshow(np.array(Image.open(path_name)))
        plt.show()
    
    def pred(self, x_arr):
        x_arr = np.array(x_arr).astype(theano.config.floatX)
        self.n_batch.set_value(int(x_arr.shape[0]))
        return self.pred_func(x_arr) 
        #print(self.h.get_value())