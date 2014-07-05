from pylearn2.datasets import preprocessing
import numpy as np

class Augment(object):
    def apply(self, dataset, can_fit=False):        
        X = dataset.get_topological_view()
        y = dataset.get_targets()
        
        if(dataset.view_converter.axes==['c', 0, 1, 'b']):
            X=X.swapaxes(0,3)
        
        S = self.generateTransformations(X[0,:])
        augSize=S.shape[3]+1
        
        dataAug=np.zeros((X.shape[0],augSize,X.shape[1],X.shape[2],X.shape[3]),dtype='uint8')

        for j in range(0,X.shape[0]):
            S = self.generateTransformations(X[j,:],augSize-1)
            tempData = np.zeros((augSize,X.shape[1],X.shape[2],X.shape[3]),dtype='uint8')
            tempData[0,:]=X[j,:]
            tempData[1:augSize,:,:]=np.rollaxis(S,-1)
            dataAug[j,:,:,:,:] = tempData

        if(y.ndim==1):
            dataset.y = np.tile(y[:,np.newaxis],[1,dataAug.shape[1]]).reshape((-1,1)).squeeze()
        else:
            dataset.y = np.tile(y[:,np.newaxis,:],[1,dataAug.shape[1],1]).reshape((-1,y.shape[1]))        
        dataAug = dataAug.reshape(dataAug.shape[0]*dataAug.shape[1],dataAug.shape[2],dataAug.shape[3],dataAug.shape[4])
        if(dataset.view_converter.axes==['c', 0, 1, 'b']):
            dataAug=dataAug.swapaxes(0,3)        
        dataset.set_topological_view(dataAug, axes=dataset.view_converter.axes)

    def generateTransformations(self, img, sSize=0):
        shiftfrac=0.1; # shift fraction used for circshifting
        if(sSize is not 0):
            S = np.zeros((img.shape[0],img.shape[1],img.shape[2],sSize)) # initialize for speed
        else:
            S = np.array([]).reshape(img.shape[0],img.shape[1],img.shape[2],0)
        img2 = np.zeros(img.shape)
        
        #sigma=0.15;
        #[V,D]=eig(cov(double(squeeze(reshape(img,size(img,1)*size(img,1),size(img,3)))./255)));
        
        #for j in range(0,img.shape[2]):
            #img2[:,:,j]=np.fliplr(img[:,:,j])
        #if(sSize is not 0):
            #S[:,:,:,0]=np.uint8(img2)
        #else:
            #S=np.concatenate((S,np.uint8(img2).reshape(img2.shape+(1,))),axis=3)
        
        #for j in range(0,img.shape[2]): 
            #img2[:,:,j]=np.flipud(img[:,:,j])
        #if(sSize is not 0):
            #S[:,:,:,1]=np.uint8(img2)
        #else:
            #S=np.concatenate((S,np.uint8(img2).reshape(img2.shape+(1,))),axis=3)
            
        for j in range(0,img.shape[2]): 
            img2[:,:,j]=np.rot90(img[:,:,j],1)
        if(sSize is not 0):
            S[:,:,:,0]=img2
        else:
            S=np.concatenate((S,img2.reshape(img2.shape+(1,))),axis=3)
            
        for j in range(0,img.shape[2]):
            img2[:,:,j]=np.rot90(img[:,:,j],-1)
        if(sSize is not 0):
            S[:,:,:,1]=img2
        else:
            S=np.concatenate((S,img2.reshape(img2.shape+(1,))),axis=3)            

        for j in range(0,img.shape[2]):
            img2[:,:,j]=np.rot90(img[:,:,j],2)
        if(sSize is not 0):
            S[:,:,:,2]=img2
        else:
            S=np.concatenate((S,img2.reshape(img2.shape+(1,))),axis=3)
        
        for j in range(0,img.shape[2]):
            img2[:,:,j]=np.roll(img[:,:,j],np.int64(img.shape[0]*shiftfrac),axis=0)
            img2[:,:,j]=np.roll(img2[:,:,j],np.int64(img.shape[1]*shiftfrac),axis=1)
        if(sSize is not 0):
            S[:,:,:,3]=img2
        else:
            S=np.concatenate((S,img2.reshape(img2.shape+(1,))),axis=3)
            
        for j in range(0,img.shape[2]):
            img2[:,:,j]=np.roll(img[:,:,j],-np.int64(img.shape[0]*shiftfrac),axis=0)
            img2[:,:,j]=np.roll(img2[:,:,j],np.int64(img.shape[1]*shiftfrac),axis=1)
        if(sSize is not 0):
            S[:,:,:,4]=img2
        else:
            S=np.concatenate((S,img2.reshape(img2.shape+(1,))),axis=3)      

        for j in range(0,img.shape[2]):
            img2[:,:,j]=np.roll(img[:,:,j],np.int64(img.shape[0]*shiftfrac),axis=0)
            img2[:,:,j]=np.roll(img2[:,:,j],-np.int64(img.shape[1]*shiftfrac),axis=1)
        if(sSize is not 0):
            S[:,:,:,5]=img2
        else:
            S=np.concatenate((S,img2.reshape(img2.shape+(1,))),axis=3)          

        for j in range(0,img.shape[2]):
            img2[:,:,j]=np.roll(img[:,:,j],-np.int64(img.shape[0]*shiftfrac),axis=0)
            img2[:,:,j]=np.roll(img2[:,:,j],-np.int64(img.shape[1]*shiftfrac),axis=1)
        if(sSize is not 0):
            S[:,:,:,6]=img2
        else:
            S=np.concatenate((S,img2.reshape(img2.shape+(1,))),axis=3)      
        
        return S
        
        #l = randperm(9,3)-1;
        #for k=0:8
        #    for j=0:0
        #        v(1,1,:)=uint8(255*sum(V*(D.*(sigma*randn(size(D)))),2));
        #        S(:,:,:,k*1+j+10)=uint8(S(:,:,:,k+1)+repmat(v,size(img,1),size(img,2),1));
        #    end
        #end            
        
class AugmentAndBalance(Augment):
    def apply(self, dataset, factor=1.0, can_fit=False):
        print "Augmenting and balancing the data..."
        # factor = specifies where to cut off the largest training class (in its percent),
        # then it augments all the other sets to fit its size
        X = dataset.get_topological_view()
        y = dataset.get_targets()
        one_hot=0
        
        if(y.ndim!=1):
            one_hot=1
            y=y.argmax(axis=1)
        
        #import tools
        #tools.showData(X[0:5,:].reshape((5,1024,3)),10)
        classBatchSize=np.zeros(y.max()+1)
        for l in xrange(y.max()+1):
            classBatchSize[l]=y[y==l].shape[-1]
        
        if(dataset.view_converter.axes==['c', 0, 1, 'b']):
            X=X.swapaxes(0,3)

        X1=np.ones((y.max()+1,classBatchSize[classBatchSize.argmax()],X.shape[1],X.shape[2],X.shape[3]))
        for l in xrange(y.max()+1):
            X1[l,0:classBatchSize[l],:]=X[y==l,:]
        
        S = self.generateTransformations(X[0,:])
        augSize=S.shape[3]
        X=None

        for l in xrange(y.max()+1):
            dataAug=np.ones((np.round((X1.shape[1]-classBatchSize[l])/np.float(augSize))+1,augSize,X1.shape[2],X1.shape[3],X1.shape[4]))
            assert dataAug.shape[0] <= classBatchSize[l]
            for j in xrange(dataAug.shape[0]):
                S = self.generateTransformations(X1[l,j,:],augSize)
                dataAug[j,:,:,:,:] = np.rollaxis(S,-1)[np.newaxis,:]
            dataAug = dataAug.reshape(dataAug.shape[0]*dataAug.shape[1],dataAug.shape[2],dataAug.shape[3],dataAug.shape[4])
            X1[l,classBatchSize[l]::,:]=dataAug[0:X1.shape[1]-classBatchSize[l],:]

        y1 = np.array(range(y.max()+1))
        y1 = np.tile(y1[np.newaxis,:],[X1.shape[1],1]).T.reshape((1,-1)).squeeze()
        X1 = X1.reshape((X1.shape[0]*X1.shape[1],X1.shape[2],X1.shape[3],X1.shape[4]))
        
        if(one_hot):
            y = np.zeros((y1.shape[0],y.max()+1),dtype='float32')
            for i in xrange(y1.shape[0]):
                y[i,y1[i]] = 1.
            dataset.y=y
        else:
            dataset.y=y1

        print "Size after augmentation: {0}".format(X1.shape)
        
        if(dataset.view_converter.axes==['c', 0, 1, 'b']):
            X1=X1.swapaxes(0,3)
        dataset.set_topological_view(X1, axes=dataset.view_converter.axes)
