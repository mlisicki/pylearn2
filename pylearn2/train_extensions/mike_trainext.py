import numpy as np

class AugmentAndBalance(object):
    #def __init__(self)
    def setup(self, model, dataset, algorithm):
        X = dataset.get_topological_view()
        y = dataset.get_targets()
        I = X[0,:]
        S = generateTransformations(I,type)
        augSize=S.shape[3]+1
        
        #if(inclOrig):
            #dataAug.data=np.zeros(dataset.shape[0],augSize,dataset.shape[1],dataset.shape[2],dtype='uint8')

        #for j in range(0,dataset.shape[0]):
            #I = reshape(dataset[j,:,:],np.round(np.sqrt(dataset.shape[1])),
                           #np.round(np.sqrt(dataset.shape[1])),dataset.shape[2])
            #S = generateTransformations(I,type,augSize-1)
            #tempData = np.zeros(augSize,dataset.shape[1],dataset.shape[2],dtype='uint8')
            #tempData[1,:,:]=dataset[j,:,:]
            #tempData[2:augSize,:,:]=reshape(permute(S,[4,1,2,3]),augSize-1,-1,dataset.shape[2])
            #dataset[j,:,:,:] = tempData

        #augSize=dataset.shape[1]
        #dataAug.data = reshape(dataAug.data,size(dataStruct.data,1)*augSize,[],size(dataStruct.data,3))
        #dataAug.labels = reshape(repmat(dataStruct.labels,1,augSize).T(),1,-1).T()
        
    #def generateTransformations(img, type, sSize):
        #shiftfrac=10; % shift fraction used for circshifting
        #if(nargin>2)
            #S = zeros(size(img,1),size(img,2),size(img,3),sSize,'uint8'); % initialize for speed
        #end
        #img2 = zeros(size(img));
        
        #sigma=0.15;
        #[V,D]=eig(cov(double(squeeze(reshape(img,size(img,1)*size(img,1),size(img,3)))./255)));
        
        #for j=1:size(img,3); img2(:,:,j)=fliplr(img(:,:,j)); end;
        #S(:,:,:,1)=uint8(img2);
        
        #for j=1:size(img,3); img2(:,:,j)=flipud(img(:,:,j)); end;
        #S(:,:,:,2)=img2;
        
        #for j=1:size(img,3); img2(:,:,j)=rot90(img(:,:,j),1); end;
        #S(:,:,:,3)=img2;
        
        #for j=1:size(img,3); img2(:,:,j)=rot90(img(:,:,j),-1); end;
        #S(:,:,:,4)=img2;
        
        #for j=1:size(img,3); img2(:,:,j)=rot90(img(:,:,j),2); end;
        #S(:,:,:,5)=img2;
        
        #for j=1:size(img,3); img2(:,:,j)=circshift(img(:,:,j),[round(size(img,1)/shiftfrac) round(size(img,1)/shiftfrac)]); end;
        #S(:,:,:,6)=img2;
        
        #for j=1:size(img,3); img2(:,:,j)=circshift(img(:,:,j),[-round(size(img,1)/shiftfrac) round(size(img,1)/shiftfrac)]); end;
        #S(:,:,:,7)=img2;
        
        #for j=1:size(img,3); img2(:,:,j)=circshift(img(:,:,j),[round(size(img,1)/shiftfrac) -round(size(img,1)/shiftfrac)]); end;
        #S(:,:,:,8)=img2;
        
        #for j=1:size(img,3); img2(:,:,j)=circshift(img(:,:,j),[-round(size(img,1)/shiftfrac) -round(size(img,1)/shiftfrac)]); end;
        #S(:,:,:,9)=img2;
        
        #%l = randperm(9,3)-1;
        #for k=0:8
            #for j=0:0
                #v(1,1,:)=uint8(255*sum(V*(D.*(sigma*randn(size(D)))),2));
                #S(:,:,:,k*1+j+10)=uint8(S(:,:,:,k+1)+repmat(v,size(img,1),size(img,2),1));
            #end
        #end    
        
    def on_monitor(self, model, dataset, algorithm):
        model = None
        dataset = None
        algorithm = None

        #self.randomize_datasets(self._randomize)             