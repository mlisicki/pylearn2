from pylearn2.utils import serial
from pylearn2.datasets import mike_preprocessing
from pylearn2.datasets import preprocessing
from pylearn2.utils import string_utils
from pylearn2.datasets.mat_data import MATDATA
from pylearn2.datasets.mat_data import MATDATAPyTables
import datetime

data_dir = string_utils.preprocess('${PYLEARN2_DATA_PATH}/wbc')

output_dir = data_dir + '/augmented0407'
#serial.mkdir( output_dir )

preprocessor1 = preprocessing.Standardize(global_std=True)
#preprocessor2 = mike_preprocessing.AugmentAndBalance()
#preprocessor3 = preprocessing.ShuffleAndSplit(seed=datetime.datetime.now().microsecond, start=0, stop = 11000)

#train = MATDATA(path=data_dir+'/classData19062014PythonMyeloidsOnlyTrain.mat',
                #which_set = 'full', step=2, one_hot=1)
#train.apply_preprocessor(preprocessor = preprocessor1, can_fit = True)
#train.apply_preprocessor(preprocessor = preprocessor2)
#train.apply_preprocessor(preprocessor = preprocessor3)
#train.use_design_loc(output_dir+'/train.npy')
#serial.save(output_dir + '/train.pkl', train)

train = MATDATAPyTables(path=data_dir+'/classData30062014PythonMyeloidsOnlyTrainPT.mat', center=True, rescale=True,start=0,stop=256)
test = MATDATAPyTables(path=data_dir+'/classData30062014PythonMyeloidsOnlyTestPT.mat', center=True, rescale=True)
test.apply_preprocessor(preprocessor = preprocessor1, can_fit = False)
test.use_design_loc(output_dir+'/test.npy')
serial.save(output_dir+'/test.pkl', test)
print "Done."

#serial.save(output_dir + '/preprocessor.pkl',preprocessor)

