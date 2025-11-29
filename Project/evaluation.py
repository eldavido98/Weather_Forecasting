from forecasting import *
from utils import *
from metrics import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

res_params = [128, 28]
u_params = [64, 2]
vit_params = [8, 4, 2, 128]

constants_set = define_sets('const')
train_set = define_sets('train')
validation_set, test_set = define_sets('val_test', val=1, test=1)
latitude_weights = latitude_weighting_function(latitude_coordinates)
forecasters = Forecasting(constants_set, train_set, validation_set,
                          res_params=res_params, u_params=u_params, vit_params=vit_params).to(device)

t_6 = [test_set[12:4374], test_set[6:4368], test_set[0:4362], test_set[18:4380]]
t_24 = [test_set[12:4356], test_set[6:4350], test_set[0:4344], test_set[36:4380]]
t_72 = [test_set[12:4308], test_set[6:4302], test_set[0:4296], test_set[84:4380]]
t_120 = [test_set[12:4260], test_set[6:4254], test_set[0:4248], test_set[132:4380]]
t_240 = [test_set[12:4140], test_set[6:4134], test_set[0:4128], test_set[252:4380]]
t = CustomDataset(t_6, t_24, t_72, t_120, t_240)
test_loader = torch.utils.data.DataLoader(t, batch_size=128, shuffle=True)

forecasters.load()
forecasters.evaluate_forecasters(test_loader, constants_set)
