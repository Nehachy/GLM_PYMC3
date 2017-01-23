import utils as ut
from pymc3.glm import glm
from pymc3 import Model,sample,summary,NUTS,Metropolis,find_MAP,Normal,Uniform
import pandas
from collections import defaultdict
import ols_sci as ols
import numpy as np

input_file = 'required_input.csv'
sampling = 10000

#reading a data csv and creating data map
data_map,samples,tslen = ut.readCSV(input_file)

model_map = {}

#model configurations
model_type = 'co_loc_ref_i'
model = {
           'dependent':'ty_tran_cnt',
           'independents':['model_ann','cvs_model','lag_ty_tran_cnt','gt_Target_1_period',
			   'gt_Walmart_1_period','gt_Amazon_1_period','wmart_dist_model'
                           'easter_flag','mamorial_flag','independence_flag','wkly_total_prcp_ty','wkly_max_tmp_ty',
                           'event_holiday','event_holiday_ly','period_60','period_12','CPI']
        }

additional_vars ={'tgt_region':"Region",'dma_name':"DMA"}

#for creating store maps: i.e key:store_id with corresponding values
for i in range(samples-1):
   dependent = defaultdict(list)
   independents = defaultdict(list)
   additional_vars_map = {}
   district = None

   for k,v in data_map.items():
      if k == model.get('dependent'):
         dependent[k].append(v[i])

      elif k in model.get('independents'):
         independents[k].append(v[i])

      elif k == model_type:
         district = v[i]

      elif k in additional_vars:
         additional_vars_map[additional_vars.get(k)] = v[i] 


   if district in model_map:
      dep = model_map.get(district).get('dependent')
      ind = model_map.get(district).get('independents')

      for k1,v1, in dependent.items():
        dep[k1].append(v1[0])

      for k1,v1 in independents.items():
        ind[k1].append(v1[0])
   else:
      model_map[district] = {'dependent':dependent,'independents':independents,'addvars':additional_vars_map}


def prepareInitialMap(result_map,json_std):
   init_map = {}
   for k,v in result_map.items():
      std = json_std.get(k)
      init_map[k] = Normal.dist(mu=v,sd=std)

   return init_map   
   

counter = 0

for k,v in model_map.items():
    dist = k
    dependent = v.get('dependent')
    independents = v.get('independents')
    additional = v.get('addvars')

    f = open('result.csv','a')
    if counter == 0:
       f.write('Store,DMA,Region,%s\n'%','.join(model.get('independents')))
    counter = 1

    panda_map = {}

    data_len = 0

    result_list = [str(dist)]


    #adding addition vars
    if len(additional) != 0:
       for k,v in additional.items():
          result_list.append(str(v))
    else:
       result_list.extend(['',''])

    #creating panda dataset
    for k1,v1 in independents.items():
       panda_map[k1] = v1
       data_len = len(v1)

    for k1,v1 in dependent.items():
       panda_map[k1] = v1

    #executing ols
    kwargs = {}
    kwargs['covariates_data'] = independents
    kwargs['dependent_data'] = dependent
    kwargs['intercept'] = False
    kwargs['data_len'] = data_len
   
    json,json_std = ols.execute(**kwargs)

    priors = prepareInitialMap(json,json_std)

    print "Running for store = %s , total samples = %d " %(dist,data_len)

    df = pandas.DataFrame(panda_map)

    #constructiong model expression string used in glm.
    independent_expression = '+'.join(model.get('independents'))
    model_expression = "%s ~ %s" %(model.get('dependent'),independent_expression)
    model_glm = Model()

    with model_glm:
      #calling glm routing
      glm(model_expression,df,priors=priors)
      #with metropolis hasting sampling method
      step = Metropolis()
      #running for 10000 samples
      trace = sample(sampling,step,progressbar=True)

      for var_ in model.get('independents'):
         result_list.append(str(np.mean(trace[var_])))   

      f.write("%s\n"%','.join(result_list))
      
      summary(trace)

    f.close()

