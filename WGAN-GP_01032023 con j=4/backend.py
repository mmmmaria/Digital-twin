
import pandas as pd
import numpy as np

def export_excel(array,file_name):
    param_ex = pd.DataFrame(array)
    filepath = file_name+'.xlsx'
    param_ex.to_excel(filepath,index=False)


def import_excel(file_name):
    x = pd.read_excel(file_name+'.xlsx')
    x = x.iloc[:,:].values
    return x