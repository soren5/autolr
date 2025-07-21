from sge.parameters import params

def trim_phenotype(phenotype):
    #print(params)
    if "shape" in phenotype:
        phenotype = phenotype.replace(", shape=shape, dtype=tf.float32", "")
        phenotype = phenotype.replace("tf.math.", "")
        phenotype = phenotype.replace("tf.", "")
        #print(params['GRAMMAR'])
        if 'deep_architecture_optimizer' in params['GRAMMAR']:
            functions = phenotype.split(r'lambda has_strides, strides, has_kernel_size, kernel_size, has_filters, filters, has_dilation_rate, dilation_rate, has_units, units, has_pool_size, pool_size, layer_count, layer_num, shape, alpha')
        elif 'architecture_layer_type' in params['GRAMMAR']:
            #print("THIS IS A LAYER TYPE GRAMMAR")
            functions = phenotype.split(r'lambda is_dense, units, is_pool, pool_size, is_conv, kernel_size, filters, stride, layer_count, layer_num, shape, alpha')
        elif 'architecture' in params['GRAMMAR']:
            #print("THIS IS AN ARCHITECTURAL GRAMMAR")
            functions = phenotype.split(r'lambda layer_count, layer_num, shape, alpha')
        else:
            #print("THIS IS NOT AN ARCHITECTURAL GRAMMAR")
            functions = phenotype.split(r'lambda shape, alpha')

    elif "size" in phenotype:
        phenotype = phenotype.replace(", size=size, dtype=torch.float32", "")
        phenotype = phenotype.replace("torch.", "")        
        functions = phenotype.split(r'lambda size, alpha')
    
    else:
        raise Exception()

    return functions 

def smart_phenotype(phenotype):
    functions = trim_phenotype(phenotype)
    print(functions)
    try:
        alpha_func_string = functions[1][8:-2]
        beta_func_string = functions[2][14:-2] 
        sigma_func_string =functions[3][21:-2] 
        grad_func_string = functions[-1][21:].replace('alpha', alpha_func_string).replace('beta', beta_func_string).replace('sigma', sigma_func_string)
    except IndexError:
        raise Exception("Error splitting genotype. Check that grammar name matches type of optimizer.")
    return grad_func_string

def dual_task_key(phenotype, it):
    s_phen = smart_phenotype(phenotype)
    if it % 2 == 0:
        task = 'FMNIST/VGG16: '
    else:
        task = 'CIFAR10/MOBILE: '
    task = ''
    return task + s_phen

def single_task_key(phenotype, it):
    s_phen = smart_phenotype(phenotype)
    return s_phen

def readable_phenotype(phenotype):
    functions = trim_phenotype(phenotype)
    print(f'Readable Phenotype Functions: {functions[0]}\n{functions[1]}\n{functions[2]}\n{functions[3]}\n')
    alpha_func_string = functions[1][8:-2]
    beta_func_string = functions[2][14:-2] 
    sigma_func_string = functions[3][21:-2] 
    grad_func_string = functions[-1][21:]
    readable_phenotype_string = f"{alpha_func_string}\n{beta_func_string}\n{sigma_func_string}\n{grad_func_string}\n"

    return readable_phenotype_string