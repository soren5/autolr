def trim_phenotype(phenotype):
    if "shape" in phenotype:
        phenotype = phenotype.replace(", shape=shape, dtype=tf.float32", "")
        phenotype = phenotype.replace("tf.math.", "")
        phenotype = phenotype.replace("tf.", "")
        functions = phenotype.split(r'lambda shape, alpha')
        functions = phenotype.split(r'lambda layer_count, layer_num, shape, alpha')

    elif "size" in phenotype:
        phenotype = phenotype.replace(", size=size, dtype=torch.float32", "")
        phenotype = phenotype.replace("torch.", "")        
        functions = phenotype.split(r'lambda size, alpha')
    
    else:
        raise Exception()

    return functions 

def smart_phenotype(phenotype):
    functions = trim_phenotype(phenotype)
    alpha_func_string = functions[1][8:-2]
    beta_func_string = functions[2][14:-2] 
    sigma_func_string =functions[3][21:-2] 
    grad_func_string = functions[-1][21:].replace('alpha', alpha_func_string).replace('beta', beta_func_string).replace('sigma', sigma_func_string)

    return grad_func_string

def readable_phenotype(phenotype):
    functions = trim_phenotype(phenotype)

    alpha_func_string = functions[1][8:-2]
    beta_func_string =functions[2][14:-2] 
    sigma_func_string =functions[3][21:-2] 
    grad_func_string = functions[-1][21:]
    readable_phenotype_string = f"{alpha_func_string}\n{beta_func_string}\n{sigma_func_string}\n{grad_func_string}\n"

    return readable_phenotype_string