def smart_phenotype(phenotype):
    phenotype = phenotype.replace(", shape=shape, dtype=tf.float32", "")
    phenotype = phenotype.replace("tf.math.", "")
    phenotype = phenotype.replace("tf.", "")

    functions = phenotype.split(r'lambda shape,  alpha')
    alpha_func_string = functions[1][8:-2]
    beta_func_string =functions[2][14:-2] 
    sigma_func_string =functions[3][21:-2] 
    grad_func_string = functions[-1][21:].replace('alpha', alpha_func_string).replace('beta', beta_func_string).replace('sigma', sigma_func_string)
    return grad_func_string