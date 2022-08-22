def smart_phenotype(phenotype):
    if "tf" in phenotype:
        phenotype = phenotype.replace(", shape=shape, dtype=tf.float32", "")
        phenotype = phenotype.replace("tf.math.", "")
        phenotype = phenotype.replace("tf.", "")

        functions = phenotype.split(r'lambda shape,  alpha')
        alpha_func_string = functions[1][8:-2]
        beta_func_string =functions[2][14:-2] 
        sigma_func_string =functions[3][21:-2] 
        grad_func_string = functions[-1][21:].replace('alpha', alpha_func_string).replace('beta', beta_func_string).replace('sigma', sigma_func_string)
    elif "torch" in phenotype:
        phenotype = phenotype.replace(", size=size, dtype=tf.float32", "")
        phenotype = phenotype.replace("torch.", "")

        functions = phenotype.split(r'lambda size, alpha')
        alpha_func_string = functions[1][8:-2]
        beta_func_string =functions[2][14:-2] 
        sigma_func_string =functions[3][21:-2] 
        grad_func_string = functions[-1][21:].replace('alpha', alpha_func_string).replace('beta', beta_func_string).replace('sigma', sigma_func_string)

    return grad_func_string

def readable_phenotype(phenotype):
    phenotype = phenotype.replace(", shape=shape, dtype=tf.float32", "")
    phenotype = phenotype.replace("tf.math.", "")
    phenotype = phenotype.replace("tf.", "")

    functions = phenotype.split(r'lambda shape,  alpha')
    alpha_func_string = "alpha: " + functions[1][8:-2]
    beta_func_string = "beta: " + functions[2][14:-2] 
    sigma_func_string = "sigma: " + functions[3][21:-2] 
    grad_func_string = "grad: " + functions[-1][21:]
    readable_phenotype_string = f"{alpha_func_string}\n{beta_func_string}\n{sigma_func_string}\n{grad_func_string}\n"
    return readable_phenotype_string