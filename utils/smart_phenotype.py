import re

def remove_scientific_notation_from_end(text):
    """
    Remove 3 comma-separated float values in scientific notation from the end of a string.
    
    Args:
        text (str): Input string that may end with scientific notation values
    
    Returns:
        str: String with the scientific notation values removed from the end
    """
    # Pattern to match 3 comma-separated scientific notation floats at the end
    pattern = r'[,\s]*[+-]?(?:\d+\.?\d*[eE][+-]?\d+|\d*\.\d+|\d+\.\d*)[,\s]*[+-]?(?:\d+\.?\d*[eE][+-]?\d+|\d*\.\d+|\d+\.\d*)[,\s]*[+-]?(?:\d+\.?\d*[eE][+-]?\d+|\d*\.\d+|\d+\.\d*)\s*$'    
    
    # Remove the pattern if found at the end
    result = re.sub(pattern, '', text)
    
    return result.strip()

def get_optimizer_type(phenotype):
    signature = phenotype.split('alpha,')[0]
    if 'has_strides' in signature:
        return 'deep_architecture_optimizer_with_ifs'
    elif 'momentum' in signature:
        return 'deep_architecture_optimizer_with_aggregators'
    elif 'strides' in signature:
        return "deep_architecture_optimizer"
    elif 'is_dense' in signature:
        return 'architecture_layer_type_optimizer'
    elif 'layer_count' in signature:
        return 'basic_architecture_optimizer'
    elif '  ' in signature:
        # This is not a real optimizer type, but an old grammar has a double space typo
        return 'basic_optimizer_double_space'
    elif 'shape' in signature:
        return 'basic_optimizer'
    elif 'size' in signature:
        return 'pytorch_optimizer'
    else:
        raise Exception(f"Unknown optimizer type in get_optimizer_type for {phenotype}")
        

def trim_phenotype(phenotype, debug=False):
    optimizer_type = get_optimizer_type(phenotype)
    if optimizer_type == 'pytorch_optimizer':
        phenotype = phenotype.replace(", size=size, dtype=torch.float32", "")
        phenotype = phenotype.replace("torch.", "")        
        functions = phenotype.split(r'lambda size, alpha')
    else:
        phenotype = phenotype.replace(", shape=shape, dtype=tf.float32", "")
        phenotype = phenotype.replace("tf.math.", "")
        phenotype = phenotype.replace("tf.", "")

        if optimizer_type == 'deep_architecture_optimizer_with_ifs':
                functions = phenotype.split(r'lambda has_strides, strides, has_kernel_size, kernel_size, has_filters, filters, has_dilation_rate, dilation_rate, has_units, units, has_pool_size, pool_size, layer_count, layer_num, shape, alpha')
        elif optimizer_type == 'deep_architecture_optimizer_with_aggregators':
                functions = phenotype.split(r'lambda momentum, variance, layer_wise_lr, strides, kernel_size, filters, dilation_rate, units, pool_size, layer_count, layer_num, shape, alpha')
                functions[-1] = remove_scientific_notation_from_end(functions[-1])
        elif optimizer_type == 'deep_architecture_optimizer':
                functions = phenotype.split(r'lambda strides, kernel_size, filters, dilation_rate, units, pool_size, layer_count, layer_num, shape, alpha')
        elif optimizer_type == 'architecture_layer_type_optimizer':
            functions = phenotype.split(r'lambda is_dense, units, is_pool, pool_size, is_conv, kernel_size, filters, stride, layer_count, layer_num, shape, alpha')
        elif optimizer_type == 'basic_architecture_optimizer':
            functions = phenotype.split(r'lambda layer_count, layer_num, shape, alpha')
        elif optimizer_type == 'basic_optimizer':
            functions = phenotype.split(r'lambda shape, alpha')
        elif optimizer_type == 'basic_optimizer_double_space':
            functions = phenotype.split(r'lambda shape,  alpha')
        else:
            raise Exception(f"Unknown optimizer type in trim_phenotype for {phenotype}")
    return functions, optimizer_type 

def smart_phenotype(phenotype, debug=False):
    functions, optimizer_type = trim_phenotype(phenotype, debug=debug)
    if debug:
        print(functions)
    try:
        alpha_func_string = functions[1][8:-2]
        beta_func_string = functions[2][14:-2].replace('alpha', alpha_func_string)
        sigma_func_string =functions[3][21:-2].replace('alpha', alpha_func_string).replace('beta', beta_func_string)
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

def readable_phenotype(phenotype, debug=False, full_return=False):
    functions, optimizer_type = trim_phenotype(phenotype)
    if debug:
        print(f'Readable Phenotype Functions: {functions[0]}\n{functions[1]}\n{functions[2]}\n{functions[3]}\n')
    alpha_func_string = functions[1][8:-2]
    beta_func_string = functions[2][14:-2] 
    sigma_func_string = functions[3][21:-2] 
    grad_func_string = functions[-1][21:]

    readable_phenotype_string = 'weights = weights - ' + grad_func_string + '\n'
    if 'sigma' in grad_func_string:
        readable_phenotype_string = 'sigma = sigma - ' + sigma_func_string + '\n' + readable_phenotype_string
    if 'beta' in readable_phenotype_string:
        readable_phenotype_string = 'beta = beta - ' + beta_func_string + '\n' + readable_phenotype_string
    if 'alpha' in readable_phenotype_string:
        readable_phenotype_string = 'alpha = alpha - ' + alpha_func_string + '\n' + readable_phenotype_string

    if full_return:
        return readable_phenotype_string, alpha_func_string, beta_func_string, sigma_func_string, grad_func_string
    else:
        return readable_phenotype_string

def advanced_readable_phenotype(phenotype, debug=False):
    readable_phen, alpha_func_string, beta_func_string, sigma_func_string, grad_func_string = readable_phenotype(phenotype, debug=debug, full_return=True)
    """
    redundant_patterns = {
        'alpha': ['alpha = alpha - add(alpha, grad)'],
        'sigma': ['sigma = sigma - add(sigma, grad)'],
        'beta': ['beta = beta - add(beta, grad)'],
    }
    for param, patterns in redundant_patterns.items():
        for pattern in patterns:
            if pattern in readable_phen:
                #print("Starting to remove redundant pattern.")
                #print(f"Before: {readable_phen}")
                readable_phen = readable_phen.replace(pattern, '')
                readable_phen = readable_phen.replace(param, 'grad')
                readable_phen = readable_phen.strip()
                #print(f"After: {readable_phen}")
    """ 
    return readable_phen