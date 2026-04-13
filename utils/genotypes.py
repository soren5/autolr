def get_adagrad_grad_func_genotype():

    gen_start = [0]
    gen_alpha_expr = [0]
    gen_alpha_func = [0]
    gen_alpha_terminal = [0]
    gen_alpha_const = [0]
    gen_beta_expr = [0]
    gen_beta_func = [0]
    gen_beta_terminal = [0]
    gen_beta_const = [0]
    gen_sigma_expr = [0]
    gen_sigma_func = [0]
    gen_sigma_terminal = [0]
    gen_sigma_const = [0]
    gen_grad_expr = [0,1,0,1,0,0,1,1]
    gen_grad_func = [3,6,7,5]
    gen_grad_terminal = [4,0,4,0]
    gen_grad_const = [0,0]

    genome = [
        #start
        gen_start,
        #alpha expr
        gen_alpha_expr,
        #alpha func
        gen_alpha_func,
        #alpha terminal
        gen_alpha_terminal,
        #alpha const
        gen_alpha_const,
        #beta expr
        gen_beta_expr,
        #beta func
        gen_beta_func,
        #beta terminal
        gen_beta_terminal,
        #beta const
        gen_beta_const,
        #sigma expr
        gen_sigma_expr,
        #sigma func
        gen_sigma_func,
        #sigma terminal
        gen_sigma_terminal,
        #sigma const
        gen_sigma_const,
        #grad expr
        gen_grad_expr,
        #grad func
        gen_grad_func, 
        #grad terminal
        gen_grad_terminal, 
        #grad const
        gen_grad_const, 
        ]
    raise NotImplementedError("This genotype is flawed and should not be used.")

    return genome

def get_momentum_genotype():
    #TESTED
    gen_start = [0]
    gen_alpha_expr = [0,0,0,0,1,1,1,0,1,1]
    gen_alpha_func = [0,1,3,1,3]
    gen_alpha_terminal = [0,0,1,0,2]
    gen_alpha_const = [83,99,27]
    gen_beta_expr = [0]
    gen_beta_func = [0]
    gen_beta_terminal = [0]
    gen_beta_const = [0]
    gen_sigma_expr = [0]
    gen_sigma_func = [0]
    gen_sigma_terminal = [0]
    gen_sigma_const = [0]
    gen_grad_expr = [0,1]
    gen_grad_func = [0]
    gen_grad_terminal = [1]
    gen_grad_const = [0]

    genome = [
        #start
        gen_start,
        #alpha expr
        gen_alpha_expr,
        #alpha func
        gen_alpha_func,
        #alpha terminal
        gen_alpha_terminal,
        #alpha const
        gen_alpha_const,
        #beta expr
        gen_beta_expr,
        #beta func
        gen_beta_func,
        #beta terminal
        gen_beta_terminal,
        #beta const
        gen_beta_const,
        #sigma expr
        gen_sigma_expr,
        #sigma func
        gen_sigma_func,
        #sigma terminal
        gen_sigma_terminal,
        #sigma const
        gen_sigma_const,
        #grad expr
        gen_grad_expr,
        #grad func
        gen_grad_func, 
        #grad terminal
        gen_grad_terminal, 
        #grad const
        gen_grad_const, 
        ]
    raise NotImplementedError("This genotype is flawed and should not be used.")

    return genome

def get_rmsprop_flawed_genotype():
    #TESTED
    gen_start = [0]
    gen_alpha_expr = [0,0,0,0,1,1,1,0,1,0,1]
    gen_alpha_func = [0,2,3,1,3,5]
    gen_alpha_terminal = [0,0,1,0,2]
    gen_alpha_const = [0, 39, 39]
    gen_beta_expr = [0,0,0,0,1,1,1,0,0,1,1,0,0,1,1]
    gen_beta_func = [0,2,3,1,6,3,8,2]
    gen_beta_terminal = [0,0,2,0,3,1,0]
    gen_beta_const = [0,27,16,0]
    gen_sigma_expr = [0]
    gen_sigma_func = [0]
    gen_sigma_terminal = [0]
    gen_sigma_const = [0]
    gen_grad_expr = [1]
    gen_grad_func = [0]
    gen_grad_terminal = [2]
    gen_grad_const = [0]

    genome = [
        #start
        gen_start,
        #alpha expr
        gen_alpha_expr,
        #alpha func
        gen_alpha_func,
        #alpha terminal
        gen_alpha_terminal,
        #alpha const
        gen_alpha_const,
        #beta expr
        gen_beta_expr,
        #beta func
        gen_beta_func,
        #beta terminal
        gen_beta_terminal,
        #beta const
        gen_beta_const,
        #sigma expr
        gen_sigma_expr,
        #sigma func
        gen_sigma_func,
        #sigma terminal
        gen_sigma_terminal,
        #sigma const
        gen_sigma_const,
        #grad expr
        gen_grad_expr,
        #grad func
        gen_grad_func, 
        #grad terminal
        gen_grad_terminal, 
        #grad const
        gen_grad_const, 
        ]
    raise NotImplementedError("This genotype is flawed and should not be used.")

    return genome


def get_rmsprop_genotype():
    #TESTED
    gen_start = [0]
    gen_alpha_expr = [0,0,0,0,1,1,1,0,1,0,1]
    gen_alpha_func = [0,2,3,1,3,5]
    gen_alpha_terminal = [0,0,1,0,2]
    gen_alpha_const = [0, 39, 39]
    gen_beta_expr = [0,0,0,1,0,0,1,1,0,0,1,1]
    gen_beta_func = [0,2,0,6,3,8,2]
    gen_beta_terminal = [2,0,3,1,0]
    gen_beta_const = [16,1]
    gen_sigma_expr = [0]
    gen_sigma_func = [0]
    gen_sigma_terminal = [0]
    gen_sigma_const = [0]
    gen_grad_expr = [1]
    gen_grad_func = [0]
    gen_grad_terminal = [2]
    gen_grad_const = [0]

    genome = [
        #start
        gen_start,
        #alpha expr
        gen_alpha_expr,
        #alpha func
        gen_alpha_func,
        #alpha terminal
        gen_alpha_terminal,
        #alpha const
        gen_alpha_const,
        #beta expr
        gen_beta_expr,
        #beta func
        gen_beta_func,
        #beta terminal
        gen_beta_terminal,
        #beta const
        gen_beta_const,
        #sigma expr
        gen_sigma_expr,
        #sigma func
        gen_sigma_func,
        #sigma terminal
        gen_sigma_terminal,
        #sigma const
        gen_sigma_const,
        #grad expr
        gen_grad_expr,
        #grad func
        gen_grad_func, 
        #grad terminal
        gen_grad_terminal, 
        #grad const
        gen_grad_const, 
        ]

    raise NotImplementedError("This genotype is flawed and should not be used.")
    return genome

def get_adam_genotype(grammar):
    if grammar == 'grammars/basic_optimizer.txt':
        genotype = [
        [0], # start
        [0, 1], # alpha_expr
        [0], # alpha_func
        [0], # alpha_terminal
        [0], # alpha_var_const
        [99], # alpha_const

        [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1], # beta_expr 
        [2, 0, 2, 3, 3, 1], # beta_func
        [0, 0, 0, 0, 0, 1], # beta_terminal
        [2, 0, 2, 0, 0], # beta_var_const
        [61, 99, 61], # beta_const

        [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1], # sigma_expr
        [2, 0, 2, 3, 3, 1, 5], # sigma_func
        [0, 0, 0, 0, 0, 1], # sigma_terminal
        [3, 0, 3, 0, 0], # sigma_var_const
        [84, 99, 84], # sigma_const
        
        [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1], # grad_expr
        [6, 3, 3, 6, 8, 1, 4, 1, 4, 2, 8], # grad_func
        [0, 2, 0, 0, 1, 0, 0, 1, 3, 0], # grad_terminal
        [16, 99, 84, 99, 61, 1]] # grad_const
    elif grammar == 'grammars/deep_architecture_optimizer.txt':
        genotype = [
            [0], # start
            [0, 1], # alpha_expr
            [0], # alpha_func
            [0], # alpha_terminal
            [0], # alpha_var_const
            [], # alpha_arch
            [], # alpha_layer_var
            [99], # alpha_const

            [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1], # beta_expr 
            [2, 0, 2, 3, 3, 1], # beta_func
            [0, 0, 0, 0, 0, 3], # beta_terminal
            [2, 0, 2, 0, 0], # beta_var_const
            [], # beta_arch
            [], # beta_layer_var
            [61, 99, 61], # beta_const

            [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1], # sigma_expr
            [2, 0, 2, 3, 3, 1, 5], # sigma_func
            [0, 0, 0, 0, 0, 3], # sigma_terminal
            [3, 0, 3, 0, 0], # sigma_var_const
            [], # sigma_arch
            [], # sigma_layer_var
            [84, 99, 84], # sigma_const
            
            [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1], # grad_expr
            [6, 3, 3, 6, 8, 1, 4, 1, 4, 2, 8], # grad_func
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # grad_terminal
            [0, 2, 0, 0, 1, 0, 0, 1, 3, 0], # grad_var_const
            [], # grad_arch
            [], # grad_layer_var
            [16, 99, 84, 99, 61, 1] #grad_const
            ]
    return genotype 
