In these experiments, I tried adam on multi task fitness and could not get it to work well across the board:
These were the parameters:
optimizer = Adam(learning_rate=2.0*1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=False)
