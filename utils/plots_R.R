library(here)
library(ggplot2)
library(ggpubr)
library(tidyverse)
library(viridis)

###Definitions 

plot_dir_name <- function(){
  return(paste(getwd(),"r_plots", sep = "/"))
}

create_and_move_to_plots_subdir <- function(){
  !file.exists(plot_dir_name())
  dir.create(plot_dir_name())
  setwd(plot_dir_name())
  return(plot_dir_name())
}

read_and_save_data <- function(){
  
  ###load the files for the different setups
  FM =  read.csv2("FM/FM_fmni.csv", sep = ",", header = T)
  FMX =  read.csv2("FMX/FMX_fmni.csv", sep = ",", header = T)
  OM =  read.csv2("OM/OM_fmni.csv", sep = ",", header = T)
  OMX =  read.csv2("OMX/OMX_fmni.csv", sep = ",", header = T)
  
  #bind the dataframes in a unique dataframe
  all_data = rbind(FM,FMX,OM,OMX)
  
  create_and_move_to_plots_subdir()
  
  #Save
  saveRDS(all_data, "all_data.Rds")
}

### Make boxplots for unique phenotypes above accuracy
maxe_unique_t_test_boxplot <- function(all_data, accuracy = 0){
  
  my_comparisons <- list( c("OM", "OMX"), c("FMX", "OM"), c("FM", "FMX"), c("FMX", "OMX"), c("FM", "OMX"))
  p <- ggplot(all_data %>% 
                filter(fitness < -accuracy) %>% 
                group_by(setup, run) %>% 
                distinct(smart_phenotype, .keep_all = TRUE)%>% 
                summarise(n_unique = n()),
              aes(x = as.factor(setup),
                  y = n_unique,
                  fill = setup
              )
  ) + 
    geom_violin() + 
    geom_boxplot(width = 0.1) +
    geom_point() +
    stat_compare_means(comparisons = my_comparisons)+ # Add pairwise comparisons p-value
    scale_fill_viridis(discrete = T) +
    xlab("setup")+
    ylab(paste("number of unique phenotypes above ", accuracy, " fitness per run", sep = "")) +
    theme_bw()
  
  print(p)
  ggsave(paste("number of unique phenotypes above ", accuracy, " fitness per run across setups.jpg", sep = ""))
  ggsave(paste("number of unique phenotypes above ", accuracy, " fitness per run across setups.pdf", sep = ""))
}

###Plot fitness over time
fitness_over_time_across_setups <- function(all_data){
  p <- ggplot(all_data%>% 
                group_by(setup, run, iteration) %>% 
                summarise(avg_fit = mean(fitness)),
              aes(x = iteration, 
              y = avg_fit,
              color = setup)
             )+
    # geom_line() +
    geom_smooth() + 
    # scale_color_viridis(option ="magma", discrete = T) +
    # geom_line(aes(group = interaction(setup, run))) + 
    scale_color_viridis(discrete = T) +
    ylim(c(0,-1))+
    xlab("generations")+
    ylab(paste("average fitness", sep = "")) +
    theme_bw()
  
  print(p)
  ggsave("fitness over time per run across setups.jpg")
  ggsave("fitness over time per run across setups.pdf")
}

###Plot final fitness 
make_fitness_t_test_boxplot <- function(all_data){
  
  my_comparisons <- list( c("OM", "OMX"), c("FMX", "OM"), c("FM", "FMX"), c("FMX", "OMX"), c("FM", "OMX"))
  p <- ggplot(all_data %>% 
                group_by(setup, run) %>% 
                filter(iteration == max(iteration)) %>% 
                summarise(avg_final_fitness = mean(fitness)),
              aes(x = as.factor(setup),
                  y = avg_final_fitness,
                  fill = setup
              )
  ) + 
    geom_violin() + 
    geom_boxplot(width = 0.1) +
    geom_point() +
    stat_compare_means(comparisons = my_comparisons)+ # Add pairwise comparisons p-value
    scale_fill_viridis(discrete = T) +
    xlab("setup")+
    ylab(paste("average fitness in last generation", sep = "")) +
    theme_bw()
  
  print(p)
  ggsave(paste("average fitness in last generation across setups.jpg", sep = ""))
  ggsave(paste("average fitness in last generation across setups.pdf", sep = ""))
}

###Plot unique behaviours over time
unique_behaviours_over_time_across_setups <- function(all_data, accuracy){
  p <- ggplot(all_data %>% 
                filter(fitness < -accuracy) %>% 
                group_by(setup, run, iteration) %>% 
                distinct(smart_phenotype, .keep_all = TRUE) %>% 
                summarise(uniques = n()),
              aes(x = iteration, 
                  y = uniques,
                  color = setup)
  )+
    # geom_line() +
    geom_smooth() + 
    # scale_color_viridis(option ="magma", discrete = T) +
    # geom_line(aes(group = interaction(setup, run))) + 
    scale_color_viridis(discrete = T) +
    xlab("generations")+
    ylab(paste("number of unique solutions", sep = "")) +
    theme_bw()
  
  print(p)
  ggsave(paste("number of unique solutions with fitness above ", accuracy, " over time per run across setups.jpg"))
  ggsave(paste("number of unique solutions with fitness above ", accuracy, " over time per run across setups.pdf"))
}

###Plot cumulative unique behaviours over time
cumulative_unique_behaviours_over_time_across_setups_per_run <- function(all_data, accuracy){
  p <- ggplot(all_data %>% 
                filter(fitness < -accuracy) %>% 
                group_by(setup, run, iteration) %>% 
                distinct(smart_phenotype, .keep_all = TRUE) %>% 
                summarise(uniques = n()) %>% 
                arrange(iteration) %>% 
                mutate(sum_uniques = cumsum(uniques))
                ,
              aes(x = iteration, 
                  y = sum_uniques,
                  color = setup)
  )+
    # geom_line() +
    geom_smooth() + 
    # scale_color_viridis(option ="magma", discrete = T) +
    # geom_line(aes(group = interaction(setup, run))) + 
    scale_color_viridis(discrete = T) +
    xlab("generations")+
    ylab(paste("cumulative number of unique solutions", sep = "")) +
    theme_bw()
  
  print(p)
  ggsave(paste("number of cumulative unique solutions with fitness above ", accuracy, " over time per run across setups.jpg"))
  ggsave(paste("number of cumulative unique solutions with fitness above ", accuracy, " over time per run across setups.pdf"))
}


###Plot cumulative unique behaviours over time
cumulative_unique_behaviours_over_time_across_setup <- function(all_data, accuracy){
  p <- ggplot(all_data %>% 
                filter(fitness < -accuracy) %>% 
                group_by(setup, iteration) %>% 
                distinct(smart_phenotype, .keep_all = TRUE) %>% 
                summarise(uniques = n()) %>% 
                arrange(iteration) %>% 
                mutate(sum_uniques = cumsum(uniques))
                ,
              aes(x = iteration, 
                  y = sum_uniques,
                  color = setup)
  )+
    # geom_line() +
    geom_smooth() + 
    # scale_color_viridis(option ="magma", discrete = T) +
    # geom_line(aes(group = interaction(setup, run))) + 
    scale_color_viridis(discrete = T) +
    xlab("generations")+
    ylab(paste("cumulative number of unique solutions", sep = "")) +
    theme_bw()
  
  print(p)
  ggsave(paste("number of cumulative unique solutions with fitness above ", accuracy, " over time per run across setups.jpg"))
  ggsave(paste("number of cumulative unique solutions with fitness above ", accuracy, " over time per run across setups.pdf"))
}


####Either read data from csv if not done yet or load from Rds
read_or_load <- function(){
  
  setwd(paste(dirname(rstudioapi::getSourceEditorContext()$path),"many_results/", sep = '/'))
  if(!file.exists(plot_dir_name())){
    read_and_save_data()
  }
    # Load
    setwd(plot_dir_name())
    all_data = readRDS("all_data.Rds") %>% mutate(fitness = as.numeric(fitness))
    return(all_data)
   
}
#######Running code

all_data = read_or_load()

maxe_unique_t_test_boxplot(all_data, accuracy = 0.8)  
maxe_unique_t_test_boxplot(all_data, accuracy = 0.5)  
maxe_unique_t_test_boxplot(all_data, accuracy = 0.2)  
maxe_unique_t_test_boxplot(all_data, accuracy = 0.1)  

unique_behaviours_over_time_across_setups(all_data, accuracy = 0.8)
unique_behaviours_over_time_across_setups(all_data, accuracy = 0.5)
unique_behaviours_over_time_across_setups(all_data, accuracy = 0.2)
unique_behaviours_over_time_across_setups(all_data, accuracy = 0.1)

cumulative_unique_behaviours_over_time_across_setups(all_data, accuracy = 0.8)
cumulative_unique_behaviours_over_time_across_setups(all_data, accuracy = 0.5)
cumulative_unique_behaviours_over_time_across_setups(all_data, accuracy = 0.2)
cumulative_unique_behaviours_over_time_across_setups(all_data, accuracy = 0.1)

fitness_over_time_across_setups(all_data)
make_fitness_t_test_boxplot(all_data)
