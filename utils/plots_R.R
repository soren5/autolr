library(directlabels)
library(here)
library(ggforce)
library(ggplot2)
library(ggpubr)
library(readr)
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

read_and_save_all_data <- function(){
  
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

read_and_save_post_hoc <- function(){

  post_hoc = read.csv2("best_fitnesses_post_hoc.csv", sep = ",")
  create_and_move_to_plots_subdir()
  
  #Save
  saveRDS(post_hoc, "post_hoc.Rds")
}

### Make boxplots for unique phenotypes above accuracy
maxe_unique_t_test_boxplot <- function(all_data, accuracy = 0.5){
  
  my_comparisons <- list( c("OM", "OMX"), c("OM", "FM"), c("FM", "OMX"), c("FM", "FMX"), c("FMX", "OM"), c("FMX", "OMX"))
  p <- all_data %>% 
    filter(fitness < -accuracy) %>% 
    group_by(setup, run) %>% 
    distinct(smart_phenotype, .keep_all = TRUE)%>% 
    summarise(n_unique = n()) %>% 
    ggplot(aes(x = as.factor(setup),
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

### Make boxplots for unique phenotypes above accuracy
maxe_unique_t_test_wiht_grad_boxplot <- function(all_data, accuracy){
  
  my_comparisons <- list( c("OM", "OMX"), c("OM", "FM"), c("FM", "OMX"), c("FM", "FMX"), c("FMX", "OM"), c("FMX", "OMX"))
  p <- all_data %>% 
    mutate(evaluation_type = ifelse(str_detect(smart_phenotype, "grad"),
                                    ifelse(fitness < -accuracy,
                                           paste("fitness above: ", -accuracy, sep = ""),
                                           "grad"),
                                    NA
    )) %>% 
    filter(!is.na(evaluation_type)) %>% 
    group_by(setup, run) %>% 
    distinct(smart_phenotype, .keep_all = TRUE) %>% 
    summarise(has_grad = n(),
              above_accuracy = sum(evaluation_type !=  "grad")) %>% 
    pivot_longer(c(has_grad, above_accuracy), values_to = "count", names_to = "evaluation_type") %>% 
    mutate(evaluation_type = ifelse(evaluation_type == "above_accuracy", 
                                    paste("above ", accuracy, " accuracy", sep =""),
                                    "using grad()")) %>% 
    ggplot(aes(x = as.factor(setup),
               y = count,
               fill = setup
    )
    ) + 
    geom_violin() + 
    geom_boxplot(width = 0.1) +
    geom_point() +
    stat_compare_means(comparisons = my_comparisons)+ # Add pairwise comparisons p-value
    facet_grid(. ~ evaluation_type, scales = "free") + 
    scale_fill_viridis(discrete = T) +
    xlab("setup")+
    ylab(paste("number of unique phenotypes above ", accuracy, " fitness per run", sep = "")) +
    theme_bw()
  
  print(p)
  ggsave(paste("number of evaluations per run across setups.jpg", sep = ""))
  ggsave(paste("number of evaluations per run across setups.pdf", sep = ""))
}

###Plot fitness over time
fitness_over_time_across_setups <- function(all_data){
  p <- ggplot(all_data%>% 
                group_by(setup, run, iteration) %>% 
                summarise(avg_fit = -1 * mean(fitness)),
              aes(x = iteration, 
                  y = avg_fit,
                  color = setup,
                  fill = setup)
  )+
    geom_smooth() + 
    geom_line(linewidth = 0.1,
              linetype = "dotted",
              aes(group = interaction(setup, run))) +
    scale_color_viridis(discrete = T) +
    scale_fill_viridis(discrete = T) +
    ylim(c(0, 1))+
    xlab("generations")+
    ylab(paste("average fitness", sep = "")) +
    theme_bw()
  
  print(p)
  ggsave("fitness over time per run across setups.jpg")
  ggsave("fitness over time per run across setups.pdf")
}
###Plot fitness of best solutions over time
fitness_best_over_time_across_setups <- function(all_data){
  p <- all_data%>% 
    group_by(setup, run, iteration) %>% 
    summarise(best_fit = max(-fitness)) %>% 
    filter(max(best_fit) == best_fit[max(iteration)]) %>% 
    ggplot(aes(x = iteration, 
               y = best_fit,
               color = setup,
               fill = setup)
    )+
    # geom_line(linewidth = 0.1,
    #           linetype = "dotted",
    #           aes(group = interaction(setup, run))) +
    geom_smooth(method = "loess", span = 0.4) +
    scale_color_viridis(discrete = T) +
    scale_fill_viridis(discrete = T) +
    ylim(c(0, 1))+
    xlab("generations")+
    ylab(paste("average fitness", sep = "")) +
    theme_bw()
  
  print(p)
  ggsave("fitness of best solutions over time per run across setups.jpg")
  ggsave("fitness of best solutions over time per run across setups.pdf")
}
fitness_best_over_time_across_setups(all_data )
###Plot final best fitness 
make_fitness_t_test_boxplot = function(all_data){
  
  my_comparisons <- list( c("OM", "OMX"), c("OM", "FM"), c("FM", "OMX"), c("FM", "FMX"), c("FMX", "OM"), c("FMX", "OMX"))
  p <- ggplot(all_data %>% 
                group_by(setup, run) %>% 
                filter(iteration == max(iteration)) %>% 
                summarise(avg_final_fitness = mean(-1 * fitness)),
              aes(x = as.factor(setup),
                  y = avg_final_fitness,
                  fill = setup
              )
  ) + 
    geom_violin() + 
    geom_boxplot(width = 0.1) +
    geom_point() +
    stat_compare_means(comparisons = my_comparisons) + # Add pairwise comparisons p-value
    scale_fill_viridis(discrete = T) +
    ylim(c(0,1)) +
    xlab("setup")+
    ylab(paste("average fitness in last generation", sep = "")) +
    theme_bw()
  
  print(p)
  ggsave(paste("average fitness in last generation across setups.jpg", sep = ""))
  ggsave(paste("average fitness in last generation across setups.pdf", sep = ""))
}

###Plot final best fitness 
make_best_fitness_t_test_boxplot <- function(post_hoc){
  
  my_comparisons <- list( c("OM", "OMX"), c("OM", "FM"), c("FM", "OMX"), c("FM", "FMX"), c("FMX", "OM"), c("FMX", "OMX"))
  p <- ggplot(post_hoc %>% 
                pivot_longer(c(test, val), names_to = "dataset", values_to = "fitness"),
              aes(x = as.factor(setup),
                  y = fitness,
                  fill = setup)
  ) + 
    geom_violin() + 
    geom_boxplot(width = 0.1) +
    geom_point() +
    stat_compare_means(comparisons = my_comparisons)+ # Add pairwise comparisons p-value
    scale_fill_viridis(discrete = T) +
    xlab("setup")+
    ylab(paste("average fitness in last generation", sep = "")) +
    theme_bw() +
    facet_grid(.~dataset)
  
  print(p)
  ggsave(paste("average fitness in best solutions across setups.jpg", sep = ""))
  ggsave(paste("average fitness in best solutions across setups.pdf", sep = ""))
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
                  color = setup,
                  fill = setup)
  )+
    geom_smooth() + 
    # geom_line(aes(group = interaction(setup, run))) + 
    scale_color_viridis(discrete = T) +
    scale_fill_viridis(discrete = T) +
    xlab("generations")+
    ylab(paste("number of unique solutions with fitness above ", accuracy, " over time per run across setups", sep = "")) +
    theme_bw()
  
  print(p)
  ggsave(paste("number of unique solutions with fitness above ", accuracy, " over time per run across setups.jpg"))
  ggsave(paste("number of unique solutions with fitness above ", accuracy, " over time per run across setups.pdf"))
}

###Plot cumulative unique behaviors over time
cumulative_unique_behaviours_over_time_across_setups_per_run <- function(all_data, accuracy){
  p <- all_data %>% 
    filter(fitness < -accuracy) %>%
    group_by(setup, run) %>% 
    arrange(iteration) %>% 
    distinct(smart_phenotype, .keep_all = T)  %>% 
    right_join(all_data %>%
                 group_by(setup,run) %>% 
                 distinct(iteration, .keep_all = T) %>% 
                 select(setup, run, iteration)) %>%  
    group_by(setup, run, iteration) %>% 
    summarise(uniques = n()) %>% 
    group_by(setup, run) %>% 
    mutate(sum_uniques = cumsum(uniques)) %>% 
    group_by(setup, iteration) %>% 
    summarise(sum_uniques = max(sum_uniques)) %>% 
    mutate(setup = factor(setup)) %>% 
    mutate(setup = fct_relevel(setup,c("FMX","FM","OM","OMX"))) %>% 
    ggplot(aes(x = iteration, y = sum_uniques, fill = setup)) +
    geom_area() +
    scale_fill_viridis(discrete = T) +
    scale_color_viridis(discrete = T) +
    # scale_color_viridis(option ="magma", discrete = T) +
    xlab("generations")+
    ylab(paste("cumulative number of unique solutions with fitness above ", accuracy, " over time per run across setups", sep = "")) +
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
                right_join(all_data %>%
                             group_by(setup,run) %>% 
                             distinct(iteration, .keep_all = T) %>% 
                             select(setup, run, iteration)) %>%  
                summarise(uniques = n()) %>% 
                arrange(iteration) %>% 
                mutate(sum_uniques = cumsum(uniques)),
              aes(x = iteration, 
                  y = sum_uniques,
                  color = setup,
                  fill = setup)
  )+
    geom_line(linewidth = 3) +
    # geom_smooth() + 
    # scale_color_viridis(option ="magma", discrete = T) +
    # geom_line(aes(group = interaction(setup, run))) + 
    scale_color_viridis(discrete = T) +
    scale_fill_viridis(discrete = T) +
    xlab("generations")+
    ylab(paste("cumulative number of unique solutions", accuracy, " over time across setups", sep = "")) +
    theme_bw()
  
  print(p)
  ggsave(paste("number of cumulative unique solutions with fitness above ", accuracy, " over time across setups.jpg"))
  ggsave(paste("number of cumulative unique solutions with fitness above ", accuracy, " over time across setups.pdf"))
}
####Converts smart_phenotypes constants to asterisks
convert_constants_to_asterisks = function(smart_phenotype){
  
  for (phen in smart_phenotype) {
    for (str in str_match_all(phen, "constant\\(\\s*(.*?)\\s*\\)")[[1]][,-1]){
      phen = gsub(str,"*",phen)
    }
  }
  
  return(phen)
}
#### Check for convergence over all setups
plot_convergent_phen_across_all_setups = function(all_data, accuracy){
  p = all_data %>%
    filter(fitness < -accuracy) %>% 
    group_by(setup, run) %>% 
    distinct(smart_phenotype, .keep_all = TRUE) %>% 
    ungroup() %>% 
    group_by(smart_phenotype) %>% 
    filter(n() > 1) %>%
    ggplot(aes(x = smart_phenotype_s, y = as.factor(run), fill = fitness)) +
    geom_tile() +
    scale_fill_viridis() +
    facet_grid( setup ~ .) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = -90, vjust = 0.5)) 
  
  print(p)
  ggsave(paste("solutions found in multiple runs with fitness above ", accuracy, " across all setups.jpg"))
  ggsave(paste("solutions found in multiple runs with fitness above ", accuracy, " across all setups.pdf"))
}

####Plot all uniques phenotypes that have appeared in different runs in the same setup
plot_convergent_phen_per_setup = function(all_data, accuracy){
  for ( i_setup in levels(all_data$setup)) {
    p = all_data %>%
      filter(fitness < -accuracy) %>% 
      group_by(setup, run) %>% 
      distinct(smart_phenotype, .keep_all = TRUE) %>% 
      ungroup() %>% 
      group_by(setup, smart_phenotype) %>% 
      filter(n() > 1 & setup == i_setup) %>%
      ggplot(aes(x = smart_phenotype,
                 y = as.factor(run),
                 fill = fitness)) +
      geom_tile() +
      scale_fill_viridis() +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = -90, vjust = 0.5)) +
      ggtitle(i_setup)
    print(p)
    
    print(p)
    ggsave(paste("solutions found in multiple runs with fitness above ", accuracy, " for ", i_setup," setup.jpg"))
    ggsave(paste("solutions found in multiple runs with fitness above ", accuracy, " for ", i_setup," setup.pdf"))
    
  }
}

###Plot all_unique phentoytpes that emerged in different runs 
### but considering constants as all equal
#### Check for convergence over all setups
plot_convergent_phen_across_all_setups_simplified = function(all_data, accuracy){
  p = all_data %>%
    filter(fitness < -accuracy) %>% 
    group_by(setup, run) %>% 
    distinct(smart_phenotype, .keep_all = TRUE) %>% 
    ungroup() %>% 
    group_by(smart_phenotype) %>% 
    filter(n() > 1)  %>%
    mutate(smart_phenotype_s = convert_constants_to_asterisks(smart_phenotype)) %>% 
    group_by(setup, run) %>% 
    distinct(smart_phenotype_s, .keep_all = TRUE) %>%
    ggplot(aes(x = smart_phenotype_s, y = as.factor(run), fill = fitness)) +
    geom_tile() +
    scale_fill_viridis() +
    facet_grid( setup ~ .) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = -90, vjust = 0.5)) 
  
  print(p)
  ggsave(paste("simplified solutions found in multiple runs with fitness above ", accuracy, " across all setups.jpg"),
         width = 30,
         height = 20,
         units = "cm"
  )
  ggsave(paste("simplified solutions found in multiple runs with fitness above ", accuracy, " across all setups.pdf"),
         width = 30,
         height = 20,
         units = "cm")
}

####Plot all uniques phenotypes that have appeared in different runs in the same setup
plot_convergent_phen_per_setup_simplified = function(all_data, accuracy){
  for ( i_setup in levels(all_data$setup)) {
    p = all_data %>%
      filter(fitness < -accuracy) %>% 
      group_by(setup, run) %>% 
      distinct(smart_phenotype, .keep_all = TRUE) %>% 
      ungroup() %>% 
      group_by(setup, smart_phenotype) %>% 
      filter(n() > 1 & setup == i_setup)  %>%
      mutate(smart_phenotype_s = convert_constants_to_asterisks(smart_phenotype)) %>% 
      group_by(setup, run) %>% 
      distinct(smart_phenotype_s, .keep_all = TRUE) %>%
      ggplot(aes(x = smart_phenotype_s,
                 y = as.factor(run),
                 fill = fitness)) +
      geom_tile() +
      scale_fill_viridis() +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = -90, vjust = 0.5)) +
      ggtitle(i_setup)
    print(p)
    
    print(p)
    ggsave(paste("simplified solutions found in multiple runs with fitness above ", accuracy, " for ", i_setup," setup.jpg"),
           width = 30,
           height = 20,
           units = "cm")
    ggsave(paste("simplified solutions found in multiple runs with fitness above ", accuracy, " for ", i_setup," setup.pdf"),
           width = 30,
           height = 20,
           units = "cm")
    
  }
}
###Plot percentage of convergent to percentage of unique for simplified phenotypes
plot_simplified_proportion_uniques_convergent_across_setups_and_runs = function(all_data, accuracy){
  p = all_data %>%
    filter(fitness < -0.8) %>% 
    group_by(setup, run) %>% 
    mutate(smart_phenotype = convert_constants_to_asterisks(smart_phenotype)) %>% 
    distinct(smart_phenotype, .keep_all = TRUE) %>% 
    ungroup() %>% 
    group_by(smart_phenotype) %>% 
    mutate(unique = n()==1,
           convergent = !unique) %>% 
    group_by(setup, run) %>% 
    summarise(uniques =sum(unique),
              convergents = sum(convergent)) %>% 
    pivot_longer(cols = c(uniques, convergents),
                 values_to = "value",
                 names_to = "smart_phenotype") %>% 
    ggplot(aes(x="", y=value, fill=smart_phenotype)) +
    geom_bar(stat="identity", width=1) +
    theme_bw() +
    facet_grid( setup ~ run ) +
    scale_fill_viridis(discrete = T) 
  print(p)
  ggsave(paste(" proportions of convergent and unique simplified solutions with fitness above ", accuracy, " across runs and setups.jpg"),
         width = 30,
         height = 20,
         units = "cm")
  ggsave(paste(" proportions of convergent and unique simplified solutions with fitness above ", accuracy, " across runs and setups.pdf"),
         width = 30,
         height = 20,
         units = "cm")
}

###Plot percentage of convergent to percentage of uniques
plot_proportion_uniques_convergent_across_setups_and_runs = function(all_data, accuracy){
  p = all_data %>%
    filter(fitness < -0.8) %>% 
    group_by(setup, run) %>% 
    distinct(smart_phenotype, .keep_all = TRUE) %>% 
    ungroup() %>% 
    group_by(smart_phenotype) %>% 
    mutate(unique = n()==1,
           convergent = !unique) %>% 
    group_by(setup, run) %>% 
    summarise(uniques =sum(unique),
              convergents = sum(convergent)) %>% 
    pivot_longer(cols = c(uniques, convergents),
                 values_to = "value",
                 names_to = "smart_phenotype") %>% 
    ggplot(aes(x="", y=value, fill=smart_phenotype)) +
    geom_bar(stat="identity", width=1) +
    theme_bw() +
    facet_grid( setup ~ run ) +
    scale_fill_viridis(discrete = T) 
  print(p)
  ggsave(paste(" proportions of convergent and unique solutions with fitness above ", accuracy, " across runs and setups.jpg"),
         width = 30,
         height = 20,
         units = "cm")
  ggsave(paste(" proportions of convergent and unique solutions with fitness above ", accuracy, " across runs and setups.pdf"),
         width = 30,
         height = 20,
         units = "cm")
}

###Plot percentage of convergent to percentage of uniques
plot_proportion_uniques_convergent_across_setups = function(all_data, accuracy){
  p = all_data %>%
    filter(fitness < -accuracy) %>% 
    group_by(setup) %>% 
    distinct(smart_phenotype, .keep_all = TRUE) %>% 
    ungroup() %>% 
    group_by(smart_phenotype) %>% 
    mutate(unique = n()==1,
           convergent = !unique) %>% 
    group_by(setup) %>% 
    summarise(uniques =sum(unique),
              convergents = sum(convergent)) %>% 
    pivot_longer(cols = c(uniques, convergents),
                 values_to = "value",
                 names_to = "smart_phenotype") %>% 
    ggplot(aes(x="", y=value, fill=smart_phenotype)) +
    geom_bar(stat="identity", width=1) +
    facet_grid( . ~ setup )+ 
    coord_polar(theta = "y") +
    theme_bw() +
    scale_fill_viridis(discrete = T)
  print(p)
  ggsave(paste(" proportions of convergent and unique solutions with fitness above ", accuracy, " across setups.jpg"),
         width = 30,
         height = 20,
         units = "cm")
  ggsave(paste(" proportions of convergent and unique solutions with fitness above ", accuracy, " across setups.pdf"),
         width = 30,
         height = 20,
         units = "cm")
}

####Either read data from csv if not done yet or load from Rds
read_or_load_all_data <- function(){
  
  setwd(paste(dirname(rstudioapi::getSourceEditorContext()$path),"many_results/", sep = '/'))
  if(!file.exists(plot_dir_name())){
    read_and_save_all_data()
  }else{
    setwd(plot_dir_name())
  }
  # Load
  all_data <- readRDS("all_data.Rds") %>% 
    mutate(fitness = as.numeric(fitness)) %>% 
    mutate(setup = factor(setup)) %>% 
    mutate(setup = fct_relevel(setup,c("FMX","FM","OM","OMX"))) 
}

read_or_load_post_hoc <- function(){
  
  setwd(paste(dirname(rstudioapi::getSourceEditorContext()$path),"many_results/", sep = '/'))
  if(!file.exists(plot_dir_name())){
    read_and_save_post_hoc()
  }else{
    setwd(plot_dir_name())
  }
  # Load
  
  post_hoc <- readRDS("post_hoc.Rds") %>% 
    rename(setup  =  1) %>% 
    rename(val  =  2) %>% 
    rename(test  = 3) %>% 
    mutate(val = as.numeric(val)) %>% 
    mutate(test = as.numeric(test)) %>% 
    mutate(setup = factor(setup)) %>% 
    mutate(setup = fct_relevel(setup,c("FMX","FM","OM","OMX")))  
}


#######Running code

all_data = read_or_load_all_data()
post_hoc = read_or_load_post_hoc()

maxe_unique_t_test_boxplot(all_data, accuracy = 0.8)  
maxe_unique_t_test_wiht_grad_boxplot(all_data, 0.5)

unique_behaviours_over_time_across_setups(all_data, accuracy = 0.8)
unique_behaviours_over_time_across_setups(all_data, accuracy = 0.5)
unique_behaviours_over_time_across_setups(all_data, accuracy = 0.2)
unique_behaviours_over_time_across_setups(all_data, accuracy = 0.1)

cumulative_unique_behaviours_over_time_across_setups_per_run(all_data, accuracy = 0.8)
cumulative_unique_behaviours_over_time_across_setups_per_run(all_data, accuracy = 0.5)
cumulative_unique_behaviours_over_time_across_setups_per_run(all_data, accuracy = 0.2)
cumulative_unique_behaviours_over_time_across_setups_per_run(all_data, accuracy = 0.1)

cumulative_unique_behaviours_over_time_across_setup(all_data, accuracy = 0.8)
cumulative_unique_behaviours_over_time_across_setup(all_data, accuracy = 0.5)
cumulative_unique_behaviours_over_time_across_setup(all_data, accuracy = 0.2)
cumulative_unique_behaviours_over_time_across_setup(all_data, accuracy = 0.1)

fitness_best_over_time_across_setups(all_data)
fitness_over_time_across_setups(all_data)
make_fitness_t_test_boxplot(all_data)
make_best_fitness_t_test_boxplot(post_hoc = post_hoc)

plot_convergent_phen_across_all_setups_simplified(all_data,0.8)
plot_convergent_phen_across_all_setups_simplified(all_data,0.5)
plot_convergent_phen_across_all_setups_simplified(all_data,0.2)

plot_convergent_phen_per_setup_simplified(all_data, 0.8)
plot_convergent_phen_per_setup_simplified(all_data, 0.5)
plot_convergent_phen_per_setup_simplified(all_data, 0.2)

plot_proportion_uniques_convergent_across_setups_and_runs(all_data, 0.8)
plot_proportion_uniques_convergent_across_setups(all_data, 0.8)

