library(rethinking)
library(tidyverse)
library(foreach)
library(ggrepel)
library(lubridate)
standard_theme = function () {
  theme (panel.background = element_rect (fill = 'white'),
         panel.grid.major = element_line (colour = 'grey90', size = 0.20),
         panel.grid.minor = element_line (colour = 'grey90', size = 0.10),
         plot.title = element_text (size = 18, lineheight = 1.15),
         plot.subtitle = element_text (size = 12, lineheight = 1.15),
         axis.title.y = element_text (angle = 90),
         axis.title = element_text (size = 12),
         text = element_text (family = 'Avenir', size = 12),
         legend.title = element_text (size = 12),
         strip.text = element_text (size = 10, angle = 0),
         strip.background = element_rect (fill = 'grey97'),
         plot.caption = element_text (hjust = 0, size = 9))
}
floor_line = function (yintercept = 0, ...) {
  geom_hline (yintercept = yintercept, colour = 'grey75', size = 0.3, ...)
}
save_plot = function(plot = last_plot(), path = '~/Documents/R.png'){
  ggsave(paste0(path, ".png"), plot, device=NULL, dpi=500, units="in", height = 6, width=10) 
}
beta_comparison = function(results, beta_prior = c(1, 1), sample_size=1E4){
  distribution = rbeta(sample_size, results[1] + beta_prior[1], results[2] + beta_prior[2])
  return (c(mean(distribution), quantile(distribution, c(0.05, 0.5, 0.95)))) 
}



simulations_data = read_csv('/Users/raphaeltamaki/Documents/personal_git/causal_methodologies_analysis/results/simulations_results.csv')  %>%
  mutate(
    bias = (estimated_ate - true_ate) / true_ate,
    included = (true_ate >= estimated_lower_bound) & (true_ate <= estimated_upper_bound),
    ci_size = (estimated_upper_bound - estimated_lower_bound) / reference_value,
    dataset = case_when(
      dataset == 'lifetime_value'  ~ 'Lifetime Value Dataset',
      dataset == 'supermarket_sales'  ~ 'Supermarket Sales Dataset',
      dataset == 'nifty50_stock_market'  ~ 'NIFTY50 Stock Market Value Dataset',
      TRUE ~ 'Wrong Name'
    ),
    model = factor(
      model,
      c("DifferenceInDifferencesEstimator", "SyntheticControlEstimator", "SLearner", "TLearner", "DoublyRobustEstimator", "DoWhyEstimator"),
      c("Difference-In-Differences\n(CausalPy)", "Synthetic Control\n(CausalPy)", "S-Learner\n(-)", "T-Learner\n(CausalML)", "Doubly Robust Estimator\n(-)", "Graph Causal Model\n(DoWhy)")
    )
  )

avg_model_data = simulations_data %>%
  mutate(model = 'Averaged Model',
         `...1` = NA) %>%
  group_by(`...1`, it, dataset, treated_groups, lift_size, reference_value, model, true_ate, ci, simulation_date) %>%
  summarise(
    estimated_ate = mean(estimated_ate),
    estimated_std = mean(estimated_std),
    estimated_lower_bound = mean(estimated_lower_bound),
    estimated_upper_bound = mean(estimated_upper_bound)
  ) %>%
  ungroup %>%
  mutate(
    bias = (estimated_ate - true_ate) / true_ate,
    included = (true_ate >= estimated_lower_bound) & (true_ate <= estimated_upper_bound),
    ci_size = (estimated_upper_bound - estimated_lower_bound) / reference_value
  )

binded_data = rbind(simulations_data, avg_model_data) %>%
  mutate(
    model = factor(
      model,
      c("DifferenceInDifferencesEstimator", "SyntheticControlEstimator", "SLearner", "TLearner", "DoublyRobustEstimator", "DoWhyEstimator", 'Averaged Model'),
      c("Difference-In-Differences\n(CausalPy)", "Synthetic Control\n(CausalPy)", "S-Learner\n(-)", "T-Learner\n(CausalML)", "Doubly Robust Estimator\n(-)", "Graph Causal Model\n(DoWhy)", 'Averaged Model')
    )
  )


#####
# Bias
ggplot(data = simulations_data,
       aes(x = model, y = bias, color = dataset, fill = dataset)) + 
  standard_theme() +
  theme(legend.position = 'bottom') +
  floor_line() +
  geom_boxplot(alpha = 0.7, width = 0.7, position = position_dodge(width=0.7), size=0.4) +
  scale_y_continuous(
    limits = c(-1, 1),
    minor_breaks = NULL, 
    labels = scales::percent) +
  labs(
    x = 'Algorithm',
    y = 'Bias',
    color = 'Dataset',
    fill = 'Dataset',
    )
save_plot(path='/Users/raphaeltamaki/Documents/personal_git/causal_methodologies_analysis/results/bias')


#####
# Coverage
coverage_data = simulations_data %>%
  group_by(model, dataset) %>%
  summarise(
    isin = sum(included),
    isout = n() - sum(included)
  ) %>%
  ungroup

coverage_data[c('avg', 'low', 'median', 'high')] = coverage_data %>%
  select(isin, isout) %>%
  apply(1, beta_comparison) %>% 
  t()
ggplot(data = coverage_data,
       aes(x = model, y = avg, ymin = low, ymax = high, color = dataset, fill = dataset)) + 
  standard_theme() +
  theme(legend.position = 'bottom') +
  floor_line() +
  geom_col(alpha = 0.7, width = 0.7, position = position_dodge(width=0.7), size=0.4) +
  geom_errorbar(alpha = 0.7, width = 0.35, position = position_dodge(width=0.7), size=0.4) +
  scale_y_continuous(
    limits = c(-0, 1),
    minor_breaks = NULL, 
    labels = scales::percent) +
  labs(
    x = 'Algorithm',
    y = 'Rate CI 90% includes true ATT',
    color = 'Dataset',
    fill = 'Dataset',
  )
save_plot(path='/Users/raphaeltamaki/Documents/personal_git/causal_methodologies_analysis/results/coverage')                     



#####
# CI 90% Size
ggplot(data = simulations_data,
       aes(x = model, y = ci_size, color = dataset, fill = dataset)) + 
  standard_theme() +
  theme(legend.position = 'bottom') +
  floor_line() +
  geom_boxplot(alpha = 0.7, width = 0.7, position = position_dodge(width=0.7), size=0.4) +
  scale_y_continuous(
    minor_breaks = NULL) +
  labs(
    x = 'Algorithm',
    y = 'Indexed Size of CI 90%',
    color = 'Dataset',
    fill = 'Dataset',
  )
save_plot(path='/Users/raphaeltamaki/Documents/personal_git/causal_methodologies_analysis/results/ci_size')


scatter_data = simulations_data %>%
  group_by(model, dataset) %>%
  summarise(
    isin_rate = sum(included) / n(),
    ci_size = mean(ci_size)
  ) %>%
  ungroup
ggplot(data = scatter_data,
       aes(x = ci_size, y = isin_rate, color = model, fill = model)) + 
  standard_theme() +
  theme(legend.position = "none") +
  floor_line() +
  geom_point(size=3, alpha=0.6) +
  geom_text_repel(aes(label = model), size=3) +
  facet_wrap(~dataset) +
  scale_x_continuous(minor_breaks = NULL, limits=c(-0.5, 2)) +
  scale_y_continuous(
    minor_breaks = NULL,
    labels = scales::percent) +
  scale_color_brewer(palette = "Dark2") +
  scale_fill_brewer(palette = "Dark2") +
  labs(
    x = 'Indexed Size of CI 90%',
    y = 'Rate CI 90% includes true ATT',
    color = 'Model',
    fill = 'Model',
    )
save_plot(path='/Users/raphaeltamaki/Documents/personal_git/causal_methodologies_analysis/results/compromisse_ci_coverage')






synthetic_control_data = tibble(
  x = c(seq(200), seq(200), seq(200)),
  y = c(rnorm(200, 5, 1), rnorm(200, 7, .5), rnorm(100, 10, 1.5), rnorm(100, 15, 1.5)),
  group = c(rep('A', 200), rep('B', 200), rep('C', 200))
)
ggplot(data = synthetic_control_data,
       aes(x = x, y = y, color = group)) + 
  standard_theme() +
  floor_line() +
  geom_line() +
  scale_x_continuous(minor_breaks = NULL) +
  scale_y_continuous(
    minor_breaks = NULL) +
  scale_color_brewer(palette = "Set1") +
  scale_fill_brewer(palette = "Set1") +
  labs(
    x = 'Time',
    y = 'Target Variable',
    color = 'Location',
  )
save_plot(path='/Users/raphaeltamaki/Documents/personal_git/causal_methodologies_analysis/results/synthetic_control_example')

