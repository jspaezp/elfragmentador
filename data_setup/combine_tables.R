
require(dplyr)
require(tidyr)
require(readr)
require(ggplot2)

in_tables <- dir(pattern = "trypticunmodified_irt.csv")
in_tables

in_df <- purrr::map_dfr(
  in_tables,
  ~ readr::read_csv(
    .x,
    col_types = readr::cols(
      `MS/MS Count` = col_double(),
      Intensity = col_double(),
      `Calibrated retention time` = col_double(),
      `Modified sequence` = col_character(),
      Charge = col_double(),
      `Leading proteins` = col_character(),
      `Raw file` = col_character(),
      Encoding = col_character(),
      iRT = col_double(),
      `Sequence Length` = col_double()
    )))

in_df

iqr <- function(x, ...) {
  diff(quantile(x, c(0.25, 0.75), ...))
}

robust_mean <- function(x, ...) {
  qs  <- quantile(x, c(0.1, 0.9))
  x <- x[x >= qs[1] & x <= qs[2]]
  return(mean(x))
}

peptide_run_counts <- group_by(in_df, `Modified sequence`) %>%
  summarise(
    n = n(),
    mRT = robust_mean(`Calibrated retention time`),
    mIRT = robust_mean(iRT),
    iqrRT = iqr(`Calibrated retention time`),
    iqrIRT = iqr(iRT)
  )

peptide_run_counts %>% .$n %>% log10() %>% hist()
ggplot2::qplot(peptide_run_counts %>% .$iqrIRT) +  ggplot2::scale_y_sqrt()
ggplot2::qplot(filter(peptide_run_counts, n > 200) %>% .$iqrIRT) +  ggplot2::scale_y_sqrt()
ggplot2::qplot(filter(peptide_run_counts, n > 200) %>% .$iqrRT) +  ggplot2::scale_y_sqrt()

stable_peptide_rts <- filter(peptide_run_counts, n > 200, iqrRT < 1.2, iqrIRT < 1.5)

filter(in_df, `Modified sequence` %in% stable_peptide_rts$`Modified sequence`) %>%
  ggplot(aes(x = iRT, y = `Calibrated retention time`)) +
  geom_point() + ylim(0, 80) + xlim(-50, 200)

plot(stable_peptide_rts$mRT, stable_peptide_rts$iqrRT)
plot(stable_peptide_rts$mIRT, stable_peptide_rts$iqrIRT)
ggplot2::qplot(x = stable_peptide_rts$mIRT, y = stable_peptide_rts$iqrIRT, colour = stable_peptide_rts$n)
ggplot2::qplot(x = stable_peptide_rts$mIRT, y = stable_peptide_rts$iqrIRT, label = stable_peptide_rts$`Modified sequence`, geom="label")
plot(stable_peptide_rts$iqrRT, stable_peptide_rts$iqrIRT)
plot(stable_peptide_rts$mRT, stable_peptide_rts$mIRT)

stable_peptide_rts_allruns <- filter(
    in_df,
    `Modified sequence` %in% stable_peptide_rts$`Modified sequence`)

slim_stable_peptide_rts <- select(stable_peptide_rts, `Modified sequence`, mIRT)
readr::write_csv(slim_stable_peptide_rts, "stable_irts.csv")

split_runs <- split(
    select(
        stable_peptide_rts_allruns,
        `Modified sequence`,
        `Calibrated retention time`),
    stable_peptide_rts_allruns$`Raw file`)

split_lms <- purrr::map(
    split_runs,
    ~ lm(mIRT ~ `Calibrated retention time`,
         left_join(.x, slim_stable_peptide_rts, by = "Modified sequence")))

split_lms[1:5]

split_runs <- split(
    in_df,
    in_df$`Raw file`)

recal_irt <- purrr::map2(split_lms[names(split_runs)], split_runs, ~ predict(.x, newdata = .y))
names(recal_irt) <- names(split_runs)

add_irt <- function(df, y) {
  df$recal_iRT <- y
  return(df)
}


split_runs <- purrr::map2(split_runs[names(recal_irt)], recal_irt, ~ add_irt(.x, .y))
split_runs <- bind_rows(split_runs)
split_runs

stopifnot(nrow(in_df) == nrow(split_runs))

in_df <- split_runs

# It is pretty clear that the recalculated is better ... will just use that
qplot(data = in_df, x = recal_iRT, y = iRT)
qplot(data = in_df, x = recal_iRT, y = `Calibrated retention time`)

in_df$iRT <- in_df$recal_iRT

irt_lms <- split(in_df, in_df$`Raw file`) %>%
  purrr::map(~ lm(data = .x, `Calibrated retention time` ~ iRT)$coefficients)

irt_slopes <- purrr::map_dbl(irt_lms, ~ .x[2])
irt_intercepts <- purrr::map_dbl(irt_lms, ~ .x[1])

hist(irt_slopes)
removable_runs <- irt_slopes[irt_slopes > 0.3 | irt_slopes < 0.21] %>% names()
# [1] "01640c_BA9-Thermo_SRM_Pool_65_01_01-3xHCD-1h-R2"
# [2] "01812a_GB3-TUM_third_pool_2_01_01-ETD-1h-R1"

plot(in_df$`Calibrated retention time`, in_df$iRT)

summ_in_df <- in_df %>%
  filter(! `Raw file` %in% removable_runs) %>%
  select(-Intensity) %>%
  group_by(`Modified sequence`, Encoding, `Leading proteins`) %>%
  summarise(
    mIRT = median(iRT),
    sdIRT = sd(iRT, na.rm = TRUE),
    mRT = mean(`Calibrated retention time`),
    sdRT = sd(`Calibrated retention time`, na.rm = TRUE),
    nRUNS = n(),
    nMSMS = sum(`MS/MS Count`))

summ_in_df$sdIRT[is.na(summ_in_df$sdIRT)] <- 0
summ_in_df$sdRT[is.na(summ_in_df$sdRT)] <- 0

plot(summ_in_df$sdIRT,  summ_in_df$mIRT)
plot(summ_in_df$sdRT,  summ_in_df$mRT)

plot(summ_in_df$sdRT,  log1p(summ_in_df$nMSMS))

plot(summ_in_df$mRT,  summ_in_df$mIRT)

hist(summ_in_df$mIRT)
hist(summ_in_df$mRT)

qplot(summ_in_df$sdRT) +  ggplot2::scale_y_sqrt()
qplot(summ_in_df$sdIRT, log1p(summ_in_df$nMSMS)) + scale_x_sqrt() + geom_vline(xintercept = 20)
qplot(summ_in_df$sdRT, log1p(summ_in_df$nMSMS)) + scale_x_sqrt() + geom_vline(xintercept = 5)

summ_in_df %>%
  filter(sdRT > 5)

summ_in_df <- summ_in_df %>% filter(nMSMS > 5, sdRT < 5, sdIRT < 20)
summ_in_df
plot(summ_in_df$sdRT,  log1p(summ_in_df$nMSMS))
plot(summ_in_df$sdRT,  summ_in_df$mRT)

write_csv(summ_in_df, "summarized_irt_times.csv")
