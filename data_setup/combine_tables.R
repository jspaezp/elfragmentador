
require(dplyr)
require(tidyr)
require(readr)

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
# # A tibble: 794,710 x 10
#    `MS/MS Count` Intensity `Calibrated ret… `Modified seque… Charge
#            <dbl>     <dbl>            <dbl> <chr>             <dbl>
#  1           100    2.05e8             9.57 _AAAHHYGAQCDKPN…      4
#  2            68    5.78e8            26.5  _AALHDDADLVHHCI…      3
#  3            44    2.28e8            28.3  _AASTDLGAGETVVG…      2
#  4             1    1.56e6            50.6  _AAVFWGDIALDEDD…      2
#  5            12    7.07e7            27.0  _AAVMVYDDANK_         2
#  6            12    2.41e8            48.9  _ADATLLLGPLR_         2
#  7            28    2.40e9            32.5  _AEFFSSENVK_          2
#  8            60    1.55e9            38.9  _AEIEQFLNEK_          2
#  9            32    1.79e8            48.6  _AELALSAFLK_          2
# 10             8    1.25e7            55.2  _AFLYEIIDIGK_         2
# # … with 794,700 more rows, and 5 more variables: `Leading
# #   proteins` <chr>, `Raw file` <chr>, Encoding <chr>, iRT <dbl>,
# #   `Sequence Length` <dbl>

irt_lms <- split(in_df, in_df$`Raw file`) %>% 
  purrr::map(~ lm(data = .x, `Calibrated retention time` ~ iRT)$coefficients)

irt_slopes <- purrr::map_dbl(irt_lms, ~ .x[2])
irt_intercepts <- purrr::map_dbl(irt_lms, ~ .x[1])

hist(irt_slopes)
removable_runs <- irt_slopes[irt_slopes > 0.3 | irt_slopes < 0.21] %>% names()
# [1] "01625b_GE5-TUM_first_pool_37_01_01-DDA-1h-R2"        
# [2] "01625b_GF3-TUM_first_pool_22_01_01-ETD-1h-R2"        
# [3] "01640a_BC5-Thermo_SRM_Pool_35_01_01-ETD-1h-R4"       
# [4] "01640c_BD6-Thermo_SRM_Pool_44_01_01-2xIT_2xHCD-1h-R2"
# [1] "01625b_GF3-TUM_first_pool_22_01_01-ETD-1h-R2" 
# [2] "01640a_BC5-Thermo_SRM_Pool_35_01_01-ETD-1h-R4"

plot(in_df$`Calibrated retention time`, in_df$iRT) 

summ_in_df <- in_df %>%
  filter(! `Raw file` %in% removable_runs) %>%
  select(-Intensity) %>%
  group_by(`Modified sequence`, Encoding, `Leading proteins`) %>%
  summarise(
    mIRT = median(iRT),
    sdIRT = sd(iRT, na.rm = TRUE),
    mRT = median(`Calibrated retention time`),
    sdRT = sd(`Calibrated retention time`, na.rm = TRUE),
    nMSMS = sum(`MS/MS Count`))

summ_in_df$sdIRT[is.na(summ_in_df$sdIRT)] <- 0
summ_in_df$sdRT[is.na(summ_in_df$sdRT)] <- 0

plot(summ_in_df$sdIRT,  summ_in_df$mIRT)
plot(summ_in_df$sdRT,  summ_in_df$mRT)

plot(summ_in_df$sdRT,  log1p(summ_in_df$nMSMS))

plot(summ_in_df$mRT,  summ_in_df$mIRT)

hist(summ_in_df$mIRT)
hist(summ_in_df$mRT)

hist(summ_in_df$sdRT)

summ_in_df %>%
  filter(sdRT > 5)

summ_in_df <- summ_in_df %>% filter(nMSMS > 5, sdRT < 5)
summ_in_df
plot(summ_in_df$sdRT,  log1p(summ_in_df$nMSMS))
plot(summ_in_df$sdRT,  summ_in_df$mRT)

write_csv(summ_in_df, "summarized_irt_times.csv")

filter(in_df, `Modified sequence` %in% "_APHQVPVQSEKNPARSPVTEIR_")
