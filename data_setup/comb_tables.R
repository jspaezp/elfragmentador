
library(tidyverse)

specs <- readr::read_csv("encoded_spectra.csv")
irts  <- readr::read_csv("/home/jspaezp/Downloads/summarized_irt_times.csv")

specs
# # A tibble: 264,836 x 3
#    Sequences          Encodings                                    Charges
#    <chr>              <chr>                                          <dbl>
#  1 AAAPRPPVSAASGRPQD… [0, 0, 0, 0, 0, 0.692661467621213, 0, 0, 0.…       3
#  2 AAASHLFPFEQL       [0, 0.017231523091699032, 0.015964018301691…       2
#  3 AADCAVDHHFRFCLLLR  [0, 0.2182896483357762, 0.3708611903831905,…       4
#  4 AADDFLEDLPLEETGAI… [0, 0, 0, 0, 0, 0, 0.4160648586047836, 0, 0…       3
#  5 AAEDLFVNIR         [0, 0.06340729063439986, 0.3593192273679846…       2
#  6 AAELSHPSPVPT       [0, 0, 0.019965074469737724, 1.0, 0.0205877…       2
#  7 AAETHFGFETVSEEEK   [0, 0.17566786990584485, 0.6108681863286385…       3
#  8 AAEVEPSSPEPKR      [0, 0.1614344444965644, 0.7769072839729991,…       3
#  9 AAGPISER           [0, 0.10792815576938526, 0.3847153397710498…       2
# 10 AAIFQTQTYLASGKTK   [0, 0.043307672070412265, 0.165061737047394…       3
# # … with 264,826 more rows
irts
# # A tibble: 215,908 x 9
#    `Modified seque… Encoding `Leading protei…   mIRT  sdIRT   mRT   sdRT
#    <chr>            <chr>    <chr>             <dbl>  <dbl> <dbl>  <dbl>
#  1 _AAAAAAAAAAAAAA… [1, 1, … TUM_first_pool_… 118.   0.0227  47.0 0.178
#  2 _AAAAAAAAAAGAAG… [1, 1, … TUM_first_pool_…  13.4  0.227   21.5 0.138
#  3 _AAAAAAAAAAR_    [1, 1, … TUM_first_pool_… -13.6  1.87    14.3 0.0849
#  4 _AAAAAAAAAKNGSS… [1, 1, … TUM_first_pool_… -22.6  5.04    11.5 0.956
#  5 _AAAAAAAAAVSR_   [1, 1, … TUM_first_pool_…   9.22 0       19.6 0
#  6 _AAAAAAAAGAFAGR_ [1, 1, … TUM_first_pool_…  38.7  0.202   27.1 0.482
#  7 _AAAAAAAAVPSAGP… [1, 1, … TUM_first_pool_…  38.9  0.773   27.6 0.397
#  8 _AAAAAAALQAK_    [1, 1, … TUM_first_pool_…  11.8  1.11    20.0 0.237
#  9 _AAAAAAAPSGGGGG… [1, 1, … TUM_first_pool_…   9.32 0.817   20.1 0.457
# 10 _AAAAAAARGSNSDS… [1, 1, … TUM_second_pool… -23.5  1.55    12.2 0.182
# # … with 215,898 more rows, and 2 more variables: nRUNS <dbl>,
# #   nMSMS <dbl>

combined_df <- irts %>%
  filter(nRUNS < 100) %>%
  mutate(
    Sequences = gsub("_", "", `Modified sequence`),
    SequenceEncoding = Encoding) %>%
  select(Sequences, SequenceEncoding, mIRT) %>%
  full_join(specs) %>%
  rename(SpectraEncoding = Encodings) %>%
  filter(!is.na(SpectraEncoding)) %>%
  filter(!is.na(SequenceEncoding))

combined_df

set.seed(2020)
seqs <- unique(combined_df$Sequences)
num_rows <- length(seqs)
num_15_pct <- as.integer(num_rows * 0.15)

holdout_seqs <- sample(seqs, num_15_pct)
val_seqs <- sample(seqs[!seqs %in% holdout_seqs], num_15_pct)
train_seqs <- seqs[(!seqs %in% holdout_seqs) & (!seqs %in% val_seqs)]

holdout_df <- combined_df %>%
  filter(Sequences %in% holdout_seqs)
holdout_df
# # A tibble: 38,437 x 5
#    Sequences    SequenceEncoding        mIRT SpectraEncoding       Charges
#    <chr>        <chr>                  <dbl> <chr>                   <dbl>
#  1 AAAAAAARGSN… [1, 1, 1, 1, 1, 1, 1,… -23.5 [0, 0.06499065219735…       2
#  2 AAAAAAARGSN… [1, 1, 1, 1, 1, 1, 1,… -23.5 [0, 0.03893089912459…       3
#  3 AAAAAAEAGDI… [1, 1, 1, 1, 1, 1, 4,…  86.5 [0, 0.06827073449199…       3
#  4 AAAAAATAPPS… [1, 1, 1, 1, 1, 1, 17…  15.8 [0, 0, 0.01063543936…       2
#  5 AAAAAGGAPGP… [1, 1, 1, 1, 1, 6, 6,…  37.3 [0, 0.11028480222482…       2
#  6 AAAAASASQDE… [1, 1, 1, 1, 1, 16, 1…  52.4 [0, 0.13942941877580…       2
#  7 AAAADSFSGGP… [1, 1, 1, 1, 3, 16, 5…  59.5 [0, 0.38542166233931…       2
#  8 AAAADSFSGGP… [1, 1, 1, 1, 3, 16, 5…  59.5 [0, 0.08601441098844…       3
#  9 AAAAGDADDEPR [1, 1, 1, 1, 6, 3, 1,… -20.4 [0, 0.11930653011442…       2
# 10 AAAAGEPEPPA… [1, 1, 1, 1, 6, 4, 13…  34.5 [0, 0.01957774922302…       2
# # … with 38,427 more rows

val_df <- combined_df %>%
  filter(Sequences %in% val_seqs)
val_df
# # A tibble: 38,415 x 5
#    Sequences    SequenceEncoding         mIRT SpectraEncoding      Charges
#    <chr>        <chr>                   <dbl> <chr>                  <dbl>
#  1 AAAAAAAAGAF… [1, 1, 1, 1, 1, 1, 1,…  38.7  [0, 0.0270357441655…       2
#  2 AAAAAAALQAK  [1, 1, 1, 1, 1, 1, 1,…  11.8  [0, 0.0344009820221…       2
#  3 AAAAAAASFAA… [1, 1, 1, 1, 1, 1, 1,… 122.   [0, 0.0672610462097…       4
#  4 AAAAAAASFAA… [1, 1, 1, 1, 1, 1, 1,… 122.   [0, 0, 0.1513147509…       3
#  5 AAAAASAPQQL… [1, 1, 1, 1, 1, 16, 1… 105.   [0, 0.0676950253939…       2
#  6 AAAAELQDR    [1, 1, 1, 1, 4, 10, 1…  -2.60 [0, 0.2302540941700…       2
#  7 AAAASPRPGFW… [1, 1, 1, 1, 16, 13, …  21.0  [0, 0.1463855546729…       4
#  8 AAAATAAEGVP… [1, 1, 1, 1, 17, 1, 1…   2.89 [0, 0.0622093117545…       2
#  9 AAAAWALGQIGR [1, 1, 1, 1, 19, 1, 1…  89.3  [0, 0.0429541136338…       2
# 10 AAAAWYRPAGRR [1, 1, 1, 1, 19, 20, …   6.51 [0, 0.1413981393656…       3
# # … with 38,405 more rows

train_df <- combined_df %>%
  filter(Sequences %in% train_seqs)
train_df
# # A tibble: 179,660 x 5
#    Sequences    SequenceEncoding         mIRT SpectraEncoding      Charges
#    <chr>        <chr>                   <dbl> <chr>                  <dbl>
#  1 AAAAAAAAAAA… [1, 1, 1, 1, 1, 1, 1,… 118.   [0, 0.0521470255581…       2
#  2 AAAAAAAAAAA… [1, 1, 1, 1, 1, 1, 1,… 118.   [0, 0.4324427491080…       3
#  3 AAAAAAAAAAG… [1, 1, 1, 1, 1, 1, 1,…  13.4  [0, 0.0561808225479…       2
#  4 AAAAAAAAAVSR [1, 1, 1, 1, 1, 1, 1,…   9.22 [0, 0.0408048864209…       2
#  5 AAAAAAAAVPS… [1, 1, 1, 1, 1, 1, 1,…  38.9  [0, 0, 0, 0.0135584…       2
#  6 AAAAAAAPSGG… [1, 1, 1, 1, 1, 1, 1,…   9.32 [0, 0.0210271670278…       3
#  7 AAAAAAAPSGG… [1, 1, 1, 1, 1, 1, 1,…   9.32 [0, 0, 0, 0, 0.1882…       2
#  8 AAAAAAAVGGQ… [1, 1, 1, 1, 1, 1, 1,…  92.8  [0, 0.0162064945835…       2
#  9 AAAAAAGAASG… [1, 1, 1, 1, 1, 1, 6,…  71.9  [0, 0.0787195611243…       2
# 10 AAAAAAGSGTP… [1, 1, 1, 1, 1, 1, 6,…  32.6  [0, 0.0712724334258…       3
# # … with 179,650 more rows

write_csv(holdout_df, "combined_holdout.csv")
write_csv(val_df, "combined_val.csv")
write_csv(train_df, "combined_train.csv")
