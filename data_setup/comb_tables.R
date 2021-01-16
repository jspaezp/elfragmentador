
library(tidyverse)

specs <- readr::read_csv("encoded_spectra.csv")
irts  <- readr::read_csv("/home/jspaezp/Downloads/summarized_irt_times.csv")

specs
# # A tibble: 264,836 x 3
#    Sequences          Encodings                                           Charges
#    <chr>              <chr>                                                 <dbl>
#  1 AAAPRPPVSAASGRPQD… [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.69266146…       3
#  2 AAASHLFPFEQL       [0, 0.017231523091699032, 0, 0, 0, 0, 0.0159640183…       2
#  3 AADCAVDHHFRFCLLLR  [0, 0.2182896483357762, 0, 0, 0, 0, 0.370861190383…       4
#  4 AADDFLEDLPLEETGAI… [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0…       3
#  5 AAEDLFVNIR         [0, 0.06340729063439986, 0, 0, 0, 0, 0.35931922736…       2
#  6 AAELSHPSPVPT       [0, 0, 0, 0, 0, 0, 0.019965074469737724, 1.0, 0, 0…       2
#  7 AAETHFGFETVSEEEK   [0, 0.17566786990584485, 0, 0, 0, 0, 0.61086818632…       3
#  8 AAEVEPSSPEPKR      [0, 0.1614344444965644, 0, 0, 0, 0, 0.776907283972…       3
#  9 AAGPISER           [0, 0.10792815576938526, 0, 0, 0, 0, 0.38471533977…       2
# 10 AAIFQTQTYLASGKTK   [0, 0.043307672070412265, 0, 0, 0, 0, 0.1650617370…       3
# # … with 264,826 more rows
irts
# # A tibble: 211,643 x 8
#    `Modified sequen… Encoding   `Leading protei…   mIRT  sdIRT   mRT   sdRT nMSMS
#    <chr>             <chr>      <chr>             <dbl>  <dbl> <dbl>  <dbl> <dbl>
#  1 _AAAAAAAAAAAAAAA… [1, 1, 1,… TUM_first_pool_… 117.   0.0160  47.0 0.178      8
#  2 _AAAAAAAAAAGAAGG… [1, 1, 1,… TUM_first_pool_…  13.7  0.189   21.5 0.138     28
#  3 _AAAAAAAAAAR_     [1, 1, 1,… TUM_first_pool_… -14.8  0.0136  14.3 0.0849    10
#  4 _AAAAAAAAAKNGSSG… [1, 1, 1,… TUM_first_pool_… -22.6  5.12    12.0 0.956     38
#  5 _AAAAAAAAAVSR_    [1, 1, 1,… TUM_first_pool_…   7.36 0       19.6 0         10
#  6 _AAAAAAAAGAFAGR_  [1, 1, 1,… TUM_first_pool_…  38.3  0.333   27.1 0.482     50
#  7 _AAAAAAAAVPSAGPA… [1, 1, 1,… TUM_first_pool_…  38.5  0.564   27.5 0.397     24
#  8 _AAAAAAALQAK_     [1, 1, 1,… TUM_first_pool_…  10.7  0.242   20.1 0.237     86
#  9 _AAAAAAAPSGGGGGG… [1, 1, 1,… TUM_first_pool_…   8.00 0.256   20.1 0.457     83
# 10 _AAAAAAARGSNSDSP… [1, 1, 1,… TUM_second_pool… -24.3  0.222   12.2 0.182    213
# # … with 211,633 more rows

combined_df <- irts %>%
  mutate(
    Sequences = gsub("_", "", `Modified sequence`),
    SequenceEncoding = Encoding) %>%
  select(Sequences, SequenceEncoding, mIRT) %>%
  full_join(specs) %>%
  rename(SpectraEncoding = Encodings) %>%
  filter(!is.na(SpectraEncoding)) %>%
  filter(!is.na(SequenceEncoding))

combined_df
# # A tibble: 252,008 x 5
#    Sequences   SequenceEncoding         mIRT SpectraEncoding       Charges
#    <chr>       <chr>                   <dbl> <chr>                   <dbl>
#  1 AAAAAAAAAA… [1, 1, 1, 1, 1, 1, 1,… 117.   [0, 0.05214702555813…       2
#  2 AAAAAAAAAA… [1, 1, 1, 1, 1, 1, 1,… 117.   [0, 0.43244274910802…       3
#  3 AAAAAAAAAA… [1, 1, 1, 1, 1, 1, 1,…  13.7  [0, 0.05618082254795…       2
#  4 AAAAAAAAAV… [1, 1, 1, 1, 1, 1, 1,…   7.36 [0, 0.04080488642098…       2
#  5 AAAAAAAAGA… [1, 1, 1, 1, 1, 1, 1,…  38.3  [0, 0.02703574416550…       2
#  6 AAAAAAAAVP… [1, 1, 1, 1, 1, 1, 1,…  38.5  [0, 0, 0, 0, 0, 0, 0…       2
#  7 AAAAAAALQAK [1, 1, 1, 1, 1, 1, 1,…  10.7  [0, 0.03440098202210…       2
#  8 AAAAAAAPSG… [1, 1, 1, 1, 1, 1, 1,…   8.00 [0, 0.02102716702781…       3
#  9 AAAAAAAPSG… [1, 1, 1, 1, 1, 1, 1,…   8.00 [0, 0, 0, 0, 0, 0, 0…       2
# 10 AAAAAAARGS… [1, 1, 1, 1, 1, 1, 1,… -24.3  [0, 0.06499065219735…       2
# # … with 251,998 more rows

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
# # A tibble: 38,070 x 5
#    Sequences    SequenceEncoding        mIRT SpectraEncoding       Charges
#    <chr>        <chr>                  <dbl> <chr>                   <dbl>
#  1 AAAAAAARGSN… [1, 1, 1, 1, 1, 1, 1,… -24.3 [0, 0.06499065219735…       2
#  2 AAAAAAARGSN… [1, 1, 1, 1, 1, 1, 1,… -24.3 [0, 0.03893089912459…       3
#  3 AAAAAAEAGDI… [1, 1, 1, 1, 1, 1, 4,…  87.2 [0, 0.06827073449199…       3
#  4 AAAAAATAPPS… [1, 1, 1, 1, 1, 1, 17…  16.3 [0, 0, 0, 0, 0, 0, 0…       2
#  5 AAAAAGGAPGP… [1, 1, 1, 1, 1, 6, 6,…  37.2 [0, 0.11028480222482…       2
#  6 AAAAASASQDE… [1, 1, 1, 1, 1, 16, 1…  52.2 [0, 0.13942941877580…       2
#  7 AAAADSFSGGP… [1, 1, 1, 1, 3, 16, 5…  59.4 [0, 0.38542166233931…       2
#  8 AAAADSFSGGP… [1, 1, 1, 1, 3, 16, 5…  59.4 [0, 0.08601441098844…       3
#  9 AAAAGDADDEPR [1, 1, 1, 1, 6, 3, 1,… -20.5 [0, 0.11930653011442…       2
# 10 AAAAGEPEPPA… [1, 1, 1, 1, 6, 4, 13…  34.4 [0, 0.01957774922302…       2
# # … with 38,060 more rows

val_df <- combined_df %>%
  filter(Sequences %in% val_seqs)
val_df
# # A tibble: 37,733 x 5
#    Sequences    SequenceEncoding         mIRT SpectraEncoding      Charges
#    <chr>        <chr>                   <dbl> <chr>                  <dbl>
#  1 AAAAAAAAGAF… [1, 1, 1, 1, 1, 1, 1,…  38.3  [0, 0.0270357441655…       2
#  2 AAAAAAALQAK  [1, 1, 1, 1, 1, 1, 1,…  10.7  [0, 0.0344009820221…       2
#  3 AAAAAAASFAA… [1, 1, 1, 1, 1, 1, 1,… 121.   [0, 0.0672610462097…       4
#  4 AAAAAAASFAA… [1, 1, 1, 1, 1, 1, 1,… 121.   [0, 0, 0, 0, 0, 0, …       3
#  5 AAAAASAPQQL… [1, 1, 1, 1, 1, 16, 1… 105.   [0, 0.0676950253939…       2
#  6 AAAAELQDR    [1, 1, 1, 1, 4, 10, 1…  -2.82 [0, 0.2302540941700…       2
#  7 AAAASPRPGFW… [1, 1, 1, 1, 16, 13, …  20.5  [0, 0.1463855546729…       4
#  8 AAAATAAEGVP… [1, 1, 1, 1, 17, 1, 1…   2.72 [0, 0.0622093117545…       2
#  9 AAAAWALGQIGR [1, 1, 1, 1, 19, 1, 1…  89.0  [0, 0.0429541136338…       2
# 10 AAAAWYRPAGRR [1, 1, 1, 1, 19, 20, …   6.18 [0, 0.1413981393656…       3
# # … with 37,723 more rows

train_df <- combined_df %>%
  filter(Sequences %in% train_seqs)
train_df
# # A tibble: 176,205 x 5
#    Sequences    SequenceEncoding         mIRT SpectraEncoding      Charges
#    <chr>        <chr>                   <dbl> <chr>                  <dbl>
#  1 AAAAAAAAAAA… [1, 1, 1, 1, 1, 1, 1,… 117.   [0, 0.0521470255581…       2
#  2 AAAAAAAAAAA… [1, 1, 1, 1, 1, 1, 1,… 117.   [0, 0.4324427491080…       3
#  3 AAAAAAAAAAG… [1, 1, 1, 1, 1, 1, 1,…  13.7  [0, 0.0561808225479…       2
#  4 AAAAAAAAAVSR [1, 1, 1, 1, 1, 1, 1,…   7.36 [0, 0.0408048864209…       2
#  5 AAAAAAAAVPS… [1, 1, 1, 1, 1, 1, 1,…  38.5  [0, 0, 0, 0, 0, 0, …       2
#  6 AAAAAAAPSGG… [1, 1, 1, 1, 1, 1, 1,…   8.00 [0, 0.0210271670278…       3
#  7 AAAAAAAPSGG… [1, 1, 1, 1, 1, 1, 1,…   8.00 [0, 0, 0, 0, 0, 0, …       2
#  8 AAAAAAAVGGQ… [1, 1, 1, 1, 1, 1, 1,…  92.4  [0, 0.0162064945835…       2
#  9 AAAAAAGAASG… [1, 1, 1, 1, 1, 1, 6,…  71.8  [0, 0.0787195611243…       2
# 10 AAAAAAGSGTP… [1, 1, 1, 1, 1, 1, 6,…  32.3  [0, 0.0712724334258…       3
# # … with 176,195 more rows

write_csv(holdout_df, "combined_holdout.csv")
write_csv(val_df, "combined_val.csv")
write_csv(train_df, "combined_train.csv")
