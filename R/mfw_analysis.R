library(dplyr)

mf_sarc_compare <- function(words, data_sarc, data_nonsarc) {
  mat <- matrix(nrow = length(words), ncol=2)
  i = 1
  for(word in words) {
    pattern = paste(" ", word, " ", sep="")
    sum_sarc =  length(which(grepl(pattern, data_sarc) == TRUE))
    dif1 = sum_sarc / length(data_sarc)
    sum_nonsarc =  length(which(grepl(pattern, data_nonsarc) == TRUE))
    dif2 = sum_nonsarc / length(data_nonsarc)
    mat[[i, 1]] = dif1
    mat[[i, 2]] = dif2
    i = i + 1
  }
  return(mat)
}

mfw_build_data_frame <- function(words, data_sarc, data_nonsarc) {
  mat <- matrix(nrow = length(words), ncol=2)
  data_words = c()
  ratio_sarc = c()
  ratio_nonsarc = c()
  ratio = c()

  for(word in words) {
    pattern = paste(" ", word, " ", sep="")
    sum_sarc =  length(which(grepl(pattern, data_sarc) == TRUE))
    dif_sarc = sum_sarc / length(data_sarc)
    sum_nonsarc =  length(which(grepl(pattern, data_nonsarc) == TRUE))
    dif_nonsarc = sum_nonsarc / length(data_nonsarc)
    diff = dif_sarc / dif_nonsarc
    
    data_words <- c(data_words, word)
    ratio_sarc <- c(ratio_sarc, dif_sarc)
    ratio_nonsarc <- c(ratio_nonsarc, dif_nonsarc)
    ratio <- c(ratio, diff)
  }

  return(data.frame(data_words, ratio_sarc, ratio_nonsarc, ratio))
}

sort_data_frame <- function(df) {
  new_df <- df %>%
    arrange(desc(ratio)) %>%
    select(data_words, ratio_sarc, ratio_nonsarc, ratio)
  
  return(new_df)
}

pattern = " totally "
sum_sarc = length(which(grepl(pattern, SarcComments$post) == TRUE))
dif1 = sum_sarc / length(SarcComments$post)
sum_nonsarc = sum_sarc = length(which(grepl(pattern, NonSarcComments$post) == TRUE))
dif2 = sum_nonsarc / length(NonSarcComments$post)
slices <- c(dif1, dif2)
labels <- c("Sarcastic", "Non-sarcastic")
n = nchar(pattern)
name = paste("Frequency of the word \"", substring(pattern, 2, n-1), "\"", sep = "")
pie(slices, labels, main = name, col = rainbow(3))

pattern = " obviously "
sum_sarc = length(which(grepl(pattern, SarcComments$post) == TRUE))
dif1 = sum_sarc / length(SarcComments$post)
sum_nonsarc = sum_sarc = length(which(grepl(pattern, NonSarcComments$post) == TRUE))
dif2 = sum_nonsarc / length(NonSarcComments$post)
slices <- c(dif1, dif2)
labels <- c("Sarcastic", "Non-sarcastic")
n = nchar(pattern)
name = paste("Frequency of the word \"", substring(pattern, 2, n-1), "\"", sep = "")
pie(slices, labels, main = name, col = rainbow(3))

pattern = " still "
sum_sarc = length(which(grepl(pattern, SarcComments$post) == TRUE))
dif1 = sum_sarc / length(SarcComments$post)
sum_nonsarc = sum_sarc = length(which(grepl(pattern, NonSarcComments$post) == TRUE))
dif2 = sum_nonsarc / length(NonSarcComments$post)
slices <- c(dif1, dif2)
labels <- c("Sarcastic", "Non-sarcastic")
n = nchar(pattern)
name = paste("Frequency of the word \"", substring(pattern, 2, n-1), "\"", sep = "")
pie(slices, labels, main = name, col = rainbow(3))

pattern = " think "
sum_sarc = length(which(grepl(pattern, SarcComments$post) == TRUE))
dif1 = sum_sarc / length(SarcComments$post)
sum_nonsarc = sum_sarc = length(which(grepl(pattern, NonSarcComments$post) == TRUE))
dif2 = sum_nonsarc / length(NonSarcComments$post)
slices <- c(dif1, dif2)
labels <- c("Sarcastic", "Non-sarcastic")
n = nchar(pattern)
name = paste("Frequency of the word \"", substring(pattern, 2, n-1), "\"", sep = "")
pie(slices, labels, main = name, col = rainbow(3))

mat_sarc = mf_sarc_compare(most_freq_words_sarc$word, SarcComments$post, NonSarcComments$post)
mat_sarc
mat_nonsarc = mf_sarc_compare(most_freq_words_non_sarc$word, SarcComments$post, NonSarcComments$post)
mat_nonsarc

mfsw = mfw_build_data_frame(most_freq_words_sarc$word, SarcComments$post, NonSarcComments$post)
mfsw = sort_data_frame(mfsw)
write.table(mfsw, "most_freq_words_sarc_analysis.csv", col.names = c("word", "% sarc", "% nonsarc", "ratio sarc/nonsarc"), sep = ",", row.names = FALSE)

mfnsw = mfw_build_data_frame(most_freq_words_non_sarc$word, SarcComments$post, NonSarcComments$post)
mfnsw = sort_data_frame(mfnsw)
write.table(mfnsw, "most_freq_words_nonsarc_analysis.csv", col.names = c("word", "% sarc", "% nonsarc", "ratio sarc/nonsarc"), sep = ",", row.names = FALSE)
