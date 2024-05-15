# Infos from here: https://manifesto-project.wzb.eu/down/tutorials/firststepsmanifestoR.html#downloading-the-manifesto-project-dataset
wd <- "..."
setwd(wd)


library(manifestoR)
library(dplyr)
library(writexl)

mp_setapikey(paste0(wd,"..."))

# Downloading the Manifesto Project Dataset (MPDS):
mpds <- mp_maindataset(version = "MPDS2023a")
# lets save this as an xlsx, making it easier to access in python
write_xlsx(mpds, paste0(wd, "/data/MPDS2023a.xlsx"))

germany_annotated <- mp_availability(countryname == "Germany") %>% filter(annotations == TRUE)

# download these documents
germany_corpus <- mp_corpus(germany_annotated)
head(content(germany_corpus[[2]]))

doc <- germany_corpus[[1]]
# gets text and codes for this document:
content(doc)
codes(doc)

table(codes(doc))

# 209 coded documents in english
english_annotated <- mp_availability(TRUE) %>% filter(annotations == TRUE & language == "english")
# 33 coded documents in swedish
swedish_annotated <- mp_availability(TRUE) %>% filter(annotations == TRUE & language == "swedish")
# 137 coded documents in german
german_annotated <- mp_availability(TRUE) %>% filter(annotations == TRUE & language == "german")

english_corpus <- mp_corpus(english_annotated)


# Lets build a nice dataframe with all the quasi-Sentences, codes etc. for a specific corpus:
mp_av <- mp_availability(TRUE) %>% filter(annotations == TRUE & language == "english")
filename <- "english_annotated_corpus"

corpus <- mp_corpus(mp_av)

column_names <- c("q_sentence", "q_sentence_nr", "codes", "manifesto_id", "party",
                  "date", "language", "handbook", "doc_title")
corpus_data <- data.frame(matrix(ncol = length(column_names), nrow = 0))
colnames(corpus_data) <- column_names


for(i in 1:length(corpus)){
  doc <- corpus[[i]]
  # build all the data we need for this doc:
  q_sentence <- content(doc)
  q_sentence_nr <- seq(1, length(q_sentence))
  codes <- codes(doc)
  manifesto_id <- rep(meta(doc)$manifesto_id,length(q_sentence))
  party <- rep(meta(doc)$party,length(q_sentence))
  date <- rep(meta(doc)$date,length(q_sentence))
  language <- rep(meta(doc)$language,length(q_sentence))
  handbook <- rep(meta(doc)$handbook,length(q_sentence))
  title <- rep(meta(doc)$title,length(q_sentence))
  
  # build a df:
  temp_df <- data.frame(q_sentence = q_sentence, q_sentence_nr = q_sentence_nr,
                        codes = codes, manifesto_id = manifesto_id, party = party,
                        date = date, language = language, handbook = handbook,
                        title = title)
  # append it
  corpus_data <- rbind(corpus_data, temp_df)
}

# save the result as an xlsx (now we can work in Python - or keep working in R)
write_xlsx(corpus_data, paste0(wd, "/data/", filename, ".xlsx"))



