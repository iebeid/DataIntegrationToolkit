library(readxl)
library(generator)
library(dplyr)
library(stringr)
library(rlErrorGeneratoR)
library(purrr)

#### Prepare Seed Data -- Last Name ####

#### Step 1: Load Top Last Name Table ####

LastName_Top <- read_excel("LastName_Top.xlsx")

#### Step 2: calculate cumulative freq ####

LastName_Top$cum_Freq <- cumsum(LastName_Top$Freq)

totalFreq_LastName <- sum(LastName_Top$Freq)

#### Step 3: Create range for each Last Name ####

LastName_Top$min <- NA
LastName_Top$max <- NA

min <- 1

for(i in 1:nrow(LastName_Top)){
  
  max <- LastName_Top[i,3]
  
  LastName_Top[i,4] <- min
  
  LastName_Top[i,5] <- max
  
  min <- max + 1
}

#### Step 4: generate xxxx random number between 1 and sum(freq) ####

tmp <- sample(1:totalFreq_LastName, 10000, replace=FALSE)

#### Step 5: select the Last Name if the number falls in the range ####

LastName_selected <- c()

for(i in 1:length(tmp)){
  
  for(j in 1:nrow(LastName_Top)){
    
    if(tmp[i] >= LastName_Top[j,4] & tmp[i] <= LastName_Top[j,5]){
      
      LastName <- LastName_Top[j,1]
        
      LastName_selected <- c(LastName_selected, LastName) 
      
    }
    
  }
  
}

#### Step 6: Create dataframe for selected Last Name ####

LastName_selected <- as.data.frame(LastName_selected)

LastName_selected_T <- t(LastName_selected)

LastName_selected_T <- as.data.frame(LastName_selected_T)

row.names(LastName_selected_T) <- NULL

View(table(LastName_selected_T$V1))




#### Prepare Seed Data -- First Name ####

#### Step 1: Load Top First Name Table ####

FirstName_Top <- read_excel("UALR/Research/Synthetic Data/SOG mimic/FirstName_Top.xlsx")

#### Step 2: Create First Name tables for female and male ####

FirstName_Top_F <- FirstName_Top[which(FirstName_Top$Gender == 'F'),]

FirstName_Top_M <- FirstName_Top[which(FirstName_Top$Gender == 'M'),]

#### Step 3: calculate cumulative freq ####

FirstName_Top_F$cum_Freq <- cumsum(FirstName_Top_F$Freq)

totalFreq_FirstName_F <- sum(FirstName_Top_F$Freq)

FirstName_Top_M$cum_Freq <- cumsum(FirstName_Top_M$Freq)

totalFreq_FirstName_M <- sum(FirstName_Top_M$Freq)

#### Step 4: Create range for each First Name ####

FirstName_Top_F$min <- NA
FirstName_Top_F$max <- NA

min <- 1

for(i in 1:nrow(FirstName_Top_F)){
  
  max <- FirstName_Top_F[i,4]
  
  FirstName_Top_F[i,5] <- min
  
  FirstName_Top_F[i,6] <- max
  
  min <- max + 1
}

FirstName_Top_M$min <- NA
FirstName_Top_M$max <- NA

min <- 1

for(i in 1:nrow(FirstName_Top_M)){
  
  max <- FirstName_Top_M[i,4]
  
  FirstName_Top_M[i,5] <- min
  
  FirstName_Top_M[i,6] <- max
  
  min <- max + 1
}

#### Step 5: generate xxxx random number between 1 and sum(freq) ####

tmp_F <- sample(1:totalFreq_FirstName_F, 5000, replace=FALSE)

tmp_M <- sample(1:totalFreq_FirstName_M, 5000, replace=FALSE)

#### Step 6: select the First Name if the number falls in the range ####

FirstName_F_selected <- c()

for(i in 1:length(tmp_F)){
  
  for(j in 1:nrow(FirstName_Top_F)){
    
    if(tmp_F[i] >= FirstName_Top_F[j,5] & tmp_F[i] <= FirstName_Top_F[j,6]){
      
      FirstName_F <- FirstName_Top_F[j,1]
      
      FirstName_F_selected <- c(FirstName_F_selected, FirstName_F) 
      
    }
    
  }
  
}

FirstName_M_selected <- c()

for(i in 1:length(tmp_M)){
  
  for(j in 1:nrow(FirstName_Top_M)){
    
    if(tmp_M[i] >= FirstName_Top_M[j,5] & tmp_M[i] <= FirstName_Top_M[j,6]){
      
      FirstName_M <- FirstName_Top_M[j,1]
      
      FirstName_M_selected <- c(FirstName_M_selected, FirstName_M) 
      
    }
    
  }
  
}

#### Step 7: Create dataframe for selected First Name ####

FirstName_F_selected <- as.data.frame(FirstName_F_selected)

FirstName_F_selected_T <- t(FirstName_F_selected)

FirstName_F_selected_T <- as.data.frame(FirstName_F_selected_T)

FirstName_F_selected_T$Gender <- 'F'

row.names(FirstName_F_selected_T) <- NULL

View(table(FirstName_F_selected_T$V1))

FirstName_M_selected <- as.data.frame(FirstName_M_selected)

FirstName_M_selected_T <- t(FirstName_M_selected)

FirstName_M_selected_T <- as.data.frame(FirstName_M_selected_T)

FirstName_M_selected_T$Gender <- 'M'

row.names(FirstName_M_selected_T) <- NULL

View(table(FirstName_M_selected_T$V1))

#### Step 7: combine First Name Female and Male together.

FirstName_selected <- rbind(FirstName_F_selected_T,FirstName_M_selected_T)


#### Prepare Seed Data -- Other Attributes ####

#### Step 1: Load Address List ####

AddressList <- read_excel("UALR/Research/Synthetic Data/SOG mimic/AddressList.xlsx")

AddressList[is.na(AddressList)] <- ""

#### Step 2: concatenate full address ####

AddressList$Address_1 <- paste(
  AddressList$US_Street_Number,
  AddressList$`US_Street_Pre-direction`,
  AddressList$US_Street_Name,
  AddressList$US_Street_Suffix,
  AddressList$`US_Street_Post-direction`,
  AddressList$US_Street_Secondary_Abbrev,
  AddressList$US_Street_Secondary_Number,
  sep = ' '
)

AddressList$Address_2 <- paste(
  AddressList$City,
  AddressList$State,
  AddressList$US_ZIP,
  sep = ', '
)

AddressList$Full_Address <- paste(
  AddressList$Address_1,
  AddressList$Address_2,
  sep = ', '
)

#### Step 3: select xxxx address ####

AddressList_selected <- AddressList[1:10000,14]

#### Step 4: Create Seed data(Last Name, First Name, Address etc) ####

seed <- data.frame(
  recid = seq(1:10000),
  fname = FirstName_selected$V1,
  lname = LastName_selected_T$V1,
  gender = FirstName_selected$Gender,
  address_full = AddressList_selected$Full_Address,
  ssn = r_national_identification_numbers(10000),
  dob = r_date_of_births(10000, start = as.Date("1900-01-01"), end = Sys.Date()),
  credit_card_num = r_credit_card_numbers(10000),
  homephone = r_phone_numbers(10000, use_parentheses = TRUE, use_hyphens = TRUE)
)

row.names(seed) <- NULL

View(seed)


#### Data Corruption Functions ####

#### function1: duplicate rows ####
duplicate_rows <- function(df){
  set.seed(1)
  df$reptime <- rexp(nrow(df), 0.05)
  dupl <- df[rep(seq_len(nrow(df)),df$reptime),]
  dupl$reptime <- NULL
  dupl
}

#### function 2: swap columns ####
swap_cols <- function(df,n_errors,col_names){
  lookup <- data.frame()
  for(i in seq(1,length(col_names), 2)){
    col_1 <- col_names[i]
    col_2 <- col_names[i + 1]
    rows <- sample(nrow(df), n_errors)
    for(j in 1:length(rows)){
      row <- rows[j]
      c1 <- df[row, col_1]
      c2 <- df[row, col_2]
      df[row, col_1] <- c2
      df[row, col_2] <- c1
      lookup_i <- data.frame(recid = df[row,"recid"],
                             error_type = 'swap_cols',
                             column = col_names,
                             before = str_c(c1, c2, sep = ", "),
                             after = str_c(c2, c1, sep = ", "))
      lookup <- rbind(lookup,lookup_i)
    }
  }
  swap_combo = list(df=df,lookup=lookup)
  swap_combo
}

#function3: remove hyphens
remove_hyphens <- function(df,n_errors,col_names){
  lookup <- data.frame()
  col <- col_names
  rows <- sample(nrow(df), n_errors)
  for(i in 1:length(rows)){
    row <- rows[i]
    beforestring <- df[row,col]
    df[row,col] <- gsub("-", "", df[row,col])
    afterstring <- df[row,col]
    lookup_i <- data.frame(recid = df[row,"recid"],
                           error_type = "remove_hyphens",
                           column = col_names,
                           before = beforestring,
                           after = afterstring)
    lookup <- rbind(lookup,lookup_i)
  }
  remove_hyphens_combo = list(df = df,lookup = lookup)
  remove_hyphens_combo
}

#### function 4: remove parentheses ####
remove_parentheses  <- function(df,n_errors,col_names){
  lookup <- data.frame()
  col <- col_names
  rows <- sample(nrow(df), n_errors)
  for(i in 1:length(rows)){
    row <- rows[i]
    beforestring <- df[row,col]
    df[row,col] <- gsub("[()]", "", df[row,col])
    afterstring <- df[row,col]
    lookup_i <- data.frame(recid = df[row,"recid"],
                           error_type = 'remove_parentheses',
                           column = col_names,
                           before = beforestring,
                           after = afterstring)
    lookup <- rbind(lookup,lookup_i)
  }
  remove_parentheses_combo = list(df = df,lookup = lookup)
  remove_parentheses_combo
}

#### function 5: remove forward slash ####
remove_fslash  <- function(df,n_errors,col_names){
  lookup <- data.frame()
  col <- col_names
  rows <- sample(nrow(df), n_errors)
  for(i in 1:length(rows)){
    row <- rows[i]
    beforestring <- df[row,col]
    df[row,col] <- gsub("/", "", df[row,col])
    afterstring <- df[row,col]
    lookup_i <- data.frame(recid = df[row,"recid"],
                           error_type = "remove_fslash",
                           column = col_names,
                           before = beforestring,
                           after = afterstring)
    lookup <- rbind(lookup,lookup_i)
  }
  remove_fslash_combo = list(df = df,lookup = lookup)
  remove_fslash_combo
}

#### function 6: transpose characters ####
transpose_ch <- function(df,n_errors,col_names){
  lookup <- data.frame()
  col <- col_names
  rows <- sample(nrow(df), n_errors)
  for(i in 1:length(rows)){
    row <- rows[i]
    beforestring <- df[row,col]
    df[row,col] <- transpose.base(df[row,col])
    afterstring <- df[row,col]
    lookup_i <- data.frame(recid = df[row,"recid"],
                           error_type = "transpose_ch",
                           column = col_names,
                           before = beforestring,
                           after = afterstring)
    lookup <- rbind(lookup,lookup_i)
  }
  transpose_ch_combo = list(df = df,lookup = lookup)
  transpose_ch_combo
}

#### function 7: repeate one character ####
rep_onech <- function(df,n_errors,col_names){
  lookup <- data.frame()
  col <- col_names
  rows <- sample(nrow(df), n_errors)
  for(i in 1:length(rows)){
    row <- rows[i]
    beforestring <- df[row,col]
    n <- sample(nchar(beforestring),1)
    letter <- substr(beforestring,n,n)
    df[row,col] <- paste(substr(beforestring, 1, n-1), 
                         strrep(letter, 1), 
                         substr(beforestring, n, nchar(beforestring)), sep = "")
    afterstring <- df[row,col]
    lookup_i <- data.frame(recid = df[row,"recid"],
                           error_type = "rep_onech",
                           column = col_names,
                           before = beforestring,
                           after = afterstring)
    lookup <- rbind(lookup,lookup_i)
  }
  rep_onech_combo = list(df = df,lookup = lookup)
  rep_onech_combo
}

#### function 8: only keep the initial ####
initials <- function(df,n_errors,col_names){
  lookup <- data.frame()
  col <- col_names
  rows <- sample(nrow(df), n_errors)
  for(i in 1:length(rows)){
    row <- rows[i]
    beforestring <- df[row,col]
    df[row,col] <- substr(beforestring,1,1)
    afterstring <- df[row,col]
    lookup_i <- data.frame(recid = df[row,"recid"],
                           error_type = "initials",
                           column = col_names,
                           before = beforestring,
                           after = afterstring)
    lookup <- rbind(lookup,lookup_i)
  }
  initials_combo = list(df = df,lookup = lookup)
  initials_combo
}

#### function 10: change females lname for marriage ####
married <- function(df, n_errors, lname,gender){
  candidates_df <- df[df[[gender]] == "F",]
  if (n_errors > nrow(candidates_df)){
    warning("The number of errors assigned is greater than total number of females in the dataset.")
    n_errors <- nrow(candidates_df)
  }
  lookup <- data.frame()
  col <- lname
  rows <- sample(nrow(df), n_errors)
  for(i in 1:length(rows)){
    row <- rows[i]
    oldlname <- df[row,col]
    newlname <- LastName_Top[sample(nrow(LastName_Top),1),1]
    df[row,col] <- newlname
    lookup_i <- data.frame(recid = df[row,"recid"],
                           error_type = "married",
                           column = col,
                           before = oldlname,
                           after = newlname$Lname)
    lookup <- rbind(lookup,lookup_i)
  }
  married_combo = list(df = df,lookup = lookup)
  married_combo
}

#### function 12: create twins identitity ####
create_twins <- function(df,n_errors,colname){
  lookup <- data.frame()
  col <- colname
  rows <- sample(nrow(df), n_errors)
  for(i in 1:length(rows)){
    row <- rows[i]
    oldrec <- df[row,]
    oldrec_fname <- df[row,col]
    
    search_name <- function(name){
      fname_len = str_length(FirstName_Top$Fname)
      FirstName_Top %>%
        filter( fname_len == str_length(name),
                str_sub(FirstName_Top$Fname, 1, 1) == str_sub(name, 1, 1)) %>%
        sample_n(1) %>%
        select(-fname_len)
    }
    
    twinrec_fname <- map_df(oldrec_fname, search_name)
    print(twinrec_fname)
    twinrec_id <- paste("000", oldrec[1,1], "999",sep = "")
    
    twinrec <- data.frame(recid = twinrec_id,
                          fname = twinrec_fname$Fname,
                          lname = as.character(df[row,"lname"]),
                          gender = as.character(df[row,"gender"]),
                          address_full = as.character(df[row,"address_full"]),
                          ssn = r_national_identification_numbers(1),
                          dob = as.character(df[row,"dob"]),
                          credit_card_num = as.character(df[row,"credit_card_num"]),
                          homephone = as.character(df[row,"homephone"]),
                          idtruth = as.character(max(as.integer(df$idtruth))+1))
    df <- rbind(df,twinrec)
    
    lookup_i <- data.frame(recid = df[row,"recid"],
                           error_type = "create_twins",
                           column = "all columns",
                           before = paste(df[row,"recid"],oldrec_fname,sep = ", "),
                           after = paste(twinrec$recid,twinrec_fname,sep = ", "))
    lookup <- rbind(lookup,lookup_i)
  }
  create_twins_combo = list(df = df,lookup = lookup)
  create_twins_combo
}

#### function 13: randomly delete item ####
random_delete <- function(df,n_errors,col_names,recid){
  lookup <- data.frame()
  col <- col_names
  rows <- sample(nrow(df), n_errors)
  for(i in 1:length(rows)){
    row <- rows[i]
    beforestring <- df[row,col]
    df[row,col] <- ""
    afterstring <- df[row,col]
    lookup_i <- data.frame(recid = df$recid[row],
                           error_type = "random_delete",
                           column = col_names,
                           before = beforestring,
                           after = afterstring)
    lookup <- rbind(lookup,lookup_i)
  }
  random_delete_combo = list(df = df,lookup = lookup)
  random_delete_combo
}

#### function 14: nickname change ####
nicknames <- read_excel("UALR/Research/Synthetic Data/SOG mimic/nicknames.xlsx")
to_nicknames <- function(df,col_names,nicknames){
  lookup <- data.frame()
  col <- col_names
  col1 <- data.frame(names = unique(df[,col]))
  col2 <- data.frame(names = unique(nicknames$lookup_name))
  commons <- data.frame(merge(col1, col2,all = FALSE))
  for(i in 1:nrow(commons)){
    candidates_df <- df[which(df[,col] == commons[i,1]),]
    candidate_df <- candidates_df[sample(nrow(candidates_df), 1), ]
    beforestring <- candidate_df[1,col]
    candidates_nick <- nicknames[which(nicknames$lookup_name == commons[i,1]),]
    candidate_nick <- candidates_nick[sample(nrow(candidates_nick), 1), ]
    afterstring <- candidate_nick[1,"lookup_alternate"]
    df[rownames(candidate_df),col] <- afterstring
    lookup_i <- data.frame(recid = candidate_df$recid,
                           error_type = "to_nicknames",
                           column = col,
                           before = beforestring,
                           after = afterstring)
    lookup <- rbind(lookup,lookup_i)
  }
  
  to_nicknames_combo = list(df = df,lookup = lookup)
  to_nicknames_combo
}

#### function 15: address change ####
newaddresses <- data.frame(newaddress = AddressList$Full_Address)
newaddresses$newaddress <- as.character(newaddresses$newaddress)
change_address <- function(df, n_errors, address){
  lookup <- data.frame()
  col <- address
  rows <- sample(nrow(df), n_errors)
  for(i in 1:length(rows)){
    row <- rows[i]
    oldaddress <- df[row,col]
    newaddress <- newaddresses[sample(nrow(newaddresses),1),]
    df[row,col] <- newaddress
    lookup_i <- data.frame(recid = df[row,"recid"],
                           error_type = "change_address",
                           column = col,
                           before = oldaddress,
                           after = newaddress)
    lookup <- rbind(lookup,lookup_i)
  }
  change_address_combo = list(df = df,lookup = lookup)
  change_address_combo
}


#### function 16: to lower case ####
to_lower <- function(df,n_errors,col_names){
  col <- col_names
  lookup <- data.frame()
  rows <- sample(nrow(df), n_errors)
  for(i in 1:length(rows)){
    row <- rows[i]
    beforestring <- df[row,col]
    df[row,col] <- tolower(df[row,col])
    afterstring <- df[row,col]
    
    lookup_i <- data.frame(recid = df[row,"recid"],
                           error_type = "to_lower",
                           column = col_names,
                           before = beforestring,
                           after = afterstring)
    lookup <- rbind(lookup,lookup_i)
  }
  to_lower_combo = list(df = df,lookup = lookup)
  to_lower_combo
}


#### Data Corruption Step by Step ####

clean_dataset <- seed

#### STEP 1: Generate duplicate records ####
system.time(df_1 <-duplicate_rows(clean_dataset))
df_1$idtruth <- df_1$recid
df_1$recid <-  rownames(df_1) 
rownames(df_1) <- NULL

hist(data.frame(table(df_1$id))$Freq)

#### STEP 2: swap the value in fname & lname ####
df_1$fname <- as.character(df_1$fname)
df_1$lname <- as.character(df_1$lname)
system.time(df_2combo <- swap_cols(df_1,2000,c("fname","lname")))
df_2 <- df_2combo$df 

#### STEP 3: remove the hyphen in ssn ####
df_2$ssn <- as.character(df_2$ssn)
system.time(df_3combo <- remove_hyphens(df_2,10000 ,'ssn'))
df_3 <- df_3combo$df

#### STEP 4: remove the hyphen in dob ####
df_3$dob <- as.character(df_3$dob)
system.time(df_4combo <- remove_hyphens(df_3,10000 ,'dob'))
df_4 <- df_4combo$df

#### STEP 5: remove the hyphen in credit_card_num ####
df_4$credit_card_num <- as.character(df_4$credit_card_num)
system.time(df_5combo <- remove_hyphens(df_4,10000 ,'credit_card_num'))
df_5 <- df_5combo$df

#### STEP 6: remove the hyphen in homephone ####
df_5$homephone <- as.character(df_5$homephone)
system.time(df_6combo <- remove_hyphens(df_5,10000 ,'homephone'))
df_6 <- df_6combo$df

#### STEP 7: remove the remove_parentheses in homephone ####
df_6$homephone <- as.character(df_6$homephone)
system.time(df_7combo <- remove_parentheses(df_6,10000,'homephone'))
df_7 <- df_7combo$df

#### STEP 8: repeat one character in fname ####
df_7$fname <- as.character(df_7$fname)
system.time(df_8combo <- rep_onech(df_7,3000,"fname"))
df_8 <- df_8combo$df 

#### STEP 9: transpose two characters in ssn #### 
df_8$ssn <- as.character(df_8$ssn)
system.time(df_9combo <- transpose_ch(df_8,5000 ,"ssn"))
df_9 <- df_9combo$df

#### STEP 10: only keep the initial of fname #### 
df_9$fname <- as.character(df_9$fname)
system.time(df_10combo <- initials(df_9,5000,'fname'))
df_10 <- df_10combo$df

#### STEP 11: transpose two characters in dob #### 
df_10$dob <- as.character(df_10$dob)
system.time(df_11combo <- transpose_ch(df_10,5000 ,'dob'))
df_11 <- df_11combo$df

#### STEP 12: transpose two characters in credit_card_num #### 
df_11$credit_card_num <- as.character(df_11$credit_card_num)
system.time(df_12combo <- transpose_ch(df_11,5000 ,"credit_card_num"))
df_12 <- df_12combo$df

#### STEP 13: transpose two characters in homephone #### 
df_12$homephone <- as.character(df_12$homephone)
system.time(df_13combo <- transpose_ch(df_12,5000 ,"homephone"))
df_13 <- df_13combo$df

#### STEP 14: randomly repeat one character in lname #### 
df_13$lname <- as.character(df_13$lname)
system.time(df_14combo <- rep_onech(df_13,2000 ,"lname"))
df_14 <- df_14combo$df

#### STEP 15: change lname (females only) for marriage. #### 
df_14$lname <- as.character(df_14$lname)
system.time(df_15combo <- married(df_14,3000,"lname",'gender'))
df_15 <- df_15combo$df

#### STEP 16:change the full_address. #### 
df_15$address_full <- as.character(df_15$address_full)
system.time(df_16combo <- change_address(df_15,5000,"address_full"))
df_16 <- df_16combo$df

#### STEP 17: randomly repeat one character in fname #### 
df_16$fname <- as.character(df_16$fname)
system.time(df_17combo <- rep_onech(df_16,2000,"fname"))
df_17 <- df_17combo$df

#### STEP 18: delete gender #### 
df_17$gender <- as.character(df_17$gender)
system.time(df_18combo <- random_delete(df_17,20000 ,"gender"))
df_18 <- df_18combo$df

#### STEP 19: delete ssn  #### 
df_18$ssn <- as.character(df_18$ssn)
system.time(df_19combo <- random_delete(df_18,20000 ,"ssn"))
df_19 <- df_19combo$df

#### STEP 20: delete dob  #### 
df_19$dob <- as.character(df_19$dob)
system.time(df_20combo <- random_delete(df_19,20000 ,"dob"))
df_20 <- df_20combo$df

#### STEP 21: delete credit_card_num #### 
df_20$credit_card_num <- as.character(df_20$credit_card_num)
system.time(df_21combo <- random_delete(df_20,20000 ,"credit_card_num"))
df_21 <- df_21combo$df

#### STEP 22: delete homephone #### 
df_21$homephone <- as.character(df_21$homephone)
system.time(df_22combo <- random_delete(df_21,20000 ,"homephone"))
df_22 <- df_22combo$df

#### STEP 23: delete address_full #### 
df_22$address_full <- as.character(df_22$address_full)
system.time(df_23combo <- random_delete(df_22,20000 ,"address_full"))
df_23 <- df_23combo$df

#### Concatenate all error lookup table ####
full_error_lookup <- rbind(df_2combo$lookup,
                           df_3combo$lookup,
                           df_4combo$lookup,
                           df_5combo$lookup,
                           df_6combo$lookup,
                           df_7combo$lookup,
                           df_8combo$lookup,
                           df_9combo$lookup,
                           df_10combo$lookup,
                           df_11combo$lookup,
                           df_12combo$lookup,
                           df_13combo$lookup,
                           df_14combo$lookup,
                           df_15combo$lookup,
                           df_16combo$lookup,
                           df_17combo$lookup,
                           df_18combo$lookup,
                           df_19combo$lookup,
                           df_20combo$lookup,
                           df_21combo$lookup,
                           df_22combo$lookup,
                           df_23combo$lookup)

#### one output table ####
final_table <- df_23
final_table_shuffle <- final_table[sample(nrow(final_table)),]

#Re-format (assign new) recid for final_table (Axxxxx,A999999)
newrecidfinal_table <- seq.int(805296,999999)
final_tablerecid <- data.frame(newrecidfinal_table)
final_tablerecid$newrecidfinal_table <- as.character(final_tablerecid$newrecidfinal_table)
final_tablerecid$newrecidfinal_table <-paste('A',final_tablerecid$newrecidfinal_table,sep = '')
final_table_shuffle <- cbind(final_tablerecid, final_table_shuffle)
final_table_recid_lookup <- data.frame(recid_new = final_table_shuffle$newrecidfinal_table,
                                       recid_old = final_table_shuffle$recid)

#remove old recid in final_table
final_table_shuffle$recid <- final_table_shuffle$newrecidfinal_table
final_table_shuffle$newrecidfinal_table <- NULL
rownames(final_table_shuffle) <- NULL

#split full address
final_table_shuffle$streetnbr <- str_split_fixed(final_table_shuffle$address_full, " ", 2)[,1]
final_table_shuffle$addresstemp <- str_split_fixed(final_table_shuffle$address_full, ", ", 2)[,1]
final_table_shuffle$address <- str_split_fixed(final_table_shuffle$addresstemp, " ", 2)[,2]
final_table_shuffle$addresstemp <- NULL

final_table_shuffle$city_state_zip <- str_split_fixed(final_table_shuffle$address_full, ", ", 2)[,2]
final_table_shuffle$city <- str_split_fixed(final_table_shuffle$city_state_zip, ", ", 3)[,1]
final_table_shuffle$state <- str_split_fixed(final_table_shuffle$city_state_zip, ", ", 3)[,2]
final_table_shuffle$zip <- str_split_fixed(final_table_shuffle$city_state_zip, ", ", 3)[,3]
final_table_shuffle$address_full <- NULL
final_table_shuffle$city_state_zip <- NULL

#truth table
truth <- data.frame(recid = final_table_shuffle$recid, idtruth = final_table_shuffle$idtruth)

#remove idtruth in dataset
final_table_shuffle$idtruth <- NULL

#lname frequency
lname <- toupper(final_table_shuffle$lname)
lname <- gsub("[^A-Za-z]","",lname)
freq_lname <- data.frame(table(lname))
View(table(freq_lname$Freq))
hist(freq_lname$Freq)

#write table
write.table(truth,"C:/Users/YumengYe/Documents/UALR/Research/Synthetic Data/SOG mimic/output/194k_all/194k_all_truth.csv",sep=",",row.names = FALSE,quote = FALSE)

write.table(final_table_shuffle,"C:/Users/YumengYe/Documents/UALR/Research/Synthetic Data/SOG mimic/output/194k_all/194k_all.csv",sep=",",row.names = FALSE,quote = FALSE)

write.csv(freq_lname,"C:/Users/YumengYe/Documents/UALR/Research/Synthetic Data/SOG mimic/output/194k_all/freq_lname_freq.csv",row.names = FALSE,quote = FALSE)

