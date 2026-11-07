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