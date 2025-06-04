##############################
######Airbnb Price Prediction######
##############################

#Load related package
library(readr)
library(dplyr)
library(ggplot2)
library(tmap)
library(sf)
library(corrplot)

#Call airbnb dataset
airbnb <- read_csv('/Users/zhuzhu/Downloads/Airbnb_Open_Data.csv')

#Load the file for calculating R^2
source("/Users/zhuzhu/Downloads/DataAnalyticsFunctions.R")
##############################

##############################
#Data Visualization
##############################
#Draw map to show different availability in different neighbourhood

#Group the data by neighbourhood and calculate the average availability
neighbourhood_availability <- airbnb %>%
  group_by(neighbourhood) %>%
  summarise(avg_availability = mean(`availability 365`, na.rm = TRUE))
summary(neighbourhood_availability)

#Classify into categories (low, medium, high) based on availability
#Define breakpoints for categories
neighbourhood_availability <- neighbourhood_availability %>%
  mutate(category = case_when(
    avg_availability < 100 ~ "Low",
    avg_availability >= 100 & avg_availability < 200 ~ "Medium",
    avg_availability >= 200 ~ "High"
  ))
summary(neighbourhood_availability$avg_availability)

#Join the availability data back to the main dataset to include coordinates
airbnb_data_with_neighbourhood <- airbnb %>%
  select(lat, long, neighbourhood) %>%
  inner_join(neighbourhood_availability, by = "neighbourhood")

#Remove rows with missing latitude or longitude values
airbnb_data_with_neighbourhood <- airbnb_data_with_neighbourhood %>%
  filter(!is.na(lat) & !is.na(long))

#Convert the data to a spatial object for mapping
airbnb_sf <- st_as_sf(airbnb_data_with_neighbourhood, coords = c("long", "lat"), crs = 4326)

#Plot the neighborhoods with categories on an OpenStreetMap
tmap_mode("view")  # Set to interactive view mode
tm_shape(airbnb_sf) +
  tm_symbols(fill = "category", fill.scale = tm_scale(values = "set1"), size = 0.1) +
  tm_basemap(server = "OpenStreetMap") +
  tm_title("Neighborhood Availability Categories") +
  tm_layout(legend.outside = TRUE)

##############################

##############################
#Find the correlation between different numerical variables and draw correlation matrix

#Ensure the price column is numeric
airbnb$price <- as.numeric(gsub("[\\$,]", "", airbnb$price))

#Select the relevant numeric columns, including 'price'
numeric_columns <- airbnb %>%
  select_if(is.numeric) %>%
  filter(!is.na(price))  # Ensure 'price' is included and no NA values

#Calculate the correlation matrix
correlation_matrix <- cor(numeric_columns, use = "complete.obs")

#Adjust the plot margins to make space for the title
par(mar = c(1, 1, 4, 1))

#Plot the correlation matrix with custom color scale
corrplot(correlation_matrix, method = "color", 
         col = colorRampPalette(c("blue", "white", "red"))(200),
         col.lim = c(-1, 1),        
         addCoef.col = "black",    # Add correlation coefficients in black
         number.cex = 0.6,         # Adjust the size of the correlation numbers
         tl.col = "black",         # Text labels for variables in black
         tl.cex = 0.8,             # Size of the text labels
         tl.srt = 90,              # Rotate the x-axis labels 45 degrees
         mar = c(0, 0, 2, 0),      # Adjust title margin positioning
         title = "Correlation Heatmap")
##############################

##############################
#Data Preparation
##############################

#Recall the airbnb dataset
airbnb <- read_csv('/Users/zhuzhu/Downloads/Airbnb_Open_Data.csv')

#Drop all the unrelated columns
airbnb$id <-NULL
airbnb$`host id` <-NULL
airbnb$`host name` <-NULL
airbnb$host_identity_verified <-NULL
airbnb$country <-NULL
airbnb$`country code` <-NULL
airbnb$house_rules <-NULL
airbnb$license <-NULL
airbnb$lat <-NULL
airbnb$long <-NULL
airbnb$`last review` <-NULL
airbnb$`calculated host listings count` <-NULL
airbnb$`minimum nights` <-NULL
airbnb$`service fee` <-NULL
airbnb$NAME <-NULL

#Unify the columns name (using underscore instead of blank space, use lowercase instead of uppercase)
colnames(airbnb)[colnames(airbnb) == "neighbourhood group"] <- "neighbourhood_group"
colnames(airbnb)[colnames(airbnb) == "room type"] <- "room_type"
colnames(airbnb)[colnames(airbnb) == "Construction year"] <- "construction_year"
colnames(airbnb)[colnames(airbnb) == "number of reviews"] <- "number_of_reviews"
colnames(airbnb)[colnames(airbnb) == "reviews per month"] <- "reviews_per_month"
colnames(airbnb)[colnames(airbnb) == "review rate number"] <- "review_rate_number"
colnames(airbnb)[colnames(airbnb) == "availability 365"] <- "availability_365"

#Drop the dollar sign and comma and change the datatype to numeric
airbnb$price <- as.numeric(gsub("\\$", "", airbnb$price))
airbnb$price <- as.numeric(gsub(",", "", airbnb$price))

#Filter the values in "availability_365" column to make sure the values are reasonable
lower <- 0
upper <- 365
airbnb <- airbnb %>% filter(availability_365 >= lower & availability_365 <= upper)

#Correct spelling error
airbnb$neighbourhood_group <- gsub("brookln", "Brooklyn", airbnb$neighbourhood_group)

#remove all null value and repeated data
airbnb_clean <- na.omit(airbnb)
airbnb_clean <- unique(airbnb_clean)

#Change all character datatype into factor to facilitate more accurate modeling later
airbnb_clean$neighbourhood_group <- as.factor(airbnb_clean$neighbourhood_group)
airbnb_clean$room_type <- as.factor(airbnb_clean$room_type)
airbnb_clean$neighbourhood <- as.factor(airbnb_clean$neighbourhood)
airbnb_clean$cancellation_policy <- as.factor(airbnb_clean$cancellation_policy)
airbnb_clean$instant_bookable <- as.factor(airbnb_clean$instant_bookable)
View(airbnb_clean)
str(airbnb_clean)

#split dataset into train data and test data
set.seed(123)
train_indices <- sample(1:nrow(airbnb_clean), size = 0.8 * nrow(airbnb_clean))
train_data <- airbnb_clean[train_indices, ]
test_data <- airbnb_clean[-train_indices, ]

##############################
#Modeling 
##############################

#Use k fold cross validation with k=5
nfold <- 5

#The number of observations
n <- nrow(train_data) 

#Create a vector of fold memberships (random order)
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]

#Create an empty dataframe of results
OOS <- data.frame(Linear=rep(NA,nfold),LASSOMIN=rep(NA,nfold),LASSO1SE=rep(NA,nfold),random_forest=rep(NA,nfold))

#Use a for loop to run through the nfold trails
for(k in 1:nfold){ 
  train <- which(foldid!=k)# train on all but fold `k'
  test  <- which(foldid == k)# test on fold `k'

  #Divide the original train_data dataset into two parts: train and test datasets according to the above method
  train_train_data <- train_data[train, ]
  train_test_data <- train_data[test, ]
  
  #Change categorical variables as factors
  #and ensure the consistency of levels between the training set and the test set
  train_train_data$neighbourhood <- factor(train_train_data$neighbourhood)
  train_test_data$neighbourhood <- factor(train_test_data$neighbourhood, levels = levels(train_train_data$neighbourhood))
  
  #Train the linear regression
  model.linear <-lm(price~.,data=train_train_data)
  
  #Install package for Lasso
  installpkg("glmnet")
  library(glmnet)
  
  #Set up the data for LASSO 
  X_poly <- model.matrix(price~ .^2, data = train_data)[,-1] #Include the features and the interactions between all features to enhance the model's ability to fit complex relationships
  My<- train_data$price
  lambda_seq <- 10^seq(-5, 1, by = 0.05) #Generate a series of lambda values
  lassoCV <- cv.glmnet(X_poly,My,alpha = 0.5, lambda = lambda_seq) #Cross validation to train on different lambda values
  
  #Use different lambda to train the LASSO models
  model.lassomin <- glmnet(X_poly[train,],My[train], family="gaussian",lambda = lassoCV$lambda.min)
  model.lasso1se <- glmnet(X_poly[train,],My[train], family="gaussian",lambda = lassoCV$lambda.1se)
  
  #Install package for Random Forest
  installpkg("randomForest")
  library(randomForest)
  
  #Train Random forest model
  model.randomForest <- randomForest(price~.-neighbourhood, data=train_train_data, nodesize=6, ntree = 500, mtry = 4)
  
  
  #Get predictions: type=response so we have continuous numerical value
  pred.linear <- predict(model.linear, newdata=train_test_data, type="response")
  pred.lassomin <- predict(model.lassomin, newx=X_poly[-train,], type="response")
  pred.lasso1se <- predict(model.lasso1se, newx=X_poly[-train,], type="response")
  pred.randomForest  <- predict(model.randomForest, newdata=train_test_data)
  
  
  #Calculate and log R2
  #linear
  OOS$Linear[k] <- R2(y=train_test_data$price, pred=pred.linear,family="gaussian")
  print(OOS)
  #LASSOMIN
  OOS$LASSOMIN[k] <- R2(y=train_test_data$price, pred=pred.lassomin,family="gaussian")
  OOS$LASSOMIN[k]
  #LASSO1SE
  OOS$LASSO1SE[k] <- R2(y=train_test_data$price, pred=pred.lasso1se, family="gaussian")
  OOS$LASSO1SE[k]
  #random forest
  OOS$random_forest[k] <- R2(y=train_test_data$price, pred=pred.randomForest, family="gaussian")
  OOS$random_forest[k]

  ## this will print the progress (iteration that finished)
  print(paste("Iteration",k,"of",nfold,"(thank you for your patience)"))
}
##############################

##############################
#Evaluation 
##############################
#Use out of sample R^2 to evaluate and measure the performance of four models
a <-colMeans(OOS, na.rm = TRUE)
m.OOS <- as.matrix(OOS)
rownames(m.OOS) <- c(1:nfold)

#show the OOS for 5 folds for 4 models
colors <- c("red", "blue", "green", "purple")
barplot(t(as.matrix(OOS)), beside=TRUE,legend=TRUE,args.legend=c(xjust=0.3, yjust=-0.2,cex=0.6),col=colors,
        ylab= bquote( "Out of Sample " ~ R^2), xlab="Fold", names.arg = c(1:5))

#show the average OOS for 5 folds for 4 models
barplot(a, beside=TRUE,legend=TRUE,args.legend=c(xjust=1, yjust=-0.2,cex=0.6),col=colors,
        ylab= bquote( "Out of Sample " ~ R^2), xlab="Fold")


#predict the test dataset
prediction  <- predict(model.randomForest, newdata=test_data)
prediction

deviance <-R2(y=test_data$price, pred=prediction, family="gaussian")
deviance
difference_percentage <- abs(prediction-test_data$price)/test_data$price
mean(difference_percentage)
