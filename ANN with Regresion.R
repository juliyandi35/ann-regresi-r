library(tensorflow)
library(keras)
library(reticulate)
library(caret)

library(readxl)
AQI <- read_excel("jakarta-air-quality.xlsx")
head(AQI)
AQI <- AQI[,-3]
AQI$tahun <- format(AQI$date, "%Y")
AQI_tahunan <- aggregate(pm25 ~ tahun, data = AQI, FUN = mean)
AQI_tahunan$tahun <- as.integer(AQI_tahunan$tahun)
head(AQI_tahunan)

JKB <- read_excel("Jumlah kendaraan bermotor tahunan Jakarta.xlsx")
head(JKB)
str(JKB)
JKB$Tahun <- gsub("\\*", "", JKB$Tahun)
JKB$Tahun <- as.integer(JKB$Tahun)

AQI_tahunan$tahun
JKB$Tahun

AQI <- AQI_tahunan[-c(7),]
JKB <- JKB[-c(1,2),]
colnames(JKB) <- c("Tahun","Mobil_Penumpang","Bus","Truk","Sepeda_Motor","Jumlah_Total")

Dataset <- data.frame(Tahun = AQI$tahun,
                      AQI = AQI$pm25,
                      JKB[,-1])
summary(Dataset)
Dataset <- Dataset[,-1]

# Partisi Data dan Features Scaling
# Membagi data menjadi data latih dan data uji dengan createDataPartition
set.seed(123)

train.index <- createDataPartition(Dataset$AQI, p = 0.5, list = FALSE)
train <- Dataset[train.index, ]
test <- Dataset[-train.index, ]

# Melakukan Feature Scaling min max (0, 1)
preprocessParams <- preProcess(train[, -1], method=c("range"))
train_X <- as.matrix(predict(preprocessParams, train[, -1]))
test_X <- as.matrix(predict(preprocessParams, test[, -1]))

train_y <- train[,1]
test_y <- test[,1]

### Model Regresi ###
# Membuat model neural network dengan 1 hidden layer
model_lm1 <- keras_model_sequential() %>%
  layer_dense(units = 10, activation = "relu", input_shape = ncol(train_X)) %>%
  layer_dense(units = 1, activation = "linear")

# Mengkompilasi model
model_lm1 %>% compile(
  loss = "mean_squared_error",
  optimizer = "adam",
  metrics = list("mean_squared_error", "mean_absolute_error")
)

# Melakukan tahapan pelatihan model
history_lm1 <- model_lm1 %>% fit(
  train_X, train_y,
  shuffle = T,
  epochs = 100,
  batch_size = 10,
  validation_split = 0.2
)

print(model_lm1)

# Menampilkan plot pembelajaran model pada setiap epoch
plot(history_lm1)

# Melakukan prediksi
prediksi_lm1 <- predict(model_lm1, test_X)
head(prediksi_lm1)

# Mengevaluasi model menggunakan data uji
scores_lm1 <- model_lm1 %>% evaluate(test_X, test_y)
print(scores_lm1)

# Membuat model neural network dengan 2 hidden layer
model_lm2 <- keras_model_sequential() %>%
  layer_dense(units = 20, activation = "relu", input_shape = ncol(train_X)) %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 1, activation = "linear")

# Mengkompilasi model
model_lm2 %>% compile(
  loss = "mean_squared_error",
  optimizer = "adam",
  metrics = list("mean_squared_error", "mean_absolute_error")
)

# Melakukan tahapan pelatihan model
history_lm2 <- model_lm2 %>% fit(
  train_X, train_y,
  shuffle = T,
  epochs = 100,
  batch_size = 10,
  validation_split = 0.2
)

print(model_lm2)

# Menampilkan plot pembelajaran model pada setiap epoch
plot(history_lm2)

# Melakukan prediksi
prediksi_lm2 <- predict(model_lm2, test_X)
head(prediksi_lm2)

# Mengevaluasi model menggunakan data uji
scores_lm2 <- model_lm2 %>% evaluate(test_X, test_y)
print(scores_lm2)

#  Membuat model neural network dengan 3 hidden layer
model_lm3 <- keras_model_sequential() %>%
  layer_dense(units = 20, activation = "relu", input_shape = ncol(train_X)) %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 5, activation = "relu") %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 1, activation = "linear")

# Mengkompilasi model
model_lm3 %>% compile(
  loss = "mean_squared_error",
  optimizer = "adam",
  metrics = list("mean_squared_error", "mean_absolute_error")
)

# Melakukan tahapan pelatihan model
history_lm3 <- model_lm3 %>% fit(
  train_X, train_y,
  shuffle = T,
  epochs = 100,
  batch_size = 10,
  validation_split = 0.2
)

print(model_lm3)

# Menampilkan plot pembelajaran model pada setiap epoch
plot(history_lm3)

# Melakukan prediksi
prediksi_lm3 <- predict(model_lm3, test_X)
head(prediksi_lm3)

# Mengevaluasi model menggunakan data uji
scores_lm3 <- model_lm3 %>% evaluate(test_X, test_y)
print(scores_lm3)

Comparison_table <- rbind(scores_lm1,scores_lm2,scores_lm3)
rownames(Comparison_table) <- c("1 Hidden Layer","2 Hidden Layer","3 Hidden Layer")
Comparison_table
