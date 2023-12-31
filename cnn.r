# Install necessary packages
install.packages("keras")
install.packages("reticulate")

# Load required libraries
library(keras)
library(reticulate)

# Use Python environment for keras
use_python("/path/to/your/python/executable")

# Load the CIFAR-10 dataset
cifar10 <- dataset_cifar10()
c(train_images, train_labels) %<-% cifar10$train
c(test_images, test_labels) %<-% cifar10$test

# Preprocess the data
train_images <- train_images / 255
test_images <- test_images / 255
train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

# Build a more complex CNN model
model <- keras_model_sequential() %>%
  # Convolutional layers
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), padding = "same", activation = "relu", input_shape = c(32, 32, 3)) %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(0.25) %>%

  layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same", activation = "relu") %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(0.25) %>%

  # Flatten layer
  layer_flatten() %>%
  
  # Dense layers
  layer_dense(units = 512, activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 10, activation = "softmax")

# Compile the model
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = c("accuracy")
)

# Display the model summary
summary(model)

# Train the model
history <- model %>% fit(
  train_images, train_labels,
  epochs = 50, batch_size = 64,
  validation_split = 0.2
)

# Evaluate the model on test data
test_loss_accuracy <- model %>% evaluate(test_images, test_labels)
cat("Test accuracy:", test_loss_accuracy$acc, "\n")

# Plot training history
plot(history)
