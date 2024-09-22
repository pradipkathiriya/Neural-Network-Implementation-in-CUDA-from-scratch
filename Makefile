# Compiler and flags
NVCC = nvcc
CXX = g++
CXXFLAGS := -std=c++17 -Wall -Wextra -g

# Output target
TARGET = main_cuda

# Source directory
SRC_DIR = src
INCLUDE_DIR = include
# Source files
SRCS = $(wildcard $(SRC_DIR)/*.cu)
SRCS_CPP = $(wildcard $(SRC_DIR)/*.cpp)
INCLUDES = -I./$(INCLUDE_DIR)
# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(SRCS) $(SRCS_CPP)
	$(NVCC) $(INCLUDES) $(SRCS_CPP) $(SRCS) -o $(TARGET)

# Rule to clean the build
clean:
	rm -f $(TARGET)
