# Makefile for Deep-Rung-Kutta
# Copyright (C) 2025, Shyamal Suhana Chandra

CC = gcc
CXX = g++
CFLAGS = -Wall -Wextra -O2 -fPIC -std=c11
CXXFLAGS = -Wall -Wextra -O2 -fPIC -std=c++11
LDFLAGS = -shared

SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
LIB_DIR = lib
BIN_DIR = bin

SOURCES = $(wildcard $(SRC_DIR)/*.c)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

LIBRARY = $(LIB_DIR)/libddrkam.a
SHARED_LIB = $(LIB_DIR)/libddrkam.dylib

.PHONY: all clean test

all: $(LIBRARY) $(SHARED_LIB)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -I$(INC_DIR) -c $< -o $@

$(LIBRARY): $(OBJECTS) | $(LIB_DIR)
	ar rcs $@ $(OBJECTS)

$(SHARED_LIB): $(OBJECTS) | $(LIB_DIR)
	$(CC) $(LDFLAGS) -o $@ $(OBJECTS)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(LIB_DIR):
	mkdir -p $(LIB_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR) $(BIN_DIR)

test: $(LIBRARY) | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(INC_DIR) tests/test_rk3.c -L$(LIB_DIR) -lddrkam -o $(BIN_DIR)/test_rk3
	$(BIN_DIR)/test_rk3
	$(CC) $(CFLAGS) -I$(INC_DIR) tests/test_ddmcmc.c -L$(LIB_DIR) -lddrkam -o $(BIN_DIR)/test_ddmcmc
	$(BIN_DIR)/test_ddmcmc
	$(CC) $(CFLAGS) -I$(INC_DIR) tests/test_comparison.c -L$(LIB_DIR) -lddrkam -o $(BIN_DIR)/test_comparison
	$(BIN_DIR)/test_comparison
