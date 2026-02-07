const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const testing = std.testing;

pub const matrix = @import("matrix");
pub const Matrix = matrix.Matrix(f32);

pub const Activation = enum {
    relu,
    sigmoid,

    pub fn apply(activation: Activation, x: f32) f32 {
        return switch (activation) {
            .relu => @max(0, x),
            .sigmoid => 1 / (1 + @exp(-x)),
        };
    }
};

pub const NeuralNetwork = struct {
    weights: []Matrix,
    biases: []Matrix,
    activations: []Matrix,
    activation_fn: Activation,
    learning_rate: f32,
    eps: f32,

    //pub fn forward(self: *NeuralNetwork, input: Matrix) !Matrix
    //pub fn backward(self: *NeuralNetwork, target: Matrix) !void
    //pub fn train(self: *NeuralNetwork, input: Matrix, target: Matrix) !f32
    //pub fn predict(self: *NeuralNetwork, input: Matrix) !Matrix  // forward without caching
    //pub fn print()
};

//pub fn cost(predicted: Matrix, actual: Matrix) f32 {}
