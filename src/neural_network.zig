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

pub const Layer = struct {
    weights: Matrix,
    biases: Matrix,
    activation: Activation,

    // init()
    // deinit()
    // forward()
};

pub const Network = struct {
    layers: []Layer,
    learning_rate: f32 = 1e-2,
    eps: f32 = 1e-8,

    // init()
    // deinit()
    // forward()
    // train()
    // predict()
};

//pub fn cost(actual: Matrix, predicted: Matrix) f32 {}
