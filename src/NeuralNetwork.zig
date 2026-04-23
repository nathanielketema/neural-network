const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;

const NeuralNetwork = @This();
pub const Matrix = @import("matrix.zig").Matrix(f32);

const Io = std.Io;
const Allocator = std.mem.Allocator;

weights: []Matrix,
biases: []Matrix,
deltas: []Matrix,
activations: []Matrix,
activation_fn: Activation,

pub const Activation = enum {
    relu,
    sigmoid,

    pub fn apply(activation_fn: Activation, matrix: *Matrix) void {
        return switch (activation_fn) {
            .relu => {
                for (matrix.data) |*data| {
                    data.* = relu_fn(data.*);
                }
            },
            .sigmoid => {
                for (matrix.data) |*data| {
                    data.* = sigmoid_fn(data.*);
                }
            },
        };
    }

    pub fn derivative(activation_fn: Activation, x: f32) f32 {
        return switch (activation_fn) {
            .relu => if (x > 0) @as(f32, 1.0) else @as(f32, 0.0),
            .sigmoid => x * (1 - x),
        };
    }

    fn relu_fn(x: f32) f32 {
        return @max(0, x);
    }

    fn sigmoid_fn(x: f32) f32 {
        return 1 / (1 + @exp(-x));
    }
};

// Caller must free memory by calling deinit()
pub fn init(gpa: Allocator, options: struct {
    architecture: []const usize,
    activation_fn: Activation = .sigmoid,
}) !NeuralNetwork {
    assert(options.architecture.len >= 2);

    const layer_count = options.architecture.len;
    const edge_count = layer_count - 1;

    const weights = try gpa.alloc(Matrix, edge_count);
    const biases = try gpa.alloc(Matrix, edge_count);
    const deltas = try gpa.alloc(Matrix, edge_count);
    const activations = try gpa.alloc(Matrix, layer_count);

    for (0..edge_count) |i| {
        activations[i] = try .init(gpa, .init(1, options.architecture[i]));
        activations[i].fill(0);

        weights[i] = try .init(gpa, .init(
            options.architecture[i],
            options.architecture[i + 1],
        ));
        weights[i].fill(0);

        biases[i] = try .init(gpa, .init(1, options.architecture[i + 1]));
        biases[i].fill(0);

        deltas[i] = try .init(gpa, .init(1, options.architecture[i + 1]));
        deltas[i].fill(0);
    }
    activations[edge_count] = try .init(gpa, .init(1, options.architecture[edge_count]));
    activations[edge_count].fill(0);

    return .{
        .weights = weights,
        .biases = biases,
        .activations = activations,
        .deltas = deltas,
        .activation_fn = options.activation_fn,
    };
}

pub fn deinit(nn: *NeuralNetwork, gpa: Allocator) void {
    for (nn.weights) |*wgt| wgt.deinit(gpa);
    for (nn.biases) |*bias| bias.deinit(gpa);
    for (nn.activations) |*lyer| lyer.deinit(gpa);
    for (nn.deltas) |*dlta| dlta.deinit(gpa);

    gpa.free(nn.weights);
    gpa.free(nn.biases);
    gpa.free(nn.activations);
    gpa.free(nn.deltas);
}

pub fn fill_rand(nn: *NeuralNetwork, random: std.Random) void {
    for (nn.weights) |*wght| wght.fill_random(random);
    for (nn.biases) |*bias| bias.fill_random(random);
}

pub fn print(
    nn: *const NeuralNetwork,
    writer: *Io.Writer,
    name: []const u8,
) !void {
    try writer.print("{s}:\n", .{name});

    // Currently limited to [0..10) weights and biases
    // TODO: extend it if need be
    for (nn.weights, 0..) |wght, i| {
        try writer.print("    ", .{});
        const num = [2]u8{
            '_',
            @intCast('0' + i % 10),
        };
        const wght_name = "wght" ++ num;
        try wght.print(writer, wght_name);
    }

    for (nn.biases, 0..) |bias, i| {
        try writer.print("    ", .{});
        const num = [2]u8{
            '_',
            @intCast('0' + i % 10),
        };
        const bias_name = "bias" ++ num;
        try bias.print(writer, bias_name);
    }
}

pub fn forward(nn: *NeuralNetwork, training_input: Matrix) void {
    nn.activations[0].copy(training_input);
    for (0..nn.activations.len - 1) |i| {
        nn.activations[i + 1].mul(nn.activations[i], nn.weights[i]);
        nn.activations[i + 1].add(nn.activations[i + 1], nn.biases[i]);
        nn.activation_fn.apply(&nn.activations[i + 1]);
    }
}

pub fn backward(nn: *NeuralNetwork, training_target: Matrix) void {
    const output_index = nn.activations.len - 1;
    var output_layer = nn.activations[output_index];

    assert(training_target.shape.row == output_layer.shape.row);
    assert(training_target.shape.col == output_layer.shape.col);

    var output_delta = nn.deltas[output_index - 1];
    for (0..output_delta.shape.row) |r| {
        for (0..output_delta.shape.col) |c| {
            const pred = output_layer.at(r, c);
            const goal = training_target.at(r, c);

            const loss = pred - goal;
            output_delta.ptr(r, c).* = loss * nn.activation_fn.derivative(pred);
        }
    }

    var i: usize = nn.weights.len;
    while (i > 0) {
        i -= 1;

        if (i > 0) {
            for (0..nn.deltas[i - 1].shape.row) |r| {
                for (0..nn.deltas[i - 1].shape.col) |c| {
                    const sum = b: {
                        var sum_tmp: f32 = 0;
                        for (0..nn.deltas[i].shape.col) |k| {
                            sum_tmp += nn.deltas[i].at(r, k) * nn.weights[i].at(c, k);
                        }
                        break :b sum_tmp;
                    };

                    const output = nn.activations[i].at(r, c);
                    nn.deltas[i - 1].ptr(r, c).* = sum * nn.activation_fn.derivative(output);
                }
            }
        }
    }
}

pub fn learn(nn: *NeuralNetwork, alpha: f32) void {
    for (0..nn.weights.len) |i| {
        for (0..nn.weights[i].shape.row) |r| {
            for (0..nn.weights[i].shape.col) |c| {
                const gradient = nn.activations[i].at(0, r) * nn.deltas[i].at(0, c);
                nn.weights[i].ptr(r, c).* -= alpha * gradient;
            }
        }

        for (0..nn.biases[i].shape.row) |r| {
            for (0..nn.biases[i].shape.col) |c| {
                nn.biases[i].ptr(r, c).* -= alpha * nn.deltas[i].at(r, c);
            }
        }
    }
}

test "smoke" {
    const io = testing.io;
    var arena_instance: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_instance.deinit();
    const arena = arena_instance.allocator();

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stderr(), io, &stdout_buffer);
    const stdout_writer = &stdout_file_writer.interface;

    var prng: std.Random.DefaultPrng = .init(67);
    const random = prng.random();

    var nn: NeuralNetwork = try .init(arena, .{ .architecture = &.{ 2, 3, 1 } });
    nn.fill_rand(random);
    try nn.print(stdout_writer, "nn");
    try stdout_writer.flush();
}

test "learn reduces loss" {
    const gpa = testing.allocator;

    var input_data = [_]f32{ 1, 0 };
    const input: Matrix = .init_from_slice(&input_data, .init(1, 2));

    var target_data = [_]f32{1};
    const target: Matrix = .init_from_slice(&target_data, .init(1, 1));

    var nn: NeuralNetwork = try .init(gpa, .{ .architecture = &.{ 2, 1 } });
    defer nn.deinit(gpa);

    nn.forward(input);
    const before = @abs(target.at(0, 0) - nn.activations[nn.activations.len - 1].at(0, 0));

    nn.backward(target);
    nn.learn(0.1);

    nn.forward(input);
    const after = @abs(target.at(0, 0) - nn.activations[nn.activations.len - 1].at(0, 0));

    try testing.expect(after < before);
}
