const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;

const NeuralNetwork = @This();

const Io = std.Io;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

pub const Matrix = @import("matrix").Matrix(f32);

weigths: []Matrix,
biases: []Matrix,
layers: []Matrix,
activation: Activation,
alpha: f32,

pub const Activation = enum {
    relu,
    sigmoid,

    pub fn apply(activation: Activation, matrix: *Matrix) void {
        return switch (activation) {
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

    fn relu_fn(x: f32) f32 {
        return @max(0, x);
    }

    fn sigmoid_fn(x: f32) f32 {
        return 1 / (1 + @exp(-x));
    }
};

pub const Options = struct {
    architecture: []const usize,
    activation: Activation = .sigmoid,
    alpha: f32 = 0.01,
};

// Caller must free memory by calling deinit()
pub fn init(gpa: Allocator, options: Options) !NeuralNetwork {
    assert(options.architecture.len >= 2);

    const count = options.architecture.len;

    var layers: ArrayList(Matrix) = try .initCapacity(gpa, count);
    var weigths: ArrayList(Matrix) = try .initCapacity(gpa, count - 1);
    var biases: ArrayList(Matrix) = try .initCapacity(gpa, count - 1);

    for (0..count - 1) |i| {
        var lyer: Matrix = try .init(gpa, .init(1, options.architecture[i]));
        lyer.fill(0);

        var wght: Matrix = try .init(gpa, .init(
            options.architecture[i],
            options.architecture[i + 1],
        ));
        wght.fill(0);

        var bias: Matrix = try .init(gpa, .init(1, options.architecture[i + 1]));
        bias.fill(0);

        try layers.append(gpa, lyer);
        try weigths.append(gpa, wght);
        try biases.append(gpa, bias);
    }
    try layers.append(gpa, try .init(gpa, .init(1, options.architecture[count - 1])));
    layers.items[layers.items.len - 1].fill(0);

    return .{
        .weigths = try weigths.toOwnedSlice(gpa),
        .biases = try biases.toOwnedSlice(gpa),
        .layers = try layers.toOwnedSlice(gpa),
        .activation = options.activation,
        .alpha = options.alpha,
    };
}

pub fn deinit(nn: *NeuralNetwork, gpa: Allocator) void {
    for (nn.weigths) |*wght| wght.deinit(gpa);
    for (nn.biases) |*bias| bias.deinit(gpa);
    for (nn.layers) |*lyer| lyer.deinit(gpa);

    gpa.free(nn.weigths);
    gpa.free(nn.biases);
    gpa.free(nn.layers);
}

pub fn fill_rand(nn: *NeuralNetwork, random: std.Random) void {
    for (nn.weigths) |*wght| wght.fill_random(random);
    for (nn.biases) |*bias| bias.fill_random(random);
}

// No need to free memory, as it will free the memory used by itself
pub fn print(
    nn: *const NeuralNetwork,
    writer: *Io.Writer,
    name: []const u8,
) !void {
    try writer.print("{s}:\n", .{name});

    // Currently limited to [0..10) weigths and biases
    // FUTURE: extend it if need be
    for (nn.weigths, 0..) |wght, i| {
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

pub fn forward(nn: *NeuralNetwork) void {
    for (0..nn.layers.len - 1) |i| {
        nn.layers[i + 1].mul(nn.layers[i], nn.weigths[i]);
        nn.layers[i + 1].add(nn.layers[i + 1], nn.biases[i]);
        nn.activation.apply(&nn.layers[i + 1]);
    }
}

pub fn backward(nn: *NeuralNetwork, training_input: Matrix, training_otput: Matrix) void {
}

//Possible APIs:
//- forward
//- bckward
//- predict
//- grdient

test "smoke" {
    const io = testing.io;
    const gpa = testing.allocator;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stderr(), io, &stdout_buffer);
    const stdout_writer = &stdout_file_writer.interface;

    var prng: std.Random.DefaultPrng = .init(67);
    const random = prng.random();

    var nn: NeuralNetwork = try .init(gpa, .{ .architecture = &.{ 2, 3, 3, 2 } });
    defer nn.deinit(gpa);
    nn.fill_rand(random);

    try nn.print(stdout_writer, "nn");
    try stdout_writer.flush();
}
