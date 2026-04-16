const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;

const Io = std.Io;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

pub const Matrix = @import("matrix").Matrix(f32);

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

pub const Options = struct {
    architecture: []const usize,
    activation: Activation = .relu,
    alpha: f32 = 0.01,
    epsln: f32 = 0.01,
};

pub const NeuralNetwork = struct {
    weigts: []Matrix,
    biases: []Matrix,
    layers: []Matrix,
    activation: Activation,
    alpha: f32,
    epsln: f32,

    // .init(gpa, &.{2, 3, 2, 2});
    pub fn init(gpa: Allocator, options: Options) !NeuralNetwork {
        assert(options.architecture.len >= 2);

        const count = options.architecture.len;

        var layers: ArrayList(Matrix) = try .initCapacity(gpa, count);
        var weigts: ArrayList(Matrix) = try .initCapacity(gpa, count - 1);
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
            try weigts.append(gpa, wght);
            try biases.append(gpa, bias);
        }
        try layers.append(gpa, try .init(gpa, .init(1, options.architecture[count - 1])));
        layers.items[layers.items.len - 1].fill(0);

        return .{
            .weigts = try weigts.toOwnedSlice(gpa),
            .biases = try biases.toOwnedSlice(gpa),
            .layers = try layers.toOwnedSlice(gpa),
            .activation = options.activation,
            .alpha = options.alpha,
            .epsln = options.epsln,
        };
    }

    pub fn deinit(nn: *NeuralNetwork, gpa: Allocator) void {
        for (nn.weigts) |*wght| wght.deinit(gpa);
        for (nn.biases) |*bias| bias.deinit(gpa);
        for (nn.layers) |*lyer| lyer.deinit(gpa);

        gpa.free(nn.weigts);
        gpa.free(nn.biases);
        gpa.free(nn.layers);
    }

    pub fn fill_rand(nn: *NeuralNetwork, random: std.Random) void {
        for (nn.weigts) |*wght| wght.fill_random(random);
        for (nn.biases) |*bias| bias.fill_random(random);
    }

    pub fn print(
        nn: *const NeuralNetwork,
        writer: *Io.Writer,
        gpa: Allocator,
        name: []const u8,
    ) !void {
        try writer.print("{s}:\n", .{name});

        for (0..nn.weigts.len) |i| {
            try writer.print("    ", .{});

            const wght_name = try std.fmt.allocPrint(gpa, "wght_{d}", .{i});
            defer gpa.free(wght_name);
            try nn.weigts[i].print(writer, wght_name);

            try writer.print("    ", .{});

            const bias_name = try std.fmt.allocPrint(gpa, "bias_{d}", .{i});
            defer gpa.free(bias_name);
            try nn.biases[i].print(writer, bias_name);
        }
    }

    //pub fn forward(self: *NeuralNetwork, input: Matrix) !Matrix
    //pub fn backward(self: *NeuralNetwork, target: Matrix) !void
    //pub fn train(self: *NeuralNetwork, input: Matrix, target: Matrix) !f32
    //pub fn predict(self: *NeuralNetwork, input: Matrix) !Matrix  // forward without caching
};

//pub fn cost(predicted: Matrix, actual: Matrix) f32 {}
//

test "smoke" {
    const io = testing.io;
    const gpa = testing.allocator;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stderr(), io, &stdout_buffer);
    const stdout_writer = &stdout_file_writer.interface;

    var prng: std.Random.DefaultPrng = .init(67);
    const random = prng.random();

    var nn: NeuralNetwork = try .init(gpa, .{ .architecture = &.{ 2, 2, 1 } });
    defer nn.deinit(gpa);
    nn.fill_rand(random);

    try nn.print(stdout_writer, gpa, "nn");
    try stdout_writer.flush();
}
