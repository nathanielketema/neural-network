pub const matrix = @import("matrix.zig");
pub const NeuralNetwork = @import("NeuralNetwork.zig");

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
