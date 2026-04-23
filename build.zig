const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const nn_mod = b.addModule("neural_network", .{
        .root_source_file = b.path("src/neural_network.zig"),
        .target = target,
        .optimize = optimize,
    });

    const nn_test = b.addTest(.{
        .root_module = nn_mod,
    });
    const run_nn_test = b.addRunArtifact(nn_test);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_nn_test.step);
}
