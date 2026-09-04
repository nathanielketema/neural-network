const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("znn", .{
        .root_source_file = b.path("src/znn.zig"),
        .target = target,
        .optimize = optimize,
    });

    const mod_test = b.addTest(.{ .root_module = mod });
    const mod_test_run = b.addRunArtifact(mod_test);

    const mod_test_step = b.step("test", "Run tests");
    mod_test_step.dependOn(&mod_test_run.step);
}
