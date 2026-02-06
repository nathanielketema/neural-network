const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const matrix_mod = b.addModule("matrix", .{
        .root_source_file = b.path("src/matrix.zig"),
        .target = target,
        .optimize = optimize,
    });

    const nn_mod = b.addModule("neural_network", .{
        .root_source_file = b.path("src/neural_network.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "matrix", .module = matrix_mod },
        },
    });

    const exe = b.addExecutable(.{
        .name = "neural_network",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "nn", .module = nn_mod },
            },
        }),
    });
    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const matrix_mod_tests = b.addTest(.{
        .root_module = matrix_mod,
    });
    const run_matrix_mod_tests = b.addRunArtifact(matrix_mod_tests);

    const nn_mod_tests = b.addTest(.{
        .root_module = nn_mod,
    });
    const run_nn_mod_tests = b.addRunArtifact(nn_mod_tests);

    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });
    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_nn_mod_tests.step);
    test_step.dependOn(&run_matrix_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);
}
