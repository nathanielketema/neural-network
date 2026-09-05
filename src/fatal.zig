const std = @import("std");

pub fn fatal(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print(fmt, args);
    std.process.exit(1);
}

pub fn oom(_: error{OutOfMemory}) noreturn {
    fatal("oom\n", .{});
}
