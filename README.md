# znn

`znn` is a neural network library written in Zig.

## Start

1. Fetch the library:

   ```console
   zig fetch --save git+https://github.com/nathanielketema/znn.git
   ```

2. Add the below to your `build.zig`:

   ```zig
   var nn_mod = b.dependency("znn", .{
       .target = target,
       .optimize = optimize,
   });
   ```

   And import it to your executable. Example:

   ```zig
   const exe = b.addExecutable(.{
       .name = "exe",
       .root_module = b.createModule(.{
           .root_source_file = b.path("src/main.zig"),
           .target = target,
           .optimize = optimize,
           .imports = &.{
               .{ .name = "znn", .module = nn_mod.module("znn") },
           },
       }),
   });
   ```
