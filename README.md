# neural-network

`neural-network` is a small neural network and matrix library written in Zig. It's mostly for
educational purposes, and for quick proof of concepts.

## Start

1. Fetch the library:

   ```console
   zig fetch --save git+https://github.com/nathanielketema/neural-network.git
   ```

2. Add the below to your `build.zig`:

   ```zig
   var nn_mod = b.dependency("neural_network", .{
       .target = target,
       .optimize = optimize,
   });
   ```

   And import it to your executable. Example:

   ```zig
   const exe = b.addExecutable(.{
       .name = "foo",
       .root_module = b.createModule(.{
           .root_source_file = b.path("src/main.zig"),
           .target = target,
           .optimize = optimize,
           .imports = &.{
               .{ .name = "neural_network", .module = nn_mod.module("neural_network") },
           },
       }),
   });
   ```

## Reference

- [Tsoding](https://www.youtube.com/playlist?list=PLpM-Dvs8t0VZPZKggcql-MmjaBdZKeDMw)
