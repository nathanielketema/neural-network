const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;

const Io = std.Io;
const Allocator = std.mem.Allocator;

/// Matrix is capped to 2^16 by 2^16 dimension
pub fn Matrix(comptime T: type) type {
    return struct {
        const Self = @This();

        data: []T,
        shape: Shape,
        stride: u16,

        comptime {
            assert(@sizeOf(Self) == 24);
        }

        pub const Shape = struct {
            row: u16,
            col: u16,

            pub fn init(row: usize, col: usize) Shape {
                const upper_bound = std.math.pow(usize, 2, @typeInfo(u16).int.bits);
                assert(row < upper_bound and col < upper_bound);
                assert(row != 0 and col != 0);

                return .{
                    .row = @intCast(row),
                    .col = @intCast(col),
                };
            }
        };

        pub const SubmatrixOptions = struct {
            inital: usize,
            stride: usize,

            pub fn validate(options: SubmatrixOptions, cols: usize) void {
                assert(options.inital < cols);
                assert(options.stride < cols);
                assert(options.stride > 0);
            }
        };

        /// No need to call deinit as caller owns memory
        pub fn init_from_slice(data: []T, shape: Shape) Self {
            return .{
                .data = data,
                .shape = shape,
                .stride = shape.col,
            };
        }

        /// Caller must free memory using deinit()
        pub fn init(gpa: Allocator, shape: Shape) !Self {
            const data = try gpa.alloc(T, shape.row * shape.col);
            errdefer gpa.free(data);

            return init_from_slice(data, shape);
        }

        pub fn deinit(matrix: *Self, gpa: Allocator) void {
            matrix.assert_matrix();

            if (matrix.data.len > 0) {
                gpa.free(matrix.data);
            }
        }

        inline fn assert_matrix(matrix: Self) void {
            assert(matrix.data.len >= matrix.shape.row * matrix.stride);
        }

        inline fn assert_shape(matrix: Self, row: usize, col: usize) void {
            assert(row <= matrix.shape.row);
            assert(col <= matrix.shape.col);
        }

        inline fn index(matrix: Self, row: usize, col: usize) usize {
            matrix.assert_matrix();
            matrix.assert_shape(row, col);
            return row * matrix.stride + col;
        }

        pub fn ptr(matrix: *Self, row: usize, col: usize) *T {
            matrix.assert_matrix();
            matrix.assert_shape(row, col);
            return &matrix.data[matrix.index(row, col)];
        }

        pub fn at(matrix: Self, row: usize, col: usize) T {
            matrix.assert_matrix();
            matrix.assert_shape(row, col);
            return matrix.data[matrix.index(row, col)];
        }

        pub fn copy(target: *Self, source: Self) void {
            assert(target.data.len == source.data.len);
            assert(target.shape.row == source.shape.row);
            assert(target.shape.col == source.shape.col);

            @memcpy(target.data, source.data);
        }

        pub fn copy_row(target: *Self, source: Self, row: usize) void {
            source.assert_matrix();
            assert(row < source.shape.row);

            for (0..source.shape.col) |c| {
                target.ptr(0, c).* = source.at(row, c);
            }
        }

        pub fn copy_submatrix(target: *Self, source: Self, options: SubmatrixOptions) void {
            target.assert_matrix();
            source.assert_matrix();
            options.validate(source.shape.col);

            const new_col = target.shape.col;
            for (0..target.shape.row) |r| {
                for (0..new_col) |c| {
                    const src_new_col = options.inital + (c * options.stride);
                    target.ptr(r, c).* = source.at(r, src_new_col);
                }
            }
        }

        // Caller is responsible for freeing memeory
        pub fn copy_transpose(source: Self, gpa: Allocator) !Self {
            source.assert_matrix();

            var target: Matrix(T) = try .init(gpa, .init(
                source.shape.col,
                source.shape.row,
            ));

            for (0..target.shape.row) |r| {
                for (0..target.shape.col) |c| {
                    target.ptr(r, c).* = source.at(c, r);
                }
            }

            return target;
        }

        pub fn fill(matrix: *Self, number: T) void {
            matrix.assert_matrix();
            @memset(matrix.data, number);
        }

        pub fn fill_random(matrix: *Self, random: std.Random) void {
            matrix.assert_matrix();

            for (0..matrix.shape.row) |r| {
                for (0..matrix.shape.col) |c| {
                    matrix.ptr(r, c).* = random.float(T);
                }
            }
        }

        pub fn scale(matrix: *Self, scalar: f32) void {
            matrix.assert_matrix();

            for (0..matrix.shape.row) |r| {
                for (0..matrix.shape.col) |c| {
                    matrix.ptr(r, c).* = scalar * matrix.at(r, c);
                }
            }
        }

        pub fn add(res: *Self, mt1: Self, mt2: Self) void {
            assert(mt1.shape.row == mt2.shape.row);
            assert(mt1.shape.col == mt2.shape.col);
            assert(res.shape.row == mt1.shape.row);
            assert(res.shape.col == mt1.shape.col);

            for (0..mt1.shape.row) |r| {
                for (0..mt2.shape.col) |c| {
                    res.ptr(r, c).* = mt1.at(r, c) + mt2.at(r, c);
                }
            }
        }

        pub fn sub(res: *Self, mt1: Self, mt2: Self) void {
            assert(mt1.shape.row == mt2.shape.row);
            assert(mt1.shape.col == mt2.shape.col);
            assert(res.shape.row == mt1.shape.row);
            assert(res.shape.col == mt1.shape.col);

            for (0..mt1.shape.row) |r| {
                for (0..mt2.shape.col) |c| {
                    res.ptr(r, c).* = mt1.at(r, c) - mt2.at(r, c);
                }
            }
        }

        pub fn mul(res: *Self, mt1: Self, mt2: Self) void {
            assert(mt1.shape.col == mt2.shape.row);
            assert(res.shape.row == mt1.shape.row);
            assert(res.shape.col == mt2.shape.col);

            for (0..mt1.shape.row) |i| {
                for (0..mt2.shape.col) |j| {
                    var sum: T = 0;
                    for (0..mt1.shape.col) |k| {
                        sum += mt1.at(i, k) * mt2.at(k, j);
                    }
                    res.ptr(i, j).* = sum;
                }
            }
        }

        pub fn print(matrix: Self, writer: *Io.Writer, name: []const u8) !void {
            matrix.assert_matrix();

            try writer.print("{s}:\n", .{name});
            for (0..matrix.shape.row) |r| {
                const start = r * matrix.stride;
                const end = start + matrix.shape.col;
                try writer.print("        {any:2.8}\n", .{matrix.data[start..end]});
            }
            try writer.print("\n", .{});
        }
    };
}

test "smoke" {
    const io = testing.io;
    const allocator = testing.allocator;

    var stderr_file_writer: Io.File.Writer = .init(.stderr(), io, &.{});
    const stderr_writer = &stderr_file_writer.interface;

    var matrix = try create_matrix(f32, allocator, .init(3, 3), null);
    defer matrix.deinit(allocator);

    try matrix.print(stderr_writer, "matrix");
    try stderr_writer.flush();
}

test "matrix access" {
    const allocator = testing.allocator;

    var matrix = try create_matrix(f32, allocator, .init(15, 15), null);
    defer matrix.deinit(allocator);

    for (0..15) |r| {
        for (0..15) |c| {
            try testing.expect(matrix.at(r, c) == matrix.ptr(r, c).*);
        }
    }
}

test "operations" {
    const allocator = testing.allocator;

    var mt1 = try create_matrix(f32, allocator, .init(15, 15), null);
    defer mt1.deinit(allocator);

    var mt2 = try create_matrix(f32, allocator, .init(15, 15), null);
    defer mt2.deinit(allocator);

    var res = try create_matrix(f32, allocator, .init(15, 15), 0);
    defer res.deinit(allocator);

    try check(f32, allocator, .add, &res, mt1, mt2);
    try check(f32, allocator, .mul, &res, mt1, mt2);
    try check(f32, allocator, .scale, &res, mt1, mt2);
}

fn create_matrix(
    comptime T: type,
    gpa: Allocator,
    size: Matrix(T).Shape,
    fill_val: ?T,
) !Matrix(T) {
    var prng: std.Random.DefaultPrng = .init(testing.random_seed);
    const random = prng.random();

    var matrix: Matrix(f32) = try .init(gpa, size);

    if (fill_val) |val| {
        matrix.fill(val);
    } else {
        matrix.fill_random(random);
    }

    return matrix;
}

fn check(
    comptime T: type,
    gpa: Allocator,
    opr: enum { add, mul, scale },
    res: *Matrix(T),
    mt1: Matrix(T),
    mt2: Matrix(T),
) !void {
    switch (opr) {
        .add => {
            res.add(mt1, mt2);
            for (0..res.shape.row) |r| {
                for (0..res.shape.col) |c| {
                    try testing.expectEqual(mt1.at(r, c), res.at(r, c) - mt2.at(r, c));
                }
            }
        },
        .mul => {
            // This also tests the matrix transpose
            // (A × B)^T = B^T × A^T
            res.mul(mt1, mt2);

            var left = try res.copy_transpose(gpa);
            defer left.deinit(gpa);

            var mt2_trans = try mt2.copy_transpose(gpa);
            defer mt2_trans.deinit(gpa);
            var mt1_trans = try mt1.copy_transpose(gpa);
            defer mt1_trans.deinit(gpa);

            var right: Matrix(T) = try create_matrix(T, gpa, left.shape, 0);
            defer right.deinit(gpa);

            right.mul(mt2_trans, mt1_trans);

            try testing.expectEqualSlices(T, left.data, right.data);
        },
        .scale => {
            // c(A + B) = cA + cB
            var right = try create_matrix(T, gpa, res.shape, 0);
            defer right.deinit(gpa);

            var a = try create_matrix(T, gpa, res.shape, 0);
            defer a.deinit(gpa);

            var b = try create_matrix(T, gpa, res.shape, 0);
            defer b.deinit(gpa);

            var prng = std.Random.DefaultPrng.init(testing.random_seed);
            const c = prng.random().float(f32);

            a.copy(mt1);
            b.copy(mt2);

            res.add(mt1, mt2);
            res.scale(c);

            a.scale(c);
            b.scale(c);

            right.add(a, b);

            try testing.expectEqualSlices(T, res.data, right.data);
        },
    }
}
