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
            start_col: usize,
            stride: usize,

            pub fn validate(options: SubmatrixOptions, cols: usize) void {
                assert(options.start_col < cols);
                assert(options.stride <= cols);
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

        pub fn copy_row(matrix: Self, gpa: Allocator, row: usize) !Self {
            matrix.assert_matrix();
            assert(row < matrix.shape.row);

            var new_matrix: Matrix(T) = try .init(gpa, Shape.init(1, matrix.shape.col));
            errdefer new_matrix.deinit(gpa);

            for (0..matrix.shape.col) |c| {
                new_matrix.ptr(0, c).* = matrix.at(row, c);
            }

            return new_matrix;
        }

        pub fn copy_submatrix(matrix: Self, gpa: Allocator, options: SubmatrixOptions) !Self {
            matrix.assert_matrix();
            options.validate(matrix.shape.col);

            const new_col = 1 + (matrix.shape.col - options.start_col - 1) / options.stride;

            var new_matrix: Self = try .init(gpa, Shape.init(matrix.shape.row, new_col));
            errdefer new_matrix.deinit(gpa);

            for (0..matrix.shape.row) |r| {
                for (0..new_col) |c| {
                    const src_new_col = options.start_col + (c * options.stride);
                    new_matrix.ptr(r, c).* = matrix.at(r, src_new_col);
                }
            }
            return new_matrix;
        }

        pub fn copy_transpose(matrix: Self, gpa: Allocator) !Self {
            matrix.assert_matrix();

            var new_matrix: Matrix(T) = try .init(gpa, Shape.init(
                matrix.shape.col,
                matrix.shape.row,
            ));
            errdefer new_matrix.deinit(gpa);

            for (0..new_matrix.shape.row) |r| {
                for (0..new_matrix.shape.col) |c| {
                    new_matrix.ptr(r, c).* = matrix.at(c, r);
                }
            }

            return new_matrix;
        }

        pub fn fill(matrix: *Self, value: T) void {
            matrix.assert_matrix();
            @memset(matrix.data, value);
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

        pub fn add(result: *Self, a: Self, b: Self) void {
            assert(a.shape.row == b.shape.row);
            assert(a.shape.col == b.shape.col);
            assert(result.shape.row == a.shape.row);
            assert(result.shape.col == a.shape.col);

            for (0..a.shape.row) |r| {
                for (0..b.shape.col) |c| {
                    result.ptr(r, c).* = a.at(r, c) + b.at(r, c);
                }
            }
        }

        pub fn mul(result: *Self, a: Self, b: Self) void {
            assert(a.shape.col == b.shape.row);
            assert(result.shape.row == a.shape.row);
            assert(result.shape.col == b.shape.col);

            for (0..a.shape.row) |i| {
                for (0..b.shape.col) |j| {
                    for (0..a.shape.col) |k| {
                        result.ptr(i, j).* = a.at(i, k) * b.at(k, j);
                    }
                }
            }
        }

        pub fn copy(dest: *Self, src: Self) void {
            assert(dest.data.len == src.data.len);
            assert(dest.shape.row == src.shape.row);
            assert(dest.shape.col == src.shape.col);

            @memcpy(dest.data, src.data);
        }

        pub fn print(matrix: Self, writer: *Io.Writer, matrix_name: []const u8) !void {
            matrix.assert_matrix();

            try writer.print("{s}:\n", .{matrix_name});
            for (0..matrix.shape.row) |r| {
                const start = r * matrix.stride;
                const end = start + matrix.shape.col;
                try writer.print("    {any:2.8}\n", .{matrix.data[start..end]});
            }
            try writer.print("\n", .{});

            try writer.flush();
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

test "add" {
    const allocator = testing.allocator;

    var mt1 = try create_matrix(f32, allocator, .init(15, 15), null);
    defer mt1.deinit(allocator);

    var mt2 = try create_matrix(f32, allocator, .init(15, 15), null);
    defer mt2.deinit(allocator);

    var res = try create_matrix(f32, allocator, .init(15, 15), 0);
    defer res.deinit(allocator);

    try check(f32, allocator, .add, &res, mt1, mt2);
}

test "mul" {
    const allocator = testing.allocator;

    var mt1 = try create_matrix(f32, allocator, .init(15, 15), null);
    defer mt1.deinit(allocator);

    var mt2 = try create_matrix(f32, allocator, .init(15, 15), null);
    defer mt2.deinit(allocator);

    var res = try create_matrix(f32, allocator, .init(15, 15), 0);
    defer res.deinit(allocator);

    try check(f32, allocator, .mul, &res, mt1, mt2);
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
    opr: enum { add, mul },
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
            // (A × B)^T = B^T × A^T
            res.mul(mt1, mt2);

            var left = try res.copy_transpose(gpa);
            defer left.deinit(gpa);

            var mt2_trans = try mt2.copy_transpose(gpa);
            defer mt2_trans.deinit(gpa);

            var mt1_trans = try mt1.copy_transpose(gpa);
            defer mt1_trans.deinit(gpa);

            res.mul(mt2_trans, mt1_trans);

            try testing.expectEqualSlices(T, left.data, res.data);
        },
    }
}
