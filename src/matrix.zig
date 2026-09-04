const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const Io = std.Io;
const Allocator = std.mem.Allocator;
const mulWide = std.math.mulWide;

const Self = @This();

/// Matrix dimensions are capped at `2^16 - 1`
pub const Matrix = struct {
    data: []f32,
    shape: Shape,
    stride: usize,
    gpa: Allocator,

    comptime {
        assert(@sizeOf(Matrix) == 48);
    }

    pub const Shape = struct {
        row: u16,
        col: u16,
    };

    pub const SubmatrixOptions = struct {
        start: u16 = 0,
        stride: u16 = 1,
        col_count: u16,

        pub fn validate(options: SubmatrixOptions, cols: u16) void {
            assert(options.col_count > 0);
            assert(options.stride > 0);
            assert(options.start < cols);
            assert(options.start + mulWide(u16, (options.col_count - 1), options.stride) < cols);
        }
    };

    /// Caller owns the returned matrix and must call deinit(gpa).
    pub fn init(gpa: Allocator, shape: Shape) Matrix {
        assert(shape.row > 0);
        assert(shape.col > 0);

        const data = gpa.alloc(f32, mulWide(u16, shape.row, shape.col)) catch oom();
        @memset(data, 0);

        return .{
            .data = data,
            .shape = shape,
            .stride = shape.col,
            .gpa = gpa,
        };
    }

    /// Caller owns the returned matrix and must call deinit(gpa).
    pub fn init_from_slice(gpa: Allocator, data: []const f32, shape: Shape) Matrix {
        assert(data.len == @as(usize, shape.row) * shape.col);

        var matrix = Matrix.init(gpa, shape);
        errdefer matrix.deinit();
        @memcpy(matrix.data, data);

        return matrix;
    }

    pub fn deinit(matrix: *Matrix) void {
        matrix.assert_matrix();
        matrix.gpa.free(matrix.data);
        matrix.* = undefined;
    }

    inline fn assert_matrix(matrix: Matrix) void {
        assert(matrix.shape.row > 0);
        assert(matrix.shape.col > 0);
        assert(matrix.stride >= matrix.shape.col);

        const last =
            (@as(usize, matrix.shape.row) - 1) * matrix.stride +
            (@as(usize, matrix.shape.col) - 1);

        assert(matrix.data.len > last);
    }

    inline fn assert_shape(matrix: Matrix, row: usize, col: usize) void {
        assert(row < matrix.shape.row);
        assert(col < matrix.shape.col);
    }

    inline fn index(matrix: Matrix, row: usize, col: usize) usize {
        matrix.assert_matrix();
        matrix.assert_shape(row, col);
        return row * matrix.stride + col;
    }

    pub fn ptr(matrix: Matrix, row: usize, col: usize) *f32 {
        return &matrix.data[matrix.index(row, col)];
    }

    pub fn at(matrix: Matrix, row: usize, col: usize) f32 {
        return matrix.data[matrix.index(row, col)];
    }

    pub fn copy(target: *Matrix, source: Matrix) void {
        source.assert_matrix();
        target.assert_matrix();

        assert(target.shape.row == source.shape.row);
        assert(target.shape.col == source.shape.col);

        for (0..source.shape.row) |r| {
            for (0..source.shape.col) |c| {
                target.ptr(r, c).* = source.at(r, c);
            }
        }
    }

    pub fn row_view(source: Matrix, row: usize) Matrix {
        source.assert_matrix();
        assert(row < source.shape.row);

        const lo = @as(usize, row) * source.stride;
        const hi = lo + source.shape.col;

        return .{
            .data = source.data[lo..hi],
            .shape = .{
                .row = 1,
                .col = source.shape.col,
            },
            .stride = source.stride,
            .gpa = source.gpa,
        };
    }

    pub fn copy_row(source: Matrix, row: usize) Matrix {
        source.assert_matrix();
        assert(row < source.shape.row);
        const gpa = source.gpa;

        var target: Matrix = .init(gpa, .{
            .row = 1,
            .col = source.shape.col,
        });
        errdefer target.deinit();

        const lo = @as(usize, row) * source.stride;
        const hi = lo + source.shape.col;

        @memcpy(
            target.data[0..source.shape.col],
            source.data[lo..hi],
        );
        return target;
    }

    pub fn copy_cols(source: Matrix, options: SubmatrixOptions) Matrix {
        source.assert_matrix();
        options.validate(source.shape.col);
        const gpa = source.gpa;

        var target: Matrix = .init(gpa, .{
            .row = source.shape.row,
            .col = options.col_count,
        });
        errdefer target.deinit();

        for (0..source.shape.row) |row| {
            for (0..options.col_count) |col| {
                const start: usize = @intCast(options.start);
                const stride: usize = @intCast(options.stride);

                const new_col = start + (col * stride);
                target.ptr(row, col).* = source.at(row, new_col);
            }
        }
        return target;
    }

    pub fn transpose(source: Matrix) Matrix {
        source.assert_matrix();
        const gpa = source.gpa;

        var target: Matrix = .init(gpa, .{
            .row = source.shape.col,
            .col = source.shape.row,
        });
        errdefer target.deinit();

        for (0..target.shape.row) |r| {
            for (0..target.shape.col) |c| {
                target.ptr(r, c).* = source.at(c, r);
            }
        }
        return target;
    }

    pub fn fill(matrix: *Matrix, number: f32) void {
        matrix.assert_matrix();
        for (0..matrix.shape.row) |r| {
            const lo = r * matrix.stride;
            const hi = lo + matrix.shape.col;
            @memset(matrix.data[lo..hi], number);
        }
    }

    pub fn fill_random(matrix: *Matrix, random: std.Random) void {
        matrix.assert_matrix();
        for (0..matrix.shape.row) |r| {
            const lo = r * matrix.stride;
            const hi = lo + matrix.shape.col;
            for (matrix.data[lo..hi]) |*value| {
                value.* = random.float(f32);
            }
        }
    }

    pub fn scale(matrix: Matrix, scalar: f32) Matrix {
        matrix.assert_matrix();
        const gpa = matrix.gpa;

        var result: Matrix = .init(gpa, matrix.shape);
        errdefer result.deinit();

        for (0..matrix.shape.row) |r| {
            for (0..matrix.shape.col) |c| {
                result.ptr(r, c).* = scalar * matrix.at(r, c);
            }
        }
        return result;
    }

    pub fn add(mt1: Matrix, mt2: Matrix) Matrix {
        return Self.add(mt1.gpa, mt1, mt2);
    }

    pub fn add_into(res: *Matrix, mt1: Matrix, mt2: Matrix) void {
        res.assert_matrix();
        mt1.assert_matrix();
        mt2.assert_matrix();
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

    pub fn sub(mt1: Matrix, mt2: Matrix) Matrix {
        return Self.sub(mt1.gpa, mt1, mt2);
    }

    pub fn sub_into(res: *Matrix, mt1: Matrix, mt2: Matrix) void {
        res.assert_matrix();
        mt1.assert_matrix();
        mt2.assert_matrix();
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

    pub fn mul(mt1: Matrix, mt2: Matrix) Matrix {
        return Self.mul(mt1.gpa, mt1, mt2);
    }

    pub fn mul_into(res: *Matrix, mt1: Matrix, mt2: Matrix) void {
        res.assert_matrix();
        mt1.assert_matrix();
        mt2.assert_matrix();
        assert(mt1.shape.col == mt2.shape.row);
        assert(res.shape.row == mt1.shape.row);
        assert(res.shape.col == mt2.shape.col);

        for (0..mt1.shape.row) |i| {
            for (0..mt2.shape.col) |j| {
                var sum: f32 = 0;
                for (0..mt1.shape.col) |k| {
                    sum += mt1.at(i, k) * mt2.at(k, j);
                }
                res.ptr(i, j).* = sum;
            }
        }
    }

    pub fn format(matrix: Matrix, writer: *std.Io.Writer) !void {
        matrix.assert_matrix();

        for (0..matrix.shape.row) |r| {
            const lo = r * matrix.stride;
            const hi = lo + matrix.shape.col;

            try writer.print("    .{any:.8}", .{matrix.data[lo..hi]});
            if (r + 1 < matrix.shape.row) try writer.print(",", .{});
            try writer.print("\n", .{});
        }
    }
};

pub fn add(gpa: Allocator, mt1: Matrix, mt2: Matrix) Matrix {
    mt1.assert_matrix();
    mt2.assert_matrix();
    assert(mt1.shape.row == mt2.shape.row);
    assert(mt1.shape.col == mt2.shape.col);

    var res: Matrix = .init(gpa, mt1.shape);
    errdefer res.deinit();

    for (0..mt1.shape.row) |r| {
        for (0..mt2.shape.col) |c| {
            res.ptr(r, c).* = mt1.at(r, c) + mt2.at(r, c);
        }
    }
    return res;
}

pub fn sub(gpa: Allocator, mt1: Matrix, mt2: Matrix) Matrix {
    mt1.assert_matrix();
    mt2.assert_matrix();
    assert(mt1.shape.row == mt2.shape.row);
    assert(mt1.shape.col == mt2.shape.col);

    var res: Matrix = .init(gpa, mt1.shape);
    errdefer res.deinit();

    for (0..mt1.shape.row) |r| {
        for (0..mt2.shape.col) |c| {
            res.ptr(r, c).* = mt1.at(r, c) - mt2.at(r, c);
        }
    }
    return res;
}

pub fn mul(gpa: Allocator, mt1: Matrix, mt2: Matrix) Matrix {
    mt1.assert_matrix();
    mt2.assert_matrix();
    assert(mt1.shape.col == mt2.shape.row);

    var res: Matrix = .init(gpa, .{
        .row = mt1.shape.row,
        .col = mt2.shape.col,
    });
    errdefer res.deinit();

    for (0..mt1.shape.row) |i| {
        for (0..mt1.shape.col) |k| {
            const a = mt1.at(i, k);
            for (0..mt2.shape.col) |j| {
                res.ptr(i, j).* += a * mt2.at(k, j);
            }
        }
    }
    return res;
}

fn fatal(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print(fmt, args);
    std.process.exit(1);
}

fn oom() noreturn {
    fatal("oom\n", .{});
}

fn expect_approx_slices(comptime T: type, expected: []const T, actual: []const T) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, actual, 0..) |exp, act, i| {
        testing.expectApproxEqRel(exp, act, 0.0001) catch |err| {
            std.debug.print(
                \\
                \\Mismatch at slice index {d}:
                \\    expected: {d}
                \\      actual: {d}
                \\
            , .{ i, exp, act });
            return err;
        };
    }
}

fn create_matrix(gpa: Allocator, size: Matrix.Shape, num_fill: ?f32, random: ?std.Random) Matrix {
    var matrix: Matrix = .init(gpa, size);
    errdefer matrix.deinit();

    if (num_fill) |num| {
        matrix.fill(num);
    } else matrix.fill_random(random.?);

    return matrix;
}

test "init matrix" {
    const gpa = testing.allocator;
    var matrix: Matrix = .init(gpa, .{
        .row = 3,
        .col = 4,
    });
    defer matrix.deinit();
    matrix.fill(2);

    try testing.expectEqual(3, matrix.shape.row);
    try testing.expectEqual(4, matrix.shape.col);
    try testing.expectEqual(4, matrix.stride);
    try testing.expectEqual(12, matrix.data.len);
    std.log.debug("matrix:\n{f}", .{matrix});
}

test "fill_random produces varying values" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(12345);
    const random = prng.random();

    var matrix: Matrix = .init(gpa, .{
        .row = 4,
        .col = 4,
    });
    defer matrix.deinit();

    matrix.fill_random(random);
    var all_equal = true;
    for (matrix.data[1..]) |value| {
        if (value != matrix.data[0]) {
            all_equal = false;
            break;
        }
    }
    try testing.expect(!all_equal);
}

test "smoke test" {
    var arena_instance: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_instance.deinit();
    const arena = arena_instance.allocator();

    const mt1 = create_matrix(arena, .{ .row = 3, .col = 3 }, 4, null);
    const mt2 = create_matrix(arena, .{ .row = 3, .col = 3 }, 2, null);
    const mt1_scaled = mt1.scale(2);

    for (0..3) |r| {
        for (0..3) |c| {
            try testing.expect(mt1_scaled.at(r, c) == mt1_scaled.ptr(r, c).*);
        }
    }

    const sum = add(arena, mt1, mt2);
    const diff = sub(arena, mt1, mt2);
    const prod = mul(arena, mt1, mt2);
    const mt1_row_vector = mt1.copy_row(2);
    const mt1_submatrix = mt1.copy_cols(.{ .col_count = 2 });
    const mt1_transpose = mt1.transpose();

    try expect_approx_slices(
        f32,
        mt1_scaled.data,
        create_matrix(arena, .{ .row = 3, .col = 3 }, 8, null).data,
    );
    try expect_approx_slices(
        f32,
        sum.data,
        create_matrix(arena, .{ .row = 3, .col = 3 }, 6, null).data,
    );
    try expect_approx_slices(
        f32,
        diff.data,
        create_matrix(arena, .{ .row = 3, .col = 3 }, 2, null).data,
    );
    try expect_approx_slices(
        f32,
        prod.data,
        create_matrix(arena, .{ .row = 3, .col = 3 }, 24, null).data,
    );
    try expect_approx_slices(
        f32,
        mt1_row_vector.data,
        create_matrix(arena, .{ .row = 1, .col = 3 }, 4, null).data,
    );
    try expect_approx_slices(
        f32,
        mt1_submatrix.data,
        create_matrix(arena, .{ .row = 3, .col = 2 }, 4, null).data,
    );
    try expect_approx_slices(
        f32,
        mt1_transpose.data,
        create_matrix(arena, .{ .row = 3, .col = 3 }, 4, null).data,
    );

    var data = [_]f32{
        0, 0, 0,
        0, 1, 0,
        1, 0, 0,
        1, 1, 1,
    };
    const matrix: Matrix = .init_from_slice(arena, &data, .{ .row = 4, .col = 3 });

    const matrix_row = matrix.copy_row(2);
    try expect_approx_slices(
        f32,
        matrix_row.data,
        &.{ 1, 0, 0 },
    );

    const matrix_cols = matrix.copy_cols(.{ .col_count = 2, .stride = 2 });
    try expect_approx_slices(
        f32,
        matrix_cols.data,
        &.{ 0, 0, 0, 0, 1, 0, 1, 1 },
    );
}

test "property based fuzzing" {
    const T = struct {
        fn fuzz_test(_: void, smith: *testing.Smith) !void {
            var arena_instance: std.heap.ArenaAllocator = .init(testing.allocator);
            defer arena_instance.deinit();
            const arena = arena_instance.allocator();
            var prng: std.Random.DefaultPrng = .init(testing.random_seed);
            const random = prng.random();

            while (!smith.eos()) {
                defer _ = arena_instance.reset(.free_all);
                const row = smith.value(u16) % 8 + 1;
                const col = smith.value(u16) % 8 + 1;

                const A = create_matrix(arena, .{ .row = row, .col = col }, null, random);
                const B = create_matrix(arena, .{ .row = row, .col = col }, null, random);
                const C = create_matrix(arena, .{ .row = row, .col = col }, null, random);

                // For mul
                const m = smith.value(u16) % 8 + 1;
                const n = smith.value(u16) % 8 + 1;
                const p = smith.value(u16) % 8 + 1;
                const q = smith.value(u16) % 8 + 1;

                const D = create_matrix(arena, .{ .row = m, .col = n }, null, random);
                const E = create_matrix(arena, .{ .row = n, .col = p }, null, random);
                const F = create_matrix(arena, .{ .row = p, .col = q }, null, random);

                // A + B = B + A
                try expect_approx_slices(
                    f32,
                    A.add(B).data,
                    B.add(A).data,
                );

                // A + (B + C) = (A + B) + C
                try expect_approx_slices(
                    f32,
                    A.add(.add(B, C)).data,
                    Matrix.add(.add(A, B), C).data,
                );

                // (DE)F = D(EF)
                try expect_approx_slices(
                    f32,
                    Matrix.mul(.mul(D, E), F).data,
                    D.mul(.mul(E, F)).data,
                );

                // X(Y + Z) = XY + XZ
                const X = create_matrix(arena, .{ .row = row, .col = row }, null, random);
                const Y = create_matrix(arena, .{ .row = row, .col = row }, null, random);
                const Z = create_matrix(arena, .{ .row = row, .col = row }, null, random);
                try expect_approx_slices(
                    f32,
                    X.mul(.add(Y, Z)).data,
                    Matrix.add(X.mul(Y), X.mul(Z)).data,
                );

                // (A^T)^T = A
                try expect_approx_slices(
                    f32,
                    A.transpose().transpose().data,
                    A.data,
                );

                // (A + B)^T = A^T + B^T
                try expect_approx_slices(
                    f32,
                    Matrix.add(A, B).transpose().data,
                    Matrix.add(A.transpose(), B.transpose()).data,
                );

                // (DE)^T = (E^T)(D^T)
                try expect_approx_slices(
                    f32,
                    Matrix.mul(D, E).transpose().data,
                    Matrix.mul(E.transpose(), D.transpose()).data,
                );

                // r(A)^T = (rA)^T
                const r = smith.value(u8);
                try expect_approx_slices(
                    f32,
                    Matrix.scale(A.transpose(), r).data,
                    Matrix.transpose(A.scale(r)).data,
                );
            }
        }
    };

    try testing.fuzz({}, T.fuzz_test, .{});
}
