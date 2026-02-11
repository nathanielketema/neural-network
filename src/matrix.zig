const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;

const Io = std.Io;
const Allocator = std.mem.Allocator;

/// Matrix is limited to dimensions 2^16 by 2^16
pub fn Matrix(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const Row = enum(u16) {
            _,

            pub fn value(row: Row) u16 {
                return @intFromEnum(row);
            }

            pub fn to_enum(row: u16) Row {
                return @enumFromInt(row);
            }
        };

        pub const Col = enum(u16) {
            _,

            pub fn value(col: Col) u16 {
                return @intFromEnum(col);
            }

            pub fn to_enum(col: u16) Col {
                return @enumFromInt(col);
            }
        };

        pub const Stride = enum(u16) {
            _,

            pub fn value(stride: Stride) u16 {
                return @intFromEnum(stride);
            }

            pub fn to_enum(stride: u16) Stride {
                return @enumFromInt(stride);
            }
        };

        data: []T,
        row: Row,
        col: Col,
        stride: Stride,

        comptime {
            assert(@sizeOf(Self) == 24);
        }

        /// No need to call deinit as caller owns memory
        pub fn init_from_slice(data: []T, row: Row, col: Col) Self {
            return .{
                .data = data,
                .row = row,
                .col = col,
                .stride = .to_enum(col.value()),
            };
        }

        /// Caller must call deinit
        pub fn init(gpa: Allocator, row: Row, col: Col) !Self {
            const data = try gpa.alloc(T, row.value() * col.value());

            return init_from_slice(data, row, col);
        }

        inline fn index(matrix: Self, row: usize, col: usize) usize {
            assert(matrix.data.len >= matrix.row.value() * matrix.stride.value());
            assert(row < matrix.row.value());
            assert(col < matrix.col.value());

            return row * matrix.stride.value() + col;
        }

        pub fn deinit(matrix: *Self, gpa: Allocator) void {
            assert(matrix.data.len >= matrix.row.value() * matrix.stride.value());
            if (matrix.data.len > 0) {
                gpa.free(matrix.data);
            }
        }

        pub fn ptr(matrix: Self, row: usize, col: usize) *T {
            assert(matrix.data.len >= matrix.row.value() * matrix.stride.value());
            assert(row < matrix.row.value());
            assert(col < matrix.col.value());

            return &matrix.data[matrix.index(row, col)];
        }

        pub fn get(matrix: Self, row: usize, col: usize) T {
            assert(matrix.data.len >= matrix.row.value() * matrix.stride.value());
            assert(row < matrix.row.value());
            assert(col < matrix.col.value());

            return matrix.data[matrix.index(row, col)];
        }

        pub fn copy_row(matrix: Self, gpa: Allocator, row: Row) !Self {
            assert(matrix.data.len >= matrix.row.value() * matrix.stride.value());
            assert(row.value() < matrix.row.value());

            var new_matrix: Matrix(T) = try .init(
                gpa,
                .to_enum(1),
                matrix.col,
            );
            for (0..new_matrix.col.value()) |c| {
                new_matrix.ptr(0, c).* = matrix.get(row.value(), c);
            }

            return new_matrix;
        }

        pub fn copy_submatrix(matrix: Self, gpa: Allocator, start: usize, stride: Stride) !Self {
            assert(matrix.data.len >= matrix.row.value() * matrix.stride.value());
            assert(start < matrix.col.value());
            assert(stride.value() <= matrix.col.value());
            assert(stride.value() > 0);

            const row_count = matrix.row.value();
            const col_count: u16 = @intCast(1 + (matrix.col.value() - start - 1) / stride.value());

            var new_matrix: Self = try .init(gpa, matrix.row, .to_enum(col_count));
            for (0..row_count) |r| {
                for (0..col_count) |c| {
                    const source_col = start + (c * stride.value());
                    new_matrix.ptr(r, c).* = matrix.get(r, source_col);
                }
            }
            return new_matrix;
        }

        pub fn print(matrix: Self, writer: *Io.Writer, matrix_name: []const u8) !void {
            const row = matrix.row.value();
            const col = matrix.col.value();
            const stride = matrix.stride.value();
            try writer.print("{s}:\n", .{matrix_name});
            for (0..row) |r| {
                try writer.print(
                    "    {any:2}\n",
                    .{
                        matrix.data[r * stride .. (r * stride) + col],
                    },
                );
            }
            try writer.print("\n", .{});
        }

        pub fn fill(matrix: *Self, value: T) void {
            assert(matrix.data.len >= matrix.row.value() * matrix.stride.value());
            @memset(matrix.data, value);
        }

        pub fn fill_random(matrix: *Self, random: std.Random) void {
            assert(matrix.data.len >= matrix.row.value() * matrix.stride.value());

            const row_count = matrix.row.value();
            const col_count = matrix.col.value();

            for (0..row_count) |r| {
                for (0..col_count) |c| {
                    matrix.ptr(r, c).* = random.float(T);
                }
            }
        }

        pub fn copy_transpose(matrix: Self, gpa: Allocator) !Self {
            assert(matrix.data.len >= matrix.row.value() * matrix.stride.value());
            var new_matrix: Matrix(T) = try .init(
                gpa,
                .to_enum(matrix.col.value()),
                .to_enum(matrix.row.value()),
            );

            const row_count = new_matrix.row.value();
            const col_count = new_matrix.col.value();
            for (0..row_count) |r| {
                for (0..col_count) |c| {
                    new_matrix.ptr(r, c).* = matrix.get(c, r);
                }
            }

            return new_matrix;
        }
    };
}

pub fn add(
    comptime T: type,
    result: *Matrix(T),
    a: Matrix(T),
    b: Matrix(T),
) void {
    assert(a.row.value() == b.row.value());
    assert(a.col.value() == b.col.value());
    assert(result.row.value() == a.row.value());
    assert(result.col.value() == a.col.value());

    const row_count_a = a.row.value();
    const col_count_b = a.col.value();
    for (0..row_count_a) |r| {
        for (0..col_count_b) |c| {
            result.ptr(r, c).* = a.get(r, c) + b.get(r, c);
        }
    }
}

pub fn mul(
    comptime T: type,
    result: *Matrix(T),
    a: Matrix(T),
    b: Matrix(T),
) void {
    assert(a.col.value() == b.row.value());
    assert(result.row.value() == a.row.value());
    assert(result.col.value() == b.col.value());

    const row_count_a = a.row.value();
    const col_count_b = b.col.value();
    const col_count_a = a.col.value();
    for (0..row_count_a) |i| {
        for (0..col_count_b) |j| {
            for (0..col_count_a) |k| {
                result.ptr(i, j).* += a.get(i, k) * b.get(k, j);
            }
        }
    }
}

pub fn copy(comptime T: type, dest: *Matrix(T), src: Matrix(T)) void {
    assert(dest.data.len == src.data.len);
    assert(dest.row.value() == src.row.value());
    assert(dest.col.value() == src.col.value());

    @memcpy(dest.data, src.data);
}

test Matrix {
    const io = testing.io;
    const allocator = testing.allocator;
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const stdout_writer = &stdout_file_writer.interface;

    var data = [_]f32{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };
    const row: Matrix(f32).Row = .to_enum(3);
    const col: Matrix(f32).Col = .to_enum(3);
    assert(data.len == row.value() * col.value());

    var matrix: Matrix(f32) = .init_from_slice(&data, row, col);

    try testing.expectEqual(5, matrix.get(1, 1));
    try testing.expectEqual(7, matrix.get(2, 0));

    matrix.ptr(2, 1).* = 6;
    try testing.expectEqual(6, matrix.get(2, 1));

    try matrix.print(stdout_writer, "matrix");

    std.debug.print("-----------\n", .{});
    var new_matrix = try matrix.copy_row(allocator, .to_enum(0));
    defer new_matrix.deinit(allocator);

    try new_matrix.print(stdout_writer, "new_matrix");
    std.debug.print("-----------\n", .{});

    var matrix_2: Matrix(f32) = try .init(allocator, row, col);
    defer matrix_2.deinit(allocator);

    var prng: std.Random.DefaultPrng = .init(testing.random_seed);
    const random = prng.random();
    matrix_2.fill_random(random);
    try matrix_2.print(stdout_writer, "matrix_2");
    std.debug.print("-----------\n", .{});

    std.debug.print("Dimensions = {d} x {d}\n", .{ matrix.row.value(), matrix.col.value() });
    std.debug.print("data_count = {d}\n", .{matrix.data.len});

    try stdout_writer.flush();
}

test "add" {
    const allocator = testing.allocator;

    const row: Matrix(f32).Row = .to_enum(3);
    const col: Matrix(f32).Col = .to_enum(3);

    var a: Matrix(f32) = try .init(allocator, row, col);
    defer a.deinit(allocator);
    a.fill(2);

    var b: Matrix(f32) = try .init(allocator, row, col);
    defer b.deinit(allocator);
    b.fill(3);

    var result: Matrix(f32) = try .init(allocator, row, col);
    defer result.deinit(allocator);

    add(f32, &result, a, b);

    for (result.data) |val| {
        try testing.expectEqual(5, val);
    }
}

test "mul" {
    var data: [4096]u8 = undefined;

    var fixed_buffer: std.heap.FixedBufferAllocator = .init(&data);
    const fba = fixed_buffer.allocator();

    var a_data = [_]f32{ 4, 2, 2, 1 };
    const a: Matrix(f32) = .init_from_slice(&a_data, .to_enum(2), .to_enum(2));

    var b_data = [_]f32{ 0, 1, 2, 2, 1, 1 };
    const b: Matrix(f32) = .init_from_slice(&b_data, .to_enum(2), .to_enum(3));

    var result: Matrix(f32) = try .init(fba, .to_enum(2), .to_enum(3));
    defer result.deinit(fba);

    mul(f32, &result, a, b);

    try testing.expectEqualSlices(f32, &[_]f32{ 4, 6, 10, 2, 3, 5 }, result.data);
}

test "clone" {
    var data: [4096]u8 = undefined;

    var fixed_buffer: std.heap.FixedBufferAllocator = .init(&data);
    const fba = fixed_buffer.allocator();

    const row: Matrix(f32).Row = .to_enum(3);
    const col: Matrix(f32).Col = .to_enum(3);

    var a: Matrix(f32) = try .init(fba, row, col);
    defer a.deinit(fba);
    a.fill(23);

    var b: Matrix(f32) = try .init(fba, row, col);
    defer b.deinit(fba);

    copy(f32, &b, a);

    try testing.expectEqualSlices(f32, a.data, b.data);
}

test "copy_transpose" {
    var data: [4096]u8 = undefined;

    var fixed_buffer: std.heap.FixedBufferAllocator = .init(&data);

    const gpa = fixed_buffer.allocator();

    const row: Matrix(f32).Row = .to_enum(3);
    const col: Matrix(f32).Col = .to_enum(3);

    var a: Matrix(f32) = try .init(gpa, row, col);
    defer a.deinit(gpa);
    a.fill(2);

    var a_transpose = try a.copy_transpose(gpa);
    defer a_transpose.deinit(gpa);

    var b = try a_transpose.copy_transpose(gpa);
    defer b.deinit(gpa);

    try testing.expectEqualSlices(f32, a.data, b.data);
}

test "copy_submatrix" {
    const io = testing.io;
    const allocator = testing.allocator;
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const stdout_writer = &stdout_file_writer.interface;

    var data = [_]f32{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };
    const row: Matrix(f32).Row = .to_enum(3);
    const col: Matrix(f32).Col = .to_enum(3);

    var matrix: Matrix(f32) = .init_from_slice(&data, row, col);

    var copy_submatrix = try matrix.copy_submatrix(allocator, 1, .to_enum(3));
    defer copy_submatrix.deinit(allocator);

    try matrix.print(stdout_writer, "matrix");

    try copy_submatrix.print(stdout_writer, "copy_submatrix");
    try stdout_writer.flush();
}
