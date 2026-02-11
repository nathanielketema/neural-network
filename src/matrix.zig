const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;

const Io = std.Io;
const Allocator = std.mem.Allocator;

/// Matrix is limited to dimensions 2^16 by 2^16
pub fn Matrix(comptime T: type) type {
    return struct {
        const Self = @This();

        data: []T,
        row: Row,
        col: Col,
        stride: Stride,

        comptime {
            assert(@sizeOf(Self) == 24);
        }

        pub const Row = enum(u16) {
            _,

            pub fn val(row: Row) u16 {
                return @intFromEnum(row);
            }

            pub fn at(row: anytype) Row {
                return @enumFromInt(row);
            }
        };

        pub const Col = enum(u16) {
            _,

            pub fn val(col: Col) u16 {
                return @intFromEnum(col);
            }

            pub fn at(col: anytype) Col {
                return @enumFromInt(col);
            }
        };

        pub const Stride = enum(u16) {
            _,

            pub fn val(stride: Stride) u16 {
                return @intFromEnum(stride);
            }

            pub fn at(stride: anytype) Stride {
                return @enumFromInt(stride);
            }
        };


        /// No need to call deinit as caller owns memory
        pub fn init_from_slice(data: []T, row: usize, col: usize) Self {
            return .{
                .data = data,
                .row = .at(row),
                .col = .at(col),
                .stride = .at(col),
            };
        }

        /// Caller must call deinit
        pub fn init(gpa: Allocator, row: usize, col: usize) !Self {
            const data = try gpa.alloc(T, row * col);

            return init_from_slice(data, row, col);
        }

        inline fn index(matrix: Self, row: usize, col: usize) usize {
            assert(matrix.data.len >= matrix.row.val() * matrix.stride.val());
            assert(row < matrix.row.val());
            assert(col < matrix.col.val());

            return row * matrix.stride.val() + col;
        }

        pub fn deinit(matrix: *Self, gpa: Allocator) void {
            assert(matrix.data.len >= matrix.row.val() * matrix.stride.val());
            if (matrix.data.len > 0) {
                gpa.free(matrix.data);
            }
        }

        pub fn ptr(matrix: Self, row: Row, col: Col) *T {
            assert(matrix.data.len >= matrix.row.val() * matrix.stride.val());
            assert(row.val() < matrix.row.val());
            assert(col.val() < matrix.col.val());

            return &matrix.data[matrix.index(row.val(), col.val())];
        }

        pub fn get(matrix: Self, row: Row, col: Col) T {
            assert(matrix.data.len >= matrix.row.val() * matrix.stride.val());
            assert(row.val() < matrix.row.val());
            assert(col.val() < matrix.col.val());

            return matrix.data[matrix.index(row.val(), col.val())];
        }

        pub fn copy_row(matrix: Self, gpa: Allocator, row: Row) !Self {
            assert(matrix.data.len >= matrix.row.val() * matrix.stride.val());
            assert(row.val() < matrix.row.val());

            var new_matrix: Matrix(T) = try .init(gpa, 1, matrix.col.val());
            for (0..new_matrix.col.val()) |c| {
                const col = Col.at(c);
                new_matrix.ptr(.at(0), col).* = matrix.get(row, col);
            }

            return new_matrix;
        }

        pub fn copy_submatrix(matrix: Self, gpa: Allocator, start: usize, stride: Stride) !Self {
            assert(matrix.data.len >= matrix.row.val() * matrix.stride.val());
            assert(start < matrix.col.val());
            assert(stride.val() <= matrix.col.val());
            assert(stride.val() > 0);

            const row_count = matrix.row.val();
            const col_count = 1 + (matrix.col.val() - start - 1) / stride.val();

            var new_matrix: Self = try .init(gpa, matrix.row.val(), col_count);
            for (0..row_count) |r| {
                const row = Row.at(r);
                for (0..col_count) |c| {
                    const col = Col.at(c);
                    const source_col = Col.at(start + (c * stride.val()));
                    new_matrix.ptr(row, col).* = matrix.get(row, source_col);
                }
            }
            return new_matrix;
        }

        pub fn print(matrix: Self, writer: *Io.Writer, matrix_name: []const u8) !void {
            const row = matrix.row.val();
            const col = matrix.col.val();
            const stride = matrix.stride.val();
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
            assert(matrix.data.len >= matrix.row.val() * matrix.stride.val());
            @memset(matrix.data, value);
        }

        pub fn fill_random(matrix: *Self, random: std.Random) void {
            assert(matrix.data.len >= matrix.row.val() * matrix.stride.val());

            const row_count = matrix.row.val();
            const col_count = matrix.col.val();

            for (0..row_count) |r| {
                const row = Row.at(r);
                for (0..col_count) |c| {
                    const col = Col.at(c);
                    matrix.ptr(row, col).* = random.float(T);
                }
            }
        }

        pub fn copy_transpose(matrix: Self, gpa: Allocator) !Self {
            assert(matrix.data.len >= matrix.row.val() * matrix.stride.val());
            var new_matrix: Matrix(T) = try .init(gpa, matrix.col.val(), matrix.row.val());

            const row_count = new_matrix.row.val();
            const col_count = new_matrix.col.val();
            for (0..row_count) |r| {
                const row = Row.at(r);
                const old_col = Col.at(r);
                for (0..col_count) |c| {
                    const col = Col.at(c);
                    const old_row = Row.at(c);
                    new_matrix.ptr(row, col).* = matrix.get(old_row, old_col);
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
    assert(a.row.val() == b.row.val());
    assert(a.col.val() == b.col.val());
    assert(result.row.val() == a.row.val());
    assert(result.col.val() == a.col.val());

    const row_count_a = a.row.val();
    const col_count_b = a.col.val();
    for (0..row_count_a) |r| {
        const row = Matrix(T).Row.at(r);
        for (0..col_count_b) |c| {
            const col = Matrix(T).Col.at(c);
            result.ptr(row, col).* = a.get(row, col) + b.get(row, col);
        }
    }
}

pub fn mul(
    comptime T: type,
    result: *Matrix(T),
    a: Matrix(T),
    b: Matrix(T),
) void {
    assert(a.col.val() == b.row.val());
    assert(result.row.val() == a.row.val());
    assert(result.col.val() == b.col.val());

    const row_count_a = a.row.val();
    const col_count_b = b.col.val();
    const col_count_a = a.col.val();
    for (0..row_count_a) |i| {
        const row = Matrix(T).Row.at(i);
        for (0..col_count_b) |j| {
            const col = Matrix(T).Col.at(j);
            for (0..col_count_a) |k| {
                result.ptr(row, col).* += a.get(row, .at(k)) * b.get(.at(k), col);
            }
        }
    }
}

pub fn copy(comptime T: type, dest: *Matrix(T), src: Matrix(T)) void {
    assert(dest.data.len == src.data.len);
    assert(dest.row.val() == src.row.val());
    assert(dest.col.val() == src.col.val());

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
    const row = 3;
    const col = 3;

    var matrix: Matrix(f32) = .init_from_slice(&data, row, col);

    try testing.expectEqual(5, matrix.get(.at(1), .at(1)));
    try testing.expectEqual(7, matrix.get(.at(2), .at(0)));

    matrix.ptr(.at(2), .at(1)).* = 6;
    try testing.expectEqual(6, matrix.get(.at(2), .at(1)));

    try matrix.print(stdout_writer, "matrix");

    var new_matrix = try matrix.copy_row(allocator, .at(0));
    defer new_matrix.deinit(allocator);

    try new_matrix.print(stdout_writer, "new_matrix");

    var matrix_2: Matrix(f32) = try .init(allocator, row, col);
    defer matrix_2.deinit(allocator);

    var prng: std.Random.DefaultPrng = .init(testing.random_seed);
    const random = prng.random();
    matrix_2.fill_random(random);
    try matrix_2.print(stdout_writer, "matrix_2");

    std.debug.print("Dimensions = {d} x {d}\n", .{ matrix.row.val(), matrix.col.val() });
    std.debug.print("data_count = {d}\n", .{matrix.data.len});

    try stdout_writer.flush();
}

test "get and ptr access same element" {
    var matrix = try Matrix(i32).init(testing.allocator, 2, 2);
    defer matrix.deinit(testing.allocator);
    
    matrix.ptr(.at(0), .at(0)).* = 42;
    matrix.ptr(.at(1), .at(1)).* = 99;
    
    try testing.expectEqual(42, matrix.get(.at(0), .at(0)));
    try testing.expectEqual(99, matrix.get(.at(1), .at(1)));
}

test "add" {
    const allocator = testing.allocator;

    const row = 3;
    const col = 3;

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
    const a: Matrix(f32) = .init_from_slice(&a_data, 2, 2);

    var b_data = [_]f32{ 0, 1, 2, 2, 1, 1 };
    const b: Matrix(f32) = .init_from_slice(&b_data, 2, 3);

    var result: Matrix(f32) = try .init(fba, 2, 3);
    defer result.deinit(fba);

    mul(f32, &result, a, b);

    try testing.expectEqualSlices(f32, &[_]f32{ 4, 6, 10, 2, 3, 5 }, result.data);
}

test "clone" {
    var data: [4096]u8 = undefined;

    var fixed_buffer: std.heap.FixedBufferAllocator = .init(&data);
    const fba = fixed_buffer.allocator();

    const row = 3;
    const col = 3;

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

    const row = 3;
    const col = 3;

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
    const row = 3;
    const col = 3;

    var matrix: Matrix(f32) = .init_from_slice(&data, row, col);

    var copy_submatrix = try matrix.copy_submatrix(allocator, 1, .at(3));
    defer copy_submatrix.deinit(allocator);

    try matrix.print(stdout_writer, "matrix");

    try copy_submatrix.print(stdout_writer, "copy_submatrix");
    try stdout_writer.flush();
}
