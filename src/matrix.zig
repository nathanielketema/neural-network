const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;

const Io = std.Io;
const Allocator = std.mem.Allocator;

pub fn Matrix(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const Row = enum(u32) {
            _,

            pub fn value(row: Row) u32 {
                return @intFromEnum(row);
            }

            pub fn to_enum(row: u32) Row {
                return @enumFromInt(row);
            }
        };

        pub const Col = enum(u32) {
            _,

            pub fn value(row: Col) u32 {
                return @intFromEnum(row);
            }

            pub fn to_enum(row: u32) Col {
                return @enumFromInt(row);
            }
        };

        data: []T,
        row: Row,
        col: Col,

        comptime {
            assert(@sizeOf(Self) == 24);
        }

        pub fn create(data: []T, row: Row, col: Col) Self {
            return .{
                .data = data,
                .row = row,
                .col = col,
            };
        }

        pub fn init(gpa: Allocator, row: Row, col: Col) !Self {
            const data = try gpa.alloc(T, row.value() * col.value());

            return .{
                .data = data,
                .row = row,
                .col = col,
            };
        }

        pub fn deinit(matrix: *Self, gpa: Allocator) void {
            assert(matrix.data.len == matrix.row.value() * matrix.col.value());
            if (matrix.data.len > 0) {
                gpa.free(matrix.data);
            }
        }

        pub fn at(matrix: *const Self, row: usize, col: usize) *T {
            assert(matrix.data.len == matrix.row.value() * matrix.col.value());
            assert(row <= matrix.row.value());
            assert(col <= matrix.col.value());

            return &matrix.data[row * matrix.col.value() + col];
        }

        pub fn get(matrix: *const Self, row: usize, col: usize) T {
            assert(matrix.data.len == matrix.row.value() * matrix.col.value());
            assert(row <= matrix.row.value());
            assert(col <= matrix.col.value());

            return matrix.data[row * matrix.col.value() + col];
        }

        pub fn print(matrix: *const Self, writer: *Io.Writer) !void {
            assert(matrix.data.len == matrix.row.value() * matrix.col.value());

            const row = matrix.row.value();
            const col = matrix.col.value();
            for (0..row) |r| {
                try writer.print(
                    "{any:2}\n",
                    .{
                        matrix.data[r * col .. (r + 1) * col],
                    },
                );
            }
        }

        pub fn fill(matrix: *Self, value: T) void {
            assert(matrix.data.len == matrix.row.value() * matrix.col.value());
            @memset(matrix.data, value);
        }

        pub fn fill_random(matrix: *Self, random: std.Random) void {
            assert(matrix.data.len == matrix.row.value() * matrix.col.value());
            @memset(matrix.data, random.float(T));
        }
    };
}

pub fn add(
    comptime T: type,
    result: *Matrix(T),
    a: *const Matrix(T),
    b: *const Matrix(T),
) void {
    assert(a.row.value() == b.row.value());
    assert(a.col.value() == b.col.value());
    assert(result.row.value() == a.row.value());
    assert(result.col.value() == a.col.value());

    const row_count_a = a.row.value();
    const col_count_b = a.col.value();
    for (0..row_count_a) |r| {
        for (0..col_count_b) |c| {
            result.at(r, c).* = a.get(r, c) + b.get(r, c);
        }
    }
}

pub fn multiply(
    comptime T: type,
    result: *Matrix(T),
    a: *const Matrix(T),
    b: *const Matrix(T),
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
                result.at(i, j).* += a.get(i, k) * b.get(k, j);
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

    var matrix: Matrix(f32) = .create(&data, row, col);
    try testing.expectEqual(5, matrix.get(1, 1));
    try testing.expectEqual(7, matrix.get(2, 0));

    matrix.at(2, 1).* = 6;
    try testing.expectEqual(6, matrix.get(2, 1));

    try matrix.print(stdout_writer);
    try stdout_writer.flush();

    var matrix_2: Matrix(f32) = try .init(allocator, row, col);
    defer matrix_2.deinit(allocator);

    var prng: std.Random.DefaultPrng = .init(testing.random_seed);
    const random = prng.random();
    matrix_2.fill_random(random);
    try matrix_2.print(stdout_writer);
    try stdout_writer.flush();

    std.debug.print("Dimensions = {d} x {d}\n", .{ matrix.row.value(), matrix.col.value() });
    std.debug.print("data_count = {d}\n", .{matrix.data.len});
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

    add(f32, &result, &a, &b);

    for (result.data) |val| {
        try testing.expectEqual(5, val);
    }
}

test "multiply" {
    var data: [4096]u8 = undefined;

    var fixed_buffer: std.heap.FixedBufferAllocator = .init(&data);
    const fba = fixed_buffer.allocator();

    var a_data = [_]f32{ 4, 2, 2, 1 };
    var a: Matrix(f32) = .create(&a_data, .to_enum(2), .to_enum(2));

    var b_data = [_]f32{ 0, 1, 2, 2, 1, 1 };
    var b: Matrix(f32) = .create(&b_data, .to_enum(2), .to_enum(3));

    var result: Matrix(f32) = try .init(fba, .to_enum(2), .to_enum(3));
    defer result.deinit(fba);

    multiply(f32, &result, &a, &b);

    try testing.expectEqualSlices(f32, &[_]f32{4, 6, 10, 2, 3, 5}, result.data);
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
