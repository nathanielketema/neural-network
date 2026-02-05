const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;

const assert = std.debug.assert;
const testing = std.testing;

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

        pub fn create(data: []T, row: Row, col: Col) Self {
            return .{
                .data = data,
                .row = row,
                .col = col,
            };
        }

        pub fn init(allocator: Allocator, row: Row, col: Col) !Self {
            const data = try allocator.alloc(T, row.value() * col.value());

            return .{
                .data = data,
                .row = row,
                .col = col,
            };
        }

        pub fn deinit(matrix: *Self, allocator: Allocator) void {
            assert(matrix.data.len == matrix.row.value() * matrix.col.value());
            if (matrix.data.len > 0) {
                allocator.free(matrix.data);
            }
        }

        pub fn at(matrix: *Self, row: usize, col: usize) *T {
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
                    "{any}\n",
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

//pub fn matrix_add(comptime T: type, dest: *Matrix(T), a: *const Matrix(T), b: *const Matrix(T)) void {}
//pub fn matrix_multiply(comptime T: type, dest: *Matrix(T), a: Matrix(T), b: *const Matrix(T)) void {}
//pub fn matrix_copy(comptime T: type, dest: *Matrix(T), src: *const Matrix(T)) void {}

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

    std.debug.print("New matrix\n", .{});
    var matrix: Matrix(f32) = .create(&data, row, col);
    try testing.expectEqual(5, matrix.get(1, 1));
    try testing.expectEqual(7, matrix.get(2, 0));

    matrix.at(2, 1).* = 6;
    try testing.expectEqual(6, matrix.get(2, 1));

    try matrix.print(stdout_writer);
    try stdout_writer.flush();

    std.debug.print("New matrix\n", .{});
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
