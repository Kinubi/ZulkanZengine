const std = @import("std");
const Texture = @import("texture.zig").Texture;
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;

pub const AssetManager = struct {
    allocator: std.mem.Allocator,
    thread_pool: std.Thread.Pool,
    gc: *GraphicsContext,

    pub fn init(allocator: std.mem.Allocator, gc: *GraphicsContext, n_threads: usize) !AssetManager {
        return AssetManager{
            .allocator = allocator,
            .thread_pool = try std.Thread.Pool.init(.{ .allocator = allocator, .n_threads = n_threads }),
            .gc = gc,
        };
    }

    pub fn deinit(self: *AssetManager) void {
        self.thread_pool.deinit();
    }

    pub fn loadTextures(self: *AssetManager, paths: [][]const u8) ![]?Texture {
        var loaded = try self.allocator.alloc(?Texture, paths.len);
        var errors = try self.allocator.alloc(?anyerror, paths.len);
        for (paths, 0..) |path, i| {
            try self.thread_pool.spawn(
                struct {
                    gc: *GraphicsContext,
                    path: []const u8,
                    out: *?Texture,
                    err: *?anyerror,
                }{
                    .gc = self.gc,
                    .path = path,
                    .out = &loaded[i],
                    .err = &errors[i],
                },
                struct {
                    fn run(ctx: *@This()) void {
                        ctx.out.* = null;
                        ctx.err.* = null;
                        ctx.out.* = Texture.initFromFile(ctx.gc, ctx.path, .rgba8) catch |e| {
                            ctx.err.* = e;
                            return;
                        };
                    }
                }.run,
            );
        }
        self.thread_pool.wait();
        // Optionally, handle errors here or return them
        return loaded;
    }
};
