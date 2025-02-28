const std = @import("std");
const Window = @import("window.zig").Window;
const glfw = @import("mach-glfw");
const Pipeline = @import("pipeline.zig").Pipeline;
const simple_vert align(@alignOf(u32)) = @embedFile("simple_vert").*;
const simple_frag align(@alignOf(u32)) = @embedFile("simple_frag").*;
const point_light_vert align(@alignOf(u32)) = @embedFile("point_light_vert").*;
const point_light_frag align(@alignOf(u32)) = @embedFile("point_light_frag").*;
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const Swapchain = @import("swapchain.zig").Swapchain;
const MAX_FRAMES_IN_FLIGHT = @import("swapchain.zig").MAX_FRAMES_IN_FLIGHT;
const vk = @import("vulkan");
const ShaderLibrary = @import("shader.zig").ShaderLibrary;
const Vertex = @import("mesh.zig").Vertex;
const Mesh = @import("mesh.zig").Mesh;
const Model = @import("mesh.zig").Model;
const Scene = @import("scene.zig").Scene;
const SimpleRenderer = @import("renderer.zig").SimpleRenderer;
const PointLightRenderer = @import("renderer.zig").PointLightRenderer;
const Math = @import("mach").math;
const Camera = @import("camera.zig").Camera;
const GameObject = @import("game_object.zig").GameObject;
const KeyboardMovementController = @import("keyboard_movement_controller.zig").KeyboardMovementController;
const FrameInfo = @import("frameinfo.zig").FrameInfo;
const DescriptorSetLayout = @import("descriptors.zig").DescriptorSetLayout;
const DescriptorPool = @import("descriptors.zig").DescriptorPool;
const DescriptorSetWriter = @import("descriptors.zig").DescriptorWriter;
const Buffer = @import("buffer.zig").Buffer;
const GlobalUbo = @import("frameinfo.zig").GlobalUbo;

pub const App = struct {
    window: Window = undefined,

    gc: GraphicsContext = undefined,
    allocator: std.mem.Allocator = undefined,
    var current_frame: u32 = 0;
    var swapchain: Swapchain = undefined;
    var cmdbufs: []vk.CommandBuffer = undefined;
    var simple_renderer: SimpleRenderer = undefined;
    var point_light_renderer: PointLightRenderer = undefined;
    var last_frame_time: f64 = undefined;
    var camera: Camera = undefined;
    var viewer_object: *GameObject = undefined;
    var camera_controller: KeyboardMovementController = undefined;
    var global_UBO_buffers: ?[]Buffer = undefined;

    var frame_info: FrameInfo = FrameInfo{};
    var global_pool: *DescriptorPool = undefined;
    var global_set_layout: *DescriptorSetLayout = undefined;
    var global_descriptor_set: vk.DescriptorSet = undefined;

    pub fn init(self: *@This()) !void {
        self.window = try Window.init(.{ .width = 1280, .height = 720 });

        self.allocator = std.heap.page_allocator;

        self.gc = try GraphicsContext.init(self.allocator, self.window.window_props.title, self.window.window.?);
        std.log.debug("Using device: {s}", .{self.gc.deviceName()});
        swapchain = try Swapchain.init(&self.gc, self.allocator, .{ .width = self.window.window_props.width, .height = self.window.window_props.height });

        try swapchain.createRenderPass();

        try swapchain.createFramebuffers();
        try self.gc.createCommandPool();

        var mesh2 = Mesh.init(self.allocator);
        var mesh3 = Mesh.init(self.allocator);
        var mesh = Mesh.init(self.allocator);

        try mesh.vertices.appendSlice(&.{
            // Left Face
            Vertex{ .pos = .{ -0.5, -0.5, -0.5 }, .color = .{ 0.9, 0.9, 0.9 } },
            Vertex{ .pos = .{ -0.5, 0.5, 0.5 }, .color = .{ 0.9, 0.9, 0.9 } },
            Vertex{ .pos = .{ -0.5, -0.5, 0.5 }, .color = .{ 0.9, 0.9, 0.9 } },
            Vertex{ .pos = .{ -0.5, 0.5, -0.5 }, .color = .{ 0.9, 0.9, 0.9 } },

            // Right face (yellow)
            Vertex{ .pos = .{ 0.5, -0.5, -0.5 }, .color = .{ 0.8, 0.8, 0.1 } },
            Vertex{ .pos = .{ 0.5, 0.5, 0.5 }, .color = .{ 0.8, 0.8, 0.1 } },
            Vertex{ .pos = .{ 0.5, -0.5, 0.5 }, .color = .{ 0.8, 0.8, 0.1 } },
            Vertex{ .pos = .{ 0.5, 0.5, -0.5 }, .color = .{ 0.8, 0.8, 0.1 } },

            // Top face (orange, remember y axis points down)
            Vertex{ .pos = .{ -0.5, -0.5, -0.5 }, .color = .{ 0.9, 0.6, 0.1 } },
            Vertex{ .pos = .{ 0.5, -0.5, 0.5 }, .color = .{ 0.9, 0.6, 0.1 } },
            Vertex{ .pos = .{ -0.5, -0.5, 0.5 }, .color = .{ 0.9, 0.6, 0.1 } },
            Vertex{ .pos = .{ 0.5, -0.5, -0.5 }, .color = .{ 0.9, 0.6, 0.1 } },

            // Bottom face (red)
            Vertex{ .pos = .{ -0.5, 0.5, -0.5 }, .color = .{ 0.8, 0.1, 0.1 } },
            Vertex{ .pos = .{ 0.5, 0.5, 0.5 }, .color = .{ 0.8, 0.1, 0.1 } },
            Vertex{ .pos = .{ -0.5, 0.5, 0.5 }, .color = .{ 0.8, 0.1, 0.1 } },
            Vertex{ .pos = .{ 0.5, 0.5, -0.5 }, .color = .{ 0.8, 0.1, 0.1 } },

            // Front Face
            Vertex{ .pos = .{ -0.5, -0.5, 0.5 }, .color = .{ 0.1, 0.8, 0.1 } },
            Vertex{ .pos = .{ 0.5, 0.5, 0.5 }, .color = .{ 0.1, 0.8, 0.1 } },
            Vertex{ .pos = .{ -0.5, 0.5, 0.5 }, .color = .{ 0.1, 0.8, 0.1 } },
            Vertex{ .pos = .{ 0.5, -0.5, 0.5 }, .color = .{ 0.1, 0.8, 0.1 } },

            // Back Face
            Vertex{ .pos = .{ -0.5, -0.5, -0.5 }, .color = .{ 0.1, 0.1, 0.8 } },
            Vertex{ .pos = .{ 0.5, 0.5, -0.5 }, .color = .{ 0.1, 0.1, 0.8 } },
            Vertex{ .pos = .{ -0.5, 0.5, -0.5 }, .color = .{ 0.1, 0.1, 0.8 } },
            Vertex{ .pos = .{ 0.5, -0.5, -0.5 }, .color = .{ 0.1, 0.1, 0.8 } },
        });

        try mesh.indices.appendSlice(&.{ 0, 1, 2, 0, 3, 1, 4, 5, 6, 4, 7, 5, 8, 9, 10, 8, 11, 9, 12, 13, 14, 12, 15, 13, 16, 17, 18, 16, 19, 17, 20, 21, 22, 20, 23, 21 });

        try mesh.createVertexBuffers(&self.gc);
        try mesh.createIndexBuffers(&self.gc);

        try mesh2.loadFromObj(self.allocator, @embedFile("smooth_vase"));
        try mesh2.createVertexBuffers(&self.gc);
        try mesh2.createIndexBuffers(&self.gc);
        try mesh3.loadFromObj(self.allocator, @embedFile("cube"));
        try mesh3.createVertexBuffers(&self.gc);
        try mesh3.createIndexBuffers(&self.gc);
        const model = Model.init(mesh2);
        const model2 = Model.init(mesh3);

        var scene: Scene = Scene.init();

        const object = try scene.addObject(model, null);
        object.transform.translate(Math.Vec3.init(0, 0.5, 0.5));
        object.transform.scale(Math.Vec3.init(0.5, 0.5, 0.5));

        const object2 = try scene.addObject(model2, null);

        object2.transform.translate(Math.Vec3.init(0, 0.5, 0.5));
        object2.transform.scale(Math.Vec3.init(0.5, 0.001, 0.5));

        const object3 = try scene.addObject(null, .{ .color = Math.Vec3.init(0.2, 0.5, 1.0), .intensity = 1.0 });
        object3.transform.translate(Math.Vec3.init(0, -1, 1.5));
        object3.transform.scale(Math.Vec3.init(0.05, 0, 0));

        const object4 = try scene.addObject(null, .{ .color = Math.Vec3.init(0.5, 0.2, 0.2), .intensity = 1.0 });
        object4.transform.translate(Math.Vec3.init(0, -1, 0.5));
        object4.transform.scale(Math.Vec3.init(0.05, 0, 0));

        cmdbufs = try self.gc.createCommandBuffers(
            self.allocator,
        );

        viewer_object = try scene.addEmpty();
        camera_controller = KeyboardMovementController.init();

        camera = Camera{ .fov = 75.0, .window = self.window };

        camera.setOrthographicProjection(-camera.aspectRatio, camera.aspectRatio, -1, 1, -1, 1);
        camera.updateProjectionMatrix();
        camera.setViewDirection(Math.Vec3.init(0, 0, 0), @constCast(&Math.Vec3.init(0, 0, 0)), Math.Vec3.init(0, 1, 0));

        global_UBO_buffers = try self.allocator.alloc(Buffer, MAX_FRAMES_IN_FLIGHT);

        for (0..global_UBO_buffers.?.len) |i| {
            global_UBO_buffers.?[i] = try Buffer.init(
                &self.gc,
                @sizeOf(GlobalUbo),
                1,
                .{ .uniform_buffer_bit = true },
                .{ .host_visible_bit = true, .host_coherent_bit = true },
            );
            try global_UBO_buffers.?[i].map(vk.WHOLE_SIZE, 0);
        }

        frame_info.camera = &camera;
        var descriptor_pool_builder = DescriptorPool.Builder{ .gc = &self.gc, .poolSizes = std.ArrayList(vk.DescriptorPoolSize).init(self.allocator), .poolFlags = .{}, .maxSets = 0 };
        global_pool = @constCast(&try descriptor_pool_builder.setMaxSets(MAX_FRAMES_IN_FLIGHT).addPoolSize(.uniform_buffer, MAX_FRAMES_IN_FLIGHT + 1).build());
        var descriptor_set_layout_builder = DescriptorSetLayout.Builder{ .gc = &self.gc, .bindings = std.AutoHashMap(u32, vk.DescriptorSetLayoutBinding).init(self.allocator) };
        global_set_layout = @constCast(&try descriptor_set_layout_builder.addBinding(0, .uniform_buffer, .{ .vertex_bit = true, .fragment_bit = true }, 1).build());

        var descriptor_set_writer = DescriptorSetWriter.init(self.gc, global_set_layout, global_pool);
        for (0..MAX_FRAMES_IN_FLIGHT) |i| {
            const bufferInfo = global_UBO_buffers.?[i].descriptor_info;
            try descriptor_set_writer.writeBuffer(0, @constCast(&bufferInfo)).build(&global_descriptor_set);
        }

        var shader_library = ShaderLibrary.init(self.gc, self.allocator);

        try shader_library.add(&.{ &simple_frag, &simple_vert }, &.{ vk.ShaderStageFlags{ .fragment_bit = true }, vk.ShaderStageFlags{ .vertex_bit = true } });

        simple_renderer = try SimpleRenderer.init(@constCast(&self.gc), swapchain.render_pass, scene, shader_library, self.allocator, @constCast(&camera), global_set_layout.descriptor_set_layout);

        var shader_library_point_light = ShaderLibrary.init(self.gc, self.allocator);

        try shader_library_point_light.add(&.{ &point_light_frag, &point_light_vert }, &.{ vk.ShaderStageFlags{ .fragment_bit = true }, vk.ShaderStageFlags{ .vertex_bit = true } });

        point_light_renderer = try PointLightRenderer.init(@constCast(&self.gc), swapchain.render_pass, scene, shader_library_point_light, self.allocator, @constCast(&camera), global_set_layout.descriptor_set_layout);

        last_frame_time = glfw.getTime();
        frame_info.global_descriptor_set = global_descriptor_set;
    }

    pub fn onUpdate(self: *@This()) !bool {
        const current_time = glfw.getTime();
        const dt = current_time - last_frame_time;
        const cmdbuf = cmdbufs[current_frame];
        frame_info.command_buffer = cmdbuf;
        frame_info.dt = @floatCast(dt);
        frame_info.current_frame = current_frame;
        frame_info.extent = .{ .width = self.window.window.?.getSize().width, .height = self.window.window.?.getSize().height };

        try swapchain.beginFrame(frame_info);
        swapchain.beginSwapChainRenderPass(frame_info);
        camera_controller.processInput(&self.window, viewer_object, dt);
        frame_info.camera.viewMatrix = viewer_object.transform.local2world;
        var ubo = GlobalUbo{
            .view = frame_info.camera.viewMatrix,
            .projection = frame_info.camera.projectionMatrix,
        };

        try point_light_renderer.update_point_lights(&frame_info, &ubo);

        global_UBO_buffers.?[frame_info.current_frame].writeToBuffer(std.mem.asBytes(&ubo), vk.WHOLE_SIZE, 0);
        try global_UBO_buffers.?[frame_info.current_frame].flush(vk.WHOLE_SIZE, 0);

        try simple_renderer.render(frame_info);
        try point_light_renderer.render(frame_info);

        swapchain.endSwapChainRenderPass(frame_info);
        try swapchain.endFrame(frame_info.command_buffer, &current_frame, frame_info.extent);
        last_frame_time = current_time;

        return self.window.isRunning();
    }

    pub fn deinit(self: @This()) void {
        try swapchain.waitForAllFences();
        for (0..MAX_FRAMES_IN_FLIGHT) |i| {
            global_UBO_buffers.?[i].deinit();
        }
        self.gc.destroyCommandBuffers(cmdbufs, self.allocator);
        point_light_renderer.deinit();
        simple_renderer.deinit();

        swapchain.deinit();
        self.gc.deinit();
        self.window.deinit();
    }
};
