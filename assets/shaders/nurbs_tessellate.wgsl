struct Params {
    u_steps: u32,
    v_steps: u32,
    _pad0: u32,
    _pad1: u32,
}

struct Vertex {
    position: vec4<f32>,
    normal: vec4<f32>,
}

struct ControlPoints {
    points: array<vec4<f32>, 16>,
}

@group(0) @binding(0) var<uniform> control: ControlPoints;
@group(0) @binding(1) var<storage, read_write> vertices: array<Vertex>;
@group(0) @binding(2) var<uniform> params: Params;

fn bernstein3(t: f32) -> vec4<f32> {
    let omt = 1.0 - t;
    return vec4<f32>(
        omt * omt * omt,
        3.0 * t * omt * omt,
        3.0 * t * t * omt,
        t * t * t
    );
}

fn d_bernstein3(t: f32) -> vec4<f32> {
    let omt = 1.0 - t;
    return vec4<f32>(
        -3.0 * omt * omt,
        3.0 * omt * omt - 6.0 * t * omt,
        6.0 * t * omt - 3.0 * t * t,
        3.0 * t * t
    );
}

fn eval_patch(u: f32, v: f32, du: bool, dv: bool) -> vec4<f32> {
    var bu = bernstein3(u);
    var bv = bernstein3(v);
    if (du) {
        bu = d_bernstein3(u);
    }
    if (dv) {
        bv = d_bernstein3(v);
    }
    var sum = vec4<f32>(0.0);
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            sum = sum + control.points[j * 4u + i] * bu[i] * bv[j];
        }
    }
    return sum;
}

fn rational_derivative(c: vec4<f32>, dc: vec4<f32>) -> vec3<f32> {
    return (dc.xyz * c.w - c.xyz * dc.w) / max(c.w * c.w, 1.0e-8);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x > params.u_steps || gid.y > params.v_steps) {
        return;
    }
    let u = f32(gid.x) / f32(max(params.u_steps, 1u));
    let v = f32(gid.y) / f32(max(params.v_steps, 1u));
    let c = eval_patch(u, v, false, false);
    let cu = eval_patch(u, v, true, false);
    let cv = eval_patch(u, v, false, true);
    let p = c.xyz / max(c.w, 1.0e-8);
    let du = rational_derivative(c, cu);
    let dv = rational_derivative(c, cv);
    let raw_normal = cross(du, dv);
    let normal_length = length(raw_normal);
    let normal = select(vec3<f32>(0.0, 0.0, 1.0), raw_normal / normal_length, normal_length > 1.0e-8);
    let row = params.u_steps + 1u;
    let index = gid.y * row + gid.x;
    vertices[index].position = vec4<f32>(p, 1.0);
    vertices[index].normal = vec4<f32>(normal, 0.0);
}
