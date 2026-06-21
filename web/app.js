const canvas = document.querySelector("#viewport");
const modePatch = document.querySelector("#modePatch");
const modeBoolean = document.querySelector("#modeBoolean");
const gridSlider = document.querySelector("#grid");
const segmentsSlider = document.querySelector("#segments");
const stats = document.querySelector("#stats");
const meshName = document.querySelector("#meshName");

let device;
let context;
let format;
let depthTexture;
let uniformBuffer;
let renderPipeline;
let computePipeline;
let computeBindGroup;
let controlBuffer;
let paramsBuffer;
let vertexBuffer;
let indexBuffer;
let indexCount = 0;
let vertexCount = 0;
let drawIndexed = true;
let mode = "patch";
let yaw = -0.55;
let pitch = 0.55;
let distance = 4.7;
let dragging = false;
let lastPointer = [0, 0];

const renderShader = `
struct Uniforms {
  mvp: mat4x4<f32>,
  model: mat4x4<f32>,
  light: vec4<f32>,
}

struct VsOut {
  @builtin(position) position: vec4<f32>,
  @location(0) normal: vec3<f32>,
  @location(1) world: vec3<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vs(@location(0) position: vec4<f32>, @location(1) normal: vec4<f32>) -> VsOut {
  var out: VsOut;
  let world = uniforms.model * vec4<f32>(position.xyz, 1.0);
  out.position = uniforms.mvp * vec4<f32>(position.xyz, 1.0);
  out.normal = normalize((uniforms.model * vec4<f32>(normal.xyz, 0.0)).xyz);
  out.world = world.xyz;
  return out;
}

@fragment
fn fs(input: VsOut, @builtin(front_facing) frontFacing: bool) -> @location(0) vec4<f32> {
  let n = select(-normalize(input.normal), normalize(input.normal), frontFacing);
  let l = normalize(uniforms.light.xyz);
  let diffuse = max(dot(n, l), 0.0);
  let rim = pow(1.0 - max(dot(n, vec3<f32>(0.0, 0.0, 1.0)), 0.0), 2.0);
  let base = vec3<f32>(0.40, 0.84, 0.76);
  let warm = vec3<f32>(0.94, 0.63, 0.22);
  let color = base * (0.22 + 0.68 * diffuse) + warm * rim * 0.18;
  return vec4<f32>(color, 1.0);
}
`;

async function main() {
  if (!navigator.gpu) {
    stats.textContent = "WebGPU unavailable";
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    stats.textContent = "no adapter";
    return;
  }
  device = await adapter.requestDevice();
  context = canvas.getContext("webgpu");
  format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });

  uniformBuffer = device.createBuffer({
    size: 144,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const renderModule = device.createShaderModule({ code: renderShader });
  renderPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: renderModule,
      entryPoint: "vs",
      buffers: [
        {
          arrayStride: 32,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x4" },
            { shaderLocation: 1, offset: 16, format: "float32x4" },
          ],
        },
      ],
    },
    fragment: {
      module: renderModule,
      entryPoint: "fs",
      targets: [{ format }],
    },
    primitive: {
      topology: "triangle-list",
      cullMode: "none",
    },
    depthStencil: {
      format: "depth24plus",
      depthWriteEnabled: true,
      depthCompare: "less",
    },
  });

  await buildPatchResources();
  bindEvents();
  resize();
  await setMode("patch");
  requestAnimationFrame(frame);
}

async function buildPatchResources() {
  const shaderCode = await fetch("../assets/shaders/nurbs_tessellate.wgsl").then((r) => r.text());
  const computeModule = device.createShaderModule({ code: shaderCode });
  computePipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: computeModule, entryPoint: "main" },
  });

  const control = rationalPatchControl();
  controlBuffer = device.createBuffer({
    size: control.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(controlBuffer, 0, control);

  paramsBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

async function setMode(nextMode) {
  mode = nextMode;
  modePatch.classList.toggle("active", mode === "patch");
  modeBoolean.classList.toggle("active", mode === "boolean");
  gridSlider.disabled = mode !== "patch";
  segmentsSlider.disabled = mode !== "boolean";
  if (mode === "patch") {
    meshName.textContent = "GPU NURBS";
    await rebuildPatch();
  } else {
    meshName.textContent = "B-rep Boolean";
    await rebuildBoolean();
  }
}

async function rebuildPatch() {
  const steps = Number(gridSlider.value);
  const row = steps + 1;
  const vertices = row * row;
  vertexBuffer = device.createBuffer({
    size: vertices * 32,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  indexBuffer = device.createBuffer({
    size: steps * steps * 6 * 4,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  const indices = gridIndices(steps, steps);
  device.queue.writeBuffer(indexBuffer, 0, indices);
  indexCount = indices.length;
  vertexCount = vertices;
  drawIndexed = true;
  device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([steps, steps, 0, 0]));
  computeBindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: controlBuffer } },
      { binding: 1, resource: { buffer: vertexBuffer } },
      { binding: 2, resource: { buffer: paramsBuffer } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(computePipeline);
  pass.setBindGroup(0, computeBindGroup);
  pass.dispatchWorkgroups(Math.ceil(row / 8), Math.ceil(row / 8));
  pass.end();
  device.queue.submit([encoder.finish()]);
  stats.textContent = `${vertices.toLocaleString()} vertices, ${Math.floor(indexCount / 3).toLocaleString()} triangles`;
}

async function rebuildBoolean() {
  const segments = Number(segmentsSlider.value);
  const raw = (await loadWasmBoolean(segments)) ?? buildBooleanFallback(segments);
  vertexBuffer = device.createBuffer({
    size: raw.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, raw);
  indexBuffer = undefined;
  vertexCount = raw.length / 8;
  indexCount = 0;
  drawIndexed = false;
  stats.textContent = `${vertexCount.toLocaleString()} vertices, ${Math.floor(vertexCount / 3).toLocaleString()} triangles`;
}

async function loadWasmBoolean(segments) {
  try {
    const response = await fetch("../target/wasm32-unknown-unknown/release/brep_kernel.wasm");
    if (!response.ok) {
      return null;
    }
    const { instance } = await WebAssembly.instantiateStreaming(response, {});
    const exports = instance.exports;
    const len = exports.brep_demo_mesh(segments);
    const ptr = exports.brep_buffer_ptr();
    const memory = exports.memory;
    if (!len || !ptr || !memory) {
      return null;
    }
    const packed = new Float32Array(memory.buffer, ptr, len).slice();
    const expanded = new Float32Array((packed.length / 6) * 8);
    for (let i = 0, j = 0; i < packed.length; i += 6, j += 8) {
      expanded[j + 0] = packed[i + 0];
      expanded[j + 1] = packed[i + 1];
      expanded[j + 2] = packed[i + 2];
      expanded[j + 3] = 1;
      expanded[j + 4] = packed[i + 3];
      expanded[j + 5] = packed[i + 4];
      expanded[j + 6] = packed[i + 5];
      expanded[j + 7] = 0;
    }
    return expanded;
  } catch {
    return null;
  }
}

function buildBooleanFallback(requestedSegments) {
  const segments = Math.ceil(Math.max(8, requestedSegments) / 8) * 8;
  const h = 1;
  const r = 0.45;
  const points = [];
  const it = [];
  const ib = [];
  const ot = [];
  const ob = [];
  for (let i = 0; i < segments; i++) {
    const theta = Math.PI * 2 * i / segments;
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    const scale = h / Math.max(Math.abs(c), Math.abs(s));
    it.push(pushPoint(points, [r * c, r * s, h]));
    ib.push(pushPoint(points, [r * c, r * s, -h]));
    ot.push(pushPoint(points, [scale * c, scale * s, h]));
    ob.push(pushPoint(points, [scale * c, scale * s, -h]));
  }
  const tris = [];
  for (let i = 0; i < segments; i++) {
    const j = (i + 1) % segments;
    tris.push([it[i], ot[i], ot[j]], [it[i], ot[j], it[j]]);
    tris.push([ib[i], ob[j], ob[i]], [ib[i], ib[j], ob[j]]);
    tris.push([ib[i], it[i], it[j]], [ib[i], it[j], ib[j]]);
    tris.push([ob[i], ob[j], ot[j]], [ob[i], ot[j], ot[i]]);
  }
  const out = new Float32Array(tris.length * 3 * 8);
  let k = 0;
  for (const tri of tris) {
    const a = points[tri[0]];
    const b = points[tri[1]];
    const c = points[tri[2]];
    const n = normalize(cross(sub(b, a), sub(c, a)));
    for (const p of [a, b, c]) {
      out[k++] = p[0];
      out[k++] = p[1];
      out[k++] = p[2];
      out[k++] = 1;
      out[k++] = n[0];
      out[k++] = n[1];
      out[k++] = n[2];
      out[k++] = 0;
    }
  }
  return out;
}

function frame() {
  resize();
  updateUniforms();
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.06, g: 0.07, b: 0.085, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
    depthStencilAttachment: {
      view: depthTexture.createView(),
      depthClearValue: 1,
      depthLoadOp: "clear",
      depthStoreOp: "store",
    },
  });
  pass.setPipeline(renderPipeline);
  pass.setBindGroup(0, device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  }));
  pass.setVertexBuffer(0, vertexBuffer);
  if (drawIndexed) {
    pass.setIndexBuffer(indexBuffer, "uint32");
    pass.drawIndexed(indexCount);
  } else {
    pass.draw(vertexCount);
  }
  pass.end();
  device.queue.submit([encoder.finish()]);
  requestAnimationFrame(frame);
}

function updateUniforms() {
  const aspect = canvas.width / Math.max(1, canvas.height);
  const projection = mat4Perspective(45 * Math.PI / 180, aspect, 0.01, 100);
  const eye = [
    distance * Math.cos(pitch) * Math.sin(yaw),
    distance * Math.sin(pitch),
    distance * Math.cos(pitch) * Math.cos(yaw),
  ];
  const view = mat4LookAt(eye, [0, 0, 0], [0, 1, 0]);
  const model = mat4Identity();
  const mvp = mat4Mul(projection, mat4Mul(view, model));
  const data = new Float32Array(36);
  data.set(mvp, 0);
  data.set(model, 16);
  data.set(normalize([0.45, 0.8, 0.4]).concat([0]), 32);
  device.queue.writeBuffer(uniformBuffer, 0, data);
}

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width === width && canvas.height === height && depthTexture) {
    return;
  }
  canvas.width = width;
  canvas.height = height;
  depthTexture = device.createTexture({
    size: [width, height],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
}

function bindEvents() {
  modePatch.addEventListener("click", () => setMode("patch"));
  modeBoolean.addEventListener("click", () => setMode("boolean"));
  gridSlider.addEventListener("input", () => mode === "patch" && rebuildPatch());
  segmentsSlider.addEventListener("input", () => mode === "boolean" && rebuildBoolean());
  canvas.addEventListener("pointerdown", (event) => {
    dragging = true;
    lastPointer = [event.clientX, event.clientY];
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const dx = event.clientX - lastPointer[0];
    const dy = event.clientY - lastPointer[1];
    yaw += dx * 0.008;
    pitch = clamp(pitch + dy * 0.008, -1.35, 1.35);
    lastPointer = [event.clientX, event.clientY];
  });
  canvas.addEventListener("pointerup", () => {
    dragging = false;
  });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    distance = clamp(distance * Math.exp(event.deltaY * 0.001), 2.2, 10);
  }, { passive: false });
}

function rationalPatchControl() {
  const out = new Float32Array(16 * 4);
  const weights = [
    1.0, 0.92, 0.92, 1.0,
    0.94, 0.62, 0.62, 0.94,
    0.94, 0.62, 0.62, 0.94,
    1.0, 0.92, 0.92, 1.0,
  ];
  let k = 0;
  for (let j = 0; j < 4; j++) {
    const y = -1.35 + j * 0.9;
    for (let i = 0; i < 4; i++) {
      const x = -1.35 + i * 0.9;
      const z = 0.35 * Math.sin(i * 1.25) - 0.28 * Math.cos(j * 1.4);
      const w = weights[j * 4 + i];
      out[k++] = x * w;
      out[k++] = y * w;
      out[k++] = z * w;
      out[k++] = w;
    }
  }
  return out;
}

function gridIndices(uSteps, vSteps) {
  const indices = new Uint32Array(uSteps * vSteps * 6);
  const row = uSteps + 1;
  let k = 0;
  for (let j = 0; j < vSteps; j++) {
    for (let i = 0; i < uSteps; i++) {
      const a = j * row + i;
      const b = a + 1;
      const c = (j + 1) * row + i;
      const d = c + 1;
      indices[k++] = a;
      indices[k++] = b;
      indices[k++] = d;
      indices[k++] = a;
      indices[k++] = d;
      indices[k++] = c;
    }
  }
  return indices;
}

function pushPoint(points, p) {
  points.push(p);
  return points.length - 1;
}

function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function normalize(v) {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
}

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

function mat4Identity() {
  return [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ];
}

function mat4Perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return [
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ];
}

function mat4LookAt(eye, center, up) {
  const z = normalize(sub(eye, center));
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return [
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ];
}

function mat4Mul(a, b) {
  const out = new Array(16).fill(0);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      for (let k = 0; k < 4; k++) {
        out[col * 4 + row] += a[k * 4 + row] * b[col * 4 + k];
      }
    }
  }
  return out;
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

main();
