let wasm;

function getArrayF64FromWasm0(ptr, len) {
  ptr = ptr >>> 0;
  return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
  if (
    cachedDataViewMemory0 === null ||
    cachedDataViewMemory0.buffer.detached === true ||
    (cachedDataViewMemory0.buffer.detached === undefined &&
      cachedDataViewMemory0.buffer !== wasm.memory.buffer)
  ) {
    cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
  }
  return cachedDataViewMemory0;
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
  if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
    cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
  }
  return cachedFloat64ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
  ptr = ptr >>> 0;
  return decodeText(ptr, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
  if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
    cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
  }
  return cachedUint8ArrayMemory0;
}

function passArrayF64ToWasm0(arg, malloc) {
  const ptr = malloc(arg.length * 8, 8) >>> 0;
  getFloat64ArrayMemory0().set(arg, ptr / 8);
  WASM_VECTOR_LEN = arg.length;
  return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
  if (realloc === undefined) {
    const buf = cachedTextEncoder.encode(arg);
    const ptr = malloc(buf.length, 1) >>> 0;
    getUint8ArrayMemory0()
      .subarray(ptr, ptr + buf.length)
      .set(buf);
    WASM_VECTOR_LEN = buf.length;
    return ptr;
  }

  let len = arg.length;
  let ptr = malloc(len, 1) >>> 0;

  const mem = getUint8ArrayMemory0();

  let offset = 0;

  for (; offset < len; offset++) {
    const code = arg.charCodeAt(offset);
    if (code > 0x7f) break;
    mem[ptr + offset] = code;
  }
  if (offset !== len) {
    if (offset !== 0) {
      arg = arg.slice(offset);
    }
    ptr = realloc(ptr, len, (len = offset + arg.length * 3), 1) >>> 0;
    const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
    const ret = cachedTextEncoder.encodeInto(arg, view);

    offset += ret.written;
    ptr = realloc(ptr, len, offset, 1) >>> 0;
  }

  WASM_VECTOR_LEN = offset;
  return ptr;
}

function takeFromExternrefTable0(idx) {
  const value = wasm.__wbindgen_externrefs.get(idx);
  wasm.__externref_table_dealloc(idx);
  return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
  numBytesDecoded += len;
  if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
    cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
    cachedTextDecoder.decode();
    numBytesDecoded = len;
  }
  return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
  cachedTextEncoder.encodeInto = function (arg, view) {
    const buf = cachedTextEncoder.encode(arg);
    view.set(buf);
    return {
      read: arg.length,
      written: buf.length,
    };
  };
}

let WASM_VECTOR_LEN = 0;

const WasmSwarmFinalization =
  typeof FinalizationRegistry === 'undefined'
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry((ptr) => wasm.__wbg_wasmswarm_free(ptr >>> 0, 1));

/**
 * WASM-friendly swarm wrapper
 */
export class WasmSwarm {
  __destroy_into_raw() {
    const ptr = this.__wbg_ptr;
    this.__wbg_ptr = 0;
    WasmSwarmFinalization.unregister(this);
    return ptr;
  }
  free() {
    const ptr = this.__destroy_into_raw();
    wasm.__wbg_wasmswarm_free(ptr, 0);
  }
  /**
   * Get current metrics
   * @returns {any}
   */
  get_metrics() {
    const ret = wasm.wasmswarm_get_metrics(this.__wbg_ptr);
    return ret;
  }
  /**
   * Get current positions as flat array
   * @returns {Float64Array}
   */
  get_positions() {
    const ret = wasm.wasmswarm_get_positions(this.__wbg_ptr);
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
  }
  /**
   * Set attractor position
   * @param {number} x
   * @param {number} y
   */
  set_attractor(x, y) {
    wasm.wasmswarm_set_attractor(this.__wbg_ptr, x, y);
  }
  /**
   * Get number of particles
   * @returns {number}
   */
  get_n_particles() {
    const ret = wasm.wasmswarm_get_n_particles(this.__wbg_ptr);
    return ret >>> 0;
  }
  /**
   * Create new swarm
   * @param {number} n_particles
   * @param {number} dt
   * @param {number} diffusion
   * @param {number} interaction_strength
   * @param {number} attractor_x
   * @param {number} attractor_y
   * @param {number} attractor_strength
   * @param {bigint} seed
   */
  constructor(
    n_particles,
    dt,
    diffusion,
    interaction_strength,
    attractor_x,
    attractor_y,
    attractor_strength,
    seed
  ) {
    const ret = wasm.wasmswarm_new(
      n_particles,
      dt,
      diffusion,
      interaction_strength,
      attractor_x,
      attractor_y,
      attractor_strength,
      seed
    );
    this.__wbg_ptr = ret >>> 0;
    WasmSwarmFinalization.register(this, this.__wbg_ptr, this);
    return this;
  }
  /**
   * Run multiple steps
   * @param {number} n_steps
   * @returns {any}
   */
  run(n_steps) {
    const ret = wasm.wasmswarm_run(this.__wbg_ptr, n_steps);
    return ret;
  }
  /**
   * Step simulation
   */
  step() {
    wasm.wasmswarm_step(this.__wbg_ptr);
  }
  /**
   * Get time
   * @returns {number}
   */
  get_time() {
    const ret = wasm.wasmswarm_get_time(this.__wbg_ptr);
    return ret;
  }
}
if (Symbol.dispose) WasmSwarm.prototype[Symbol.dispose] = WasmSwarm.prototype.free;

/**
 * Initialize WASM module
 */
export function init() {
  wasm.init();
}

/**
 * Analyze Q-matrix (WASM)
 * @param {Float64Array} q_flat
 * @param {number} n
 * @returns {any}
 */
export function wasm_analyze_q(q_flat, n) {
  const ptr0 = passArrayF64ToWasm0(q_flat, wasm.__wbindgen_malloc);
  const len0 = WASM_VECTOR_LEN;
  const ret = wasm.wasm_analyze_q(ptr0, len0, n);
  if (ret[2]) {
    throw takeFromExternrefTable0(ret[1]);
  }
  return takeFromExternrefTable0(ret[0]);
}

/**
 * Build Q-matrix from rates (WASM)
 * @param {Float64Array} rates_flat
 * @param {number} n
 * @returns {Float64Array}
 */
export function wasm_build_q_matrix(rates_flat, n) {
  const ptr0 = passArrayF64ToWasm0(rates_flat, wasm.__wbindgen_malloc);
  const len0 = WASM_VECTOR_LEN;
  const ret = wasm.wasm_build_q_matrix(ptr0, len0, n);
  if (ret[3]) {
    throw takeFromExternrefTable0(ret[2]);
  }
  var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
  wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
  return v2;
}

/**
 * Compute persistence diagram (WASM)
 * @param {Float64Array} points_flat
 * @param {number} n_points
 * @param {number} max_edge
 * @returns {any}
 */
export function wasm_compute_persistence(points_flat, n_points, max_edge) {
  const ptr0 = passArrayF64ToWasm0(points_flat, wasm.__wbindgen_malloc);
  const len0 = WASM_VECTOR_LEN;
  const ret = wasm.wasm_compute_persistence(ptr0, len0, n_points, max_edge);
  if (ret[2]) {
    throw takeFromExternrefTable0(ret[1]);
  }
  return takeFromExternrefTable0(ret[0]);
}

/**
 * Compute distance matrix (WASM)
 * @param {Float64Array} points_flat
 * @param {number} n_points
 * @returns {Float64Array}
 */
export function wasm_distance_matrix(points_flat, n_points) {
  const ptr0 = passArrayF64ToWasm0(points_flat, wasm.__wbindgen_malloc);
  const len0 = WASM_VECTOR_LEN;
  const ret = wasm.wasm_distance_matrix(ptr0, len0, n_points);
  var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
  wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
  return v2;
}

/**
 * Integrate geodesic on Fisher metric (WASM)
 * @param {Float64Array} x0
 * @param {Float64Array} v0
 * @param {number} dt
 * @param {number} n_steps
 * @returns {any}
 */
export function wasm_integrate_geodesic_fisher(x0, v0, dt, n_steps) {
  const ptr0 = passArrayF64ToWasm0(x0, wasm.__wbindgen_malloc);
  const len0 = WASM_VECTOR_LEN;
  const ptr1 = passArrayF64ToWasm0(v0, wasm.__wbindgen_malloc);
  const len1 = WASM_VECTOR_LEN;
  const ret = wasm.wasm_integrate_geodesic_fisher(ptr0, len0, ptr1, len1, dt, n_steps);
  if (ret[2]) {
    throw takeFromExternrefTable0(ret[1]);
  }
  return takeFromExternrefTable0(ret[0]);
}

/**
 * Compute persistent entropy (WASM)
 * @param {Float64Array} births
 * @param {Float64Array} deaths
 * @returns {number}
 */
export function wasm_persistent_entropy(births, deaths) {
  const ptr0 = passArrayF64ToWasm0(births, wasm.__wbindgen_malloc);
  const len0 = WASM_VECTOR_LEN;
  const ptr1 = passArrayF64ToWasm0(deaths, wasm.__wbindgen_malloc);
  const len1 = WASM_VECTOR_LEN;
  const ret = wasm.wasm_persistent_entropy(ptr0, len0, ptr1, len1);
  return ret;
}

/**
 * Simulate Markov chain (WASM)
 * @param {Float64Array} q_flat
 * @param {number} n
 * @param {number} r0
 * @param {number} total_time
 * @param {number} dt
 * @param {bigint} seed
 * @returns {any}
 */
export function wasm_simulate_markov_chain(q_flat, n, r0, total_time, dt, seed) {
  const ptr0 = passArrayF64ToWasm0(q_flat, wasm.__wbindgen_malloc);
  const len0 = WASM_VECTOR_LEN;
  const ret = wasm.wasm_simulate_markov_chain(ptr0, len0, n, r0, total_time, dt, seed);
  if (ret[2]) {
    throw takeFromExternrefTable0(ret[1]);
  }
  return takeFromExternrefTable0(ret[0]);
}

const EXPECTED_RESPONSE_TYPES = new Set(['basic', 'cors', 'default']);

async function __wbg_load(module, imports) {
  if (typeof Response === 'function' && module instanceof Response) {
    if (typeof WebAssembly.instantiateStreaming === 'function') {
      try {
        return await WebAssembly.instantiateStreaming(module, imports);
      } catch (e) {
        const validResponse = module.ok && EXPECTED_RESPONSE_TYPES.has(module.type);

        if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
          console.warn(
            '`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n',
            e
          );
        } else {
          throw e;
        }
      }
    }

    const bytes = await module.arrayBuffer();
    return await WebAssembly.instantiate(bytes, imports);
  } else {
    const instance = await WebAssembly.instantiate(module, imports);

    if (instance instanceof WebAssembly.Instance) {
      return { instance, module };
    } else {
      return instance;
    }
  }
}

function __wbg_get_imports() {
  const imports = {};
  imports.wbg = {};
  imports.wbg.__wbg_Error_52673b7de5a0ca89 = function (arg0, arg1) {
    const ret = Error(getStringFromWasm0(arg0, arg1));
    return ret;
  };
  imports.wbg.__wbg_String_8f0eb39a4a4c2f66 = function (arg0, arg1) {
    const ret = String(arg1);
    const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
    getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
  };
  imports.wbg.__wbg___wbindgen_throw_dd24417ed36fc46e = function (arg0, arg1) {
    throw new Error(getStringFromWasm0(arg0, arg1));
  };
  imports.wbg.__wbg_error_7534b8e9a36f1ab4 = function (arg0, arg1) {
    let deferred0_0;
    let deferred0_1;
    try {
      deferred0_0 = arg0;
      deferred0_1 = arg1;
      console.error(getStringFromWasm0(arg0, arg1));
    } finally {
      wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
    }
  };
  imports.wbg.__wbg_new_1ba21ce319a06297 = function () {
    const ret = new Object();
    return ret;
  };
  imports.wbg.__wbg_new_25f239778d6112b9 = function () {
    const ret = new Array();
    return ret;
  };
  imports.wbg.__wbg_new_8a6f238a6ece86ea = function () {
    const ret = new Error();
    return ret;
  };
  imports.wbg.__wbg_set_3f1d0b984ed272ed = function (arg0, arg1, arg2) {
    arg0[arg1] = arg2;
  };
  imports.wbg.__wbg_set_7df433eea03a5c14 = function (arg0, arg1, arg2) {
    arg0[arg1 >>> 0] = arg2;
  };
  imports.wbg.__wbg_stack_0ed75d68575b0f3c = function (arg0, arg1) {
    const ret = arg1.stack;
    const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
    getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
  };
  imports.wbg.__wbindgen_cast_2241b6af4c4b2941 = function (arg0, arg1) {
    // Cast intrinsic for `Ref(String) -> Externref`.
    const ret = getStringFromWasm0(arg0, arg1);
    return ret;
  };
  imports.wbg.__wbindgen_cast_4625c577ab2ec9ee = function (arg0) {
    // Cast intrinsic for `U64 -> Externref`.
    const ret = BigInt.asUintN(64, arg0);
    return ret;
  };
  imports.wbg.__wbindgen_cast_d6cd19b81560fd6e = function (arg0) {
    // Cast intrinsic for `F64 -> Externref`.
    const ret = arg0;
    return ret;
  };
  imports.wbg.__wbindgen_init_externref_table = function () {
    const table = wasm.__wbindgen_externrefs;
    const offset = table.grow(4);
    table.set(0, undefined);
    table.set(offset + 0, undefined);
    table.set(offset + 1, null);
    table.set(offset + 2, true);
    table.set(offset + 3, false);
  };

  return imports;
}

function __wbg_finalize_init(instance, module) {
  wasm = instance.exports;
  __wbg_init.__wbindgen_wasm_module = module;
  cachedDataViewMemory0 = null;
  cachedFloat64ArrayMemory0 = null;
  cachedUint8ArrayMemory0 = null;

  wasm.__wbindgen_start();
  return wasm;
}

function initSync(module) {
  if (wasm !== undefined) return wasm;

  if (typeof module !== 'undefined') {
    if (Object.getPrototypeOf(module) === Object.prototype) {
      ({ module } = module);
    } else {
      console.warn('using deprecated parameters for `initSync()`; pass a single object instead');
    }
  }

  const imports = __wbg_get_imports();
  if (!(module instanceof WebAssembly.Module)) {
    module = new WebAssembly.Module(module);
  }
  const instance = new WebAssembly.Instance(module, imports);
  return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
  if (wasm !== undefined) return wasm;

  if (typeof module_or_path !== 'undefined') {
    if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
      ({ module_or_path } = module_or_path);
    } else {
      console.warn(
        'using deprecated parameters for the initialization function; pass a single object instead'
      );
    }
  }

  if (typeof module_or_path === 'undefined') {
    module_or_path = new URL('latticeforge_core_bg.wasm', import.meta.url);
  }
  const imports = __wbg_get_imports();

  if (
    typeof module_or_path === 'string' ||
    (typeof Request === 'function' && module_or_path instanceof Request) ||
    (typeof URL === 'function' && module_or_path instanceof URL)
  ) {
    module_or_path = fetch(module_or_path);
  }

  const { instance, module } = await __wbg_load(await module_or_path, imports);

  return __wbg_finalize_init(instance, module);
}

export { initSync };
export default __wbg_init;
