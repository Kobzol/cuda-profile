/******/ (function(modules) { // webpackBootstrap
/******/ 	// The module cache
/******/ 	var installedModules = {};
/******/
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/
/******/ 		// Check if module is in cache
/******/ 		if(installedModules[moduleId]) {
/******/ 			return installedModules[moduleId].exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = installedModules[moduleId] = {
/******/ 			i: moduleId,
/******/ 			l: false,
/******/ 			exports: {}
/******/ 		};
/******/
/******/ 		// Execute the module function
/******/ 		modules[moduleId].call(module.exports, module, module.exports, __webpack_require__);
/******/
/******/ 		// Flag the module as loaded
/******/ 		module.l = true;
/******/
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/
/******/
/******/ 	// expose the modules object (__webpack_modules__)
/******/ 	__webpack_require__.m = modules;
/******/
/******/ 	// expose the module cache
/******/ 	__webpack_require__.c = installedModules;
/******/
/******/ 	// define getter function for harmony exports
/******/ 	__webpack_require__.d = function(exports, name, getter) {
/******/ 		if(!__webpack_require__.o(exports, name)) {
/******/ 			Object.defineProperty(exports, name, {
/******/ 				configurable: false,
/******/ 				enumerable: true,
/******/ 				get: getter
/******/ 			});
/******/ 		}
/******/ 	};
/******/
/******/ 	// getDefaultExport function for compatibility with non-harmony modules
/******/ 	__webpack_require__.n = function(module) {
/******/ 		var getter = module && module.__esModule ?
/******/ 			function getDefault() { return module['default']; } :
/******/ 			function getModuleExports() { return module; };
/******/ 		__webpack_require__.d(getter, 'a', getter);
/******/ 		return getter;
/******/ 	};
/******/
/******/ 	// Object.prototype.hasOwnProperty.call
/******/ 	__webpack_require__.o = function(object, property) { return Object.prototype.hasOwnProperty.call(object, property); };
/******/
/******/ 	// __webpack_public_path__
/******/ 	__webpack_require__.p = "/cuda-profile/";
/******/
/******/ 	// Load entry module and return exports
/******/ 	return __webpack_require__(__webpack_require__.s = 3);
/******/ })
/************************************************************************/
/******/ ([
/* 0 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";
/* WEBPACK VAR INJECTION */(function(global) {
var util = exports;

// used to return a Promise where callback is omitted
util.asPromise = __webpack_require__(8);

// converts to / from base64 encoded strings
util.base64 = __webpack_require__(9);

// base class of rpc.Service
util.EventEmitter = __webpack_require__(10);

// float handling accross browsers
util.float = __webpack_require__(11);

// requires modules optionally and hides the call from bundlers
util.inquire = __webpack_require__(12);

// converts to / from utf8 encoded strings
util.utf8 = __webpack_require__(13);

// provides a node-like buffer pool in the browser
util.pool = __webpack_require__(14);

// utility to work with the low and high bits of a 64 bit value
util.LongBits = __webpack_require__(15);

/**
 * An immuable empty array.
 * @memberof util
 * @type {Array.<*>}
 * @const
 */
util.emptyArray = Object.freeze ? Object.freeze([]) : /* istanbul ignore next */ []; // used on prototypes

/**
 * An immutable empty object.
 * @type {Object}
 * @const
 */
util.emptyObject = Object.freeze ? Object.freeze({}) : /* istanbul ignore next */ {}; // used on prototypes

/**
 * Whether running within node or not.
 * @memberof util
 * @type {boolean}
 * @const
 */
util.isNode = Boolean(global.process && global.process.versions && global.process.versions.node);

/**
 * Tests if the specified value is an integer.
 * @function
 * @param {*} value Value to test
 * @returns {boolean} `true` if the value is an integer
 */
util.isInteger = Number.isInteger || /* istanbul ignore next */ function isInteger(value) {
    return typeof value === "number" && isFinite(value) && Math.floor(value) === value;
};

/**
 * Tests if the specified value is a string.
 * @param {*} value Value to test
 * @returns {boolean} `true` if the value is a string
 */
util.isString = function isString(value) {
    return typeof value === "string" || value instanceof String;
};

/**
 * Tests if the specified value is a non-null object.
 * @param {*} value Value to test
 * @returns {boolean} `true` if the value is a non-null object
 */
util.isObject = function isObject(value) {
    return value && typeof value === "object";
};

/**
 * Checks if a property on a message is considered to be present.
 * This is an alias of {@link util.isSet}.
 * @function
 * @param {Object} obj Plain object or message instance
 * @param {string} prop Property name
 * @returns {boolean} `true` if considered to be present, otherwise `false`
 */
util.isset =

/**
 * Checks if a property on a message is considered to be present.
 * @param {Object} obj Plain object or message instance
 * @param {string} prop Property name
 * @returns {boolean} `true` if considered to be present, otherwise `false`
 */
util.isSet = function isSet(obj, prop) {
    var value = obj[prop];
    if (value != null && obj.hasOwnProperty(prop)) // eslint-disable-line eqeqeq, no-prototype-builtins
        return typeof value !== "object" || (Array.isArray(value) ? value.length : Object.keys(value).length) > 0;
    return false;
};

/**
 * Any compatible Buffer instance.
 * This is a minimal stand-alone definition of a Buffer instance. The actual type is that exported by node's typings.
 * @interface Buffer
 * @extends Uint8Array
 */

/**
 * Node's Buffer class if available.
 * @type {Constructor<Buffer>}
 */
util.Buffer = (function() {
    try {
        var Buffer = util.inquire("buffer").Buffer;
        // refuse to use non-node buffers if not explicitly assigned (perf reasons):
        return Buffer.prototype.utf8Write ? Buffer : /* istanbul ignore next */ null;
    } catch (e) {
        /* istanbul ignore next */
        return null;
    }
})();

// Internal alias of or polyfull for Buffer.from.
util._Buffer_from = null;

// Internal alias of or polyfill for Buffer.allocUnsafe.
util._Buffer_allocUnsafe = null;

/**
 * Creates a new buffer of whatever type supported by the environment.
 * @param {number|number[]} [sizeOrArray=0] Buffer size or number array
 * @returns {Uint8Array|Buffer} Buffer
 */
util.newBuffer = function newBuffer(sizeOrArray) {
    /* istanbul ignore next */
    return typeof sizeOrArray === "number"
        ? util.Buffer
            ? util._Buffer_allocUnsafe(sizeOrArray)
            : new util.Array(sizeOrArray)
        : util.Buffer
            ? util._Buffer_from(sizeOrArray)
            : typeof Uint8Array === "undefined"
                ? sizeOrArray
                : new Uint8Array(sizeOrArray);
};

/**
 * Array implementation used in the browser. `Uint8Array` if supported, otherwise `Array`.
 * @type {Constructor<Uint8Array>}
 */
util.Array = typeof Uint8Array !== "undefined" ? Uint8Array /* istanbul ignore next */ : Array;

/**
 * Any compatible Long instance.
 * This is a minimal stand-alone definition of a Long instance. The actual type is that exported by long.js.
 * @interface Long
 * @property {number} low Low bits
 * @property {number} high High bits
 * @property {boolean} unsigned Whether unsigned or not
 */

/**
 * Long.js's Long class if available.
 * @type {Constructor<Long>}
 */
util.Long = /* istanbul ignore next */ global.dcodeIO && /* istanbul ignore next */ global.dcodeIO.Long || util.inquire("long");

/**
 * Regular expression used to verify 2 bit (`bool`) map keys.
 * @type {RegExp}
 * @const
 */
util.key2Re = /^true|false|0|1$/;

/**
 * Regular expression used to verify 32 bit (`int32` etc.) map keys.
 * @type {RegExp}
 * @const
 */
util.key32Re = /^-?(?:0|[1-9][0-9]*)$/;

/**
 * Regular expression used to verify 64 bit (`int64` etc.) map keys.
 * @type {RegExp}
 * @const
 */
util.key64Re = /^(?:[\\x00-\\xff]{8}|-?(?:0|[1-9][0-9]*))$/;

/**
 * Converts a number or long to an 8 characters long hash string.
 * @param {Long|number} value Value to convert
 * @returns {string} Hash
 */
util.longToHash = function longToHash(value) {
    return value
        ? util.LongBits.from(value).toHash()
        : util.LongBits.zeroHash;
};

/**
 * Converts an 8 characters long hash string to a long or number.
 * @param {string} hash Hash
 * @param {boolean} [unsigned=false] Whether unsigned or not
 * @returns {Long|number} Original value
 */
util.longFromHash = function longFromHash(hash, unsigned) {
    var bits = util.LongBits.fromHash(hash);
    if (util.Long)
        return util.Long.fromBits(bits.lo, bits.hi, unsigned);
    return bits.toNumber(Boolean(unsigned));
};

/**
 * Merges the properties of the source object into the destination object.
 * @memberof util
 * @param {Object.<string,*>} dst Destination object
 * @param {Object.<string,*>} src Source object
 * @param {boolean} [ifNotSet=false] Merges only if the key is not already set
 * @returns {Object.<string,*>} Destination object
 */
function merge(dst, src, ifNotSet) { // used by converters
    for (var keys = Object.keys(src), i = 0; i < keys.length; ++i)
        if (dst[keys[i]] === undefined || !ifNotSet)
            dst[keys[i]] = src[keys[i]];
    return dst;
}

util.merge = merge;

/**
 * Converts the first character of a string to lower case.
 * @param {string} str String to convert
 * @returns {string} Converted string
 */
util.lcFirst = function lcFirst(str) {
    return str.charAt(0).toLowerCase() + str.substring(1);
};

/**
 * Creates a custom error constructor.
 * @memberof util
 * @param {string} name Error name
 * @returns {Constructor<Error>} Custom error constructor
 */
function newError(name) {

    function CustomError(message, properties) {

        if (!(this instanceof CustomError))
            return new CustomError(message, properties);

        // Error.call(this, message);
        // ^ just returns a new error instance because the ctor can be called as a function

        Object.defineProperty(this, "message", { get: function() { return message; } });

        /* istanbul ignore next */
        if (Error.captureStackTrace) // node
            Error.captureStackTrace(this, CustomError);
        else
            Object.defineProperty(this, "stack", { value: (new Error()).stack || "" });

        if (properties)
            merge(this, properties);
    }

    (CustomError.prototype = Object.create(Error.prototype)).constructor = CustomError;

    Object.defineProperty(CustomError.prototype, "name", { get: function() { return name; } });

    CustomError.prototype.toString = function toString() {
        return this.name + ": " + this.message;
    };

    return CustomError;
}

util.newError = newError;

/**
 * Constructs a new protocol error.
 * @classdesc Error subclass indicating a protocol specifc error.
 * @memberof util
 * @extends Error
 * @template T extends Message<T>
 * @constructor
 * @param {string} message Error message
 * @param {Object.<string,*>} [properties] Additional properties
 * @example
 * try {
 *     MyMessage.decode(someBuffer); // throws if required fields are missing
 * } catch (e) {
 *     if (e instanceof ProtocolError && e.instance)
 *         console.log("decoded so far: " + JSON.stringify(e.instance));
 * }
 */
util.ProtocolError = newError("ProtocolError");

/**
 * So far decoded message instance.
 * @name util.ProtocolError#instance
 * @type {Message<T>}
 */

/**
 * A OneOf getter as returned by {@link util.oneOfGetter}.
 * @typedef OneOfGetter
 * @type {function}
 * @returns {string|undefined} Set field name, if any
 */

/**
 * Builds a getter for a oneof's present field name.
 * @param {string[]} fieldNames Field names
 * @returns {OneOfGetter} Unbound getter
 */
util.oneOfGetter = function getOneOf(fieldNames) {
    var fieldMap = {};
    for (var i = 0; i < fieldNames.length; ++i)
        fieldMap[fieldNames[i]] = 1;

    /**
     * @returns {string|undefined} Set field name, if any
     * @this Object
     * @ignore
     */
    return function() { // eslint-disable-line consistent-return
        for (var keys = Object.keys(this), i = keys.length - 1; i > -1; --i)
            if (fieldMap[keys[i]] === 1 && this[keys[i]] !== undefined && this[keys[i]] !== null)
                return keys[i];
    };
};

/**
 * A OneOf setter as returned by {@link util.oneOfSetter}.
 * @typedef OneOfSetter
 * @type {function}
 * @param {string|undefined} value Field name
 * @returns {undefined}
 */

/**
 * Builds a setter for a oneof's present field name.
 * @param {string[]} fieldNames Field names
 * @returns {OneOfSetter} Unbound setter
 */
util.oneOfSetter = function setOneOf(fieldNames) {

    /**
     * @param {string} name Field name
     * @returns {undefined}
     * @this Object
     * @ignore
     */
    return function(name) {
        for (var i = 0; i < fieldNames.length; ++i)
            if (fieldNames[i] !== name)
                delete this[fieldNames[i]];
    };
};

/**
 * Default conversion options used for {@link Message#toJSON} implementations.
 *
 * These options are close to proto3's JSON mapping with the exception that internal types like Any are handled just like messages. More precisely:
 *
 * - Longs become strings
 * - Enums become string keys
 * - Bytes become base64 encoded strings
 * - (Sub-)Messages become plain objects
 * - Maps become plain objects with all string keys
 * - Repeated fields become arrays
 * - NaN and Infinity for float and double fields become strings
 *
 * @type {IConversionOptions}
 * @see https://developers.google.com/protocol-buffers/docs/proto3?hl=en#json
 */
util.toJSONOptions = {
    longs: String,
    enums: String,
    bytes: String,
    json: true
};

util._configure = function() {
    var Buffer = util.Buffer;
    /* istanbul ignore if */
    if (!Buffer) {
        util._Buffer_from = util._Buffer_allocUnsafe = null;
        return;
    }
    // because node 4.x buffers are incompatible & immutable
    // see: https://github.com/dcodeIO/protobuf.js/pull/665
    util._Buffer_from = Buffer.from !== Uint8Array.from && Buffer.from ||
        /* istanbul ignore next */
        function Buffer_from(value, encoding) {
            return new Buffer(value, encoding);
        };
    util._Buffer_allocUnsafe = Buffer.allocUnsafe ||
        /* istanbul ignore next */
        function Buffer_allocUnsafe(size) {
            return new Buffer(size);
        };
};

/* WEBPACK VAR INJECTION */}.call(exports, __webpack_require__(7)))

/***/ }),
/* 1 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";

module.exports = Writer;

var util      = __webpack_require__(0);

var BufferWriter; // cyclic

var LongBits  = util.LongBits,
    base64    = util.base64,
    utf8      = util.utf8;

/**
 * Constructs a new writer operation instance.
 * @classdesc Scheduled writer operation.
 * @constructor
 * @param {function(*, Uint8Array, number)} fn Function to call
 * @param {number} len Value byte length
 * @param {*} val Value to write
 * @ignore
 */
function Op(fn, len, val) {

    /**
     * Function to call.
     * @type {function(Uint8Array, number, *)}
     */
    this.fn = fn;

    /**
     * Value byte length.
     * @type {number}
     */
    this.len = len;

    /**
     * Next operation.
     * @type {Writer.Op|undefined}
     */
    this.next = undefined;

    /**
     * Value to write.
     * @type {*}
     */
    this.val = val; // type varies
}

/* istanbul ignore next */
function noop() {} // eslint-disable-line no-empty-function

/**
 * Constructs a new writer state instance.
 * @classdesc Copied writer state.
 * @memberof Writer
 * @constructor
 * @param {Writer} writer Writer to copy state from
 * @ignore
 */
function State(writer) {

    /**
     * Current head.
     * @type {Writer.Op}
     */
    this.head = writer.head;

    /**
     * Current tail.
     * @type {Writer.Op}
     */
    this.tail = writer.tail;

    /**
     * Current buffer length.
     * @type {number}
     */
    this.len = writer.len;

    /**
     * Next state.
     * @type {State|null}
     */
    this.next = writer.states;
}

/**
 * Constructs a new writer instance.
 * @classdesc Wire format writer using `Uint8Array` if available, otherwise `Array`.
 * @constructor
 */
function Writer() {

    /**
     * Current length.
     * @type {number}
     */
    this.len = 0;

    /**
     * Operations head.
     * @type {Object}
     */
    this.head = new Op(noop, 0, 0);

    /**
     * Operations tail
     * @type {Object}
     */
    this.tail = this.head;

    /**
     * Linked forked states.
     * @type {Object|null}
     */
    this.states = null;

    // When a value is written, the writer calculates its byte length and puts it into a linked
    // list of operations to perform when finish() is called. This both allows us to allocate
    // buffers of the exact required size and reduces the amount of work we have to do compared
    // to first calculating over objects and then encoding over objects. In our case, the encoding
    // part is just a linked list walk calling operations with already prepared values.
}

/**
 * Creates a new writer.
 * @function
 * @returns {BufferWriter|Writer} A {@link BufferWriter} when Buffers are supported, otherwise a {@link Writer}
 */
Writer.create = util.Buffer
    ? function create_buffer_setup() {
        return (Writer.create = function create_buffer() {
            return new BufferWriter();
        })();
    }
    /* istanbul ignore next */
    : function create_array() {
        return new Writer();
    };

/**
 * Allocates a buffer of the specified size.
 * @param {number} size Buffer size
 * @returns {Uint8Array} Buffer
 */
Writer.alloc = function alloc(size) {
    return new util.Array(size);
};

// Use Uint8Array buffer pool in the browser, just like node does with buffers
/* istanbul ignore else */
if (util.Array !== Array)
    Writer.alloc = util.pool(Writer.alloc, util.Array.prototype.subarray);

/**
 * Pushes a new operation to the queue.
 * @param {function(Uint8Array, number, *)} fn Function to call
 * @param {number} len Value byte length
 * @param {number} val Value to write
 * @returns {Writer} `this`
 * @private
 */
Writer.prototype._push = function push(fn, len, val) {
    this.tail = this.tail.next = new Op(fn, len, val);
    this.len += len;
    return this;
};

function writeByte(val, buf, pos) {
    buf[pos] = val & 255;
}

function writeVarint32(val, buf, pos) {
    while (val > 127) {
        buf[pos++] = val & 127 | 128;
        val >>>= 7;
    }
    buf[pos] = val;
}

/**
 * Constructs a new varint writer operation instance.
 * @classdesc Scheduled varint writer operation.
 * @extends Op
 * @constructor
 * @param {number} len Value byte length
 * @param {number} val Value to write
 * @ignore
 */
function VarintOp(len, val) {
    this.len = len;
    this.next = undefined;
    this.val = val;
}

VarintOp.prototype = Object.create(Op.prototype);
VarintOp.prototype.fn = writeVarint32;

/**
 * Writes an unsigned 32 bit value as a varint.
 * @param {number} value Value to write
 * @returns {Writer} `this`
 */
Writer.prototype.uint32 = function write_uint32(value) {
    // here, the call to this.push has been inlined and a varint specific Op subclass is used.
    // uint32 is by far the most frequently used operation and benefits significantly from this.
    this.len += (this.tail = this.tail.next = new VarintOp(
        (value = value >>> 0)
                < 128       ? 1
        : value < 16384     ? 2
        : value < 2097152   ? 3
        : value < 268435456 ? 4
        :                     5,
    value)).len;
    return this;
};

/**
 * Writes a signed 32 bit value as a varint.
 * @function
 * @param {number} value Value to write
 * @returns {Writer} `this`
 */
Writer.prototype.int32 = function write_int32(value) {
    return value < 0
        ? this._push(writeVarint64, 10, LongBits.fromNumber(value)) // 10 bytes per spec
        : this.uint32(value);
};

/**
 * Writes a 32 bit value as a varint, zig-zag encoded.
 * @param {number} value Value to write
 * @returns {Writer} `this`
 */
Writer.prototype.sint32 = function write_sint32(value) {
    return this.uint32((value << 1 ^ value >> 31) >>> 0);
};

function writeVarint64(val, buf, pos) {
    while (val.hi) {
        buf[pos++] = val.lo & 127 | 128;
        val.lo = (val.lo >>> 7 | val.hi << 25) >>> 0;
        val.hi >>>= 7;
    }
    while (val.lo > 127) {
        buf[pos++] = val.lo & 127 | 128;
        val.lo = val.lo >>> 7;
    }
    buf[pos++] = val.lo;
}

/**
 * Writes an unsigned 64 bit value as a varint.
 * @param {Long|number|string} value Value to write
 * @returns {Writer} `this`
 * @throws {TypeError} If `value` is a string and no long library is present.
 */
Writer.prototype.uint64 = function write_uint64(value) {
    var bits = LongBits.from(value);
    return this._push(writeVarint64, bits.length(), bits);
};

/**
 * Writes a signed 64 bit value as a varint.
 * @function
 * @param {Long|number|string} value Value to write
 * @returns {Writer} `this`
 * @throws {TypeError} If `value` is a string and no long library is present.
 */
Writer.prototype.int64 = Writer.prototype.uint64;

/**
 * Writes a signed 64 bit value as a varint, zig-zag encoded.
 * @param {Long|number|string} value Value to write
 * @returns {Writer} `this`
 * @throws {TypeError} If `value` is a string and no long library is present.
 */
Writer.prototype.sint64 = function write_sint64(value) {
    var bits = LongBits.from(value).zzEncode();
    return this._push(writeVarint64, bits.length(), bits);
};

/**
 * Writes a boolish value as a varint.
 * @param {boolean} value Value to write
 * @returns {Writer} `this`
 */
Writer.prototype.bool = function write_bool(value) {
    return this._push(writeByte, 1, value ? 1 : 0);
};

function writeFixed32(val, buf, pos) {
    buf[pos    ] =  val         & 255;
    buf[pos + 1] =  val >>> 8   & 255;
    buf[pos + 2] =  val >>> 16  & 255;
    buf[pos + 3] =  val >>> 24;
}

/**
 * Writes an unsigned 32 bit value as fixed 32 bits.
 * @param {number} value Value to write
 * @returns {Writer} `this`
 */
Writer.prototype.fixed32 = function write_fixed32(value) {
    return this._push(writeFixed32, 4, value >>> 0);
};

/**
 * Writes a signed 32 bit value as fixed 32 bits.
 * @function
 * @param {number} value Value to write
 * @returns {Writer} `this`
 */
Writer.prototype.sfixed32 = Writer.prototype.fixed32;

/**
 * Writes an unsigned 64 bit value as fixed 64 bits.
 * @param {Long|number|string} value Value to write
 * @returns {Writer} `this`
 * @throws {TypeError} If `value` is a string and no long library is present.
 */
Writer.prototype.fixed64 = function write_fixed64(value) {
    var bits = LongBits.from(value);
    return this._push(writeFixed32, 4, bits.lo)._push(writeFixed32, 4, bits.hi);
};

/**
 * Writes a signed 64 bit value as fixed 64 bits.
 * @function
 * @param {Long|number|string} value Value to write
 * @returns {Writer} `this`
 * @throws {TypeError} If `value` is a string and no long library is present.
 */
Writer.prototype.sfixed64 = Writer.prototype.fixed64;

/**
 * Writes a float (32 bit).
 * @function
 * @param {number} value Value to write
 * @returns {Writer} `this`
 */
Writer.prototype.float = function write_float(value) {
    return this._push(util.float.writeFloatLE, 4, value);
};

/**
 * Writes a double (64 bit float).
 * @function
 * @param {number} value Value to write
 * @returns {Writer} `this`
 */
Writer.prototype.double = function write_double(value) {
    return this._push(util.float.writeDoubleLE, 8, value);
};

var writeBytes = util.Array.prototype.set
    ? function writeBytes_set(val, buf, pos) {
        buf.set(val, pos); // also works for plain array values
    }
    /* istanbul ignore next */
    : function writeBytes_for(val, buf, pos) {
        for (var i = 0; i < val.length; ++i)
            buf[pos + i] = val[i];
    };

/**
 * Writes a sequence of bytes.
 * @param {Uint8Array|string} value Buffer or base64 encoded string to write
 * @returns {Writer} `this`
 */
Writer.prototype.bytes = function write_bytes(value) {
    var len = value.length >>> 0;
    if (!len)
        return this._push(writeByte, 1, 0);
    if (util.isString(value)) {
        var buf = Writer.alloc(len = base64.length(value));
        base64.decode(value, buf, 0);
        value = buf;
    }
    return this.uint32(len)._push(writeBytes, len, value);
};

/**
 * Writes a string.
 * @param {string} value Value to write
 * @returns {Writer} `this`
 */
Writer.prototype.string = function write_string(value) {
    var len = utf8.length(value);
    return len
        ? this.uint32(len)._push(utf8.write, len, value)
        : this._push(writeByte, 1, 0);
};

/**
 * Forks this writer's state by pushing it to a stack.
 * Calling {@link Writer#reset|reset} or {@link Writer#ldelim|ldelim} resets the writer to the previous state.
 * @returns {Writer} `this`
 */
Writer.prototype.fork = function fork() {
    this.states = new State(this);
    this.head = this.tail = new Op(noop, 0, 0);
    this.len = 0;
    return this;
};

/**
 * Resets this instance to the last state.
 * @returns {Writer} `this`
 */
Writer.prototype.reset = function reset() {
    if (this.states) {
        this.head   = this.states.head;
        this.tail   = this.states.tail;
        this.len    = this.states.len;
        this.states = this.states.next;
    } else {
        this.head = this.tail = new Op(noop, 0, 0);
        this.len  = 0;
    }
    return this;
};

/**
 * Resets to the last state and appends the fork state's current write length as a varint followed by its operations.
 * @returns {Writer} `this`
 */
Writer.prototype.ldelim = function ldelim() {
    var head = this.head,
        tail = this.tail,
        len  = this.len;
    this.reset().uint32(len);
    if (len) {
        this.tail.next = head.next; // skip noop
        this.tail = tail;
        this.len += len;
    }
    return this;
};

/**
 * Finishes the write operation.
 * @returns {Uint8Array} Finished buffer
 */
Writer.prototype.finish = function finish() {
    var head = this.head.next, // skip noop
        buf  = this.constructor.alloc(this.len),
        pos  = 0;
    while (head) {
        head.fn(head.val, buf, pos);
        pos += head.len;
        head = head.next;
    }
    // this.head = this.tail = null;
    return buf;
};

Writer._configure = function(BufferWriter_) {
    BufferWriter = BufferWriter_;
};


/***/ }),
/* 2 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";

module.exports = Reader;

var util      = __webpack_require__(0);

var BufferReader; // cyclic

var LongBits  = util.LongBits,
    utf8      = util.utf8;

/* istanbul ignore next */
function indexOutOfRange(reader, writeLength) {
    return RangeError("index out of range: " + reader.pos + " + " + (writeLength || 1) + " > " + reader.len);
}

/**
 * Constructs a new reader instance using the specified buffer.
 * @classdesc Wire format reader using `Uint8Array` if available, otherwise `Array`.
 * @constructor
 * @param {Uint8Array} buffer Buffer to read from
 */
function Reader(buffer) {

    /**
     * Read buffer.
     * @type {Uint8Array}
     */
    this.buf = buffer;

    /**
     * Read buffer position.
     * @type {number}
     */
    this.pos = 0;

    /**
     * Read buffer length.
     * @type {number}
     */
    this.len = buffer.length;
}

var create_array = typeof Uint8Array !== "undefined"
    ? function create_typed_array(buffer) {
        if (buffer instanceof Uint8Array || Array.isArray(buffer))
            return new Reader(buffer);
        throw Error("illegal buffer");
    }
    /* istanbul ignore next */
    : function create_array(buffer) {
        if (Array.isArray(buffer))
            return new Reader(buffer);
        throw Error("illegal buffer");
    };

/**
 * Creates a new reader using the specified buffer.
 * @function
 * @param {Uint8Array|Buffer} buffer Buffer to read from
 * @returns {Reader|BufferReader} A {@link BufferReader} if `buffer` is a Buffer, otherwise a {@link Reader}
 * @throws {Error} If `buffer` is not a valid buffer
 */
Reader.create = util.Buffer
    ? function create_buffer_setup(buffer) {
        return (Reader.create = function create_buffer(buffer) {
            return util.Buffer.isBuffer(buffer)
                ? new BufferReader(buffer)
                /* istanbul ignore next */
                : create_array(buffer);
        })(buffer);
    }
    /* istanbul ignore next */
    : create_array;

Reader.prototype._slice = util.Array.prototype.subarray || /* istanbul ignore next */ util.Array.prototype.slice;

/**
 * Reads a varint as an unsigned 32 bit value.
 * @function
 * @returns {number} Value read
 */
Reader.prototype.uint32 = (function read_uint32_setup() {
    var value = 4294967295; // optimizer type-hint, tends to deopt otherwise (?!)
    return function read_uint32() {
        value = (         this.buf[this.pos] & 127       ) >>> 0; if (this.buf[this.pos++] < 128) return value;
        value = (value | (this.buf[this.pos] & 127) <<  7) >>> 0; if (this.buf[this.pos++] < 128) return value;
        value = (value | (this.buf[this.pos] & 127) << 14) >>> 0; if (this.buf[this.pos++] < 128) return value;
        value = (value | (this.buf[this.pos] & 127) << 21) >>> 0; if (this.buf[this.pos++] < 128) return value;
        value = (value | (this.buf[this.pos] &  15) << 28) >>> 0; if (this.buf[this.pos++] < 128) return value;

        /* istanbul ignore if */
        if ((this.pos += 5) > this.len) {
            this.pos = this.len;
            throw indexOutOfRange(this, 10);
        }
        return value;
    };
})();

/**
 * Reads a varint as a signed 32 bit value.
 * @returns {number} Value read
 */
Reader.prototype.int32 = function read_int32() {
    return this.uint32() | 0;
};

/**
 * Reads a zig-zag encoded varint as a signed 32 bit value.
 * @returns {number} Value read
 */
Reader.prototype.sint32 = function read_sint32() {
    var value = this.uint32();
    return value >>> 1 ^ -(value & 1) | 0;
};

/* eslint-disable no-invalid-this */

function readLongVarint() {
    // tends to deopt with local vars for octet etc.
    var bits = new LongBits(0, 0);
    var i = 0;
    if (this.len - this.pos > 4) { // fast route (lo)
        for (; i < 4; ++i) {
            // 1st..4th
            bits.lo = (bits.lo | (this.buf[this.pos] & 127) << i * 7) >>> 0;
            if (this.buf[this.pos++] < 128)
                return bits;
        }
        // 5th
        bits.lo = (bits.lo | (this.buf[this.pos] & 127) << 28) >>> 0;
        bits.hi = (bits.hi | (this.buf[this.pos] & 127) >>  4) >>> 0;
        if (this.buf[this.pos++] < 128)
            return bits;
        i = 0;
    } else {
        for (; i < 3; ++i) {
            /* istanbul ignore if */
            if (this.pos >= this.len)
                throw indexOutOfRange(this);
            // 1st..3th
            bits.lo = (bits.lo | (this.buf[this.pos] & 127) << i * 7) >>> 0;
            if (this.buf[this.pos++] < 128)
                return bits;
        }
        // 4th
        bits.lo = (bits.lo | (this.buf[this.pos++] & 127) << i * 7) >>> 0;
        return bits;
    }
    if (this.len - this.pos > 4) { // fast route (hi)
        for (; i < 5; ++i) {
            // 6th..10th
            bits.hi = (bits.hi | (this.buf[this.pos] & 127) << i * 7 + 3) >>> 0;
            if (this.buf[this.pos++] < 128)
                return bits;
        }
    } else {
        for (; i < 5; ++i) {
            /* istanbul ignore if */
            if (this.pos >= this.len)
                throw indexOutOfRange(this);
            // 6th..10th
            bits.hi = (bits.hi | (this.buf[this.pos] & 127) << i * 7 + 3) >>> 0;
            if (this.buf[this.pos++] < 128)
                return bits;
        }
    }
    /* istanbul ignore next */
    throw Error("invalid varint encoding");
}

/* eslint-enable no-invalid-this */

/**
 * Reads a varint as a signed 64 bit value.
 * @name Reader#int64
 * @function
 * @returns {Long} Value read
 */

/**
 * Reads a varint as an unsigned 64 bit value.
 * @name Reader#uint64
 * @function
 * @returns {Long} Value read
 */

/**
 * Reads a zig-zag encoded varint as a signed 64 bit value.
 * @name Reader#sint64
 * @function
 * @returns {Long} Value read
 */

/**
 * Reads a varint as a boolean.
 * @returns {boolean} Value read
 */
Reader.prototype.bool = function read_bool() {
    return this.uint32() !== 0;
};

function readFixed32_end(buf, end) { // note that this uses `end`, not `pos`
    return (buf[end - 4]
          | buf[end - 3] << 8
          | buf[end - 2] << 16
          | buf[end - 1] << 24) >>> 0;
}

/**
 * Reads fixed 32 bits as an unsigned 32 bit integer.
 * @returns {number} Value read
 */
Reader.prototype.fixed32 = function read_fixed32() {

    /* istanbul ignore if */
    if (this.pos + 4 > this.len)
        throw indexOutOfRange(this, 4);

    return readFixed32_end(this.buf, this.pos += 4);
};

/**
 * Reads fixed 32 bits as a signed 32 bit integer.
 * @returns {number} Value read
 */
Reader.prototype.sfixed32 = function read_sfixed32() {

    /* istanbul ignore if */
    if (this.pos + 4 > this.len)
        throw indexOutOfRange(this, 4);

    return readFixed32_end(this.buf, this.pos += 4) | 0;
};

/* eslint-disable no-invalid-this */

function readFixed64(/* this: Reader */) {

    /* istanbul ignore if */
    if (this.pos + 8 > this.len)
        throw indexOutOfRange(this, 8);

    return new LongBits(readFixed32_end(this.buf, this.pos += 4), readFixed32_end(this.buf, this.pos += 4));
}

/* eslint-enable no-invalid-this */

/**
 * Reads fixed 64 bits.
 * @name Reader#fixed64
 * @function
 * @returns {Long} Value read
 */

/**
 * Reads zig-zag encoded fixed 64 bits.
 * @name Reader#sfixed64
 * @function
 * @returns {Long} Value read
 */

/**
 * Reads a float (32 bit) as a number.
 * @function
 * @returns {number} Value read
 */
Reader.prototype.float = function read_float() {

    /* istanbul ignore if */
    if (this.pos + 4 > this.len)
        throw indexOutOfRange(this, 4);

    var value = util.float.readFloatLE(this.buf, this.pos);
    this.pos += 4;
    return value;
};

/**
 * Reads a double (64 bit float) as a number.
 * @function
 * @returns {number} Value read
 */
Reader.prototype.double = function read_double() {

    /* istanbul ignore if */
    if (this.pos + 8 > this.len)
        throw indexOutOfRange(this, 4);

    var value = util.float.readDoubleLE(this.buf, this.pos);
    this.pos += 8;
    return value;
};

/**
 * Reads a sequence of bytes preceeded by its length as a varint.
 * @returns {Uint8Array} Value read
 */
Reader.prototype.bytes = function read_bytes() {
    var length = this.uint32(),
        start  = this.pos,
        end    = this.pos + length;

    /* istanbul ignore if */
    if (end > this.len)
        throw indexOutOfRange(this, length);

    this.pos += length;
    if (Array.isArray(this.buf)) // plain array
        return this.buf.slice(start, end);
    return start === end // fix for IE 10/Win8 and others' subarray returning array of size 1
        ? new this.buf.constructor(0)
        : this._slice.call(this.buf, start, end);
};

/**
 * Reads a string preceeded by its byte length as a varint.
 * @returns {string} Value read
 */
Reader.prototype.string = function read_string() {
    var bytes = this.bytes();
    return utf8.read(bytes, 0, bytes.length);
};

/**
 * Skips the specified number of bytes if specified, otherwise skips a varint.
 * @param {number} [length] Length if known, otherwise a varint is assumed
 * @returns {Reader} `this`
 */
Reader.prototype.skip = function skip(length) {
    if (typeof length === "number") {
        /* istanbul ignore if */
        if (this.pos + length > this.len)
            throw indexOutOfRange(this, length);
        this.pos += length;
    } else {
        do {
            /* istanbul ignore if */
            if (this.pos >= this.len)
                throw indexOutOfRange(this);
        } while (this.buf[this.pos++] & 128);
    }
    return this;
};

/**
 * Skips the next element of the specified wire type.
 * @param {number} wireType Wire type received
 * @returns {Reader} `this`
 */
Reader.prototype.skipType = function(wireType) {
    switch (wireType) {
        case 0:
            this.skip();
            break;
        case 1:
            this.skip(8);
            break;
        case 2:
            this.skip(this.uint32());
            break;
        case 3:
            do { // eslint-disable-line no-constant-condition
                if ((wireType = this.uint32() & 7) === 4)
                    break;
                this.skipType(wireType);
            } while (true);
            break;
        case 5:
            this.skip(4);
            break;

        /* istanbul ignore next */
        default:
            throw Error("invalid wire type " + wireType + " at offset " + this.pos);
    }
    return this;
};

Reader._configure = function(BufferReader_) {
    BufferReader = BufferReader_;

    var fn = util.Long ? "toLong" : /* istanbul ignore next */ "toNumber";
    util.merge(Reader.prototype, {

        int64: function read_int64() {
            return readLongVarint.call(this)[fn](false);
        },

        uint64: function read_uint64() {
            return readLongVarint.call(this)[fn](true);
        },

        sint64: function read_sint64() {
            return readLongVarint.call(this).zzDecode()[fn](false);
        },

        fixed64: function read_fixed64() {
            return readFixed64.call(this)[fn](true);
        },

        sfixed64: function read_sfixed64() {
            return readFixed64.call(this)[fn](false);
        }

    });
};


/***/ }),
/* 3 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
Object.defineProperty(__webpack_exports__, "__esModule", { value: true });
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_0__protobuf_bundle__ = __webpack_require__(4);

var ctx = self;
ctx.onmessage = function (message) {
    ctx.postMessage(__WEBPACK_IMPORTED_MODULE_0__protobuf_bundle__["a" /* cupr */].proto.KernelTrace.decode(new Uint8Array(message.data)).toJSON());
};
/* harmony default export */ __webpack_exports__["default"] = ({});


/***/ }),
/* 4 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
/* unused harmony export default */
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_0_protobufjs_minimal__ = __webpack_require__(5);
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_0_protobufjs_minimal___default = __webpack_require__.n(__WEBPACK_IMPORTED_MODULE_0_protobufjs_minimal__);
/*eslint-disable block-scoped-var, no-redeclare, no-control-regex, no-prototype-builtins*/


// Common aliases
const $Reader = __WEBPACK_IMPORTED_MODULE_0_protobufjs_minimal__["Reader"], $Writer = __WEBPACK_IMPORTED_MODULE_0_protobufjs_minimal__["Writer"], $util = __WEBPACK_IMPORTED_MODULE_0_protobufjs_minimal__["util"];

// Exported root namespace
const $root = __WEBPACK_IMPORTED_MODULE_0_protobufjs_minimal__["roots"]["default"] || (__WEBPACK_IMPORTED_MODULE_0_protobufjs_minimal__["roots"]["default"] = {});

const cupr = $root.cupr = (() => {

    /**
     * Namespace cupr.
     * @exports cupr
     * @namespace
     */
    const cupr = {};

    cupr.proto = (function() {

        /**
         * Namespace proto.
         * @memberof cupr
         * @namespace
         */
        const proto = {};

        proto.AllocRecord = (function() {

            /**
             * Properties of an AllocRecord.
             * @memberof cupr.proto
             * @interface IAllocRecord
             * @property {string} address AllocRecord address
             * @property {number} size AllocRecord size
             * @property {number} elementSize AllocRecord elementSize
             * @property {number} space AllocRecord space
             * @property {number} [typeIndex] AllocRecord typeIndex
             * @property {string} [typeString] AllocRecord typeString
             * @property {string} name AllocRecord name
             * @property {string} location AllocRecord location
             * @property {boolean} active AllocRecord active
             */

            /**
             * Constructs a new AllocRecord.
             * @memberof cupr.proto
             * @classdesc Represents an AllocRecord.
             * @constructor
             * @param {cupr.proto.IAllocRecord=} [properties] Properties to set
             */
            function AllocRecord(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * AllocRecord address.
             * @member {string}address
             * @memberof cupr.proto.AllocRecord
             * @instance
             */
            AllocRecord.prototype.address = "";

            /**
             * AllocRecord size.
             * @member {number}size
             * @memberof cupr.proto.AllocRecord
             * @instance
             */
            AllocRecord.prototype.size = 0;

            /**
             * AllocRecord elementSize.
             * @member {number}elementSize
             * @memberof cupr.proto.AllocRecord
             * @instance
             */
            AllocRecord.prototype.elementSize = 0;

            /**
             * AllocRecord space.
             * @member {number}space
             * @memberof cupr.proto.AllocRecord
             * @instance
             */
            AllocRecord.prototype.space = 0;

            /**
             * AllocRecord typeIndex.
             * @member {number}typeIndex
             * @memberof cupr.proto.AllocRecord
             * @instance
             */
            AllocRecord.prototype.typeIndex = 0;

            /**
             * AllocRecord typeString.
             * @member {string}typeString
             * @memberof cupr.proto.AllocRecord
             * @instance
             */
            AllocRecord.prototype.typeString = "";

            /**
             * AllocRecord name.
             * @member {string}name
             * @memberof cupr.proto.AllocRecord
             * @instance
             */
            AllocRecord.prototype.name = "";

            /**
             * AllocRecord location.
             * @member {string}location
             * @memberof cupr.proto.AllocRecord
             * @instance
             */
            AllocRecord.prototype.location = "";

            /**
             * AllocRecord active.
             * @member {boolean}active
             * @memberof cupr.proto.AllocRecord
             * @instance
             */
            AllocRecord.prototype.active = false;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * AllocRecord type.
             * @member {string|undefined} type
             * @memberof cupr.proto.AllocRecord
             * @instance
             */
            Object.defineProperty(AllocRecord.prototype, "type", {
                get: $util.oneOfGetter($oneOfFields = ["typeIndex", "typeString"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new AllocRecord instance using the specified properties.
             * @function create
             * @memberof cupr.proto.AllocRecord
             * @static
             * @param {cupr.proto.IAllocRecord=} [properties] Properties to set
             * @returns {cupr.proto.AllocRecord} AllocRecord instance
             */
            AllocRecord.create = function create(properties) {
                return new AllocRecord(properties);
            };

            /**
             * Encodes the specified AllocRecord message. Does not implicitly {@link cupr.proto.AllocRecord.verify|verify} messages.
             * @function encode
             * @memberof cupr.proto.AllocRecord
             * @static
             * @param {cupr.proto.IAllocRecord} message AllocRecord message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            AllocRecord.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                writer.uint32(/* id 1, wireType 2 =*/10).string(message.address);
                writer.uint32(/* id 2, wireType 0 =*/16).int32(message.size);
                writer.uint32(/* id 3, wireType 0 =*/24).int32(message.elementSize);
                writer.uint32(/* id 4, wireType 0 =*/32).int32(message.space);
                if (message.typeIndex != null && message.hasOwnProperty("typeIndex"))
                    writer.uint32(/* id 5, wireType 0 =*/40).int32(message.typeIndex);
                if (message.typeString != null && message.hasOwnProperty("typeString"))
                    writer.uint32(/* id 6, wireType 2 =*/50).string(message.typeString);
                writer.uint32(/* id 7, wireType 2 =*/58).string(message.name);
                writer.uint32(/* id 8, wireType 2 =*/66).string(message.location);
                writer.uint32(/* id 9, wireType 0 =*/72).bool(message.active);
                return writer;
            };

            /**
             * Encodes the specified AllocRecord message, length delimited. Does not implicitly {@link cupr.proto.AllocRecord.verify|verify} messages.
             * @function encodeDelimited
             * @memberof cupr.proto.AllocRecord
             * @static
             * @param {cupr.proto.IAllocRecord} message AllocRecord message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            AllocRecord.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes an AllocRecord message from the specified reader or buffer.
             * @function decode
             * @memberof cupr.proto.AllocRecord
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {cupr.proto.AllocRecord} AllocRecord
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            AllocRecord.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.cupr.proto.AllocRecord();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.address = reader.string();
                        break;
                    case 2:
                        message.size = reader.int32();
                        break;
                    case 3:
                        message.elementSize = reader.int32();
                        break;
                    case 4:
                        message.space = reader.int32();
                        break;
                    case 5:
                        message.typeIndex = reader.int32();
                        break;
                    case 6:
                        message.typeString = reader.string();
                        break;
                    case 7:
                        message.name = reader.string();
                        break;
                    case 8:
                        message.location = reader.string();
                        break;
                    case 9:
                        message.active = reader.bool();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                if (!message.hasOwnProperty("address"))
                    throw $util.ProtocolError("missing required 'address'", { instance: message });
                if (!message.hasOwnProperty("size"))
                    throw $util.ProtocolError("missing required 'size'", { instance: message });
                if (!message.hasOwnProperty("elementSize"))
                    throw $util.ProtocolError("missing required 'elementSize'", { instance: message });
                if (!message.hasOwnProperty("space"))
                    throw $util.ProtocolError("missing required 'space'", { instance: message });
                if (!message.hasOwnProperty("name"))
                    throw $util.ProtocolError("missing required 'name'", { instance: message });
                if (!message.hasOwnProperty("location"))
                    throw $util.ProtocolError("missing required 'location'", { instance: message });
                if (!message.hasOwnProperty("active"))
                    throw $util.ProtocolError("missing required 'active'", { instance: message });
                return message;
            };

            /**
             * Decodes an AllocRecord message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof cupr.proto.AllocRecord
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {cupr.proto.AllocRecord} AllocRecord
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            AllocRecord.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies an AllocRecord message.
             * @function verify
             * @memberof cupr.proto.AllocRecord
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            AllocRecord.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                let properties = {};
                if (!$util.isString(message.address))
                    return "address: string expected";
                if (!$util.isInteger(message.size))
                    return "size: integer expected";
                if (!$util.isInteger(message.elementSize))
                    return "elementSize: integer expected";
                if (!$util.isInteger(message.space))
                    return "space: integer expected";
                if (message.typeIndex != null && message.hasOwnProperty("typeIndex")) {
                    properties.type = 1;
                    if (!$util.isInteger(message.typeIndex))
                        return "typeIndex: integer expected";
                }
                if (message.typeString != null && message.hasOwnProperty("typeString")) {
                    if (properties.type === 1)
                        return "type: multiple values";
                    properties.type = 1;
                    if (!$util.isString(message.typeString))
                        return "typeString: string expected";
                }
                if (!$util.isString(message.name))
                    return "name: string expected";
                if (!$util.isString(message.location))
                    return "location: string expected";
                if (typeof message.active !== "boolean")
                    return "active: boolean expected";
                return null;
            };

            /**
             * Creates an AllocRecord message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof cupr.proto.AllocRecord
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {cupr.proto.AllocRecord} AllocRecord
             */
            AllocRecord.fromObject = function fromObject(object) {
                if (object instanceof $root.cupr.proto.AllocRecord)
                    return object;
                let message = new $root.cupr.proto.AllocRecord();
                if (object.address != null)
                    message.address = String(object.address);
                if (object.size != null)
                    message.size = object.size | 0;
                if (object.elementSize != null)
                    message.elementSize = object.elementSize | 0;
                if (object.space != null)
                    message.space = object.space | 0;
                if (object.typeIndex != null)
                    message.typeIndex = object.typeIndex | 0;
                if (object.typeString != null)
                    message.typeString = String(object.typeString);
                if (object.name != null)
                    message.name = String(object.name);
                if (object.location != null)
                    message.location = String(object.location);
                if (object.active != null)
                    message.active = Boolean(object.active);
                return message;
            };

            /**
             * Creates a plain object from an AllocRecord message. Also converts values to other types if specified.
             * @function toObject
             * @memberof cupr.proto.AllocRecord
             * @static
             * @param {cupr.proto.AllocRecord} message AllocRecord
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            AllocRecord.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.address = "";
                    object.size = 0;
                    object.elementSize = 0;
                    object.space = 0;
                    object.name = "";
                    object.location = "";
                    object.active = false;
                }
                if (message.address != null && message.hasOwnProperty("address"))
                    object.address = message.address;
                if (message.size != null && message.hasOwnProperty("size"))
                    object.size = message.size;
                if (message.elementSize != null && message.hasOwnProperty("elementSize"))
                    object.elementSize = message.elementSize;
                if (message.space != null && message.hasOwnProperty("space"))
                    object.space = message.space;
                if (message.typeIndex != null && message.hasOwnProperty("typeIndex")) {
                    object.typeIndex = message.typeIndex;
                    if (options.oneofs)
                        object.type = "typeIndex";
                }
                if (message.typeString != null && message.hasOwnProperty("typeString")) {
                    object.typeString = message.typeString;
                    if (options.oneofs)
                        object.type = "typeString";
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.location != null && message.hasOwnProperty("location"))
                    object.location = message.location;
                if (message.active != null && message.hasOwnProperty("active"))
                    object.active = message.active;
                return object;
            };

            /**
             * Converts this AllocRecord to JSON.
             * @function toJSON
             * @memberof cupr.proto.AllocRecord
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            AllocRecord.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, __WEBPACK_IMPORTED_MODULE_0_protobufjs_minimal__["util"].toJSONOptions);
            };

            return AllocRecord;
        })();

        proto.Dim3 = (function() {

            /**
             * Properties of a Dim3.
             * @memberof cupr.proto
             * @interface IDim3
             * @property {number} x Dim3 x
             * @property {number} y Dim3 y
             * @property {number} z Dim3 z
             */

            /**
             * Constructs a new Dim3.
             * @memberof cupr.proto
             * @classdesc Represents a Dim3.
             * @constructor
             * @param {cupr.proto.IDim3=} [properties] Properties to set
             */
            function Dim3(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Dim3 x.
             * @member {number}x
             * @memberof cupr.proto.Dim3
             * @instance
             */
            Dim3.prototype.x = 0;

            /**
             * Dim3 y.
             * @member {number}y
             * @memberof cupr.proto.Dim3
             * @instance
             */
            Dim3.prototype.y = 0;

            /**
             * Dim3 z.
             * @member {number}z
             * @memberof cupr.proto.Dim3
             * @instance
             */
            Dim3.prototype.z = 0;

            /**
             * Creates a new Dim3 instance using the specified properties.
             * @function create
             * @memberof cupr.proto.Dim3
             * @static
             * @param {cupr.proto.IDim3=} [properties] Properties to set
             * @returns {cupr.proto.Dim3} Dim3 instance
             */
            Dim3.create = function create(properties) {
                return new Dim3(properties);
            };

            /**
             * Encodes the specified Dim3 message. Does not implicitly {@link cupr.proto.Dim3.verify|verify} messages.
             * @function encode
             * @memberof cupr.proto.Dim3
             * @static
             * @param {cupr.proto.IDim3} message Dim3 message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Dim3.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                writer.uint32(/* id 1, wireType 0 =*/8).int32(message.x);
                writer.uint32(/* id 2, wireType 0 =*/16).int32(message.y);
                writer.uint32(/* id 3, wireType 0 =*/24).int32(message.z);
                return writer;
            };

            /**
             * Encodes the specified Dim3 message, length delimited. Does not implicitly {@link cupr.proto.Dim3.verify|verify} messages.
             * @function encodeDelimited
             * @memberof cupr.proto.Dim3
             * @static
             * @param {cupr.proto.IDim3} message Dim3 message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Dim3.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a Dim3 message from the specified reader or buffer.
             * @function decode
             * @memberof cupr.proto.Dim3
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {cupr.proto.Dim3} Dim3
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Dim3.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.cupr.proto.Dim3();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.x = reader.int32();
                        break;
                    case 2:
                        message.y = reader.int32();
                        break;
                    case 3:
                        message.z = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                if (!message.hasOwnProperty("x"))
                    throw $util.ProtocolError("missing required 'x'", { instance: message });
                if (!message.hasOwnProperty("y"))
                    throw $util.ProtocolError("missing required 'y'", { instance: message });
                if (!message.hasOwnProperty("z"))
                    throw $util.ProtocolError("missing required 'z'", { instance: message });
                return message;
            };

            /**
             * Decodes a Dim3 message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof cupr.proto.Dim3
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {cupr.proto.Dim3} Dim3
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Dim3.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a Dim3 message.
             * @function verify
             * @memberof cupr.proto.Dim3
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            Dim3.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (!$util.isInteger(message.x))
                    return "x: integer expected";
                if (!$util.isInteger(message.y))
                    return "y: integer expected";
                if (!$util.isInteger(message.z))
                    return "z: integer expected";
                return null;
            };

            /**
             * Creates a Dim3 message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof cupr.proto.Dim3
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {cupr.proto.Dim3} Dim3
             */
            Dim3.fromObject = function fromObject(object) {
                if (object instanceof $root.cupr.proto.Dim3)
                    return object;
                let message = new $root.cupr.proto.Dim3();
                if (object.x != null)
                    message.x = object.x | 0;
                if (object.y != null)
                    message.y = object.y | 0;
                if (object.z != null)
                    message.z = object.z | 0;
                return message;
            };

            /**
             * Creates a plain object from a Dim3 message. Also converts values to other types if specified.
             * @function toObject
             * @memberof cupr.proto.Dim3
             * @static
             * @param {cupr.proto.Dim3} message Dim3
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            Dim3.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.x = 0;
                    object.y = 0;
                    object.z = 0;
                }
                if (message.x != null && message.hasOwnProperty("x"))
                    object.x = message.x;
                if (message.y != null && message.hasOwnProperty("y"))
                    object.y = message.y;
                if (message.z != null && message.hasOwnProperty("z"))
                    object.z = message.z;
                return object;
            };

            /**
             * Converts this Dim3 to JSON.
             * @function toJSON
             * @memberof cupr.proto.Dim3
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            Dim3.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, __WEBPACK_IMPORTED_MODULE_0_protobufjs_minimal__["util"].toJSONOptions);
            };

            return Dim3;
        })();

        proto.KernelTrace = (function() {

            /**
             * Properties of a KernelTrace.
             * @memberof cupr.proto
             * @interface IKernelTrace
             * @property {Array.<cupr.proto.IMemoryAccess>} [accesses] KernelTrace accesses
             * @property {Array.<cupr.proto.IAllocRecord>} [allocations] KernelTrace allocations
             * @property {string} kernel KernelTrace kernel
             * @property {number} start KernelTrace start
             * @property {number} end KernelTrace end
             * @property {string} type KernelTrace type
             * @property {cupr.proto.IDim3} gridDim KernelTrace gridDim
             * @property {cupr.proto.IDim3} blockDim KernelTrace blockDim
             * @property {number} warpSize KernelTrace warpSize
             */

            /**
             * Constructs a new KernelTrace.
             * @memberof cupr.proto
             * @classdesc Represents a KernelTrace.
             * @constructor
             * @param {cupr.proto.IKernelTrace=} [properties] Properties to set
             */
            function KernelTrace(properties) {
                this.accesses = [];
                this.allocations = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * KernelTrace accesses.
             * @member {Array.<cupr.proto.IMemoryAccess>}accesses
             * @memberof cupr.proto.KernelTrace
             * @instance
             */
            KernelTrace.prototype.accesses = $util.emptyArray;

            /**
             * KernelTrace allocations.
             * @member {Array.<cupr.proto.IAllocRecord>}allocations
             * @memberof cupr.proto.KernelTrace
             * @instance
             */
            KernelTrace.prototype.allocations = $util.emptyArray;

            /**
             * KernelTrace kernel.
             * @member {string}kernel
             * @memberof cupr.proto.KernelTrace
             * @instance
             */
            KernelTrace.prototype.kernel = "";

            /**
             * KernelTrace start.
             * @member {number}start
             * @memberof cupr.proto.KernelTrace
             * @instance
             */
            KernelTrace.prototype.start = 0;

            /**
             * KernelTrace end.
             * @member {number}end
             * @memberof cupr.proto.KernelTrace
             * @instance
             */
            KernelTrace.prototype.end = 0;

            /**
             * KernelTrace type.
             * @member {string}type
             * @memberof cupr.proto.KernelTrace
             * @instance
             */
            KernelTrace.prototype.type = "";

            /**
             * KernelTrace gridDim.
             * @member {cupr.proto.IDim3}gridDim
             * @memberof cupr.proto.KernelTrace
             * @instance
             */
            KernelTrace.prototype.gridDim = null;

            /**
             * KernelTrace blockDim.
             * @member {cupr.proto.IDim3}blockDim
             * @memberof cupr.proto.KernelTrace
             * @instance
             */
            KernelTrace.prototype.blockDim = null;

            /**
             * KernelTrace warpSize.
             * @member {number}warpSize
             * @memberof cupr.proto.KernelTrace
             * @instance
             */
            KernelTrace.prototype.warpSize = 0;

            /**
             * Creates a new KernelTrace instance using the specified properties.
             * @function create
             * @memberof cupr.proto.KernelTrace
             * @static
             * @param {cupr.proto.IKernelTrace=} [properties] Properties to set
             * @returns {cupr.proto.KernelTrace} KernelTrace instance
             */
            KernelTrace.create = function create(properties) {
                return new KernelTrace(properties);
            };

            /**
             * Encodes the specified KernelTrace message. Does not implicitly {@link cupr.proto.KernelTrace.verify|verify} messages.
             * @function encode
             * @memberof cupr.proto.KernelTrace
             * @static
             * @param {cupr.proto.IKernelTrace} message KernelTrace message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            KernelTrace.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.accesses != null && message.accesses.length)
                    for (let i = 0; i < message.accesses.length; ++i)
                        $root.cupr.proto.MemoryAccess.encode(message.accesses[i], writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
                if (message.allocations != null && message.allocations.length)
                    for (let i = 0; i < message.allocations.length; ++i)
                        $root.cupr.proto.AllocRecord.encode(message.allocations[i], writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
                writer.uint32(/* id 3, wireType 2 =*/26).string(message.kernel);
                writer.uint32(/* id 4, wireType 1 =*/33).double(message.start);
                writer.uint32(/* id 5, wireType 1 =*/41).double(message.end);
                writer.uint32(/* id 6, wireType 2 =*/50).string(message.type);
                $root.cupr.proto.Dim3.encode(message.gridDim, writer.uint32(/* id 7, wireType 2 =*/58).fork()).ldelim();
                $root.cupr.proto.Dim3.encode(message.blockDim, writer.uint32(/* id 8, wireType 2 =*/66).fork()).ldelim();
                writer.uint32(/* id 9, wireType 0 =*/72).int32(message.warpSize);
                return writer;
            };

            /**
             * Encodes the specified KernelTrace message, length delimited. Does not implicitly {@link cupr.proto.KernelTrace.verify|verify} messages.
             * @function encodeDelimited
             * @memberof cupr.proto.KernelTrace
             * @static
             * @param {cupr.proto.IKernelTrace} message KernelTrace message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            KernelTrace.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a KernelTrace message from the specified reader or buffer.
             * @function decode
             * @memberof cupr.proto.KernelTrace
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {cupr.proto.KernelTrace} KernelTrace
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            KernelTrace.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.cupr.proto.KernelTrace();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.accesses && message.accesses.length))
                            message.accesses = [];
                        message.accesses.push($root.cupr.proto.MemoryAccess.decode(reader, reader.uint32()));
                        break;
                    case 2:
                        if (!(message.allocations && message.allocations.length))
                            message.allocations = [];
                        message.allocations.push($root.cupr.proto.AllocRecord.decode(reader, reader.uint32()));
                        break;
                    case 3:
                        message.kernel = reader.string();
                        break;
                    case 4:
                        message.start = reader.double();
                        break;
                    case 5:
                        message.end = reader.double();
                        break;
                    case 6:
                        message.type = reader.string();
                        break;
                    case 7:
                        message.gridDim = $root.cupr.proto.Dim3.decode(reader, reader.uint32());
                        break;
                    case 8:
                        message.blockDim = $root.cupr.proto.Dim3.decode(reader, reader.uint32());
                        break;
                    case 9:
                        message.warpSize = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                if (!message.hasOwnProperty("kernel"))
                    throw $util.ProtocolError("missing required 'kernel'", { instance: message });
                if (!message.hasOwnProperty("start"))
                    throw $util.ProtocolError("missing required 'start'", { instance: message });
                if (!message.hasOwnProperty("end"))
                    throw $util.ProtocolError("missing required 'end'", { instance: message });
                if (!message.hasOwnProperty("type"))
                    throw $util.ProtocolError("missing required 'type'", { instance: message });
                if (!message.hasOwnProperty("gridDim"))
                    throw $util.ProtocolError("missing required 'gridDim'", { instance: message });
                if (!message.hasOwnProperty("blockDim"))
                    throw $util.ProtocolError("missing required 'blockDim'", { instance: message });
                if (!message.hasOwnProperty("warpSize"))
                    throw $util.ProtocolError("missing required 'warpSize'", { instance: message });
                return message;
            };

            /**
             * Decodes a KernelTrace message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof cupr.proto.KernelTrace
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {cupr.proto.KernelTrace} KernelTrace
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            KernelTrace.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a KernelTrace message.
             * @function verify
             * @memberof cupr.proto.KernelTrace
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            KernelTrace.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.accesses != null && message.hasOwnProperty("accesses")) {
                    if (!Array.isArray(message.accesses))
                        return "accesses: array expected";
                    for (let i = 0; i < message.accesses.length; ++i) {
                        let error = $root.cupr.proto.MemoryAccess.verify(message.accesses[i]);
                        if (error)
                            return "accesses." + error;
                    }
                }
                if (message.allocations != null && message.hasOwnProperty("allocations")) {
                    if (!Array.isArray(message.allocations))
                        return "allocations: array expected";
                    for (let i = 0; i < message.allocations.length; ++i) {
                        error = $root.cupr.proto.AllocRecord.verify(message.allocations[i]);
                        if (error)
                            return "allocations." + error;
                    }
                }
                if (!$util.isString(message.kernel))
                    return "kernel: string expected";
                if (typeof message.start !== "number")
                    return "start: number expected";
                if (typeof message.end !== "number")
                    return "end: number expected";
                if (!$util.isString(message.type))
                    return "type: string expected";
                error = $root.cupr.proto.Dim3.verify(message.gridDim);
                if (error)
                    return "gridDim." + error;
                error = $root.cupr.proto.Dim3.verify(message.blockDim);
                if (error)
                    return "blockDim." + error;
                if (!$util.isInteger(message.warpSize))
                    return "warpSize: integer expected";
                return null;
            };

            /**
             * Creates a KernelTrace message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof cupr.proto.KernelTrace
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {cupr.proto.KernelTrace} KernelTrace
             */
            KernelTrace.fromObject = function fromObject(object) {
                if (object instanceof $root.cupr.proto.KernelTrace)
                    return object;
                let message = new $root.cupr.proto.KernelTrace();
                if (object.accesses) {
                    if (!Array.isArray(object.accesses))
                        throw TypeError(".cupr.proto.KernelTrace.accesses: array expected");
                    message.accesses = [];
                    for (let i = 0; i < object.accesses.length; ++i) {
                        if (typeof object.accesses[i] !== "object")
                            throw TypeError(".cupr.proto.KernelTrace.accesses: object expected");
                        message.accesses[i] = $root.cupr.proto.MemoryAccess.fromObject(object.accesses[i]);
                    }
                }
                if (object.allocations) {
                    if (!Array.isArray(object.allocations))
                        throw TypeError(".cupr.proto.KernelTrace.allocations: array expected");
                    message.allocations = [];
                    for (let i = 0; i < object.allocations.length; ++i) {
                        if (typeof object.allocations[i] !== "object")
                            throw TypeError(".cupr.proto.KernelTrace.allocations: object expected");
                        message.allocations[i] = $root.cupr.proto.AllocRecord.fromObject(object.allocations[i]);
                    }
                }
                if (object.kernel != null)
                    message.kernel = String(object.kernel);
                if (object.start != null)
                    message.start = Number(object.start);
                if (object.end != null)
                    message.end = Number(object.end);
                if (object.type != null)
                    message.type = String(object.type);
                if (object.gridDim != null) {
                    if (typeof object.gridDim !== "object")
                        throw TypeError(".cupr.proto.KernelTrace.gridDim: object expected");
                    message.gridDim = $root.cupr.proto.Dim3.fromObject(object.gridDim);
                }
                if (object.blockDim != null) {
                    if (typeof object.blockDim !== "object")
                        throw TypeError(".cupr.proto.KernelTrace.blockDim: object expected");
                    message.blockDim = $root.cupr.proto.Dim3.fromObject(object.blockDim);
                }
                if (object.warpSize != null)
                    message.warpSize = object.warpSize | 0;
                return message;
            };

            /**
             * Creates a plain object from a KernelTrace message. Also converts values to other types if specified.
             * @function toObject
             * @memberof cupr.proto.KernelTrace
             * @static
             * @param {cupr.proto.KernelTrace} message KernelTrace
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            KernelTrace.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.arrays || options.defaults) {
                    object.accesses = [];
                    object.allocations = [];
                }
                if (options.defaults) {
                    object.kernel = "";
                    object.start = 0;
                    object.end = 0;
                    object.type = "";
                    object.gridDim = null;
                    object.blockDim = null;
                    object.warpSize = 0;
                }
                if (message.accesses && message.accesses.length) {
                    object.accesses = [];
                    for (let j = 0; j < message.accesses.length; ++j)
                        object.accesses[j] = $root.cupr.proto.MemoryAccess.toObject(message.accesses[j], options);
                }
                if (message.allocations && message.allocations.length) {
                    object.allocations = [];
                    for (let j = 0; j < message.allocations.length; ++j)
                        object.allocations[j] = $root.cupr.proto.AllocRecord.toObject(message.allocations[j], options);
                }
                if (message.kernel != null && message.hasOwnProperty("kernel"))
                    object.kernel = message.kernel;
                if (message.start != null && message.hasOwnProperty("start"))
                    object.start = options.json && !isFinite(message.start) ? String(message.start) : message.start;
                if (message.end != null && message.hasOwnProperty("end"))
                    object.end = options.json && !isFinite(message.end) ? String(message.end) : message.end;
                if (message.type != null && message.hasOwnProperty("type"))
                    object.type = message.type;
                if (message.gridDim != null && message.hasOwnProperty("gridDim"))
                    object.gridDim = $root.cupr.proto.Dim3.toObject(message.gridDim, options);
                if (message.blockDim != null && message.hasOwnProperty("blockDim"))
                    object.blockDim = $root.cupr.proto.Dim3.toObject(message.blockDim, options);
                if (message.warpSize != null && message.hasOwnProperty("warpSize"))
                    object.warpSize = message.warpSize;
                return object;
            };

            /**
             * Converts this KernelTrace to JSON.
             * @function toJSON
             * @memberof cupr.proto.KernelTrace
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            KernelTrace.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, __WEBPACK_IMPORTED_MODULE_0_protobufjs_minimal__["util"].toJSONOptions);
            };

            return KernelTrace;
        })();

        proto.MemoryAccess = (function() {

            /**
             * Properties of a MemoryAccess.
             * @memberof cupr.proto
             * @interface IMemoryAccess
             * @property {cupr.proto.IDim3} threadIdx MemoryAccess threadIdx
             * @property {cupr.proto.IDim3} blockIdx MemoryAccess blockIdx
             * @property {number} warpId MemoryAccess warpId
             * @property {number} debugId MemoryAccess debugId
             * @property {string} address MemoryAccess address
             * @property {number} size MemoryAccess size
             * @property {number} kind MemoryAccess kind
             * @property {number} space MemoryAccess space
             * @property {number} typeIndex MemoryAccess typeIndex
             * @property {number|Long} timestamp MemoryAccess timestamp
             */

            /**
             * Constructs a new MemoryAccess.
             * @memberof cupr.proto
             * @classdesc Represents a MemoryAccess.
             * @constructor
             * @param {cupr.proto.IMemoryAccess=} [properties] Properties to set
             */
            function MemoryAccess(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * MemoryAccess threadIdx.
             * @member {cupr.proto.IDim3}threadIdx
             * @memberof cupr.proto.MemoryAccess
             * @instance
             */
            MemoryAccess.prototype.threadIdx = null;

            /**
             * MemoryAccess blockIdx.
             * @member {cupr.proto.IDim3}blockIdx
             * @memberof cupr.proto.MemoryAccess
             * @instance
             */
            MemoryAccess.prototype.blockIdx = null;

            /**
             * MemoryAccess warpId.
             * @member {number}warpId
             * @memberof cupr.proto.MemoryAccess
             * @instance
             */
            MemoryAccess.prototype.warpId = 0;

            /**
             * MemoryAccess debugId.
             * @member {number}debugId
             * @memberof cupr.proto.MemoryAccess
             * @instance
             */
            MemoryAccess.prototype.debugId = 0;

            /**
             * MemoryAccess address.
             * @member {string}address
             * @memberof cupr.proto.MemoryAccess
             * @instance
             */
            MemoryAccess.prototype.address = "";

            /**
             * MemoryAccess size.
             * @member {number}size
             * @memberof cupr.proto.MemoryAccess
             * @instance
             */
            MemoryAccess.prototype.size = 0;

            /**
             * MemoryAccess kind.
             * @member {number}kind
             * @memberof cupr.proto.MemoryAccess
             * @instance
             */
            MemoryAccess.prototype.kind = 0;

            /**
             * MemoryAccess space.
             * @member {number}space
             * @memberof cupr.proto.MemoryAccess
             * @instance
             */
            MemoryAccess.prototype.space = 0;

            /**
             * MemoryAccess typeIndex.
             * @member {number}typeIndex
             * @memberof cupr.proto.MemoryAccess
             * @instance
             */
            MemoryAccess.prototype.typeIndex = 0;

            /**
             * MemoryAccess timestamp.
             * @member {number|Long}timestamp
             * @memberof cupr.proto.MemoryAccess
             * @instance
             */
            MemoryAccess.prototype.timestamp = $util.Long ? $util.Long.fromBits(0,0,false) : 0;

            /**
             * Creates a new MemoryAccess instance using the specified properties.
             * @function create
             * @memberof cupr.proto.MemoryAccess
             * @static
             * @param {cupr.proto.IMemoryAccess=} [properties] Properties to set
             * @returns {cupr.proto.MemoryAccess} MemoryAccess instance
             */
            MemoryAccess.create = function create(properties) {
                return new MemoryAccess(properties);
            };

            /**
             * Encodes the specified MemoryAccess message. Does not implicitly {@link cupr.proto.MemoryAccess.verify|verify} messages.
             * @function encode
             * @memberof cupr.proto.MemoryAccess
             * @static
             * @param {cupr.proto.IMemoryAccess} message MemoryAccess message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            MemoryAccess.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                $root.cupr.proto.Dim3.encode(message.threadIdx, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
                $root.cupr.proto.Dim3.encode(message.blockIdx, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
                writer.uint32(/* id 3, wireType 0 =*/24).int32(message.warpId);
                writer.uint32(/* id 4, wireType 0 =*/32).int32(message.debugId);
                writer.uint32(/* id 5, wireType 2 =*/42).string(message.address);
                writer.uint32(/* id 6, wireType 0 =*/48).int32(message.size);
                writer.uint32(/* id 7, wireType 0 =*/56).int32(message.kind);
                writer.uint32(/* id 8, wireType 0 =*/64).int32(message.space);
                writer.uint32(/* id 9, wireType 0 =*/72).int32(message.typeIndex);
                writer.uint32(/* id 10, wireType 0 =*/80).int64(message.timestamp);
                return writer;
            };

            /**
             * Encodes the specified MemoryAccess message, length delimited. Does not implicitly {@link cupr.proto.MemoryAccess.verify|verify} messages.
             * @function encodeDelimited
             * @memberof cupr.proto.MemoryAccess
             * @static
             * @param {cupr.proto.IMemoryAccess} message MemoryAccess message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            MemoryAccess.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a MemoryAccess message from the specified reader or buffer.
             * @function decode
             * @memberof cupr.proto.MemoryAccess
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {cupr.proto.MemoryAccess} MemoryAccess
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            MemoryAccess.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.cupr.proto.MemoryAccess();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.threadIdx = $root.cupr.proto.Dim3.decode(reader, reader.uint32());
                        break;
                    case 2:
                        message.blockIdx = $root.cupr.proto.Dim3.decode(reader, reader.uint32());
                        break;
                    case 3:
                        message.warpId = reader.int32();
                        break;
                    case 4:
                        message.debugId = reader.int32();
                        break;
                    case 5:
                        message.address = reader.string();
                        break;
                    case 6:
                        message.size = reader.int32();
                        break;
                    case 7:
                        message.kind = reader.int32();
                        break;
                    case 8:
                        message.space = reader.int32();
                        break;
                    case 9:
                        message.typeIndex = reader.int32();
                        break;
                    case 10:
                        message.timestamp = reader.int64();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                if (!message.hasOwnProperty("threadIdx"))
                    throw $util.ProtocolError("missing required 'threadIdx'", { instance: message });
                if (!message.hasOwnProperty("blockIdx"))
                    throw $util.ProtocolError("missing required 'blockIdx'", { instance: message });
                if (!message.hasOwnProperty("warpId"))
                    throw $util.ProtocolError("missing required 'warpId'", { instance: message });
                if (!message.hasOwnProperty("debugId"))
                    throw $util.ProtocolError("missing required 'debugId'", { instance: message });
                if (!message.hasOwnProperty("address"))
                    throw $util.ProtocolError("missing required 'address'", { instance: message });
                if (!message.hasOwnProperty("size"))
                    throw $util.ProtocolError("missing required 'size'", { instance: message });
                if (!message.hasOwnProperty("kind"))
                    throw $util.ProtocolError("missing required 'kind'", { instance: message });
                if (!message.hasOwnProperty("space"))
                    throw $util.ProtocolError("missing required 'space'", { instance: message });
                if (!message.hasOwnProperty("typeIndex"))
                    throw $util.ProtocolError("missing required 'typeIndex'", { instance: message });
                if (!message.hasOwnProperty("timestamp"))
                    throw $util.ProtocolError("missing required 'timestamp'", { instance: message });
                return message;
            };

            /**
             * Decodes a MemoryAccess message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof cupr.proto.MemoryAccess
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {cupr.proto.MemoryAccess} MemoryAccess
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            MemoryAccess.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a MemoryAccess message.
             * @function verify
             * @memberof cupr.proto.MemoryAccess
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            MemoryAccess.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                let error = $root.cupr.proto.Dim3.verify(message.threadIdx);
                if (error)
                    return "threadIdx." + error;
                error = $root.cupr.proto.Dim3.verify(message.blockIdx);
                if (error)
                    return "blockIdx." + error;
                if (!$util.isInteger(message.warpId))
                    return "warpId: integer expected";
                if (!$util.isInteger(message.debugId))
                    return "debugId: integer expected";
                if (!$util.isString(message.address))
                    return "address: string expected";
                if (!$util.isInteger(message.size))
                    return "size: integer expected";
                if (!$util.isInteger(message.kind))
                    return "kind: integer expected";
                if (!$util.isInteger(message.space))
                    return "space: integer expected";
                if (!$util.isInteger(message.typeIndex))
                    return "typeIndex: integer expected";
                if (!$util.isInteger(message.timestamp) && !(message.timestamp && $util.isInteger(message.timestamp.low) && $util.isInteger(message.timestamp.high)))
                    return "timestamp: integer|Long expected";
                return null;
            };

            /**
             * Creates a MemoryAccess message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof cupr.proto.MemoryAccess
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {cupr.proto.MemoryAccess} MemoryAccess
             */
            MemoryAccess.fromObject = function fromObject(object) {
                if (object instanceof $root.cupr.proto.MemoryAccess)
                    return object;
                let message = new $root.cupr.proto.MemoryAccess();
                if (object.threadIdx != null) {
                    if (typeof object.threadIdx !== "object")
                        throw TypeError(".cupr.proto.MemoryAccess.threadIdx: object expected");
                    message.threadIdx = $root.cupr.proto.Dim3.fromObject(object.threadIdx);
                }
                if (object.blockIdx != null) {
                    if (typeof object.blockIdx !== "object")
                        throw TypeError(".cupr.proto.MemoryAccess.blockIdx: object expected");
                    message.blockIdx = $root.cupr.proto.Dim3.fromObject(object.blockIdx);
                }
                if (object.warpId != null)
                    message.warpId = object.warpId | 0;
                if (object.debugId != null)
                    message.debugId = object.debugId | 0;
                if (object.address != null)
                    message.address = String(object.address);
                if (object.size != null)
                    message.size = object.size | 0;
                if (object.kind != null)
                    message.kind = object.kind | 0;
                if (object.space != null)
                    message.space = object.space | 0;
                if (object.typeIndex != null)
                    message.typeIndex = object.typeIndex | 0;
                if (object.timestamp != null)
                    if ($util.Long)
                        (message.timestamp = $util.Long.fromValue(object.timestamp)).unsigned = false;
                    else if (typeof object.timestamp === "string")
                        message.timestamp = parseInt(object.timestamp, 10);
                    else if (typeof object.timestamp === "number")
                        message.timestamp = object.timestamp;
                    else if (typeof object.timestamp === "object")
                        message.timestamp = new $util.LongBits(object.timestamp.low >>> 0, object.timestamp.high >>> 0).toNumber();
                return message;
            };

            /**
             * Creates a plain object from a MemoryAccess message. Also converts values to other types if specified.
             * @function toObject
             * @memberof cupr.proto.MemoryAccess
             * @static
             * @param {cupr.proto.MemoryAccess} message MemoryAccess
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            MemoryAccess.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.threadIdx = null;
                    object.blockIdx = null;
                    object.warpId = 0;
                    object.debugId = 0;
                    object.address = "";
                    object.size = 0;
                    object.kind = 0;
                    object.space = 0;
                    object.typeIndex = 0;
                    if ($util.Long) {
                        let long = new $util.Long(0, 0, false);
                        object.timestamp = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.timestamp = options.longs === String ? "0" : 0;
                }
                if (message.threadIdx != null && message.hasOwnProperty("threadIdx"))
                    object.threadIdx = $root.cupr.proto.Dim3.toObject(message.threadIdx, options);
                if (message.blockIdx != null && message.hasOwnProperty("blockIdx"))
                    object.blockIdx = $root.cupr.proto.Dim3.toObject(message.blockIdx, options);
                if (message.warpId != null && message.hasOwnProperty("warpId"))
                    object.warpId = message.warpId;
                if (message.debugId != null && message.hasOwnProperty("debugId"))
                    object.debugId = message.debugId;
                if (message.address != null && message.hasOwnProperty("address"))
                    object.address = message.address;
                if (message.size != null && message.hasOwnProperty("size"))
                    object.size = message.size;
                if (message.kind != null && message.hasOwnProperty("kind"))
                    object.kind = message.kind;
                if (message.space != null && message.hasOwnProperty("space"))
                    object.space = message.space;
                if (message.typeIndex != null && message.hasOwnProperty("typeIndex"))
                    object.typeIndex = message.typeIndex;
                if (message.timestamp != null && message.hasOwnProperty("timestamp"))
                    if (typeof message.timestamp === "number")
                        object.timestamp = options.longs === String ? String(message.timestamp) : message.timestamp;
                    else
                        object.timestamp = options.longs === String ? $util.Long.prototype.toString.call(message.timestamp) : options.longs === Number ? new $util.LongBits(message.timestamp.low >>> 0, message.timestamp.high >>> 0).toNumber() : message.timestamp;
                return object;
            };

            /**
             * Converts this MemoryAccess to JSON.
             * @function toJSON
             * @memberof cupr.proto.MemoryAccess
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            MemoryAccess.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, __WEBPACK_IMPORTED_MODULE_0_protobufjs_minimal__["util"].toJSONOptions);
            };

            return MemoryAccess;
        })();

        return proto;
    })();

    return cupr;
})();
/* harmony export (immutable) */ __webpack_exports__["a"] = cupr;





/***/ }),
/* 5 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";
// minimal library entry point.


module.exports = __webpack_require__(6);


/***/ }),
/* 6 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";

var protobuf = exports;

/**
 * Build type, one of `"full"`, `"light"` or `"minimal"`.
 * @name build
 * @type {string}
 * @const
 */
protobuf.build = "minimal";

// Serialization
protobuf.Writer       = __webpack_require__(1);
protobuf.BufferWriter = __webpack_require__(16);
protobuf.Reader       = __webpack_require__(2);
protobuf.BufferReader = __webpack_require__(17);

// Utility
protobuf.util         = __webpack_require__(0);
protobuf.rpc          = __webpack_require__(18);
protobuf.roots        = __webpack_require__(20);
protobuf.configure    = configure;

/* istanbul ignore next */
/**
 * Reconfigures the library according to the environment.
 * @returns {undefined}
 */
function configure() {
    protobuf.Reader._configure(protobuf.BufferReader);
    protobuf.util._configure();
}

// Configure serialization
protobuf.Writer._configure(protobuf.BufferWriter);
configure();


/***/ }),
/* 7 */
/***/ (function(module, exports) {

var g;

// This works in non-strict mode
g = (function() {
	return this;
})();

try {
	// This works if eval is allowed (see CSP)
	g = g || Function("return this")() || (1,eval)("this");
} catch(e) {
	// This works if the window reference is available
	if(typeof window === "object")
		g = window;
}

// g can still be undefined, but nothing to do about it...
// We return undefined, instead of nothing here, so it's
// easier to handle this case. if(!global) { ...}

module.exports = g;


/***/ }),
/* 8 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";

module.exports = asPromise;

/**
 * Callback as used by {@link util.asPromise}.
 * @typedef asPromiseCallback
 * @type {function}
 * @param {Error|null} error Error, if any
 * @param {...*} params Additional arguments
 * @returns {undefined}
 */

/**
 * Returns a promise from a node-style callback function.
 * @memberof util
 * @param {asPromiseCallback} fn Function to call
 * @param {*} ctx Function context
 * @param {...*} params Function arguments
 * @returns {Promise<*>} Promisified function
 */
function asPromise(fn, ctx/*, varargs */) {
    var params  = new Array(arguments.length - 1),
        offset  = 0,
        index   = 2,
        pending = true;
    while (index < arguments.length)
        params[offset++] = arguments[index++];
    return new Promise(function executor(resolve, reject) {
        params[offset] = function callback(err/*, varargs */) {
            if (pending) {
                pending = false;
                if (err)
                    reject(err);
                else {
                    var params = new Array(arguments.length - 1),
                        offset = 0;
                    while (offset < params.length)
                        params[offset++] = arguments[offset];
                    resolve.apply(null, params);
                }
            }
        };
        try {
            fn.apply(ctx || null, params);
        } catch (err) {
            if (pending) {
                pending = false;
                reject(err);
            }
        }
    });
}


/***/ }),
/* 9 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


/**
 * A minimal base64 implementation for number arrays.
 * @memberof util
 * @namespace
 */
var base64 = exports;

/**
 * Calculates the byte length of a base64 encoded string.
 * @param {string} string Base64 encoded string
 * @returns {number} Byte length
 */
base64.length = function length(string) {
    var p = string.length;
    if (!p)
        return 0;
    var n = 0;
    while (--p % 4 > 1 && string.charAt(p) === "=")
        ++n;
    return Math.ceil(string.length * 3) / 4 - n;
};

// Base64 encoding table
var b64 = new Array(64);

// Base64 decoding table
var s64 = new Array(123);

// 65..90, 97..122, 48..57, 43, 47
for (var i = 0; i < 64;)
    s64[b64[i] = i < 26 ? i + 65 : i < 52 ? i + 71 : i < 62 ? i - 4 : i - 59 | 43] = i++;

/**
 * Encodes a buffer to a base64 encoded string.
 * @param {Uint8Array} buffer Source buffer
 * @param {number} start Source start
 * @param {number} end Source end
 * @returns {string} Base64 encoded string
 */
base64.encode = function encode(buffer, start, end) {
    var parts = null,
        chunk = [];
    var i = 0, // output index
        j = 0, // goto index
        t;     // temporary
    while (start < end) {
        var b = buffer[start++];
        switch (j) {
            case 0:
                chunk[i++] = b64[b >> 2];
                t = (b & 3) << 4;
                j = 1;
                break;
            case 1:
                chunk[i++] = b64[t | b >> 4];
                t = (b & 15) << 2;
                j = 2;
                break;
            case 2:
                chunk[i++] = b64[t | b >> 6];
                chunk[i++] = b64[b & 63];
                j = 0;
                break;
        }
        if (i > 8191) {
            (parts || (parts = [])).push(String.fromCharCode.apply(String, chunk));
            i = 0;
        }
    }
    if (j) {
        chunk[i++] = b64[t];
        chunk[i++] = 61;
        if (j === 1)
            chunk[i++] = 61;
    }
    if (parts) {
        if (i)
            parts.push(String.fromCharCode.apply(String, chunk.slice(0, i)));
        return parts.join("");
    }
    return String.fromCharCode.apply(String, chunk.slice(0, i));
};

var invalidEncoding = "invalid encoding";

/**
 * Decodes a base64 encoded string to a buffer.
 * @param {string} string Source string
 * @param {Uint8Array} buffer Destination buffer
 * @param {number} offset Destination offset
 * @returns {number} Number of bytes written
 * @throws {Error} If encoding is invalid
 */
base64.decode = function decode(string, buffer, offset) {
    var start = offset;
    var j = 0, // goto index
        t;     // temporary
    for (var i = 0; i < string.length;) {
        var c = string.charCodeAt(i++);
        if (c === 61 && j > 1)
            break;
        if ((c = s64[c]) === undefined)
            throw Error(invalidEncoding);
        switch (j) {
            case 0:
                t = c;
                j = 1;
                break;
            case 1:
                buffer[offset++] = t << 2 | (c & 48) >> 4;
                t = c;
                j = 2;
                break;
            case 2:
                buffer[offset++] = (t & 15) << 4 | (c & 60) >> 2;
                t = c;
                j = 3;
                break;
            case 3:
                buffer[offset++] = (t & 3) << 6 | c;
                j = 0;
                break;
        }
    }
    if (j === 1)
        throw Error(invalidEncoding);
    return offset - start;
};

/**
 * Tests if the specified string appears to be base64 encoded.
 * @param {string} string String to test
 * @returns {boolean} `true` if probably base64 encoded, otherwise false
 */
base64.test = function test(string) {
    return /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(string);
};


/***/ }),
/* 10 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";

module.exports = EventEmitter;

/**
 * Constructs a new event emitter instance.
 * @classdesc A minimal event emitter.
 * @memberof util
 * @constructor
 */
function EventEmitter() {

    /**
     * Registered listeners.
     * @type {Object.<string,*>}
     * @private
     */
    this._listeners = {};
}

/**
 * Registers an event listener.
 * @param {string} evt Event name
 * @param {function} fn Listener
 * @param {*} [ctx] Listener context
 * @returns {util.EventEmitter} `this`
 */
EventEmitter.prototype.on = function on(evt, fn, ctx) {
    (this._listeners[evt] || (this._listeners[evt] = [])).push({
        fn  : fn,
        ctx : ctx || this
    });
    return this;
};

/**
 * Removes an event listener or any matching listeners if arguments are omitted.
 * @param {string} [evt] Event name. Removes all listeners if omitted.
 * @param {function} [fn] Listener to remove. Removes all listeners of `evt` if omitted.
 * @returns {util.EventEmitter} `this`
 */
EventEmitter.prototype.off = function off(evt, fn) {
    if (evt === undefined)
        this._listeners = {};
    else {
        if (fn === undefined)
            this._listeners[evt] = [];
        else {
            var listeners = this._listeners[evt];
            for (var i = 0; i < listeners.length;)
                if (listeners[i].fn === fn)
                    listeners.splice(i, 1);
                else
                    ++i;
        }
    }
    return this;
};

/**
 * Emits an event by calling its listeners with the specified arguments.
 * @param {string} evt Event name
 * @param {...*} args Arguments
 * @returns {util.EventEmitter} `this`
 */
EventEmitter.prototype.emit = function emit(evt) {
    var listeners = this._listeners[evt];
    if (listeners) {
        var args = [],
            i = 1;
        for (; i < arguments.length;)
            args.push(arguments[i++]);
        for (i = 0; i < listeners.length;)
            listeners[i].fn.apply(listeners[i++].ctx, args);
    }
    return this;
};


/***/ }),
/* 11 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


module.exports = factory(factory);

/**
 * Reads / writes floats / doubles from / to buffers.
 * @name util.float
 * @namespace
 */

/**
 * Writes a 32 bit float to a buffer using little endian byte order.
 * @name util.float.writeFloatLE
 * @function
 * @param {number} val Value to write
 * @param {Uint8Array} buf Target buffer
 * @param {number} pos Target buffer offset
 * @returns {undefined}
 */

/**
 * Writes a 32 bit float to a buffer using big endian byte order.
 * @name util.float.writeFloatBE
 * @function
 * @param {number} val Value to write
 * @param {Uint8Array} buf Target buffer
 * @param {number} pos Target buffer offset
 * @returns {undefined}
 */

/**
 * Reads a 32 bit float from a buffer using little endian byte order.
 * @name util.float.readFloatLE
 * @function
 * @param {Uint8Array} buf Source buffer
 * @param {number} pos Source buffer offset
 * @returns {number} Value read
 */

/**
 * Reads a 32 bit float from a buffer using big endian byte order.
 * @name util.float.readFloatBE
 * @function
 * @param {Uint8Array} buf Source buffer
 * @param {number} pos Source buffer offset
 * @returns {number} Value read
 */

/**
 * Writes a 64 bit double to a buffer using little endian byte order.
 * @name util.float.writeDoubleLE
 * @function
 * @param {number} val Value to write
 * @param {Uint8Array} buf Target buffer
 * @param {number} pos Target buffer offset
 * @returns {undefined}
 */

/**
 * Writes a 64 bit double to a buffer using big endian byte order.
 * @name util.float.writeDoubleBE
 * @function
 * @param {number} val Value to write
 * @param {Uint8Array} buf Target buffer
 * @param {number} pos Target buffer offset
 * @returns {undefined}
 */

/**
 * Reads a 64 bit double from a buffer using little endian byte order.
 * @name util.float.readDoubleLE
 * @function
 * @param {Uint8Array} buf Source buffer
 * @param {number} pos Source buffer offset
 * @returns {number} Value read
 */

/**
 * Reads a 64 bit double from a buffer using big endian byte order.
 * @name util.float.readDoubleBE
 * @function
 * @param {Uint8Array} buf Source buffer
 * @param {number} pos Source buffer offset
 * @returns {number} Value read
 */

// Factory function for the purpose of node-based testing in modified global environments
function factory(exports) {

    // float: typed array
    if (typeof Float32Array !== "undefined") (function() {

        var f32 = new Float32Array([ -0 ]),
            f8b = new Uint8Array(f32.buffer),
            le  = f8b[3] === 128;

        function writeFloat_f32_cpy(val, buf, pos) {
            f32[0] = val;
            buf[pos    ] = f8b[0];
            buf[pos + 1] = f8b[1];
            buf[pos + 2] = f8b[2];
            buf[pos + 3] = f8b[3];
        }

        function writeFloat_f32_rev(val, buf, pos) {
            f32[0] = val;
            buf[pos    ] = f8b[3];
            buf[pos + 1] = f8b[2];
            buf[pos + 2] = f8b[1];
            buf[pos + 3] = f8b[0];
        }

        /* istanbul ignore next */
        exports.writeFloatLE = le ? writeFloat_f32_cpy : writeFloat_f32_rev;
        /* istanbul ignore next */
        exports.writeFloatBE = le ? writeFloat_f32_rev : writeFloat_f32_cpy;

        function readFloat_f32_cpy(buf, pos) {
            f8b[0] = buf[pos    ];
            f8b[1] = buf[pos + 1];
            f8b[2] = buf[pos + 2];
            f8b[3] = buf[pos + 3];
            return f32[0];
        }

        function readFloat_f32_rev(buf, pos) {
            f8b[3] = buf[pos    ];
            f8b[2] = buf[pos + 1];
            f8b[1] = buf[pos + 2];
            f8b[0] = buf[pos + 3];
            return f32[0];
        }

        /* istanbul ignore next */
        exports.readFloatLE = le ? readFloat_f32_cpy : readFloat_f32_rev;
        /* istanbul ignore next */
        exports.readFloatBE = le ? readFloat_f32_rev : readFloat_f32_cpy;

    // float: ieee754
    })(); else (function() {

        function writeFloat_ieee754(writeUint, val, buf, pos) {
            var sign = val < 0 ? 1 : 0;
            if (sign)
                val = -val;
            if (val === 0)
                writeUint(1 / val > 0 ? /* positive */ 0 : /* negative 0 */ 2147483648, buf, pos);
            else if (isNaN(val))
                writeUint(2143289344, buf, pos);
            else if (val > 3.4028234663852886e+38) // +-Infinity
                writeUint((sign << 31 | 2139095040) >>> 0, buf, pos);
            else if (val < 1.1754943508222875e-38) // denormal
                writeUint((sign << 31 | Math.round(val / 1.401298464324817e-45)) >>> 0, buf, pos);
            else {
                var exponent = Math.floor(Math.log(val) / Math.LN2),
                    mantissa = Math.round(val * Math.pow(2, -exponent) * 8388608) & 8388607;
                writeUint((sign << 31 | exponent + 127 << 23 | mantissa) >>> 0, buf, pos);
            }
        }

        exports.writeFloatLE = writeFloat_ieee754.bind(null, writeUintLE);
        exports.writeFloatBE = writeFloat_ieee754.bind(null, writeUintBE);

        function readFloat_ieee754(readUint, buf, pos) {
            var uint = readUint(buf, pos),
                sign = (uint >> 31) * 2 + 1,
                exponent = uint >>> 23 & 255,
                mantissa = uint & 8388607;
            return exponent === 255
                ? mantissa
                ? NaN
                : sign * Infinity
                : exponent === 0 // denormal
                ? sign * 1.401298464324817e-45 * mantissa
                : sign * Math.pow(2, exponent - 150) * (mantissa + 8388608);
        }

        exports.readFloatLE = readFloat_ieee754.bind(null, readUintLE);
        exports.readFloatBE = readFloat_ieee754.bind(null, readUintBE);

    })();

    // double: typed array
    if (typeof Float64Array !== "undefined") (function() {

        var f64 = new Float64Array([-0]),
            f8b = new Uint8Array(f64.buffer),
            le  = f8b[7] === 128;

        function writeDouble_f64_cpy(val, buf, pos) {
            f64[0] = val;
            buf[pos    ] = f8b[0];
            buf[pos + 1] = f8b[1];
            buf[pos + 2] = f8b[2];
            buf[pos + 3] = f8b[3];
            buf[pos + 4] = f8b[4];
            buf[pos + 5] = f8b[5];
            buf[pos + 6] = f8b[6];
            buf[pos + 7] = f8b[7];
        }

        function writeDouble_f64_rev(val, buf, pos) {
            f64[0] = val;
            buf[pos    ] = f8b[7];
            buf[pos + 1] = f8b[6];
            buf[pos + 2] = f8b[5];
            buf[pos + 3] = f8b[4];
            buf[pos + 4] = f8b[3];
            buf[pos + 5] = f8b[2];
            buf[pos + 6] = f8b[1];
            buf[pos + 7] = f8b[0];
        }

        /* istanbul ignore next */
        exports.writeDoubleLE = le ? writeDouble_f64_cpy : writeDouble_f64_rev;
        /* istanbul ignore next */
        exports.writeDoubleBE = le ? writeDouble_f64_rev : writeDouble_f64_cpy;

        function readDouble_f64_cpy(buf, pos) {
            f8b[0] = buf[pos    ];
            f8b[1] = buf[pos + 1];
            f8b[2] = buf[pos + 2];
            f8b[3] = buf[pos + 3];
            f8b[4] = buf[pos + 4];
            f8b[5] = buf[pos + 5];
            f8b[6] = buf[pos + 6];
            f8b[7] = buf[pos + 7];
            return f64[0];
        }

        function readDouble_f64_rev(buf, pos) {
            f8b[7] = buf[pos    ];
            f8b[6] = buf[pos + 1];
            f8b[5] = buf[pos + 2];
            f8b[4] = buf[pos + 3];
            f8b[3] = buf[pos + 4];
            f8b[2] = buf[pos + 5];
            f8b[1] = buf[pos + 6];
            f8b[0] = buf[pos + 7];
            return f64[0];
        }

        /* istanbul ignore next */
        exports.readDoubleLE = le ? readDouble_f64_cpy : readDouble_f64_rev;
        /* istanbul ignore next */
        exports.readDoubleBE = le ? readDouble_f64_rev : readDouble_f64_cpy;

    // double: ieee754
    })(); else (function() {

        function writeDouble_ieee754(writeUint, off0, off1, val, buf, pos) {
            var sign = val < 0 ? 1 : 0;
            if (sign)
                val = -val;
            if (val === 0) {
                writeUint(0, buf, pos + off0);
                writeUint(1 / val > 0 ? /* positive */ 0 : /* negative 0 */ 2147483648, buf, pos + off1);
            } else if (isNaN(val)) {
                writeUint(0, buf, pos + off0);
                writeUint(2146959360, buf, pos + off1);
            } else if (val > 1.7976931348623157e+308) { // +-Infinity
                writeUint(0, buf, pos + off0);
                writeUint((sign << 31 | 2146435072) >>> 0, buf, pos + off1);
            } else {
                var mantissa;
                if (val < 2.2250738585072014e-308) { // denormal
                    mantissa = val / 5e-324;
                    writeUint(mantissa >>> 0, buf, pos + off0);
                    writeUint((sign << 31 | mantissa / 4294967296) >>> 0, buf, pos + off1);
                } else {
                    var exponent = Math.floor(Math.log(val) / Math.LN2);
                    if (exponent === 1024)
                        exponent = 1023;
                    mantissa = val * Math.pow(2, -exponent);
                    writeUint(mantissa * 4503599627370496 >>> 0, buf, pos + off0);
                    writeUint((sign << 31 | exponent + 1023 << 20 | mantissa * 1048576 & 1048575) >>> 0, buf, pos + off1);
                }
            }
        }

        exports.writeDoubleLE = writeDouble_ieee754.bind(null, writeUintLE, 0, 4);
        exports.writeDoubleBE = writeDouble_ieee754.bind(null, writeUintBE, 4, 0);

        function readDouble_ieee754(readUint, off0, off1, buf, pos) {
            var lo = readUint(buf, pos + off0),
                hi = readUint(buf, pos + off1);
            var sign = (hi >> 31) * 2 + 1,
                exponent = hi >>> 20 & 2047,
                mantissa = 4294967296 * (hi & 1048575) + lo;
            return exponent === 2047
                ? mantissa
                ? NaN
                : sign * Infinity
                : exponent === 0 // denormal
                ? sign * 5e-324 * mantissa
                : sign * Math.pow(2, exponent - 1075) * (mantissa + 4503599627370496);
        }

        exports.readDoubleLE = readDouble_ieee754.bind(null, readUintLE, 0, 4);
        exports.readDoubleBE = readDouble_ieee754.bind(null, readUintBE, 4, 0);

    })();

    return exports;
}

// uint helpers

function writeUintLE(val, buf, pos) {
    buf[pos    ] =  val        & 255;
    buf[pos + 1] =  val >>> 8  & 255;
    buf[pos + 2] =  val >>> 16 & 255;
    buf[pos + 3] =  val >>> 24;
}

function writeUintBE(val, buf, pos) {
    buf[pos    ] =  val >>> 24;
    buf[pos + 1] =  val >>> 16 & 255;
    buf[pos + 2] =  val >>> 8  & 255;
    buf[pos + 3] =  val        & 255;
}

function readUintLE(buf, pos) {
    return (buf[pos    ]
          | buf[pos + 1] << 8
          | buf[pos + 2] << 16
          | buf[pos + 3] << 24) >>> 0;
}

function readUintBE(buf, pos) {
    return (buf[pos    ] << 24
          | buf[pos + 1] << 16
          | buf[pos + 2] << 8
          | buf[pos + 3]) >>> 0;
}


/***/ }),
/* 12 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";

module.exports = inquire;

/**
 * Requires a module only if available.
 * @memberof util
 * @param {string} moduleName Module to require
 * @returns {?Object} Required module if available and not empty, otherwise `null`
 */
function inquire(moduleName) {
    try {
        var mod = eval("quire".replace(/^/,"re"))(moduleName); // eslint-disable-line no-eval
        if (mod && (mod.length || Object.keys(mod).length))
            return mod;
    } catch (e) {} // eslint-disable-line no-empty
    return null;
}


/***/ }),
/* 13 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


/**
 * A minimal UTF8 implementation for number arrays.
 * @memberof util
 * @namespace
 */
var utf8 = exports;

/**
 * Calculates the UTF8 byte length of a string.
 * @param {string} string String
 * @returns {number} Byte length
 */
utf8.length = function utf8_length(string) {
    var len = 0,
        c = 0;
    for (var i = 0; i < string.length; ++i) {
        c = string.charCodeAt(i);
        if (c < 128)
            len += 1;
        else if (c < 2048)
            len += 2;
        else if ((c & 0xFC00) === 0xD800 && (string.charCodeAt(i + 1) & 0xFC00) === 0xDC00) {
            ++i;
            len += 4;
        } else
            len += 3;
    }
    return len;
};

/**
 * Reads UTF8 bytes as a string.
 * @param {Uint8Array} buffer Source buffer
 * @param {number} start Source start
 * @param {number} end Source end
 * @returns {string} String read
 */
utf8.read = function utf8_read(buffer, start, end) {
    var len = end - start;
    if (len < 1)
        return "";
    var parts = null,
        chunk = [],
        i = 0, // char offset
        t;     // temporary
    while (start < end) {
        t = buffer[start++];
        if (t < 128)
            chunk[i++] = t;
        else if (t > 191 && t < 224)
            chunk[i++] = (t & 31) << 6 | buffer[start++] & 63;
        else if (t > 239 && t < 365) {
            t = ((t & 7) << 18 | (buffer[start++] & 63) << 12 | (buffer[start++] & 63) << 6 | buffer[start++] & 63) - 0x10000;
            chunk[i++] = 0xD800 + (t >> 10);
            chunk[i++] = 0xDC00 + (t & 1023);
        } else
            chunk[i++] = (t & 15) << 12 | (buffer[start++] & 63) << 6 | buffer[start++] & 63;
        if (i > 8191) {
            (parts || (parts = [])).push(String.fromCharCode.apply(String, chunk));
            i = 0;
        }
    }
    if (parts) {
        if (i)
            parts.push(String.fromCharCode.apply(String, chunk.slice(0, i)));
        return parts.join("");
    }
    return String.fromCharCode.apply(String, chunk.slice(0, i));
};

/**
 * Writes a string as UTF8 bytes.
 * @param {string} string Source string
 * @param {Uint8Array} buffer Destination buffer
 * @param {number} offset Destination offset
 * @returns {number} Bytes written
 */
utf8.write = function utf8_write(string, buffer, offset) {
    var start = offset,
        c1, // character 1
        c2; // character 2
    for (var i = 0; i < string.length; ++i) {
        c1 = string.charCodeAt(i);
        if (c1 < 128) {
            buffer[offset++] = c1;
        } else if (c1 < 2048) {
            buffer[offset++] = c1 >> 6       | 192;
            buffer[offset++] = c1       & 63 | 128;
        } else if ((c1 & 0xFC00) === 0xD800 && ((c2 = string.charCodeAt(i + 1)) & 0xFC00) === 0xDC00) {
            c1 = 0x10000 + ((c1 & 0x03FF) << 10) + (c2 & 0x03FF);
            ++i;
            buffer[offset++] = c1 >> 18      | 240;
            buffer[offset++] = c1 >> 12 & 63 | 128;
            buffer[offset++] = c1 >> 6  & 63 | 128;
            buffer[offset++] = c1       & 63 | 128;
        } else {
            buffer[offset++] = c1 >> 12      | 224;
            buffer[offset++] = c1 >> 6  & 63 | 128;
            buffer[offset++] = c1       & 63 | 128;
        }
    }
    return offset - start;
};


/***/ }),
/* 14 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";

module.exports = pool;

/**
 * An allocator as used by {@link util.pool}.
 * @typedef PoolAllocator
 * @type {function}
 * @param {number} size Buffer size
 * @returns {Uint8Array} Buffer
 */

/**
 * A slicer as used by {@link util.pool}.
 * @typedef PoolSlicer
 * @type {function}
 * @param {number} start Start offset
 * @param {number} end End offset
 * @returns {Uint8Array} Buffer slice
 * @this {Uint8Array}
 */

/**
 * A general purpose buffer pool.
 * @memberof util
 * @function
 * @param {PoolAllocator} alloc Allocator
 * @param {PoolSlicer} slice Slicer
 * @param {number} [size=8192] Slab size
 * @returns {PoolAllocator} Pooled allocator
 */
function pool(alloc, slice, size) {
    var SIZE   = size || 8192;
    var MAX    = SIZE >>> 1;
    var slab   = null;
    var offset = SIZE;
    return function pool_alloc(size) {
        if (size < 1 || size > MAX)
            return alloc(size);
        if (offset + size > SIZE) {
            slab = alloc(SIZE);
            offset = 0;
        }
        var buf = slice.call(slab, offset, offset += size);
        if (offset & 7) // align to 32 bit
            offset = (offset | 7) + 1;
        return buf;
    };
}


/***/ }),
/* 15 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";

module.exports = LongBits;

var util = __webpack_require__(0);

/**
 * Constructs new long bits.
 * @classdesc Helper class for working with the low and high bits of a 64 bit value.
 * @memberof util
 * @constructor
 * @param {number} lo Low 32 bits, unsigned
 * @param {number} hi High 32 bits, unsigned
 */
function LongBits(lo, hi) {

    // note that the casts below are theoretically unnecessary as of today, but older statically
    // generated converter code might still call the ctor with signed 32bits. kept for compat.

    /**
     * Low bits.
     * @type {number}
     */
    this.lo = lo >>> 0;

    /**
     * High bits.
     * @type {number}
     */
    this.hi = hi >>> 0;
}

/**
 * Zero bits.
 * @memberof util.LongBits
 * @type {util.LongBits}
 */
var zero = LongBits.zero = new LongBits(0, 0);

zero.toNumber = function() { return 0; };
zero.zzEncode = zero.zzDecode = function() { return this; };
zero.length = function() { return 1; };

/**
 * Zero hash.
 * @memberof util.LongBits
 * @type {string}
 */
var zeroHash = LongBits.zeroHash = "\0\0\0\0\0\0\0\0";

/**
 * Constructs new long bits from the specified number.
 * @param {number} value Value
 * @returns {util.LongBits} Instance
 */
LongBits.fromNumber = function fromNumber(value) {
    if (value === 0)
        return zero;
    var sign = value < 0;
    if (sign)
        value = -value;
    var lo = value >>> 0,
        hi = (value - lo) / 4294967296 >>> 0;
    if (sign) {
        hi = ~hi >>> 0;
        lo = ~lo >>> 0;
        if (++lo > 4294967295) {
            lo = 0;
            if (++hi > 4294967295)
                hi = 0;
        }
    }
    return new LongBits(lo, hi);
};

/**
 * Constructs new long bits from a number, long or string.
 * @param {Long|number|string} value Value
 * @returns {util.LongBits} Instance
 */
LongBits.from = function from(value) {
    if (typeof value === "number")
        return LongBits.fromNumber(value);
    if (util.isString(value)) {
        /* istanbul ignore else */
        if (util.Long)
            value = util.Long.fromString(value);
        else
            return LongBits.fromNumber(parseInt(value, 10));
    }
    return value.low || value.high ? new LongBits(value.low >>> 0, value.high >>> 0) : zero;
};

/**
 * Converts this long bits to a possibly unsafe JavaScript number.
 * @param {boolean} [unsigned=false] Whether unsigned or not
 * @returns {number} Possibly unsafe number
 */
LongBits.prototype.toNumber = function toNumber(unsigned) {
    if (!unsigned && this.hi >>> 31) {
        var lo = ~this.lo + 1 >>> 0,
            hi = ~this.hi     >>> 0;
        if (!lo)
            hi = hi + 1 >>> 0;
        return -(lo + hi * 4294967296);
    }
    return this.lo + this.hi * 4294967296;
};

/**
 * Converts this long bits to a long.
 * @param {boolean} [unsigned=false] Whether unsigned or not
 * @returns {Long} Long
 */
LongBits.prototype.toLong = function toLong(unsigned) {
    return util.Long
        ? new util.Long(this.lo | 0, this.hi | 0, Boolean(unsigned))
        /* istanbul ignore next */
        : { low: this.lo | 0, high: this.hi | 0, unsigned: Boolean(unsigned) };
};

var charCodeAt = String.prototype.charCodeAt;

/**
 * Constructs new long bits from the specified 8 characters long hash.
 * @param {string} hash Hash
 * @returns {util.LongBits} Bits
 */
LongBits.fromHash = function fromHash(hash) {
    if (hash === zeroHash)
        return zero;
    return new LongBits(
        ( charCodeAt.call(hash, 0)
        | charCodeAt.call(hash, 1) << 8
        | charCodeAt.call(hash, 2) << 16
        | charCodeAt.call(hash, 3) << 24) >>> 0
    ,
        ( charCodeAt.call(hash, 4)
        | charCodeAt.call(hash, 5) << 8
        | charCodeAt.call(hash, 6) << 16
        | charCodeAt.call(hash, 7) << 24) >>> 0
    );
};

/**
 * Converts this long bits to a 8 characters long hash.
 * @returns {string} Hash
 */
LongBits.prototype.toHash = function toHash() {
    return String.fromCharCode(
        this.lo        & 255,
        this.lo >>> 8  & 255,
        this.lo >>> 16 & 255,
        this.lo >>> 24      ,
        this.hi        & 255,
        this.hi >>> 8  & 255,
        this.hi >>> 16 & 255,
        this.hi >>> 24
    );
};

/**
 * Zig-zag encodes this long bits.
 * @returns {util.LongBits} `this`
 */
LongBits.prototype.zzEncode = function zzEncode() {
    var mask =   this.hi >> 31;
    this.hi  = ((this.hi << 1 | this.lo >>> 31) ^ mask) >>> 0;
    this.lo  = ( this.lo << 1                   ^ mask) >>> 0;
    return this;
};

/**
 * Zig-zag decodes this long bits.
 * @returns {util.LongBits} `this`
 */
LongBits.prototype.zzDecode = function zzDecode() {
    var mask = -(this.lo & 1);
    this.lo  = ((this.lo >>> 1 | this.hi << 31) ^ mask) >>> 0;
    this.hi  = ( this.hi >>> 1                  ^ mask) >>> 0;
    return this;
};

/**
 * Calculates the length of this longbits when encoded as a varint.
 * @returns {number} Length
 */
LongBits.prototype.length = function length() {
    var part0 =  this.lo,
        part1 = (this.lo >>> 28 | this.hi << 4) >>> 0,
        part2 =  this.hi >>> 24;
    return part2 === 0
         ? part1 === 0
           ? part0 < 16384
             ? part0 < 128 ? 1 : 2
             : part0 < 2097152 ? 3 : 4
           : part1 < 16384
             ? part1 < 128 ? 5 : 6
             : part1 < 2097152 ? 7 : 8
         : part2 < 128 ? 9 : 10;
};


/***/ }),
/* 16 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";

module.exports = BufferWriter;

// extends Writer
var Writer = __webpack_require__(1);
(BufferWriter.prototype = Object.create(Writer.prototype)).constructor = BufferWriter;

var util = __webpack_require__(0);

var Buffer = util.Buffer;

/**
 * Constructs a new buffer writer instance.
 * @classdesc Wire format writer using node buffers.
 * @extends Writer
 * @constructor
 */
function BufferWriter() {
    Writer.call(this);
}

/**
 * Allocates a buffer of the specified size.
 * @param {number} size Buffer size
 * @returns {Buffer} Buffer
 */
BufferWriter.alloc = function alloc_buffer(size) {
    return (BufferWriter.alloc = util._Buffer_allocUnsafe)(size);
};

var writeBytesBuffer = Buffer && Buffer.prototype instanceof Uint8Array && Buffer.prototype.set.name === "set"
    ? function writeBytesBuffer_set(val, buf, pos) {
        buf.set(val, pos); // faster than copy (requires node >= 4 where Buffers extend Uint8Array and set is properly inherited)
                           // also works for plain array values
    }
    /* istanbul ignore next */
    : function writeBytesBuffer_copy(val, buf, pos) {
        if (val.copy) // Buffer values
            val.copy(buf, pos, 0, val.length);
        else for (var i = 0; i < val.length;) // plain array values
            buf[pos++] = val[i++];
    };

/**
 * @override
 */
BufferWriter.prototype.bytes = function write_bytes_buffer(value) {
    if (util.isString(value))
        value = util._Buffer_from(value, "base64");
    var len = value.length >>> 0;
    this.uint32(len);
    if (len)
        this._push(writeBytesBuffer, len, value);
    return this;
};

function writeStringBuffer(val, buf, pos) {
    if (val.length < 40) // plain js is faster for short strings (probably due to redundant assertions)
        util.utf8.write(val, buf, pos);
    else
        buf.utf8Write(val, pos);
}

/**
 * @override
 */
BufferWriter.prototype.string = function write_string_buffer(value) {
    var len = Buffer.byteLength(value);
    this.uint32(len);
    if (len)
        this._push(writeStringBuffer, len, value);
    return this;
};


/**
 * Finishes the write operation.
 * @name BufferWriter#finish
 * @function
 * @returns {Buffer} Finished buffer
 */


/***/ }),
/* 17 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";

module.exports = BufferReader;

// extends Reader
var Reader = __webpack_require__(2);
(BufferReader.prototype = Object.create(Reader.prototype)).constructor = BufferReader;

var util = __webpack_require__(0);

/**
 * Constructs a new buffer reader instance.
 * @classdesc Wire format reader using node buffers.
 * @extends Reader
 * @constructor
 * @param {Buffer} buffer Buffer to read from
 */
function BufferReader(buffer) {
    Reader.call(this, buffer);

    /**
     * Read buffer.
     * @name BufferReader#buf
     * @type {Buffer}
     */
}

/* istanbul ignore else */
if (util.Buffer)
    BufferReader.prototype._slice = util.Buffer.prototype.slice;

/**
 * @override
 */
BufferReader.prototype.string = function read_string_buffer() {
    var len = this.uint32(); // modifies pos
    return this.buf.utf8Slice(this.pos, this.pos = Math.min(this.pos + len, this.len));
};

/**
 * Reads a sequence of bytes preceeded by its length as a varint.
 * @name BufferReader#bytes
 * @function
 * @returns {Buffer} Value read
 */


/***/ }),
/* 18 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


/**
 * Streaming RPC helpers.
 * @namespace
 */
var rpc = exports;

/**
 * RPC implementation passed to {@link Service#create} performing a service request on network level, i.e. by utilizing http requests or websockets.
 * @typedef RPCImpl
 * @type {function}
 * @param {Method|rpc.ServiceMethod<Message<{}>,Message<{}>>} method Reflected or static method being called
 * @param {Uint8Array} requestData Request data
 * @param {RPCImplCallback} callback Callback function
 * @returns {undefined}
 * @example
 * function rpcImpl(method, requestData, callback) {
 *     if (protobuf.util.lcFirst(method.name) !== "myMethod") // compatible with static code
 *         throw Error("no such method");
 *     asynchronouslyObtainAResponse(requestData, function(err, responseData) {
 *         callback(err, responseData);
 *     });
 * }
 */

/**
 * Node-style callback as used by {@link RPCImpl}.
 * @typedef RPCImplCallback
 * @type {function}
 * @param {Error|null} error Error, if any, otherwise `null`
 * @param {Uint8Array|null} [response] Response data or `null` to signal end of stream, if there hasn't been an error
 * @returns {undefined}
 */

rpc.Service = __webpack_require__(19);


/***/ }),
/* 19 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";

module.exports = Service;

var util = __webpack_require__(0);

// Extends EventEmitter
(Service.prototype = Object.create(util.EventEmitter.prototype)).constructor = Service;

/**
 * A service method callback as used by {@link rpc.ServiceMethod|ServiceMethod}.
 *
 * Differs from {@link RPCImplCallback} in that it is an actual callback of a service method which may not return `response = null`.
 * @typedef rpc.ServiceMethodCallback
 * @template TRes extends Message<TRes>
 * @type {function}
 * @param {Error|null} error Error, if any
 * @param {TRes} [response] Response message
 * @returns {undefined}
 */

/**
 * A service method part of a {@link rpc.Service} as created by {@link Service.create}.
 * @typedef rpc.ServiceMethod
 * @template TReq extends Message<TReq>
 * @template TRes extends Message<TRes>
 * @type {function}
 * @param {TReq|Properties<TReq>} request Request message or plain object
 * @param {rpc.ServiceMethodCallback<TRes>} [callback] Node-style callback called with the error, if any, and the response message
 * @returns {Promise<Message<TRes>>} Promise if `callback` has been omitted, otherwise `undefined`
 */

/**
 * Constructs a new RPC service instance.
 * @classdesc An RPC service as returned by {@link Service#create}.
 * @exports rpc.Service
 * @extends util.EventEmitter
 * @constructor
 * @param {RPCImpl} rpcImpl RPC implementation
 * @param {boolean} [requestDelimited=false] Whether requests are length-delimited
 * @param {boolean} [responseDelimited=false] Whether responses are length-delimited
 */
function Service(rpcImpl, requestDelimited, responseDelimited) {

    if (typeof rpcImpl !== "function")
        throw TypeError("rpcImpl must be a function");

    util.EventEmitter.call(this);

    /**
     * RPC implementation. Becomes `null` once the service is ended.
     * @type {RPCImpl|null}
     */
    this.rpcImpl = rpcImpl;

    /**
     * Whether requests are length-delimited.
     * @type {boolean}
     */
    this.requestDelimited = Boolean(requestDelimited);

    /**
     * Whether responses are length-delimited.
     * @type {boolean}
     */
    this.responseDelimited = Boolean(responseDelimited);
}

/**
 * Calls a service method through {@link rpc.Service#rpcImpl|rpcImpl}.
 * @param {Method|rpc.ServiceMethod<TReq,TRes>} method Reflected or static method
 * @param {Constructor<TReq>} requestCtor Request constructor
 * @param {Constructor<TRes>} responseCtor Response constructor
 * @param {TReq|Properties<TReq>} request Request message or plain object
 * @param {rpc.ServiceMethodCallback<TRes>} callback Service callback
 * @returns {undefined}
 * @template TReq extends Message<TReq>
 * @template TRes extends Message<TRes>
 */
Service.prototype.rpcCall = function rpcCall(method, requestCtor, responseCtor, request, callback) {

    if (!request)
        throw TypeError("request must be specified");

    var self = this;
    if (!callback)
        return util.asPromise(rpcCall, self, method, requestCtor, responseCtor, request);

    if (!self.rpcImpl) {
        setTimeout(function() { callback(Error("already ended")); }, 0);
        return undefined;
    }

    try {
        return self.rpcImpl(
            method,
            requestCtor[self.requestDelimited ? "encodeDelimited" : "encode"](request).finish(),
            function rpcCallback(err, response) {

                if (err) {
                    self.emit("error", err, method);
                    return callback(err);
                }

                if (response === null) {
                    self.end(/* endedByRPC */ true);
                    return undefined;
                }

                if (!(response instanceof responseCtor)) {
                    try {
                        response = responseCtor[self.responseDelimited ? "decodeDelimited" : "decode"](response);
                    } catch (err) {
                        self.emit("error", err, method);
                        return callback(err);
                    }
                }

                self.emit("data", response, method);
                return callback(null, response);
            }
        );
    } catch (err) {
        self.emit("error", err, method);
        setTimeout(function() { callback(err); }, 0);
        return undefined;
    }
};

/**
 * Ends this service and emits the `end` event.
 * @param {boolean} [endedByRPC=false] Whether the service has been ended by the RPC implementation.
 * @returns {rpc.Service} `this`
 */
Service.prototype.end = function end(endedByRPC) {
    if (this.rpcImpl) {
        if (!endedByRPC) // signal end to rpcImpl
            this.rpcImpl(null, null, null);
        this.rpcImpl = null;
        this.emit("end").off();
    }
    return this;
};


/***/ }),
/* 20 */
/***/ (function(module, exports, __webpack_require__) {

"use strict";

module.exports = {};

/**
 * Named roots.
 * This is where pbjs stores generated structures (the option `-r, --root` specifies a name).
 * Can also be used manually to make roots available accross modules.
 * @name roots
 * @type {Object.<string,Root>}
 * @example
 * // pbjs -r myroot -o compiled.js ...
 *
 * // in another module:
 * require("./compiled.js");
 *
 * // in any subsequent module:
 * var root = protobuf.roots["myroot"];
 */


/***/ })
/******/ ]);
//# sourceMappingURL=0917d06898f782e2ccf6.worker.js.map