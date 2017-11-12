/*eslint-disable block-scoped-var, no-redeclare, no-control-regex, no-prototype-builtins*/
(function($protobuf) {
    "use strict";

    // Common aliases
    var $Reader = $protobuf.Reader, $Writer = $protobuf.Writer, $util = $protobuf.util;
    
    // Exported root namespace
    var $root = $protobuf.roots["default"] || ($protobuf.roots["default"] = {});
    
    $root.cupr = (function() {
    
        /**
         * Namespace cupr.
         * @exports cupr
         * @namespace
         */
        var cupr = {};
    
        cupr.proto = (function() {
    
            /**
             * Namespace proto.
             * @memberof cupr
             * @namespace
             */
            var proto = {};
    
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
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
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
                var $oneOfFields;
    
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
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.cupr.proto.AllocRecord();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                    var properties = {};
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
                    var message = new $root.cupr.proto.AllocRecord();
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
                    var object = {};
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
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
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
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.cupr.proto.Dim3();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                    var message = new $root.cupr.proto.Dim3();
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
                    var object = {};
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
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
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
                        for (var i = 0; i < message.accesses.length; ++i)
                            $root.cupr.proto.MemoryAccess.encode(message.accesses[i], writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
                    if (message.allocations != null && message.allocations.length)
                        for (var i = 0; i < message.allocations.length; ++i)
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
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.cupr.proto.KernelTrace();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                        for (var i = 0; i < message.accesses.length; ++i) {
                            var error = $root.cupr.proto.MemoryAccess.verify(message.accesses[i]);
                            if (error)
                                return "accesses." + error;
                        }
                    }
                    if (message.allocations != null && message.hasOwnProperty("allocations")) {
                        if (!Array.isArray(message.allocations))
                            return "allocations: array expected";
                        for (var i = 0; i < message.allocations.length; ++i) {
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
                    var message = new $root.cupr.proto.KernelTrace();
                    if (object.accesses) {
                        if (!Array.isArray(object.accesses))
                            throw TypeError(".cupr.proto.KernelTrace.accesses: array expected");
                        message.accesses = [];
                        for (var i = 0; i < object.accesses.length; ++i) {
                            if (typeof object.accesses[i] !== "object")
                                throw TypeError(".cupr.proto.KernelTrace.accesses: object expected");
                            message.accesses[i] = $root.cupr.proto.MemoryAccess.fromObject(object.accesses[i]);
                        }
                    }
                    if (object.allocations) {
                        if (!Array.isArray(object.allocations))
                            throw TypeError(".cupr.proto.KernelTrace.allocations: array expected");
                        message.allocations = [];
                        for (var i = 0; i < object.allocations.length; ++i) {
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
                    var object = {};
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
                        for (var j = 0; j < message.accesses.length; ++j)
                            object.accesses[j] = $root.cupr.proto.MemoryAccess.toObject(message.accesses[j], options);
                    }
                    if (message.allocations && message.allocations.length) {
                        object.allocations = [];
                        for (var j = 0; j < message.allocations.length; ++j)
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
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
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
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.cupr.proto.MemoryAccess();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                    var error = $root.cupr.proto.Dim3.verify(message.threadIdx);
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
                    var message = new $root.cupr.proto.MemoryAccess();
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
                    var object = {};
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
                            var long = new $util.Long(0, 0, false);
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
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return MemoryAccess;
            })();
    
            return proto;
        })();
    
        return cupr;
    })();

    return $root;
})(protobuf);
