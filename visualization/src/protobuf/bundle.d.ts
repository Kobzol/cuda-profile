import * as $protobuf from "protobufjs";

/** Namespace cupr. */
export namespace cupr {

    /** Namespace proto. */
    namespace proto {

        /** Properties of an AllocRecord. */
        interface IAllocRecord {

            /** AllocRecord address */
            address: string;

            /** AllocRecord size */
            size: number;

            /** AllocRecord elementSize */
            elementSize: number;

            /** AllocRecord space */
            space: number;

            /** AllocRecord typeIndex */
            typeIndex?: (number|null);

            /** AllocRecord typeString */
            typeString?: (string|null);

            /** AllocRecord nameIndex */
            nameIndex?: (number|null);

            /** AllocRecord nameString */
            nameString?: (string|null);

            /** AllocRecord location */
            location: string;

            /** AllocRecord active */
            active: boolean;
        }

        /** Represents an AllocRecord. */
        class AllocRecord implements IAllocRecord {

            /**
             * Constructs a new AllocRecord.
             * @param [properties] Properties to set
             */
            constructor(properties?: cupr.proto.IAllocRecord);

            /** AllocRecord address. */
            public address: string;

            /** AllocRecord size. */
            public size: number;

            /** AllocRecord elementSize. */
            public elementSize: number;

            /** AllocRecord space. */
            public space: number;

            /** AllocRecord typeIndex. */
            public typeIndex: number;

            /** AllocRecord typeString. */
            public typeString: string;

            /** AllocRecord nameIndex. */
            public nameIndex: number;

            /** AllocRecord nameString. */
            public nameString: string;

            /** AllocRecord location. */
            public location: string;

            /** AllocRecord active. */
            public active: boolean;

            /** AllocRecord type. */
            public type?: ("typeIndex"|"typeString");

            /** AllocRecord name. */
            public name?: ("nameIndex"|"nameString");

            /**
             * Creates a new AllocRecord instance using the specified properties.
             * @param [properties] Properties to set
             * @returns AllocRecord instance
             */
            public static create(properties?: cupr.proto.IAllocRecord): cupr.proto.AllocRecord;

            /**
             * Encodes the specified AllocRecord message. Does not implicitly {@link cupr.proto.AllocRecord.verify|verify} messages.
             * @param message AllocRecord message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: cupr.proto.IAllocRecord, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified AllocRecord message, length delimited. Does not implicitly {@link cupr.proto.AllocRecord.verify|verify} messages.
             * @param message AllocRecord message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: cupr.proto.IAllocRecord, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes an AllocRecord message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns AllocRecord
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): cupr.proto.AllocRecord;

            /**
             * Decodes an AllocRecord message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns AllocRecord
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): cupr.proto.AllocRecord;

            /**
             * Verifies an AllocRecord message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates an AllocRecord message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns AllocRecord
             */
            public static fromObject(object: { [k: string]: any }): cupr.proto.AllocRecord;

            /**
             * Creates a plain object from an AllocRecord message. Also converts values to other types if specified.
             * @param message AllocRecord
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: cupr.proto.AllocRecord, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this AllocRecord to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
        }

        /** Properties of a Dim3. */
        interface IDim3 {

            /** Dim3 x */
            x: number;

            /** Dim3 y */
            y: number;

            /** Dim3 z */
            z: number;
        }

        /** Represents a Dim3. */
        class Dim3 implements IDim3 {

            /**
             * Constructs a new Dim3.
             * @param [properties] Properties to set
             */
            constructor(properties?: cupr.proto.IDim3);

            /** Dim3 x. */
            public x: number;

            /** Dim3 y. */
            public y: number;

            /** Dim3 z. */
            public z: number;

            /**
             * Creates a new Dim3 instance using the specified properties.
             * @param [properties] Properties to set
             * @returns Dim3 instance
             */
            public static create(properties?: cupr.proto.IDim3): cupr.proto.Dim3;

            /**
             * Encodes the specified Dim3 message. Does not implicitly {@link cupr.proto.Dim3.verify|verify} messages.
             * @param message Dim3 message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: cupr.proto.IDim3, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified Dim3 message, length delimited. Does not implicitly {@link cupr.proto.Dim3.verify|verify} messages.
             * @param message Dim3 message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: cupr.proto.IDim3, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a Dim3 message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns Dim3
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): cupr.proto.Dim3;

            /**
             * Decodes a Dim3 message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns Dim3
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): cupr.proto.Dim3;

            /**
             * Verifies a Dim3 message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a Dim3 message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns Dim3
             */
            public static fromObject(object: { [k: string]: any }): cupr.proto.Dim3;

            /**
             * Creates a plain object from a Dim3 message. Also converts values to other types if specified.
             * @param message Dim3
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: cupr.proto.Dim3, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this Dim3 to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
        }

        /** Properties of a KernelTrace. */
        interface IKernelTrace {

            /** KernelTrace warps */
            warps?: (cupr.proto.IWarp[]|null);

            /** KernelTrace allocations */
            allocations?: (cupr.proto.IAllocRecord[]|null);

            /** KernelTrace kernel */
            kernel: string;

            /** KernelTrace start */
            start: number;

            /** KernelTrace end */
            end: number;

            /** KernelTrace type */
            type: string;

            /** KernelTrace gridDim */
            gridDim: cupr.proto.IDim3;

            /** KernelTrace blockDim */
            blockDim: cupr.proto.IDim3;

            /** KernelTrace warpSize */
            warpSize: number;

            /** KernelTrace bankSize */
            bankSize: number;
        }

        /** Represents a KernelTrace. */
        class KernelTrace implements IKernelTrace {

            /**
             * Constructs a new KernelTrace.
             * @param [properties] Properties to set
             */
            constructor(properties?: cupr.proto.IKernelTrace);

            /** KernelTrace warps. */
            public warps: cupr.proto.IWarp[];

            /** KernelTrace allocations. */
            public allocations: cupr.proto.IAllocRecord[];

            /** KernelTrace kernel. */
            public kernel: string;

            /** KernelTrace start. */
            public start: number;

            /** KernelTrace end. */
            public end: number;

            /** KernelTrace type. */
            public type: string;

            /** KernelTrace gridDim. */
            public gridDim: cupr.proto.IDim3;

            /** KernelTrace blockDim. */
            public blockDim: cupr.proto.IDim3;

            /** KernelTrace warpSize. */
            public warpSize: number;

            /** KernelTrace bankSize. */
            public bankSize: number;

            /**
             * Creates a new KernelTrace instance using the specified properties.
             * @param [properties] Properties to set
             * @returns KernelTrace instance
             */
            public static create(properties?: cupr.proto.IKernelTrace): cupr.proto.KernelTrace;

            /**
             * Encodes the specified KernelTrace message. Does not implicitly {@link cupr.proto.KernelTrace.verify|verify} messages.
             * @param message KernelTrace message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: cupr.proto.IKernelTrace, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified KernelTrace message, length delimited. Does not implicitly {@link cupr.proto.KernelTrace.verify|verify} messages.
             * @param message KernelTrace message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: cupr.proto.IKernelTrace, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a KernelTrace message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns KernelTrace
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): cupr.proto.KernelTrace;

            /**
             * Decodes a KernelTrace message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns KernelTrace
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): cupr.proto.KernelTrace;

            /**
             * Verifies a KernelTrace message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a KernelTrace message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns KernelTrace
             */
            public static fromObject(object: { [k: string]: any }): cupr.proto.KernelTrace;

            /**
             * Creates a plain object from a KernelTrace message. Also converts values to other types if specified.
             * @param message KernelTrace
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: cupr.proto.KernelTrace, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this KernelTrace to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
        }

        /** Properties of a MemoryAccess. */
        interface IMemoryAccess {

            /** MemoryAccess threadIdx */
            threadIdx: cupr.proto.IDim3;

            /** MemoryAccess address */
            address: string;

            /** MemoryAccess value */
            value: string;
        }

        /** Represents a MemoryAccess. */
        class MemoryAccess implements IMemoryAccess {

            /**
             * Constructs a new MemoryAccess.
             * @param [properties] Properties to set
             */
            constructor(properties?: cupr.proto.IMemoryAccess);

            /** MemoryAccess threadIdx. */
            public threadIdx: cupr.proto.IDim3;

            /** MemoryAccess address. */
            public address: string;

            /** MemoryAccess value. */
            public value: string;

            /**
             * Creates a new MemoryAccess instance using the specified properties.
             * @param [properties] Properties to set
             * @returns MemoryAccess instance
             */
            public static create(properties?: cupr.proto.IMemoryAccess): cupr.proto.MemoryAccess;

            /**
             * Encodes the specified MemoryAccess message. Does not implicitly {@link cupr.proto.MemoryAccess.verify|verify} messages.
             * @param message MemoryAccess message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: cupr.proto.IMemoryAccess, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified MemoryAccess message, length delimited. Does not implicitly {@link cupr.proto.MemoryAccess.verify|verify} messages.
             * @param message MemoryAccess message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: cupr.proto.IMemoryAccess, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a MemoryAccess message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns MemoryAccess
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): cupr.proto.MemoryAccess;

            /**
             * Decodes a MemoryAccess message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns MemoryAccess
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): cupr.proto.MemoryAccess;

            /**
             * Verifies a MemoryAccess message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a MemoryAccess message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns MemoryAccess
             */
            public static fromObject(object: { [k: string]: any }): cupr.proto.MemoryAccess;

            /**
             * Creates a plain object from a MemoryAccess message. Also converts values to other types if specified.
             * @param message MemoryAccess
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: cupr.proto.MemoryAccess, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this MemoryAccess to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
        }

        /** Properties of a Warp. */
        interface IWarp {

            /** Warp accesses */
            accesses?: (cupr.proto.IMemoryAccess[]|null);

            /** Warp blockIdx */
            blockIdx: cupr.proto.IDim3;

            /** Warp warpId */
            warpId: number;

            /** Warp debugId */
            debugId: number;

            /** Warp size */
            size: number;

            /** Warp kind */
            kind: number;

            /** Warp space */
            space: number;

            /** Warp typeIndex */
            typeIndex: number;

            /** Warp timestamp */
            timestamp: string;
        }

        /** Represents a Warp. */
        class Warp implements IWarp {

            /**
             * Constructs a new Warp.
             * @param [properties] Properties to set
             */
            constructor(properties?: cupr.proto.IWarp);

            /** Warp accesses. */
            public accesses: cupr.proto.IMemoryAccess[];

            /** Warp blockIdx. */
            public blockIdx: cupr.proto.IDim3;

            /** Warp warpId. */
            public warpId: number;

            /** Warp debugId. */
            public debugId: number;

            /** Warp size. */
            public size: number;

            /** Warp kind. */
            public kind: number;

            /** Warp space. */
            public space: number;

            /** Warp typeIndex. */
            public typeIndex: number;

            /** Warp timestamp. */
            public timestamp: string;

            /**
             * Creates a new Warp instance using the specified properties.
             * @param [properties] Properties to set
             * @returns Warp instance
             */
            public static create(properties?: cupr.proto.IWarp): cupr.proto.Warp;

            /**
             * Encodes the specified Warp message. Does not implicitly {@link cupr.proto.Warp.verify|verify} messages.
             * @param message Warp message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: cupr.proto.IWarp, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified Warp message, length delimited. Does not implicitly {@link cupr.proto.Warp.verify|verify} messages.
             * @param message Warp message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: cupr.proto.IWarp, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a Warp message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns Warp
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): cupr.proto.Warp;

            /**
             * Decodes a Warp message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns Warp
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): cupr.proto.Warp;

            /**
             * Verifies a Warp message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a Warp message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns Warp
             */
            public static fromObject(object: { [k: string]: any }): cupr.proto.Warp;

            /**
             * Creates a plain object from a Warp message. Also converts values to other types if specified.
             * @param message Warp
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: cupr.proto.Warp, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this Warp to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };
        }
    }
}
